# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Train the next-piece model on Metal through MLX.

PyTorch's compiler cannot build the backward pass for this loop on this Mac.
MLX compiles the same Kuramoto step into one Metal graph, so the 16 GPU cores
spend their time on the spin instead of on a kernel launch per piece. Each
row is one story. A story longer than the window continues in the next
window with the same wheels. Kicking the end piece returns the wheels to rest.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import contextlib
import json
import math
from pathlib import Path
import queue
import re
import threading
import time

from mlx import nn
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map, tree_unflatten
import torch

from sidecar.beats import _content_word
from sidecar.model import (
    CHARACTER_DRIVE_SCALE,
    N_MEMORIES,
    RECOGNITION_GAMMA,
    RESOLUTION_DRIVE_SCALE,
    ROUTE_COUNT_KEEP,
    STICKY_DRIVE_SCALE,
    WRITE_ALIGN,
    _clock_split,
)
from sidecar.prepare import EOS, case_groups, chunk_ids
from sidecar.train import (
    DYNAMICS_LR_SCALE,
    GLUE_TARGET_WEIGHT,
    HEAVY_DT_SCALE,
    HEAVY_PULL,
    LATCH,
    LATCH_BLEND,
    N_ARC,
    N_OSCILLATORS,
    N_ROOTS,
    N_SLOW,
    N_STAMP,
    RANK,
    READOUT_RANK,
    STORY_WINDOW,
    StoryBatch,
    StoryLabels,
    keep_short_stories,
    label_shelf,
    limit_stories,
    load_corpus,
    prefetch_batches,
    sentence_end_ids,
    story_epoch_steps,
)

_PEAK_LR = 1e-3
_LOG_SECONDS = 15.0
_SAVE_EVERY = 10
_CATEGORY_PREVIEW = 8
# Each stage walks this many kicks. The credit, split into a direction and a
# scale, is handed to the stage behind it so the ending can still teach the
# opening. The 64-kick cut dropped that handoff. This one keeps it.
_CREDIT_SPAN = 32
_SCALE_CAP = 1e300


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help=(
            "Stop after this many seconds. 0 runs until the loss flattens. "
            "A numbered bag is written every 10 epochs, plus the best bag and the final epoch."
        ),
    )
    parser.add_argument(
        "--stories",
        type=int,
        default=None,
        help="Use only the first N packed stories. Default is the whole shelf.",
    )
    parser.add_argument(
        "--all-stories",
        action="store_true",
        help=(
            "Train every story, including ones longer than the window. "
            "A long story continues in the next window. The short shelf stays the default."
        ),
    )
    parser.add_argument(
        "--target-loss",
        type=float,
        default=None,
        help="Stop early once an epoch's average loss is at or below this value.",
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=STORY_WINDOW)
    parser.add_argument("--n-oscillators", type=int, default=N_OSCILLATORS)
    parser.add_argument("--rank", type=int, default=RANK)
    parser.add_argument("--n-roots", type=int, default=N_ROOTS)
    parser.add_argument("--n-slow", type=int, default=N_SLOW)
    parser.add_argument("--n-stamp", type=int, default=N_STAMP)
    parser.add_argument("--n-arc", type=int, default=N_ARC)
    parser.add_argument("--latch", action=argparse.BooleanOptionalAction, default=LATCH)
    parser.add_argument("--character-scale", type=float, default=CHARACTER_DRIVE_SCALE)
    parser.add_argument("--dilemma-scale", type=float, default=STICKY_DRIVE_SCALE)
    parser.add_argument("--resolution-scale", type=float, default=RESOLUTION_DRIVE_SCALE)
    parser.add_argument("--latch-blend", type=float, default=LATCH_BLEND)
    parser.add_argument("--heavy-pull", type=float, default=HEAVY_PULL)
    parser.add_argument("--heavy-dt-scale", type=float, default=HEAVY_DT_SCALE)
    parser.add_argument(
        "--plateau-delta",
        type=float,
        default=0.01,
        help="An epoch must beat the best loss by this much to count as progress.",
    )
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=6,
        help="Stop after this many epochs in a row fail to beat the best loss.",
    )
    parser.add_argument("--lr", type=float, default=_PEAK_LR)
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Keep training from a saved bag. Earlier passes stay on disk.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--sample-length", type=int, default=400)
    parser.add_argument("--sample-temperature", type=float, default=0.2)
    return parser


class KuramotoMLX(nn.Module):
    """Low-rank Kuramoto bag. One Euler kick per piece."""

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        *,
        n: int = N_OSCILLATORS,
        rank: int = RANK,
        n_roots: int = N_ROOTS,
        readout_rank: int = 0,
        n_slow: int = N_SLOW,
        n_stamp: int = N_STAMP,
        n_arc: int = N_ARC,
        heavy_pull: float = HEAVY_PULL,
        heavy_dt_scale: float = HEAVY_DT_SCALE,
        boundary_ids: tuple[int, ...] = (),
        latch: bool = LATCH,
        latch_blend: float = LATCH_BLEND,
        eos_id: int | None = None,
        beat_canon: list[int] | None = None,
    ) -> None:
        """Build the bag. The slow clock creeps. The arc wheel waits for the last sentence."""
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n = int(n)
        self.n_slow = int(n_slow)
        self.n_stamp = int(n_stamp)
        self.n_arc = int(n_arc)
        self.heavy_pull = float(heavy_pull)
        self.heavy_dt_scale = float(heavy_dt_scale)
        self.latch = bool(latch)
        self.latch_blend = float(latch_blend)
        self.latch_at = self.n - 1 if self.latch else -1
        self.eos_id = None if eos_id is None else int(eos_id)
        self.boundary_ids = tuple(int(piece) for piece in boundary_ids)
        self.dt = 0.2
        self._install_clocks()
        scale = self.n**-0.5
        self.omega = 0.01 * mx.random.normal((self.n,))
        self.U = scale * mx.random.normal((self.n, rank))
        self.V = scale * mx.random.normal((self.n, rank))
        self.root_phase = nn.Embedding(self.vocab_size, n_roots)
        self.D = 0.1 * mx.random.normal((self.n, n_roots))
        self.theta0 = mx.zeros((self.n,))
        # Zeros match a bag saved before the catalog existed. ``scatter_route``
        # splits them for a new run. Tied patterns would share one gradient.
        # Off until ``scatter_route`` or a bag that actually stored ``K``.
        # A zero catalog is the angle 0, and the nudge would pull every wheel
        # there. Tests and old bags keep the plain tug.
        self.use_route = False
        # The pose spring and the stacked per-pose couplings are not the paper.
        # They held the loss near 5, so they stay out of the step for this run.
        self.route_extras = False
        self.patterns = mx.zeros((N_MEMORIES, self.n))
        self.route_K = mx.zeros((N_MEMORIES, N_MEMORIES))
        # One low-rank coupling per stored pose. Zeros add nothing, so a bag
        # from before this map still steps under U and V alone.
        self.mem_U = mx.zeros((N_MEMORIES, self.n, rank))
        self.mem_V = mx.zeros((N_MEMORIES, self.n, rank))
        # Slot 0 is the rest pose. Later slots open when a kick reaches them.
        self.catalog_filled = 1
        self._trace = None
        self.readout_rank = int(readout_rank)
        if self.readout_rank > 0:
            self.readout_down = nn.Linear(2 * self.n, self.readout_rank, bias=False)
            self.readout_up = nn.Linear(self.readout_rank, self.vocab_size)
        else:
            self.readout = nn.Linear(2 * self.n, self.vocab_size)
        canon_ids = list(range(self.vocab_size)) if beat_canon is None else list(beat_canon)
        self._beat_canon = mx.array(canon_ids)
        self.character_scale = CHARACTER_DRIVE_SCALE
        self.dilemma_scale = STICKY_DRIVE_SCALE
        self.resolution_scale = RESOLUTION_DRIVE_SCALE

    def _install_clocks(self) -> None:
        """Lay the slow clock, the stamp clock, and the arc wheel into the bag."""
        self.slow, self.stamp, self.arc = _clock_split(
            self.n, self.n_slow, self.n_stamp, self.n_arc, latch=self.latch
        )
        if not self.n_slow and not self.n_stamp and not self.n_arc and not self.latch:
            return
        self._fill_clock_buffers()

    def _fill_clock_buffers(self) -> None:
        """Source gain and resting step for every wheel the clocks set aside."""
        bands = (
            (self.slow.start, 1.0, self.dt),
            (self.n_slow, self.heavy_pull, self.dt * self.heavy_dt_scale),
            (self.n_stamp, self.heavy_pull, 0.0),
            (self.n_arc, self.heavy_pull, 0.0),
        )
        gain_parts: list[mx.array] = []
        step_parts: list[mx.array] = []
        for count, gain, step in bands:
            if count:
                gain_parts.append(mx.full((count,), gain))
                step_parts.append(mx.full((count,), step))
        if self.latch:
            covered = self.slow.start + self.n_slow + self.n_stamp + self.n_arc
            middle = self.n - 1 - covered
            if middle:
                gain_parts.append(mx.ones((middle,)))
                step_parts.append(mx.full((middle,), self.dt))
            gain_parts.append(mx.full((1,), self.heavy_pull))
            step_parts.append(mx.full((1,), self.dt * self.heavy_dt_scale))
        self._source_gain = mx.concatenate(gain_parts)
        self._step = mx.concatenate(step_parts)
        if self.n_slow:
            self._slow_mask = _band_mask(self.n, self.slow.start, self.n_slow)
        if self.n_stamp:
            mask = [mx.zeros((self.stamp.start,)), mx.ones((self.n_stamp,))]
            if self.n_arc:
                mask.append(mx.zeros((self.n_arc,)))
            self._stamp_mask = mx.concatenate(mask)
        if self.n_arc:
            self._arc_mask = mx.concatenate(
                [mx.zeros((self.arc.start,)), mx.ones((self.n_arc,))]
            )
        if self.n_stamp or self.latch:
            self._ends = mx.array(self.boundary_ids or (-1,))

    def velocity(  # noqa: PLR0913
        self,
        theta: mx.array,
        tokens: mx.array,
        latch: mx.array | None = None,
        *,
        character: mx.array | None = None,
        dilemma: mx.array | None = None,
        resolution: mx.array | None = None,
        recognition: mx.array | None = None,
    ) -> mx.array:
        """Phase velocity. ``K = U @ V.T`` is never formed.

        The word enters as a phase offset on the angles inside the tug,
        ``shown + D @ phi(word)``. Character, dilemma, and resolution shift
        only the slow wheels when that band exists, so a word cannot tug
        them. A negative id adds nothing. With no slow wheels the specials
        fall back onto every wheel. Slow wheels couple only with each other.
        ``recognition`` is the overlap of the pose just left. ``K`` transpose
        ``r`` names the next catalog pose and nudges toward it, and it blends
        one coupling per stored pose. A missing recognition is uniform, the
        start of a story. Neither term snaps the wheels onto the pose.
        """
        offset = self._phase_offset(tokens, character, dilemma, resolution)
        shown = self.shown_state(theta, latch) + offset
        sin_src, cos_src = self._outbound(mx.sin(shown), mx.cos(shown))
        # The paper's blend is the tug. ``K`` transpose ``r`` picks the mix of
        # stored couplings. The pose spring stays behind ``route_extras``.
        if self.use_route:
            if recognition is None:
                recognition = self._uniform_recognition(theta.shape[0])
            successor = self._from_recognition(recognition)
            sin_k, cos_k = self._memory_fields(sin_src, cos_src, successor)
            if self.route_extras:
                sin_k = sin_k + successor @ mx.sin(self.patterns)
                cos_k = cos_k + successor @ mx.cos(self.patterns)
        else:
            sin_k, cos_k = self._couple(sin_src, cos_src)
        return self.omega + mx.cos(theta) * sin_k - mx.sin(theta) * cos_k

    def _root_phi(self, ids: mx.array) -> mx.array:
        """``tanh(root_phase) * pi`` for each id. A negative id is zero."""
        active = (ids >= 0).astype(self.D.dtype)[:, None]
        safe = mx.maximum(ids, 0)
        phi = mx.tanh(self.root_phase(self._beat_canon[safe])) * math.pi
        return phi * active

    def _phase_offset(
        self,
        tokens: mx.array,
        character: mx.array | None,
        dilemma: mx.array | None,
        resolution: mx.array | None,
    ) -> mx.array:
        """Word offset on the word band. Specials offset the slow band."""
        phi = mx.tanh(self.root_phase(self._beat_canon[tokens])) * math.pi
        spec = mx.zeros_like(phi)
        if character is not None:
            spec = spec + self.character_scale * self._root_phi(character)
        if dilemma is not None:
            spec = spec + self.dilemma_scale * self._root_phi(dilemma)
        if resolution is not None:
            spec = spec + self.resolution_scale * self._root_phi(resolution)
        word = phi @ self.D.T
        extra = spec @ self.D.T
        if not self.n_slow:
            return word + extra
        slow = self._slow_mask
        return word * (1.0 - slow) + extra * slow

    def _project(self, src: mx.array) -> mx.array:
        """Apply ``K = U @ V.T`` to one sine or cosine row. ``K`` is not stored."""
        return (src @ self.V) @ self.U.T

    def scatter_route(self) -> None:
        """Open every catalog slot on the rest pose and on ``U`` and ``V``.

        That is the state the bag already has. Each stored coupling is a copy
        of ``U`` and ``V``, and each stored pose is ``theta0``. Any mix of
        those slots is the plain tug. Poses are not drawn. ``K`` starts with
        no order. The spring stays off.
        """
        self.use_route = True
        memories = self.patterns.shape[0]
        self.patterns = mx.repeat(self.theta0[None, :], memories, axis=0)
        self.route_K = mx.zeros(self.route_K.shape)
        self.mem_U = mx.repeat(self.U[None, :, :], memories, axis=0)
        self.mem_V = mx.repeat(self.V[None, :, :], memories, axis=0)
        self.catalog_filled = 1

    def write_catalog(  # noqa: PLR0913
        self,
        token_ids: mx.array,
        character: mx.array | None = None,
        dilemma: mx.array | None = None,
        resolution: mx.array | None = None,
        resolution_at: mx.array | None = None,
        theta: mx.array | None = None,
        latch: mx.array | None = None,
        in_story: mx.array | None = None,
        seen: mx.array | None = None,
    ) -> None:
        """Record a nudge only when it clicks into a slot.

        Slot 0 is the rest pose. A nudge that stays there, or that never
        reaches any slot, is not a write. A nudge that lands writes the slot
        it entered and the slot it came from. The couplings stay the copies
        of ``U`` and ``V`` until a row of ``K`` actually uses one.
        """
        if not self.use_route:
            return
        self._trace = []
        try:
            self.roll(
                token_ids,
                character,
                dilemma,
                resolution,
                resolution_at,
                theta,
                latch,
                in_story,
                seen,
            )
            traced = list(self._trace)
        finally:
            self._trace = None
        if not traced:
            return
        self.route_K = self.route_K * ROUTE_COUNT_KEEP
        patterns = self.patterns
        order = self.route_K
        filled = int(self.catalog_filled)
        prev = mx.zeros((token_ids.shape[0],), dtype=mx.int32)
        for pose, take in traced:
            patterns, order, filled, prev = _store_pose(
                patterns, order, filled, prev, pose, take
            )
        self.patterns = patterns
        self.route_K = order
        self.catalog_filled = filled
        mx.eval(self.patterns, self.route_K)

    def _uniform_recognition(self, batch: int) -> mx.array:
        """No pose has been left yet, so every catalog entry is equally likely."""
        memories = self.patterns.shape[0]
        return mx.ones((batch, memories)) / memories

    def _overlap(self, pose: mx.array) -> mx.array:
        """Instantaneous recognition. ``r`` itself lags this by one kick.

        The overlap is the paper's softmax of negative squared distance
        between the pose and each stored pattern. On the circle that distance
        is ``2 - 2 cos``, so the softmax argument is ``2 γ`` times the cosine
        alignment.
        """
        sin_p = mx.sin(self.patterns)
        cos_p = mx.cos(self.patterns)
        align = mx.sin(pose) @ sin_p.T + mx.cos(pose) @ cos_p.T
        return mx.softmax(align * (2.0 * RECOGNITION_GAMMA), axis=-1)

    def _from_recognition(self, recognition: mx.array) -> mx.array:
        """``K`` transpose ``r``. ``r`` does not read ``K``.

        ``K`` is row-stochastic: row ``k`` says what follows memory ``k``.
        With ``r`` on memory ``k``, the product is that row.
        """
        routing = mx.softmax(self.route_K, axis=-1)
        return recognition @ routing

    def _blend_coupling(self, src: mx.array, successor: mx.array) -> mx.array:
        """Blend one low-rank coupling per stored pose. No dense ``n`` by ``n``."""
        coeff = mx.einsum("bn,mnr->bmr", src, self.mem_V)
        field = mx.einsum("bmr,mnr->bmn", coeff, self.mem_U)
        return mx.einsum("bm,bmn->bn", successor, field)

    def _memory_fields(
        self,
        sin_src: mx.array,
        cos_src: mx.array,
        successor: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Per-pose couplings, kept inside each timescale."""

        def blend(src: mx.array) -> mx.array:
            if not self.n_slow:
                return self._blend_coupling(src, successor)
            slow = self._slow_mask
            fast = 1.0 - slow
            return (
                self._blend_coupling(src * fast, successor) * fast
                + self._blend_coupling(src * slow, successor) * slow
            )

        return blend(sin_src), blend(cos_src)

    def _couple(self, sin_src: mx.array, cos_src: mx.array) -> tuple[mx.array, mx.array]:
        """Tug inside a timescale. Slow wheels do not tug the word wheels.

        The split is the timescale. The catalog adds its own couplings later.
        """
        if not self.n_slow:
            return self._project(sin_src), self._project(cos_src)
        slow = self._slow_mask
        fast = 1.0 - slow
        sin_k = self._project(sin_src * fast) * fast + self._project(sin_src * slow) * slow
        cos_k = self._project(cos_src * fast) * fast + self._project(cos_src * slow) * slow
        return sin_k, cos_k

    def logits_from(self, features: mx.array) -> mx.array:
        """Next-piece scores. Rank 0 keeps the full linear map."""
        if self.readout_rank > 0:
            return self.readout_up(self.readout_down(features))
        return self.readout(features)

    def _open_dilemma(
        self,
        seen: mx.array,
        tokens: mx.array,
        dilemma: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Turn the dilemma on at the piece that says it, and keep it on."""
        folded = self._beat_canon[tokens]
        seen = mx.logical_or(seen, mx.logical_and(dilemma >= 0, folded == dilemma))
        inactive = mx.zeros_like(dilemma) - 1
        return seen, mx.where(seen, dilemma, inactive)

    def _outbound(self, sin_t: mx.array, cos_t: mx.array) -> tuple[mx.array, mx.array]:
        """Multiply the tug coming from the set-aside wheels."""
        if not self.n_slow and not self.n_stamp and not self.n_arc and not self.latch:
            return sin_t, cos_t
        return sin_t * self._source_gain, cos_t * self._source_gain

    def _at_period(self, tokens: mx.array) -> mx.array:
        """Mark rows whose piece ends a sentence."""
        ends = self._ends.astype(tokens.dtype)
        return mx.any(tokens[:, None] == ends[None, :], axis=-1)

    def step_scale(
        self,
        tokens: mx.array,
        resolution_on: mx.array | None = None,
    ) -> mx.array | float:
        """Per-wheel Euler step.

        The stamp clock moves only at a sentence end. The arc wheel takes a
        full step only while the resolution drive is on for that row.
        """
        if not self.n_slow and not self.n_stamp and not self.n_arc and not self.latch:
            return self.dt
        if not self.n_stamp and not self.n_arc:
            return self._step
        step: mx.array | float = self._step
        if self.n_stamp:
            end = self._at_period(tokens)[:, None]
            stamped = mx.where(self._stamp_mask, self.dt, step)
            step = mx.where(end, stamped, step)
        if self.n_arc:
            if resolution_on is None:
                on = mx.zeros((tokens.shape[0], 1), dtype=mx.bool_)
            else:
                on = resolution_on.astype(mx.bool_)[:, None]
            moved = mx.where(self._arc_mask, self.dt, step)
            step = mx.where(on, moved, step)
        return step

    def shown_state(self, theta: mx.array, latch: mx.array | None) -> mx.array:
        """Swap the live dilemma wheel for the held angle."""
        if not self.latch:
            return theta
        if latch is None:
            latch = mx.zeros((theta.shape[0], 1))
        return mx.concatenate([theta[:, :-1], latch], axis=-1)

    def commit_sentence(
        self,
        latch: mx.array,
        theta: mx.array,
        tokens: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Store the sentence in the latch, then clear the accumulator."""
        end = self._at_period(tokens)[:, None]
        live = theta[:, self.latch_at : self.latch_at + 1]
        mixed = (1.0 - self.latch_blend) * latch + self.latch_blend * live
        held = mx.where(end, mixed, latch)
        cleared = mx.where(end, mx.zeros_like(live), live)
        return held, mx.concatenate([theta[:, :-1], cleared], axis=-1)

    def __call__(
        self,
        token_ids: mx.array,
        character: mx.array | None = None,
        dilemma: mx.array | None = None,
        resolution: mx.array | None = None,
        resolution_at: mx.array | None = None,
    ) -> mx.array:
        """Return next-piece logits ``(batch, time - 1, vocab)`` from rest."""
        logits, _theta, _latch, _in_story, _seen = self.roll(
            token_ids,
            character,
            dilemma,
            resolution,
            resolution_at,
        )
        return logits

    def roll(  # noqa: PLR0912, PLR0913, PLR0915, C901
        self,
        token_ids: mx.array,
        character: mx.array | None = None,
        dilemma: mx.array | None = None,
        resolution: mx.array | None = None,
        resolution_at: mx.array | None = None,
        theta: mx.array | None = None,
        latch: mx.array | None = None,
        in_story: mx.array | None = None,
        seen: mx.array | None = None,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        """Spin one window and return the wheels it leaves behind.

        A missing start is rest, so the first window of a story begins there.
        A later window of the same story passes the wheels, the sentence
        latch, and whether the row is still inside the story. After the end
        piece is kicked, the next kick starts at ``theta0`` again. Padding
        after that first end piece is not kicked. A sentence period still
        only moves the stamp clock. Character stays on for the story. The
        dilemma turns on at its piece and stays on, at a fifth of a word
        kick, until the resolution starts. Then the dilemma fades and the
        resolution stays on through the last sentence at three word kicks.
        Character never fades. The
        arc wheel steps while that resolution drive is on. The
        end piece rests the whole bag, including that wheel.
        """
        batch = token_ids.shape[0]
        if theta is None:
            theta = mx.broadcast_to(self.theta0, (batch, self.n))
        if latch is None:
            latch = mx.zeros((batch, 1))
        if in_story is None:
            in_story = mx.ones((batch,), dtype=mx.bool_)
        if seen is None:
            seen = mx.zeros((batch,), dtype=mx.bool_)
        # r lags the overlap by one kick. The first kick of a story has not
        # left a pose, so it starts uniform and is reset there at the end piece.
        recognition = self._uniform_recognition(batch)
        logits = []
        for t in range(token_ids.shape[1] - 1):
            tokens = token_ids[:, t]
            dilemma_now = None
            resolution_now = None
            resolution_on = None
            if dilemma is not None:
                seen, dilemma_now = self._open_dilemma(seen, tokens, dilemma)
            in_ending = None
            if resolution_at is not None:
                in_ending = mx.full((batch,), t) >= resolution_at
            if dilemma_now is not None and in_ending is not None:
                quiet = mx.zeros_like(dilemma_now) - 1
                dilemma_now = mx.where(in_ending, quiet, dilemma_now)
            if resolution is not None and in_ending is not None:
                before = mx.zeros_like(resolution) - 1
                resolution_now = mx.where(in_ending, resolution, before)
                resolution_on = in_ending
            if self.eos_id is None:
                is_end = mx.zeros((batch,), dtype=mx.bool_)
            else:
                is_end = tokens == self.eos_id
            padding = mx.logical_and(mx.logical_not(in_story), is_end)
            take = mx.logical_not(padding)[:, None]
            if self.latch:
                held, committed = self.commit_sentence(latch, theta, tokens)
                latch = mx.where(take, held, latch)
                theta = mx.where(take, committed, theta)
            left = self._overlap(self.shown_state(theta, latch))
            stepped = theta + self.step_scale(tokens, resolution_on) * self.velocity(
                theta,
                tokens,
                latch,
                character=character,
                dilemma=dilemma_now,
                resolution=resolution_now,
                recognition=recognition,
            )
            stepped = mx.arctan2(mx.sin(stepped), mx.cos(stepped))
            theta = mx.where(take, stepped, theta)
            recognition = mx.where(take, left, recognition)
            shown = self.shown_state(theta, latch)
            features = mx.concatenate([mx.sin(shown), mx.cos(shown)], axis=-1)
            logits.append(self.logits_from(features))
            if self.eos_id is not None:
                reset = mx.logical_and(take.reshape(is_end.shape), is_end)[:, None]
                rest = mx.broadcast_to(self.theta0, theta.shape)
                theta = mx.where(reset, rest, theta)
                latch = mx.where(reset, mx.zeros_like(latch), latch)
                recognition = mx.where(reset, self._uniform_recognition(batch), recognition)
                in_story = mx.logical_not(is_end)
                seen = mx.where(reset.reshape(seen.shape), mx.zeros_like(seen), seen)
            if self._trace is not None:
                self._trace.append((theta, take.reshape((batch,))))
        return mx.stack(logits, axis=1), theta, latch, in_story, seen


def _store_pose(  # noqa: PLR0913
    patterns: mx.array,
    order: mx.array,
    filled: int,
    prev: mx.array,
    pose: mx.array,
    take: mx.array,
) -> tuple[mx.array, mx.array, int, mx.array]:
    """Record a landing, and the slot it came from. A nudge that stays is nothing.

    One new slot per kick across the batch, and only for a pose that clicked
    into no open slot. Rows that land on that same pose share it. A pose that
    clicks a different open slot records that arrival without a new phase.
    """
    memories, n = patterns.shape
    align = (mx.sin(pose) @ mx.sin(patterns).T + mx.cos(pose) @ mx.cos(patterns).T) / n
    open_slot = mx.arange(memories) < filled
    align = mx.where(open_slot[None, :], align, mx.full(align.shape, -2.0))
    best = mx.argmax(align, axis=-1).astype(mx.int32)
    best_val = mx.max(align, axis=-1)
    clicked = mx.logical_and(take, best_val >= WRITE_ALIGN)
    arrived = mx.logical_and(clicked, mx.not_equal(best, prev))
    slot = mx.where(arrived, best, prev)
    missed = mx.logical_and(take, mx.logical_not(clicked))
    if filled < memories:
        score = mx.where(missed, best_val, mx.full(best_val.shape, 2.0))
        pick = int(mx.argmin(score).item())
        if bool(missed[pick].item()):
            chosen = pose[pick]
            patterns = patterns.at[filled].add(chosen - patterns[filled])
            same = (mx.sin(pose) @ mx.sin(chosen) + mx.cos(pose) @ mx.cos(chosen)) / n
            opened = mx.logical_and(take, same >= WRITE_ALIGN)
            fresh = mx.array(filled, dtype=mx.int32)
            slot = mx.where(opened, fresh, slot)
            arrived = mx.logical_or(opened, mx.logical_and(arrived, mx.logical_not(opened)))
            filled += 1
    counts = mx.zeros_like(order)
    counts = counts.at[prev, slot].add(arrived.astype(order.dtype))
    order = order + counts
    prev = mx.where(arrived, slot, prev)
    return patterns, order, filled, prev


def _story_weights(tokens: mx.array, eos_id: int | None) -> mx.array:
    """Weight next-piece targets up through the real end piece. Padding is 0."""
    width = tokens.shape[1] - 1
    if eos_id is None:
        return mx.ones((tokens.shape[0], width))
    is_end = tokens == eos_id
    has_end = mx.any(is_end, axis=1)
    first = mx.argmax(is_end.astype(mx.int32), axis=1)
    keep = mx.arange(width)[None, :] < first[:, None]
    return mx.where(has_end[:, None], keep, mx.ones_like(keep))


def _local_resolution_at(resolution_at: mx.array, offset: int, width: int) -> mx.array:
    """Shift a full-window onset into one credit stage."""
    shifted = resolution_at - offset
    return mx.clip(shifted, 0, width)


def _stage_ends(length: int, span: int) -> list[tuple[int, int]]:
    """Kick ranges ``[start, stop)`` inside a window of ``length`` tokens."""
    steps = length - 1
    if steps < 1:
        raise ValueError("A window needs at least two pieces.")
    return [(start, min(start + span, steps)) for start in range(0, steps, span)]


def _array_norm(parts: tuple[mx.array, ...]) -> mx.array:
    """Euclidean norm of a few credit tensors."""
    total = mx.array(0.0)
    for part in parts:
        total = total + mx.sum(part * part)
    return mx.sqrt(total)


def _mix_credit(  # noqa: PLR0911
    left: tuple[mx.array, mx.array] | None,
    left_scale: float,
    right: tuple[mx.array, mx.array],
    right_scale: float,
) -> tuple[tuple[mx.array, mx.array], float] | None:
    """Add two phase credits kept as direction plus log-size."""
    if right_scale <= 0.0:
        if left is None or left_scale <= 0.0:
            return None
        return left, left_scale
    if left is None or left_scale <= 0.0:
        norm = _array_norm(right)
        mx.eval(norm)
        norm_f = float(norm)
        if norm_f == 0.0 or not math.isfinite(norm_f):
            return None
        unit = tuple(mx.stop_gradient(part / norm) for part in right)
        return unit, math.log(right_scale) + math.log(norm_f)  # type: ignore[return-value]
    right_norm = _array_norm(right)
    mx.eval(right_norm)
    right_f = float(right_norm)
    if right_f == 0.0 or not math.isfinite(right_f):
        return left, left_scale
    log_right = math.log(right_scale) + math.log(right_f)
    top = max(left_scale, log_right)
    weight_left = math.exp(left_scale - top)
    weight_right = math.exp(log_right - top)
    mixed = tuple(
        weight_left * old + weight_right * (new / right_norm) for old, new in zip(left, right, strict=True)
    )
    mixed_norm = _array_norm(mixed)
    mx.eval(mixed_norm)
    mixed_f = float(mixed_norm)
    if mixed_f == 0.0 or not math.isfinite(mixed_f):
        return left, left_scale
    unit = tuple(mx.stop_gradient(part / mixed_norm) for part in mixed)
    return unit, top + math.log(mixed_f)  # type: ignore[return-value]


def _tree_norm(tree: dict) -> mx.array:
    """Euclidean norm of a parameter-gradient tree."""
    total = mx.array(0.0)
    for _, leaf in tree_flatten(tree):
        total = total + mx.sum(leaf * leaf)
    return mx.sqrt(total)


def _add_grad(
    acc: tuple[dict, float] | None,
    grad: dict,
    scale: float,
) -> tuple[dict, float] | None:
    """Add ``scale * grad`` without building the huge product in one float.

    The sum is kept as a unit direction plus the log of its size. A later
    stage whose credit is vastly larger simply outweighs an earlier one.
    """
    if scale <= 0.0:
        return acc
    norm = _tree_norm(grad)
    mx.eval(norm)
    norm_f = float(norm)
    if norm_f == 0.0 or not math.isfinite(norm_f):
        return acc
    log_b = math.log(scale) + math.log(norm_f)
    direction_b = tree_map(lambda leaf: leaf / norm, grad)
    if acc is None:
        return direction_b, log_b
    direction_a, log_a = acc
    top = max(log_a, log_b)
    weight_a = math.exp(log_a - top)
    weight_b = math.exp(log_b - top)
    mixed = tree_map(
        lambda left, right: left * weight_a + right * weight_b,
        direction_a,
        direction_b,
    )
    mixed_norm = _tree_norm(mixed)
    mx.eval(mixed_norm)
    mixed_f = float(mixed_norm)
    if mixed_f == 0.0 or not math.isfinite(mixed_f):
        return acc
    unit = tree_map(lambda leaf: leaf / mixed_norm, mixed)
    return unit, top + math.log(mixed_f)


def _clip_credit(acc: tuple[dict, float]) -> dict:
    """Return a float32 gradient whose norm is at most 1."""
    direction, log_scale = acc
    if log_scale >= 0.0:
        return tree_map(lambda leaf: leaf.astype(mx.float32), direction)
    factor = math.exp(log_scale)
    return tree_map(lambda leaf: (leaf * factor).astype(mx.float32), direction)


def window_credit(  # noqa: PLR0913, C901, PLR0915
    model: KuramotoMLX,
    tokens: mx.array,
    character: mx.array,
    dilemma: mx.array,
    resolution: mx.array,
    resolution_at: mx.array,
    theta: mx.array,
    latch: mx.array,
    in_story: mx.array,
    seen: mx.array,
    *,
    span: int = _CREDIT_SPAN,
) -> tuple[mx.array, dict | None, mx.array, mx.array, mx.array, mx.array]:
    """Teach one window in stages and hand the credit back to the opening.

    Not the default step. One epoch of this walk costs on the order of 300
    seconds, and the POC budget is one compiled graph at about 5 seconds a
    batch. Kept so a later pass can hand credit to the opening again.

    Each stage is ``span`` kicks. The stage's own next-piece loss teaches
    that stretch. The direction and scale of how much later words still
    cared are an extra push on the phase the stage ends in, then the same
    pair is saved for the stage behind this one. The wheels returned are
    the phase after the last kick, detached, ready for the next window.
    """
    ranges = _stage_ends(int(tokens.shape[1]), span)
    states: list[tuple[mx.array, mx.array, mx.array, mx.array]] = [
        (
            mx.stop_gradient(theta),
            mx.stop_gradient(latch),
            mx.stop_gradient(in_story),
            mx.stop_gradient(seen),
        )
    ]
    cursor = (theta, latch, in_story, seen)
    for start, stop in ranges:
        piece = tokens[:, start : stop + 1]
        local_at = _local_resolution_at(resolution_at, start, int(piece.shape[1]))
        _logits, *cursor = model.roll(
            piece,
            character,
            dilemma,
            resolution,
            local_at,
            *cursor,
        )
        cursor = tuple(mx.stop_gradient(part) for part in cursor)
        mx.eval(*cursor)
        states.append(cursor)  # type: ignore[arg-type]

    acc: dict | None = None
    direction: mx.array | None = None
    latch_direction: mx.array | None = None
    scale = 0.0
    weighted = mx.array(0.0)
    weight_sum = mx.array(0.0)

    def stage_value(  # noqa: PLR0913
        piece: mx.array,
        local_at: mx.array,
        theta_s: mx.array,
        latch_s: mx.array,
        in_story_s: mx.array,
        seen_s: mx.array,
        credit: mx.array | None,
        latch_credit: mx.array | None,
    ) -> tuple[mx.array, mx.array]:
        logits, theta_e, latch_e, _in_e, _seen_e = model.roll(
            piece,
            character,
            dilemma,
            resolution,
            local_at,
            theta_s,
            latch_s,
            in_story_s,
            seen_s,
        )
        targets = piece[:, 1:]
        losses = _piece_losses(logits, targets, model.vocab_size)
        weights = _story_weights(piece, model.eos_id)
        ce = (losses * weights).sum()
        handoff = mx.array(0.0)
        if credit is not None and latch_credit is not None:
            handoff = mx.sum(credit * theta_e) + mx.sum(latch_credit * latch_e)
        return ce, handoff

    def finite_parts(*parts: dict | mx.array) -> bool:
        for part in parts:
            norm = _tree_norm(part) if isinstance(part, dict) else _array_norm((part,))
            mx.eval(norm)
            if not math.isfinite(float(norm)):
                return False
        return True

    def teach(start: int, stop: int, theta_s: mx.array, latch_s: mx.array, in_story_s: mx.array, seen_s: mx.array) -> None:  # noqa: PLR0913, PLR0915, C901
        """Score kicks ``[start, stop)``. Split the stage if its gradient overflows."""
        nonlocal acc, direction, latch_direction, scale, weighted, weight_sum
        piece = tokens[:, start : stop + 1]
        local_at = _local_resolution_at(resolution_at, start, int(piece.shape[1]))

        def differentiate(  # noqa: PLR0913
            which: str,
            piece: mx.array = piece,
            local_at: mx.array = local_at,
            theta_s: mx.array = theta_s,
            latch_s: mx.array = latch_s,
            in_story_s: mx.array = in_story_s,
            seen_s: mx.array = seen_s,
            direction: mx.array | None = direction,
            latch_direction: mx.array | None = latch_direction,
        ) -> tuple[mx.array, dict, mx.array, mx.array, tuple]:
            use_credit = direction if which == "credit" else None
            use_latch = latch_direction if which == "credit" else None

            def forward(
                params: dict,
                theta_in: mx.array,
                latch_in: mx.array,
            ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
                model.update(params)
                ce, handoff = stage_value(
                    piece,
                    local_at,
                    theta_in,
                    latch_in,
                    in_story_s,
                    seen_s,
                    use_credit,
                    use_latch,
                )
                return (ce if which == "ce" else handoff), (ce, handoff)

            (scalar, aux), grads = mx.value_and_grad(forward, argnums=(0, 1, 2))(
                model.parameters(),
                theta_s,
                latch_s,
            )
            return scalar, grads[0], grads[1], grads[2], aux

        stage_weight = _story_weights(piece, model.eos_id).sum()
        mx.eval(stage_weight)
        if float(stage_weight) == 0.0 and (direction is None or scale <= 0.0):
            return
        _ce_scalar, ce_params, ce_theta, ce_latch, (ce_value, _handoff) = differentiate("ce")
        credit_pack = None
        if direction is not None and latch_direction is not None and scale > 0.0:
            credit_pack = differentiate("credit")
        overflow = not finite_parts(ce_params, ce_theta, ce_latch)
        if credit_pack is not None:
            _c_scalar, c_params, c_theta, c_latch, _c_aux = credit_pack
            overflow = overflow or not finite_parts(c_params, c_theta, c_latch)
        else:
            c_params = c_theta = c_latch = None
        if overflow:
            if stop - start <= 1:
                raise RuntimeError("A single kick gradient was not finite.")
            mid = start + (stop - start) // 2
            _logits, theta_m, latch_m, in_m, seen_m = model.roll(
                tokens[:, start : mid + 1],
                character,
                dilemma,
                resolution,
                _local_resolution_at(resolution_at, start, mid - start + 1),
                theta_s,
                latch_s,
                in_story_s,
                seen_s,
            )
            mx.eval(theta_m, latch_m, in_m, seen_m)
            teach(mid, stop, theta_m, latch_m, in_m, seen_m)
            teach(start, mid, theta_s, latch_s, in_story_s, seen_s)
            return
        weighted = weighted + ce_value
        weight_sum = weight_sum + stage_weight
        if float(stage_weight) > 0.0:
            acc = _add_grad(acc, ce_params, 1.0)
        handed = _mix_credit(None, 0.0, (ce_theta, ce_latch), 1.0)
        if c_params is not None and c_theta is not None and c_latch is not None:
            acc = _add_grad(acc, c_params, scale)
            if handed is None:
                handed = _mix_credit(None, 0.0, (c_theta, c_latch), scale)
            else:
                handed = _mix_credit(handed[0], handed[1], (c_theta, c_latch), scale)
        mx.eval(weighted, weight_sum)
        if handed is None:
            direction = None
            latch_direction = None
            scale = 0.0
        else:
            (direction, latch_direction), log_size = handed
            log_size = min(log_size, math.log(_SCALE_CAP))
            scale = math.exp(log_size)
            mx.eval(direction, latch_direction)

    for index in range(len(ranges) - 1, -1, -1):
        start, stop = ranges[index]
        teach(start, stop, *states[index])
    final_theta, final_latch, final_in_story, final_seen = states[-1]
    if acc is None:
        # Every stage was padding after the end token. The wheels are already
        # at rest, and there is no update. The caller moves to the next story.
        return mx.array(0.0), None, final_theta, final_latch, final_in_story, final_seen
    loss = weighted / mx.maximum(weight_sum, mx.array(1.0))
    mx.eval(loss)
    return loss, _clip_credit(acc), final_theta, final_latch, final_in_story, final_seen


_DYNAMICS = ("U", "V", "mem_U", "mem_V", "omega", "theta0")


def _scale_dynamics_grads(grads: dict) -> dict:
    """Learn the coupling and the spin slower than the word map and the readout."""
    scaled = []
    for key, leaf in tree_flatten(grads):
        updated = leaf * DYNAMICS_LR_SCALE if key in _DYNAMICS else leaf
        scaled.append((key, updated))
    return tree_unflatten(scaled)


def _clip_global_norm(grads: dict, max_norm: float = 1.0) -> dict:
    """Scale a gradient tree so its Euclidean norm is at most ``max_norm``."""
    norm = _tree_norm(grads)
    cap = mx.array(max_norm, dtype=norm.dtype)
    floor = mx.array(1e-12, dtype=norm.dtype)
    scale = mx.minimum(norm, cap) / mx.maximum(norm, floor)
    return tree_map(lambda leaf: leaf * scale, grads)


class _CaseHold:
    """Case groups for the loss. Unset means each surface piece is its own target."""

    members: mx.array | None = None


class _GlueHold:
    """Per-piece loss weights. Unset means every target counts as 1."""

    table: mx.array | None = None


_CASE = _CaseHold()
_GLUE = _GlueHold()


def _band_mask(n: int, start: int, count: int) -> mx.array:
    """Ones on ``[start, start + count)`` and zeros on the other wheels."""
    parts = []
    if start:
        parts.append(mx.zeros((start,)))
    parts.append(mx.ones((count,)))
    tail = n - start - count
    if tail:
        parts.append(mx.zeros((tail,)))
    return mx.concatenate(parts)


def set_glue_weights(itos: list[str] | None, weight: float = 1.0) -> None:
    """Give glue and sentence marks ``weight``. Content words and a blank line stay at 1."""
    if itos is None or weight == 1.0:
        _GLUE.table = None
        return
    values = [1.0 if _takes_repetition_tax(piece) else float(weight) for piece in itos]
    _GLUE.table = mx.array(values)


def set_case_members(itos: list[str] | None) -> None:
    """Score ``The`` and ``the`` as one word. ``None`` keeps the surface target."""
    if itos is None:
        _CASE.members = None
        return
    groups = case_groups(itos)
    by_surface: list[list[int]] = [[] for _ in itos]
    for group in groups:
        for index in group:
            by_surface[index] = group
    width = max(len(group) for group in by_surface)
    rows = [group + [-1] * (width - len(group)) for group in by_surface]
    _CASE.members = mx.array(rows)


def _piece_losses(logits: mx.array, targets: mx.array, vocab: int) -> mx.array:
    """Cross-entropy. With case groups, either case of the word is the target."""
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)
    members = _CASE.members
    if members is None:
        losses = nn.losses.cross_entropy(flat_logits, flat_targets, reduction="none")
        return losses.reshape(targets.shape)
    idx = members[flat_targets]
    safe = mx.maximum(idx, 0)
    rows = mx.arange(flat_logits.shape[0])[:, None]
    picked = flat_logits[rows, safe]
    picked = mx.where(idx >= 0, picked, mx.full(picked.shape, -1e9))
    group = mx.logsumexp(picked, axis=1)
    full = mx.logsumexp(flat_logits, axis=1)
    return (full - group).reshape(targets.shape)


def _masked_cross_entropy(model: KuramotoMLX, tokens: mx.array, logits: mx.array) -> mx.array:
    """Mean next-piece cross-entropy. Padding after the end piece adds nothing."""
    targets = tokens[:, 1:]
    losses = _piece_losses(logits, targets, model.vocab_size)
    weights = _story_weights(tokens, model.eos_id).astype(losses.dtype)
    if _GLUE.table is not None:
        weights = weights * _GLUE.table[targets]
    total = mx.sum(weights)
    return mx.sum(losses * weights) / mx.maximum(total, mx.ones_like(total))


_WindowStep = Callable[
    [mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array, mx.array],
    tuple[mx.array, mx.array, mx.array, mx.array, mx.array],
]


def compile_window_step(model: KuramotoMLX, optimizer: optim.Optimizer) -> _WindowStep:
    """Compile one AdamW update of one window.

    A full-window backward overflows float32. Staged credit (``window_credit``)
    reaches the opening, but one epoch of that walk took about 300 seconds,
    against about 5 seconds a batch for one compiled graph. That graph is the
    POC budget, so it is the default. ``window_credit`` stays in this file for
    later and is not the default, because of that cost. The shelf is the short
    stories that fit in this one window.

    The compiled function is loss-and-grad, clip at 1, then ``optimizer.update``,
    with inputs and outputs ``[model.state, optimizer.state]``.
    """
    state = [model.state, optimizer.state]

    def apply_window(  # noqa: PLR0913
        tokens: mx.array,
        character: mx.array,
        dilemma: mx.array,
        resolution: mx.array,
        resolution_at: mx.array,
        theta: mx.array,
        latch: mx.array,
        in_story: mx.array,
        seen: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        def forward(
            params: dict,
            theta_in: mx.array,
            latch_in: mx.array,
            in_story_in: mx.array,
            seen_in: mx.array,
        ) -> tuple[mx.array, tuple[mx.array, mx.array, mx.array, mx.array]]:
            model.update(params)
            logits, theta_e, latch_e, in_e, seen_e = model.roll(
                tokens,
                character,
                dilemma,
                resolution,
                resolution_at,
                theta_in,
                latch_in,
                in_story_in,
                seen_in,
            )
            loss = _masked_cross_entropy(model, tokens, logits)
            carry = (
                mx.stop_gradient(theta_e),
                mx.stop_gradient(latch_e),
                mx.stop_gradient(in_e),
                mx.stop_gradient(seen_e),
            )
            return loss, carry

        (loss, carry), grads = mx.value_and_grad(forward)(
            model.parameters(),
            mx.stop_gradient(theta),
            mx.stop_gradient(latch),
            mx.stop_gradient(in_story),
            mx.stop_gradient(seen),
        )
        # The write owns the poses and the order. Adam must not invent either.
        kept_patterns = mx.stop_gradient(model.patterns)
        kept_route = mx.stop_gradient(model.route_K)
        optimizer.update(model, _scale_dynamics_grads(_clip_global_norm(grads, 1.0)))
        model.patterns = kept_patterns
        model.route_K = kept_route
        theta_e, latch_e, in_e, seen_e = carry
        return loss, theta_e, latch_e, in_e, seen_e

    return mx.compile(apply_window, inputs=state, outputs=state)


def run_compiled_window(  # noqa: PLR0913
    model: KuramotoMLX,
    compiled_step: _WindowStep,
    optimizer: optim.Optimizer,
    tokens: mx.array,
    character: mx.array,
    dilemma: mx.array,
    resolution: mx.array,
    resolution_at: mx.array,
    theta: mx.array,
    latch: mx.array,
    in_story: mx.array,
    seen: mx.array,
) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
    """One default step. An all-pad window is not an update and not an error."""
    weight = _story_weights(tokens, model.eos_id).sum()
    mx.eval(weight)
    if float(weight) == 0.0:
        _logits, theta, latch, in_story, seen = model.roll(
            tokens,
            character,
            dilemma,
            resolution,
            resolution_at,
            theta,
            latch,
            in_story,
            seen,
        )
        loss = mx.array(0.0)
        mx.eval(loss, theta, latch, in_story, seen)
        return loss, theta, latch, in_story, seen
    loss, theta, latch, in_story, seen = compiled_step(
        tokens,
        character,
        dilemma,
        resolution,
        resolution_at,
        theta,
        latch,
        in_story,
        seen,
    )
    mx.eval(loss, model.parameters(), optimizer.state, theta, latch, in_story, seen)
    return loss, theta, latch, in_story, seen


def loss_fn(  # noqa: PLR0913
    model: KuramotoMLX,
    tokens: mx.array,
    character: mx.array | None = None,
    dilemma: mx.array | None = None,
    resolution: mx.array | None = None,
    resolution_at: mx.array | None = None,
) -> mx.array:
    """Mean next-piece cross-entropy. Padding after the end piece adds nothing."""
    logits = model(tokens, character, dilemma, resolution, resolution_at)
    return _masked_cross_entropy(model, tokens, logits)


def _piece_name(itos: list[str], piece: int) -> str:
    """Surface form of one category id. A missing id reads as none."""
    if piece < 0:
        return "(none)"
    text = itos[piece].strip()
    return text or "(none)"


def _resolution_text(itos: list[str], pieces: tuple[int, ...]) -> str:
    """Join the last sentence into one readable line."""
    if not pieces:
        return "(none)"
    text = " ".join("".join(itos[piece] for piece in pieces).split())
    return text or "(none)"


def print_categories(labels: list[StoryLabels], itos: list[str]) -> None:
    """Print the first few training stories and whether the ending names them."""
    shown = labels[:_CATEGORY_PREVIEW]
    print(f"categories {len(shown)}", flush=True)
    for index, label in enumerate(shown, start=1):
        character = _piece_name(itos, label.character)
        dilemma = _piece_name(itos, label.dilemma)
        resolution = _resolution_text(itos, label.resolution)
        print(
            f"{index} character {character} | dilemma {dilemma} | "
            f"resolution {resolution} | ending_names {label.arc.ending_names}",
            flush=True,
        )


def resolution_columns(
    tokens: torch.Tensor,
    resolutions: tuple[tuple[int, ...], ...],
    itos: list[str],
    canon: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the last sentence's first content piece and where that sentence starts.

    A window that does not contain the sentence keeps a negative id, so the
    drive stays off. The id is the folded piece.
    """
    width = int(tokens.shape[1])
    ids: list[int] = []
    starts: list[int] = []
    for row, resolution in zip(tokens.tolist(), resolutions, strict=True):
        piece = -1
        for token in resolution:
            if _content_word(itos[token]):
                piece = canon[token]
                break
        start = width
        span = list(resolution)
        if piece >= 0 and span:
            limit = len(row) - len(span) + 1
            for index in range(max(limit, 0)):
                if row[index : index + len(span)] == span:
                    start = index
                    break
            if start >= width:
                piece = -1
        ids.append(piece)
        starts.append(start)
    return torch.tensor(ids, dtype=torch.long), torch.tensor(starts, dtype=torch.long)


def require_metal() -> None:
    """Stop unless this process is on the Metal GPU."""
    device = mx.default_device()
    if device.type != mx.DeviceType.gpu:
        raise RuntimeError(f"Refusing to train on {device}. Metal GPU is required.")


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Compile the step and train until the time budget is spent."""
    args = build_parser().parse_args()
    require_metal()
    mx.random.seed(int(args.seed))
    itos, stories, chunk = load_corpus(args.data_dir)
    stories = limit_stories(stories, args.stories)
    loaded = len(stories)
    if args.all_stories:
        shelf = "all"
    else:
        stories = keep_short_stories(stories, STORY_WINDOW)
        shelf = f"cap {STORY_WINDOW}"
    if not stories:
        raise ValueError(f"No story fits in {STORY_WINDOW} pieces.")
    labels, canon = label_shelf(stories, itos)
    print_categories(labels, itos)
    seq_len = int(args.seq_len)
    batch_size = int(args.batch_size)
    total_tokens = sum(len(story) for story in stories)
    eos_id = itos.index(EOS)
    steps_per_epoch = story_epoch_steps(len(stories), batch_size)

    model = KuramotoMLX(
        len(itos),
        n=int(args.n_oscillators),
        rank=int(args.rank),
        n_roots=int(args.n_roots),
        n_slow=int(args.n_slow),
        n_stamp=int(args.n_stamp),
        n_arc=int(args.n_arc),
        heavy_pull=float(args.heavy_pull),
        heavy_dt_scale=float(args.heavy_dt_scale),
        boundary_ids=tuple(sentence_end_ids(itos)),
        latch=bool(args.latch),
        latch_blend=float(args.latch_blend),
        eos_id=eos_id,
        beat_canon=canon,
        readout_rank=READOUT_RANK,
    )
    model.character_scale = float(args.character_scale)
    model.dilemma_scale = float(args.dilemma_scale)
    model.resolution_scale = float(args.resolution_scale)
    if args.resume is not None:
        resume_from = Path(args.resume)
        if not resume_from.exists():
            raise FileNotFoundError(f"No saved bag at {resume_from}.")
        saved = mx.load(str(resume_from))
        saved_vocab, saved_readout = readout_spec(saved)
        if saved_vocab != len(itos):
            raise ValueError(
                f"Saved bag has {saved_vocab} pieces; the vocabulary on disk has {len(itos)}."
            )
        n, rank, n_roots = widths(saved)
        if (n, rank, n_roots, saved_readout) != (
            model.n,
            int(args.rank),
            int(args.n_roots),
            model.readout_rank,
        ):
            raise ValueError(
                f"Saved bag is {n} wheels, rank {rank}, roots {n_roots}, "
                f"readout {saved_readout}; this run is {model.n}, "
                f"{int(args.rank)}, {int(args.n_roots)}, {model.readout_rank}."
            )
        load_bag_weights(model, resume_from)
    # A fresh run leaves the route off. Velocity stays the plain tug, U @ V.
    # scatter_route and write_catalog stay on the class for the wheel tests.
    mx.eval(model.parameters())
    set_case_members(itos)
    set_glue_weights(itos, GLUE_TARGET_WEIGHT)

    optimizer = optim.AdamW(learning_rate=float(args.lr), weight_decay=0.01)
    compiled_step = compile_window_step(model, optimizer)

    def step(  # noqa: PLR0913
        tokens: mx.array,
        character: mx.array,
        dilemma: mx.array,
        resolution: mx.array,
        resolution_at: mx.array,
        theta: mx.array,
        latch: mx.array,
        in_story: mx.array,
        seen: mx.array,
    ) -> tuple[mx.array, mx.array, mx.array, mx.array, mx.array]:
        return run_compiled_window(
            model,
            compiled_step,
            optimizer,
            tokens,
            character,
            dilemma,
            resolution,
            resolution_at,
            theta,
            latch,
            in_story,
            seen,
        )

    def rest_state(batch: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
        return (
            mx.broadcast_to(model.theta0, (batch, model.n)),
            mx.zeros((batch, 1)),
            mx.ones((batch,), dtype=mx.bool_),
            mx.zeros((batch,), dtype=mx.bool_),
        )

    def run_windows(batch: StoryBatch) -> mx.array:
        """Update once per window. Later windows keep the wheels of this story."""
        windows = batch.windows or (batch.tokens,)
        flags = batch.active or tuple(torch.ones(window.shape[0], dtype=torch.bool) for window in windows)
        theta, latch, in_story, seen = rest_state(windows[0].shape[0])
        total = 0.0
        taken = 0
        for window, flag in zip(windows, flags, strict=True):
            if not bool(flag.any()):
                continue
            resolution, resolution_at = resolution_columns(window, batch.resolution, itos, canon)
            tokens_mx = mx.array(window.numpy())
            character_mx = mx.array(batch.character.numpy())
            dilemma_mx = mx.array(batch.dilemma.numpy())
            resolution_mx = mx.array(resolution.numpy())
            resolution_at_mx = mx.array(resolution_at.numpy())
            loss, theta, latch, in_story, seen = step(
                tokens_mx,
                character_mx,
                dilemma_mx,
                resolution_mx,
                resolution_at_mx,
                theta,
                latch,
                in_story,
                seen,
            )
            mx.eval(loss, theta, latch, in_story, seen, model.parameters(), optimizer.state)
            if float(loss) == 0.0:
                continue
            total += float(loss)
            taken += 1
        if taken == 0:
            return mx.array(0.0)
        return mx.array(total / taken)
    flat = sum(value.size for _, value in tree_flatten(model.parameters()))
    arc = f"arc {model.arc.start} " if model.n_arc else ""
    print(
        f"metal {mx.default_device()} stories {len(stories)} of {loaded} "
        f"{shelf} tokens {total_tokens} "
        f"chunk {chunk} vocab {len(itos)} params {flat} "
        f"wheels {model.n} rank {int(args.rank)} roots {int(args.n_roots)} "
        f"readout {model.readout_rank} dynamics {DYNAMICS_LR_SCALE:g} "
        f"memories {N_MEMORIES} "
        f"glue {GLUE_TARGET_WEIGHT:g} "
        f"slow {model.slow.start}-{model.slow.stop - 1} step {model.dt * model.heavy_dt_scale:g} "
        f"stamp {model.stamp.start}-{model.stamp.stop - 1} pull {model.heavy_pull:g} "
        f"{arc}"
        f"window {seq_len} one graph "
        f"pull character {model.character_scale:g} dilemma {model.dilemma_scale:g} "
        f"resolution {model.resolution_scale:g} "
        f"batch {batch_size} steps/epoch {steps_per_epoch} "
        f"windows story follow-on categories on resolution on",
        flush=True,
    )

    # One compiled step pays for the graph. The budget starts after that.
    warm_batches, warm_worker = prefetch_batches(
        stories,
        steps=1,
        batch_size=batch_size,
        seq_len=seq_len,
        eos_id=eos_id,
        seed=0,
        labels=labels,
    )
    warmup = warm_batches.get()
    warm_worker.join()
    # The compiled step writes the optimizer too. Leave that update unevaluated
    # and the next step reads a half-built Adam state and the loss goes NaN.
    warmup_loss = run_windows(warmup)
    print(f"compiled warmup loss {float(warmup_loss):.3f}", flush=True)

    started = time.perf_counter()
    deadline = math.inf if float(args.seconds) <= 0 else started + float(args.seconds)
    next_log = started + _LOG_SECONDS
    epoch = 0
    seen_steps = 0
    mean_loss = float("nan")
    best_loss = float("inf")
    best_epoch = 0
    quiet_epochs = 0
    epoch_cost = 0.0
    stop = ""
    scales_default = (
        float(args.character_scale) == CHARACTER_DRIVE_SCALE
        and float(args.dilemma_scale) == STICKY_DRIVE_SCALE
        and float(args.resolution_scale) == RESOLUTION_DRIVE_SCALE
    )
    pull_default = float(args.heavy_pull) == HEAVY_PULL
    default_pull = scales_default and pull_default
    if args.all_stories:
        shelf_name = "all" if default_pull else "all_pull"
    elif default_pull:
        shelf_name = "fullstory"
    else:
        shelf_name = (
            f"c{args.character_scale:g}_d{args.dilemma_scale:g}_"
            f"r{args.resolution_scale:g}_pull{float(args.heavy_pull):g}"
        ).replace(".", "p")
    # Rank and the slow band are in the directory. This restore run writes
    # beside the rank-64 bag that reached loss 1.715, and beside the ktr_*
    # runs, instead of over them.
    ckpt_dir = (
        args.data_dir
        / f"checkpoints_n{model.n}_r{int(args.rank)}_slow{model.n_slow}_restore_{shelf_name}"
    )
    print(
        f"marker: plain tug is the step, route off, "
        f"rank-{RANK} / dynamics-{DYNAMICS_LR_SCALE:g}",
        flush=True,
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    index_path = ckpt_dir / "index.jsonl"
    if args.resume is None:
        index_path.write_text("", encoding="utf-8")
    else:
        epoch, best_loss, quiet_epochs, best_epoch = prior_progress(
            index_path,
            float(args.plateau_delta),
        )
        numbered = _epoch_number(Path(args.resume))
        if numbered is not None:
            epoch = max(epoch, numbered)
        print(
            f"resuming after epoch {epoch} best {best_loss:.3f} "
            f"lr {float(args.lr):.4g} from {args.resume}",
            flush=True,
        )
    while time.perf_counter() < deadline:
        epoch += 1
        taken = 0
        batches, worker = prefetch_batches(
            stories,
            steps=steps_per_epoch,
            batch_size=batch_size,
            seq_len=seq_len,
            eos_id=eos_id,
            seed=int(args.seed) + epoch,
            labels=labels,
        )
        total = 0.0
        epoch_started = time.perf_counter()
        for _ in range(steps_per_epoch):
            batch = batches.get()
            loss = run_windows(batch)
            value = float(loss)
            if not math.isfinite(value):
                raise RuntimeError(f"loss became {value} at epoch {epoch}")
            total += value
            taken += 1
            seen_steps += 1
            now = time.perf_counter()
            if now >= next_log:
                print(
                    f"epoch {epoch} step {taken}/{steps_per_epoch} "
                    f"loss {value:.3f} {now - started:.0f}s",
                    flush=True,
                )
                next_log = now + _LOG_SECONDS
            if now >= deadline:
                break
        if taken < steps_per_epoch:
            _drain_prefetch(batches, worker)
            stop = "time"
        else:
            worker.join()
        mean_loss = total / taken
        epoch_cost = time.perf_counter() - epoch_started
        now = time.perf_counter()
        improved = mean_loss < best_loss - float(args.plateau_delta)
        if improved:
            best_loss = mean_loss
            best_epoch = epoch
            quiet_epochs = 0
            save_bag(model, ckpt_dir / "best.safetensors")
        else:
            quiet_epochs += 1
        save_bag(model, args.data_dir / "kuramoto_mlx.safetensors")
        hit_target = args.target_loss is not None and mean_loss <= float(args.target_loss)
        if hit_target:
            stop = "target"
        elif quiet_epochs >= int(args.plateau_patience):
            stop = "plateau"
        overrun = (
            not stop
            and bool(epoch_cost)
            and time.perf_counter() + epoch_cost > deadline
        )
        if overrun:
            stop = "time"
        numbered = ckpt_dir / f"epoch_{epoch:04d}.safetensors"
        keep_numbered = epoch % _SAVE_EVERY == 0 or bool(stop)
        if keep_numbered:
            save_bag(model, numbered)
        record = {
            "epoch": epoch,
            "loss": round(mean_loss, 4),
            "seconds": round(now - started),
            "n_slow": model.n_slow,
            "n_stamp": model.n_stamp,
            "n_arc": model.n_arc,
            "stories": len(stories),
        }
        if keep_numbered:
            record["weights"] = str(numbered)
        with (ckpt_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        print(
            f"epoch {epoch} loss {mean_loss:.3f} best {best_loss:.3f} "
            f"slots {model.catalog_filled} quiet {quiet_epochs} {now - started:.0f}s"
            + (f" saved {numbered.name}" if keep_numbered else ""),
            flush=True,
        )
        next_log = now + _LOG_SECONDS
        if stop == "plateau":
            print(
                f"stopping: loss did not beat {best_loss:.3f} by "
                f"{float(args.plateau_delta):.3f} for {quiet_epochs} epochs",
                flush=True,
            )
        elif overrun:
            print("stopping before the next pass would pass the time budget", flush=True)
        if stop:
            break
    if not stop:
        stop = "time"
    if epoch:
        final_path = ckpt_dir / f"epoch_{epoch:04d}.safetensors"
        if not final_path.is_file():
            save_bag(model, final_path)
    if best_epoch:
        load_bag_weights(model, ckpt_dir / "best.safetensors")
        weights = save_bag(model, args.data_dir / "kuramoto_mlx.safetensors")
    else:
        weights = args.data_dir / "kuramoto_mlx.safetensors"
    print(
        f"epoch {epoch} loss {mean_loss:.3f} final stop {stop} "
        f"best epoch {best_epoch} loss {best_loss:.3f}",
        flush=True,
    )

    text = sample(
        model,
        itos,
        chunk=chunk,
        prompt=args.prompt,
        length=int(args.sample_length),
        temperature=float(args.sample_temperature),
        repetition_penalty=1.3,
    )
    elapsed = time.perf_counter() - started
    print(f"sample:\n{text}", flush=True)
    from sidecar.query import report_cohesion  # noqa: PLC0415

    report_cohesion(
        model,
        itos,
        stories,
        chunk=chunk,
        count=4,
        length=int(args.sample_length),
        temperature=float(args.sample_temperature),
        repetition_penalty=1.3,
    )
    print(f"trained {seen_steps} steps in {elapsed:.0f}s", flush=True)
    print(f"saved {weights}", flush=True)


def _epoch_number(path: Path) -> int | None:
    """Read the pass number from ``epoch_0010.safetensors``."""
    match = re.search(r"epoch_(\d+)", path.name)
    if match is None:
        return None
    return int(match.group(1))


def prior_progress(index_path: Path, delta: float) -> tuple[int, float, int, int]:
    """Replay the checkpoint log: last pass, best loss, quiet passes, best pass."""
    rows: list[dict] = []
    if index_path.exists():
        rows.extend(
            json.loads(line)
            for line in index_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    if not rows:
        return 0, float("inf"), 0, 0
    best_loss = float("inf")
    best_epoch = 0
    quiet = 0
    for row in rows:
        loss = float(row["loss"])
        finished = int(row["epoch"])
        if loss < best_loss - delta:
            best_loss = loss
            best_epoch = finished
            quiet = 0
        else:
            quiet += 1
    return int(rows[-1]["epoch"]), best_loss, quiet, best_epoch


def widths(saved: dict) -> tuple[int, int, int]:
    """Read wheel count, tug rank, and piece-drive width from a saved bag."""
    n, rank = saved["U"].shape
    n_roots = saved["D"].shape[1]
    return int(n), int(rank), int(n_roots)


def load_bag_weights(model: KuramotoMLX, path: Path) -> None:
    """Load a bag. Missing ``route_K`` leaves the memory route off.

    A file from before the catalog has no ``K`` and no patterns, so the step
    is the low-rank tug alone. ``readout_up.bias`` is the factored readout.
    Otherwise the file has ``readout.bias`` and the readout rank is 0. Keys
    this model does not own are ignored.
    """
    saved = dict(mx.load(str(path)))
    present = "route_K" in saved and "patterns" in saved
    if "route_K" not in saved:
        saved["route_K"] = mx.zeros(model.route_K.shape)
    if "patterns" not in saved:
        saved["patterns"] = mx.zeros(model.patterns.shape)
    if "mem_U" not in saved:
        saved["mem_U"] = mx.zeros(model.mem_U.shape)
    if "mem_V" not in saved:
        saved["mem_V"] = mx.zeros(model.mem_V.shape)
    model.load_weights([(key, saved[key]) for key, _ in tree_flatten(model.parameters())])
    model.use_route = present


def readout_spec(saved: dict) -> tuple[int, int]:
    """Vocab size and narrow-readout width. Width 0 is the full linear map."""
    if "readout_up.bias" in saved:
        vocab = int(saved["readout_up.bias"].shape[0])
        rank = int(saved["readout_down.weight"].shape[0])
        return vocab, rank
    return int(saved["readout.bias"].shape[0]), 0


def read_bag_meta(path: Path) -> dict[str, float]:
    """Read the clock record saved next to a bag. Missing means an even kick."""
    meta_path = path.with_suffix(".json")
    if not meta_path.is_file():
        return {
            "n_slow": 0,
            "n_stamp": 0,
            "n_arc": 0,
            "heavy_pull": 1.0,
            "heavy_dt_scale": 1.0,
            "latch": 0,
            "latch_blend": 0.25,
        }
    raw = json.loads(meta_path.read_text(encoding="utf-8"))
    # The previous run used one slow clock for every set-aside wheel.
    n_slow = int(raw.get("n_slow", raw.get("n_heavy", 0)))
    n_stamp = int(raw.get("n_stamp", 0))
    return {
        "n_slow": n_slow,
        "n_stamp": n_stamp,
        "n_arc": int(raw.get("n_arc", 0)),
        "heavy_pull": float(raw.get("heavy_pull", 1.0)),
        "heavy_dt_scale": float(raw.get("heavy_dt_scale", 1.0)),
        "latch": float(raw.get("latch", 0)),
        "latch_blend": float(raw.get("latch_blend", 0.25)),
    }


def save_bag(model: KuramotoMLX, path: Path) -> Path:
    """Write the wheel weights. The file is a usable query checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_weights(str(path))
    meta = {
        "n_slow": model.n_slow,
        "n_stamp": model.n_stamp,
        "n_arc": model.n_arc,
        "heavy_pull": model.heavy_pull,
        "heavy_dt_scale": model.heavy_dt_scale,
        "slow_from": model.slow.start,
        "stamp_from": model.stamp.start,
        "arc_from": model.arc.start,
        "latch": int(model.latch),
        "latch_blend": model.latch_blend,
    }
    path.with_suffix(".json").write_text(json.dumps(meta) + "\n", encoding="utf-8")
    return path


def _drain_prefetch(batches: queue.Queue, worker: threading.Thread) -> None:
    """Empty a producer that was stopped before its last batch."""
    while worker.is_alive():
        with contextlib.suppress(queue.Empty):
            batches.get(timeout=0.2)
    worker.join()
    while True:
        try:
            batches.get_nowait()
        except queue.Empty:
            return


def _takes_repetition_tax(piece: str) -> bool:
    """Content words and a paragraph break take the tax.

    Glue, including ``and``, and sentence marks stay at full score. A blank
    line is once a paragraph, so it takes the same tax as a content word.
    """
    if "\n" in piece:
        return True
    if piece == EOS:
        return False
    return _content_word(piece) != ""


def apply_repetition_penalty(
    logits: mx.array,
    seen: list[int],
    penalty: float,
    itos: list[str],
) -> mx.array:
    """Down-weight taxed pieces already used. Glue and sentence marks stay.

    ``Bird`` and ``bird`` are one word, so using either taxes both.
    """
    if penalty == 1.0 or not seen:
        return logits
    groups = {piece.casefold(): [] for piece in itos}
    for index, piece in enumerate(itos):
        groups[piece.casefold()].append(index)
    taxed = [
        sibling
        for token in set(seen)
        for sibling in groups[itos[token].casefold()]
        if _takes_repetition_tax(itos[sibling])
    ]
    if not taxed:
        return logits
    mask = mx.zeros((logits.shape[0],))
    mask = mask.at[mx.array(list(set(taxed)))].add(1)
    adjusted = mx.where(logits < 0, logits * penalty, logits / penalty)
    return mx.where(mask > 0, adjusted, logits)


def _opens_sentence(piece: str | None) -> bool:
    """Return whether the next piece opens a sentence."""
    if piece is None:
        return True
    if any(ch.isalpha() for ch in piece):
        return False
    return any(ch in ".!?\n" for ch in piece)


def _capital_piece(piece: str) -> bool:
    stripped = piece.lstrip()
    return bool(stripped) and stripped[0].isupper()


def _realize_case(
    itos: list[str],
    group: list[int],
    logits: mx.array,
    *,
    sentence_start: bool,
) -> int:
    """Write the capital after a sentence start, and the lower form inside one."""
    if sentence_start:
        pool = [index for index in group if _capital_piece(itos[index])] or group
    else:
        pool = [index for index in group if not _capital_piece(itos[index])] or group
    best = pool[0]
    best_score = float(logits[best].item())
    for index in pool[1:]:
        score = float(logits[index].item())
        if score > best_score:
            best = index
            best_score = score
    return best


def choose_piece(  # noqa: PLR0913
    logits: mx.array,
    itos: list[str],
    groups: list[list[int]],
    table: mx.array,
    *,
    previous: str | None,
    temperature: float,
) -> int:
    """Pick a word without regard to case, then write the case the sentence needs.

    ``The`` and ``the`` share one score. After a period the written piece is
    ``The`` when that piece exists. Inside a sentence it is ``the``.
    """
    safe = mx.maximum(table, 0)
    picked = logits[safe]
    picked = mx.where(table >= 0, picked, mx.full(picked.shape, -1e9))
    scores = mx.logsumexp(picked, axis=1)
    if temperature <= 0:
        pick = int(mx.argmax(scores).item())
    else:
        pick = int(mx.random.categorical(scores / temperature).item())
    return _realize_case(
        itos,
        groups[pick],
        logits,
        sentence_start=_opens_sentence(previous),
    )


def _case_table(itos: list[str]) -> tuple[list[list[int]], mx.array]:
    groups = case_groups(itos)
    width = max(len(group) for group in groups)
    rows = [group + [-1] * (width - len(group)) for group in groups]
    return groups, mx.array(rows)


def continue_ids(  # noqa: PLR0913
    model: KuramotoMLX,
    itos: list[str],
    *,
    chunk: int,
    prompt: str,
    length: int,
    temperature: float = 0.0,
    repetition_penalty: float = 1.3,
    token_ids: list[int] | None = None,
    character: int = -1,
    dilemma: int = -1,
    canon: list[int] | None = None,
) -> list[int]:
    """Continue a prompt and return the piece ids, including the opening.

    ``character`` drives from the first kick. ``dilemma`` drives from the kick
    that says it. A negative id adds nothing, so a bare lead-in has neither.
    """
    stoi = {gram: index for index, gram in enumerate(itos)}
    eos_id = stoi.get(EOS)
    ids = list(token_ids) if token_ids is not None else chunk_ids(prompt, chunk=chunk, stoi=stoi)
    folded = canon or list(range(len(itos)))
    groups, table = _case_table(itos)
    theta = model.theta0
    hold = mx.zeros((1, 1))
    seen = False
    recognition = None
    for token in ids:
        seen, dilemma_now = _dilemma_now(token, dilemma, folded, seen=seen)
        theta, hold, recognition = spin(
            model,
            theta,
            hold,
            token,
            character=character,
            dilemma=dilemma_now,
            recognition=recognition,
        )
    for _ in range(length):
        shown = model.shown_state(theta[None, :], hold)[0]
        features = mx.concatenate([mx.sin(shown), mx.cos(shown)], axis=-1)
        logits = apply_repetition_penalty(
            model.logits_from(features[None, :])[0],
            ids,
            repetition_penalty,
            itos,
        )
        previous = None if not ids else itos[ids[-1]]
        nxt = choose_piece(
            logits,
            itos,
            groups,
            table,
            previous=previous,
            temperature=temperature,
        )
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
        seen, dilemma_now = _dilemma_now(nxt, dilemma, folded, seen=seen)
        theta, hold, recognition = spin(
            model,
            theta,
            hold,
            nxt,
            character=character,
            dilemma=dilemma_now,
            recognition=recognition,
        )
    return ids


def sample(  # noqa: PLR0913
    model: KuramotoMLX,
    itos: list[str],
    *,
    chunk: int,
    prompt: str,
    length: int,
    temperature: float = 0.0,
    repetition_penalty: float = 1.3,
    token_ids: list[int] | None = None,
    character: int = -1,
    dilemma: int = -1,
    canon: list[int] | None = None,
) -> str:
    """Continue a prompt. Stop on the end piece. Temperature 0 takes the top piece."""
    ids = continue_ids(
        model,
        itos,
        chunk=chunk,
        prompt=prompt,
        length=length,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        token_ids=token_ids,
        character=character,
        dilemma=dilemma,
        canon=canon,
    )
    return "".join(itos[index] for index in ids)


def _dilemma_now(token: int, dilemma: int, canon: list[int], *, seen: bool) -> tuple[bool, int]:
    """Turn the dilemma drive on at the kick that says it."""
    if dilemma >= 0 and canon[token] == dilemma:
        seen = True
    return seen, dilemma if seen else -1


def spin(  # noqa: PLR0913
    model: KuramotoMLX,
    theta: mx.array,
    hold: mx.array,
    token: int,
    *,
    character: int = -1,
    dilemma: int = -1,
    recognition: mx.array | None = None,
) -> tuple[mx.array, mx.array, mx.array]:
    """Advance one token and fold the phase back onto the circle.

    The returned recognition is the overlap of the pose this kick left, so
    the next kick's ``K`` transpose ``r`` lags by one word. The end piece
    returns the story to a uniform recognition.
    """
    tokens = mx.array([token])
    if model.latch:
        hold, theta_b = model.commit_sentence(hold, theta[None, :], tokens)
        theta = theta_b[0]
    else:
        theta_b = theta[None, :]
    left = model._overlap(model.shown_state(theta_b, hold))  # noqa: SLF001
    nxt = theta_b + model.step_scale(tokens) * model.velocity(
        theta_b,
        tokens,
        hold,
        character=mx.array([character]),
        dilemma=mx.array([dilemma]),
        recognition=recognition,
    )
    nxt = mx.arctan2(mx.sin(nxt), mx.cos(nxt))[0]
    if model.eos_id is not None and token == model.eos_id:
        left = model._uniform_recognition(1)  # noqa: SLF001
    mx.eval(nxt, hold, left)
    return nxt, hold, left


if __name__ == "__main__":
    main()
