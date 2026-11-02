# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Train the next-piece model on Metal through MLX.

PyTorch's compiler cannot build the backward pass for this loop on this Mac.
MLX compiles the same Kuramoto step into one Metal graph, so the 16 GPU cores
spend their time on the spin instead of on a kernel launch per piece. Each
row is one story. Kicking the end piece returns the wheels to rest.
"""

from __future__ import annotations

import argparse
import contextlib
from functools import partial
import json
import math
from pathlib import Path
import queue
import re
import threading
import time
from typing import Any

from mlx import nn
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten, tree_map

from sidecar.model import clock_ranges
from sidecar.prepare import EOS, chunk_ids
from sidecar.train import (
    HEAVY_DT_SCALE,
    HEAVY_PULL,
    LATCH,
    LATCH_BLEND,
    N_OSCILLATORS,
    N_ROOTS,
    N_SLOW,
    N_STAMP,
    RANK,
    STORY_WINDOW,
    StoryLabels,
    label_shelf,
    limit_stories,
    load_corpus,
    pack_stream,
    prefetch_batches,
    prefetch_stream,
    sentence_end_ids,
    story_epoch_steps,
)

_PEAK_LR = 3e-3
_LOG_SECONDS = 15.0
_CATEGORY_PREVIEW = 8


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="Stop after this many seconds. 0 runs until the loss flattens. Each finished epoch is already saved.",
    )
    parser.add_argument(
        "--stories",
        type=int,
        default=None,
        help="Use only the first N packed stories. Default is the whole shelf.",
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
    parser.add_argument("--latch", action=argparse.BooleanOptionalAction, default=LATCH)
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
    parser.add_argument(
        "--no-beat-drives",
        action="store_true",
        help="Do not add the character or dilemma kick. The current word still drives the wheels.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Train the old way: random crops through one joined stream, no beat drives, no end-piece reset.",
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
        n_slow: int = N_SLOW,
        n_stamp: int = N_STAMP,
        heavy_pull: float = HEAVY_PULL,
        heavy_dt_scale: float = HEAVY_DT_SCALE,
        boundary_ids: tuple[int, ...] = (),
        latch: bool = LATCH,
        latch_blend: float = LATCH_BLEND,
        eos_id: int | None = None,
        beat_canon: list[int] | None = None,
    ) -> None:
        """Build the bag. The slow clock creeps. The latch holds one sentence angle."""
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.n = int(n)
        self.n_slow = int(n_slow)
        self.n_stamp = int(n_stamp)
        self.heavy_pull = float(heavy_pull)
        self.heavy_dt_scale = float(heavy_dt_scale)
        self.latch = bool(latch)
        self.latch_blend = float(latch_blend)
        self.latch_at = self.n - 1 if self.latch else -1
        self.eos_id = None if eos_id is None else int(eos_id)
        if self.latch and self.n_stamp:
            raise ValueError("The latch replaces the stamp clock.")
        self.slow, self.stamp = clock_ranges(self.n - int(self.latch), self.n_slow, self.n_stamp)
        self.boundary_ids = tuple(int(piece) for piece in boundary_ids)
        self.dt = 0.2
        if self.n_slow or self.n_stamp or self.latch:
            gain_parts = [mx.ones((self.slow.start,))]
            step_parts = [mx.full((self.slow.start,), self.dt)]
            if self.n_slow:
                gain_parts.append(mx.full((self.n_slow,), self.heavy_pull))
                step_parts.append(mx.full((self.n_slow,), self.dt * self.heavy_dt_scale))
            if self.n_stamp:
                gain_parts.append(mx.full((self.n_stamp,), self.heavy_pull))
                step_parts.append(mx.zeros((self.n_stamp,)))
            elif self.latch:
                middle = self.n - 1 - (self.slow.start + self.n_slow)
                if middle:
                    gain_parts.append(mx.ones((middle,)))
                    step_parts.append(mx.full((middle,), self.dt))
                gain_parts.append(mx.full((1,), self.heavy_pull))
                step_parts.append(mx.full((1,), self.dt * self.heavy_dt_scale))
            self._source_gain = mx.concatenate(gain_parts)
            self._step = mx.concatenate(step_parts)
            if self.n_stamp:
                self._stamp_mask = mx.concatenate(
                    [mx.zeros((self.n - self.n_stamp,)), mx.ones((self.n_stamp,))]
                )
            if self.n_stamp or self.latch:
                self._ends = mx.array(self.boundary_ids or (-1,))
        scale = self.n**-0.5
        self.omega = 0.01 * mx.random.normal((self.n,))
        self.U = scale * mx.random.normal((self.n, rank))
        self.V = scale * mx.random.normal((self.n, rank))
        self.root_phase = nn.Embedding(self.vocab_size, n_roots)
        self.D = 0.1 * mx.random.normal((self.n, n_roots))
        self.theta0 = mx.zeros((self.n,))
        self.readout = nn.Linear(2 * self.n, self.vocab_size)
        canon_ids = list(range(self.vocab_size)) if beat_canon is None else list(beat_canon)
        self._beat_canon = mx.array(canon_ids)

    def velocity(
        self,
        theta: mx.array,
        tokens: mx.array,
        latch: mx.array | None = None,
        *,
        character: mx.array | None = None,
        dilemma: mx.array | None = None,
    ) -> mx.array:
        """Phase velocity. ``K = U @ V.T`` is never formed.

        A non-negative character id adds that root drive beside the current
        piece. A non-negative dilemma id does the same. A negative id adds
        nothing. Resolution has no drive.
        """
        shown = self.shown_state(theta, latch)
        sin_src, cos_src = self._outbound(mx.sin(shown), mx.cos(shown))
        sin_k = (sin_src @ self.V) @ self.U.T
        cos_k = (cos_src @ self.V) @ self.U.T
        vel = self.omega + mx.cos(theta) * sin_k - mx.sin(theta) * cos_k
        phi = mx.tanh(self.root_phase(tokens)) * math.pi
        sin_p = mx.sin(phi)
        cos_p = mx.cos(phi)
        if character is not None:
            char_sin, char_cos = self._root_trig(character)
            sin_p = sin_p + char_sin
            cos_p = cos_p + char_cos
        if dilemma is not None:
            dil_sin, dil_cos = self._root_trig(dilemma)
            sin_p = sin_p + dil_sin
            cos_p = cos_p + dil_cos
        drive_sin = sin_p @ self.D.T
        drive_cos = cos_p @ self.D.T
        return vel + mx.cos(theta) * drive_sin - mx.sin(theta) * drive_cos

    def _root_trig(self, ids: mx.array) -> tuple[mx.array, mx.array]:
        """Sine and cosine of ``tanh(root_phase)`` for each id. Negatives are zero."""
        active = (ids >= 0).astype(self.D.dtype)[:, None]
        safe = mx.maximum(ids, 0)
        phi = mx.tanh(self.root_phase(safe)) * math.pi
        return mx.sin(phi) * active, mx.cos(phi) * active

    def _open_dilemma(
        self,
        seen: mx.array,
        tokens: mx.array,
        dilemma: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Mark rows whose dilemma piece has appeared, including this kick."""
        folded = self._beat_canon[tokens]
        seen = mx.logical_or(seen, mx.logical_and(dilemma >= 0, folded == dilemma))
        inactive = mx.zeros_like(dilemma) - 1
        return seen, mx.where(seen, dilemma, inactive)

    def _outbound(self, sin_t: mx.array, cos_t: mx.array) -> tuple[mx.array, mx.array]:
        """Multiply the tug coming from the set-aside wheels."""
        if not self.n_slow and not self.n_stamp and not self.latch:
            return sin_t, cos_t
        return sin_t * self._source_gain, cos_t * self._source_gain

    def _at_period(self, tokens: mx.array) -> mx.array:
        """Mark rows whose piece ends a sentence."""
        ends = self._ends.astype(tokens.dtype)
        return mx.any(tokens[:, None] == ends[None, :], axis=-1)

    def step_scale(self, tokens: mx.array) -> mx.array | float:
        """Per-wheel Euler step. The stamp clock moves only at a sentence end."""
        if not self.n_slow and not self.n_stamp and not self.latch:
            return self.dt
        if not self.n_stamp:
            return self._step
        end = self._at_period(tokens)[:, None]
        stamped = mx.where(self._stamp_mask, self.dt, self._step)
        return mx.where(end, stamped, self._step)

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
    ) -> mx.array:
        """Return next-piece logits ``(batch, time - 1, vocab)``.

        The bag starts at ``theta0``. After the end piece is kicked, the next
        kick starts at ``theta0`` again and the latch is clear. A run of end
        pieces after that first one is padding and is not kicked. A sentence
        period still only latches; it does not rest the bag. ``character`` is
        on from the first kick. ``dilemma`` turns on at the first piece in
        this window that folds to that id. Omit either one and that drive
        stays off.
        """
        batch = token_ids.shape[0]
        theta = mx.broadcast_to(self.theta0, (batch, self.n))
        latch = mx.zeros((batch, 1))
        # True until this row has kicked its end piece and not yet started another.
        in_story = mx.ones((batch,), dtype=mx.bool_)
        seen = mx.zeros((batch,), dtype=mx.bool_)
        logits = []
        for t in range(token_ids.shape[1] - 1):
            tokens = token_ids[:, t]
            dilemma_now = None
            if dilemma is not None:
                seen, dilemma_now = self._open_dilemma(seen, tokens, dilemma)
            if self.eos_id is None:
                is_end = mx.zeros((batch,), dtype=mx.bool_)
            else:
                is_end = tokens == self.eos_id
            # Padding is an end piece that follows the end piece. A later real
            # piece opens the next story and is kicked from rest.
            padding = mx.logical_and(mx.logical_not(in_story), is_end)
            take = mx.logical_not(padding)[:, None]
            if self.latch:
                held, committed = self.commit_sentence(latch, theta, tokens)
                latch = mx.where(take, held, latch)
                theta = mx.where(take, committed, theta)
            stepped = theta + self.step_scale(tokens) * self.velocity(
                theta,
                tokens,
                latch,
                character=character,
                dilemma=dilemma_now,
            )
            stepped = mx.arctan2(mx.sin(stepped), mx.cos(stepped))
            theta = mx.where(take, stepped, theta)
            shown = self.shown_state(theta, latch)
            features = mx.concatenate([mx.sin(shown), mx.cos(shown)], axis=-1)
            logits.append(self.readout(features))
            if self.eos_id is not None:
                reset = mx.logical_and(take.reshape(is_end.shape), is_end)[:, None]
                rest = mx.broadcast_to(self.theta0, theta.shape)
                theta = mx.where(reset, rest, theta)
                latch = mx.where(reset, mx.zeros_like(latch), latch)
                in_story = mx.logical_not(is_end)
        return mx.stack(logits, axis=1)


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


def loss_fn(
    model: KuramotoMLX,
    tokens: mx.array,
    character: mx.array | None = None,
    dilemma: mx.array | None = None,
) -> mx.array:
    """Mean next-piece cross-entropy. Padding after the end piece adds nothing."""
    logits = model(tokens, character, dilemma)
    targets = tokens[:, 1:]
    losses = nn.losses.cross_entropy(
        logits.reshape(-1, model.vocab_size),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    weights = _story_weights(tokens, model.eos_id).astype(losses.dtype)
    # A pad target can be non-finite. Multiplying by zero would keep that NaN.
    kept = mx.where(weights > 0, losses, mx.zeros_like(losses))
    total = mx.sum(weights)
    return mx.sum(kept) / mx.maximum(total, mx.ones_like(total))


def _finite_grads(grads: dict[str, Any]) -> dict[str, Any]:
    """Drop an update whose backward overflowed.

    A long story can make part of the backward non-finite while the loss is
    still finite. ``clip_grad_norm`` then multiplies every weight by NaN.
    The entries that stayed finite are overflow residue, so a clipped step
    along them drives the loss up. Skip the whole update instead.
    """
    flags: list[mx.array] = []

    def mark(value: mx.array) -> mx.array:
        if isinstance(value, mx.array) and mx.issubdtype(value.dtype, mx.floating):
            flags.append(mx.any(~mx.isfinite(value)))
        return value

    tree_map(mark, grads)
    if not flags:
        return grads
    bad = mx.any(mx.stack(flags))

    def clean(value: mx.array) -> mx.array:
        if not isinstance(value, mx.array) or not mx.issubdtype(value.dtype, mx.floating):
            return value
        return mx.where(bad, mx.zeros_like(value), value)

    return tree_map(clean, grads)


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
    labels, canon = label_shelf(stories, itos)
    print_categories(labels, itos)
    seq_len = int(args.seq_len)
    batch_size = int(args.batch_size)
    eos_id = itos.index(EOS)
    total_tokens = sum(len(story) for story in stories)
    stream = pack_stream(stories) if args.stream else []
    if args.stream:
        steps_per_epoch = max(1, total_tokens // (batch_size * seq_len))
    else:
        steps_per_epoch = story_epoch_steps(len(stories), batch_size)

    model = KuramotoMLX(
        len(itos),
        n=int(args.n_oscillators),
        rank=int(args.rank),
        n_roots=int(args.n_roots),
        n_slow=int(args.n_slow),
        n_stamp=int(args.n_stamp),
        heavy_pull=float(args.heavy_pull),
        heavy_dt_scale=float(args.heavy_dt_scale),
        boundary_ids=tuple(sentence_end_ids(itos)),
        latch=bool(args.latch),
        latch_blend=float(args.latch_blend),
        eos_id=None if args.stream else eos_id,
        beat_canon=canon,
    )
    if args.resume is not None:
        resume_from = Path(args.resume)
        if not resume_from.exists():
            raise FileNotFoundError(f"No saved bag at {resume_from}.")
        saved = mx.load(str(resume_from))
        saved_vocab = int(saved["readout.bias"].shape[0])
        if saved_vocab != len(itos):
            raise ValueError(
                f"Saved bag has {saved_vocab} pieces; the vocabulary on disk has {len(itos)}."
            )
        n, rank, n_roots = widths(saved)
        if (n, rank, n_roots) != (model.n, int(args.rank), int(args.n_roots)):
            raise ValueError(
                f"Saved bag is {n} wheels, rank {rank}, roots {n_roots}; "
                f"this run is {model.n}, {int(args.rank)}, {int(args.n_roots)}."
            )
        model.load_weights(str(resume_from))
    mx.eval(model.parameters())

    def raw_loss(tokens: mx.array, character: mx.array, dilemma: mx.array) -> mx.array:
        return loss_fn(model, tokens, character, dilemma)

    loss_and_grad = nn.value_and_grad(model, raw_loss)
    optimizer = optim.AdamW(learning_rate=float(args.lr), weight_decay=0.01)

    def step(tokens: mx.array, character: mx.array, dilemma: mx.array) -> mx.array:
        loss, grads = loss_and_grad(tokens, character, dilemma)
        # Full backprop through a long story can overflow. Skip that update
        # so the next batch is not trained on NaN weights.
        grads = _finite_grads(grads)
        grads, _norm = optim.clip_grad_norm(grads, 1.0)
        optimizer.update(model, grads)
        return loss

    state = [model.state, optimizer.state]
    compiled = partial(mx.compile, inputs=state, outputs=state)(step)
    flat = sum(value.size for _, value in tree_flatten(model.parameters()))
    print(
        f"metal {mx.default_device()} stories {len(stories)} tokens {total_tokens} "
        f"chunk {chunk} vocab {len(itos)} params {flat} "
        f"wheels {model.n} rank {int(args.rank)} roots {int(args.n_roots)} "
        f"slow {model.slow.start}-{model.slow.stop - 1} step {model.dt * model.heavy_dt_scale:g} "
        f"latch {model.latch_at} blend {model.latch_blend:g} pull {model.heavy_pull:g} "
        f"batch {batch_size} steps/epoch {steps_per_epoch} "
        f"beat-drives {'off' if args.no_beat_drives or args.stream else 'on'} "
        f"windows {'stream' if args.stream else 'story'}",
        flush=True,
    )

    # One compiled step pays for the graph. The budget starts after that.
    if args.stream:
        warm_batches, warm_worker = prefetch_stream(
            stream, steps=1, batch_size=batch_size, seq_len=seq_len, seed=0,
        )
        warmup_tokens = mx.array(warm_batches.get().numpy())
        warm_worker.join()
        warmup_character = mx.full((batch_size,), -1)
        warmup_dilemma = mx.full((batch_size,), -1)
    else:
        warmup_tokens, warmup_character, warmup_dilemma = prefetch_one(
            stories,
            batch_size=batch_size,
            seq_len=seq_len,
            eos_id=eos_id,
            seed=0,
            labels=labels,
        )
    if args.no_beat_drives:
        warmup_character = mx.full(warmup_character.shape, -1)
        warmup_dilemma = mx.full(warmup_dilemma.shape, -1)
    warmup_loss = compiled(warmup_tokens, warmup_character, warmup_dilemma)
    # The compiled step writes the optimizer too. Leave that update unevaluated
    # and the next step reads a half-built Adam state and the loss goes NaN.
    mx.eval(warmup_loss, model.parameters(), optimizer.state)
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
    if args.stream:
        ckpt_name = "checkpoints_stream"
    elif args.no_beat_drives:
        ckpt_name = "checkpoints_nobeat"
    else:
        ckpt_name = "checkpoints"
    ckpt_dir = args.data_dir / ckpt_name
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
        if epoch_cost and time.perf_counter() + epoch_cost > deadline:
            stop = "time"
            print("stopping before the next pass would pass the time budget", flush=True)
            break
        epoch += 1
        taken = 0
        if args.stream:
            batches, worker = prefetch_stream(
                stream,
                steps=steps_per_epoch,
                batch_size=batch_size,
                seq_len=seq_len,
                seed=int(args.seed) + epoch,
            )
        else:
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
            if args.stream:
                tokens = mx.array(batch.numpy())
                character = mx.full((batch_size,), -1)
                dilemma = mx.full((batch_size,), -1)
            else:
                tokens = mx.array(batch.tokens.numpy())
                character = mx.array(batch.character.numpy())
                dilemma = mx.array(batch.dilemma.numpy())
            if args.no_beat_drives:
                character = mx.full(character.shape, -1)
                dilemma = mx.full(dilemma.shape, -1)
            loss = compiled(tokens, character, dilemma)
            mx.eval(loss, model.parameters(), optimizer.state)
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
        epoch_path = save_bag(model, ckpt_dir / f"epoch_{epoch:04d}.safetensors")
        record = {
            "epoch": epoch,
            "loss": round(mean_loss, 4),
            "seconds": round(now - started),
            "weights": str(epoch_path),
            "n_slow": model.n_slow,
            "n_stamp": model.n_stamp,
            "stories": len(stories),
        }
        with (ckpt_dir / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        improved = mean_loss < best_loss - float(args.plateau_delta)
        if improved:
            best_loss = mean_loss
            best_epoch = epoch
            quiet_epochs = 0
            save_bag(model, ckpt_dir / "best.safetensors")
        else:
            quiet_epochs += 1
        if not args.no_beat_drives and not args.stream:
            save_bag(model, args.data_dir / "kuramoto_mlx.safetensors")
        hit_target = args.target_loss is not None and mean_loss <= float(args.target_loss)
        print(
            f"epoch {epoch} loss {mean_loss:.3f} best {best_loss:.3f} "
            f"quiet {quiet_epochs} {now - started:.0f}s saved {epoch_path.name}",
            flush=True,
        )
        next_log = now + _LOG_SECONDS
        if hit_target:
            stop = "target"
            break
        if quiet_epochs >= int(args.plateau_patience):
            stop = "plateau"
            print(
                f"stopping: loss did not beat {best_loss:.3f} by "
                f"{float(args.plateau_delta):.3f} for {quiet_epochs} epochs",
                flush=True,
            )
            break
        if stop == "time":
            break
    if not stop:
        stop = "time"
    if best_epoch:
        model.load_weights(str(ckpt_dir / "best.safetensors"))
        if not args.no_beat_drives and not args.stream:
            weights = save_bag(model, args.data_dir / "kuramoto_mlx.safetensors")
        else:
            weights = ckpt_dir / "best.safetensors"
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
        repetition_penalty=1.1,
    )
    elapsed = time.perf_counter() - started
    print(f"sample:\n{text}", flush=True)
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


def read_bag_meta(path: Path) -> dict[str, float]:
    """Read the clock record saved next to a bag. Missing means an even kick."""
    meta_path = path.with_suffix(".json")
    if not meta_path.is_file():
        return {
            "n_slow": 0,
            "n_stamp": 0,
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
        "heavy_pull": model.heavy_pull,
        "heavy_dt_scale": model.heavy_dt_scale,
        "slow_from": model.slow.start,
        "stamp_from": model.stamp.start,
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


def prefetch_one(  # noqa: PLR0913
    stories: list[list[int]],
    *,
    batch_size: int,
    seq_len: int,
    eos_id: int,
    seed: int,
    labels: list[StoryLabels],
) -> tuple[mx.array, mx.array, mx.array]:
    """Draw one compiled-warmup batch of whole stories and their beat ids."""
    batches, worker = prefetch_batches(
        stories,
        steps=1,
        batch_size=batch_size,
        seq_len=seq_len,
        eos_id=eos_id,
        seed=seed,
        labels=labels,
    )
    batch = batches.get()
    worker.join()
    return (
        mx.array(batch.tokens.numpy()),
        mx.array(batch.character.numpy()),
        mx.array(batch.dilemma.numpy()),
    )


def apply_repetition_penalty(logits: mx.array, seen: list[int], penalty: float) -> mx.array:
    """Down-weight pieces already used, the usual decoder repetition penalty."""
    if penalty == 1.0 or not seen:
        return logits
    mask = mx.zeros((logits.shape[0],))
    mask = mask.at[mx.array(list(set(seen)))].add(1)
    adjusted = mx.where(logits < 0, logits * penalty, logits / penalty)
    return mx.where(mask > 0, adjusted, logits)


def sample(  # noqa: PLR0913
    model: KuramotoMLX,
    itos: list[str],
    *,
    chunk: int,
    prompt: str,
    length: int,
    temperature: float = 0.0,
    repetition_penalty: float = 1.0,
    token_ids: list[int] | None = None,
    character: int = -1,
    dilemma: int = -1,
    canon: list[int] | None = None,
) -> str:
    """Continue a prompt. Stop on the end piece. Temperature 0 takes the top piece.

    ``character`` drives from the first kick. ``dilemma`` drives from the kick
    where that piece appears, including in ``token_ids``. A negative id adds
    nothing. A bare lead-in has neither until the words are in the text.
    """
    stoi = {gram: index for index, gram in enumerate(itos)}
    eos_id = stoi.get(EOS)
    ids = list(token_ids) if token_ids is not None else chunk_ids(prompt, chunk=chunk, stoi=stoi)
    folded = canon or list(range(len(itos)))
    theta = model.theta0
    hold = mx.zeros((1, 1))
    seen = False
    for token in ids:
        seen, dilemma_now = _dilemma_now(token, dilemma, folded, seen=seen)
        theta, hold = spin(model, theta, hold, token, character=character, dilemma=dilemma_now)
    for _ in range(length):
        shown = model.shown_state(theta[None, :], hold)[0]
        features = mx.concatenate([mx.sin(shown), mx.cos(shown)], axis=-1)
        logits = apply_repetition_penalty(
            model.readout(features[None, :])[0],
            ids,
            repetition_penalty,
        )
        if temperature <= 0:
            nxt = int(mx.argmax(logits).item())
        else:
            nxt = int(mx.random.categorical(logits / temperature).item())
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
        seen, dilemma_now = _dilemma_now(nxt, dilemma, folded, seen=seen)
        theta, hold = spin(model, theta, hold, nxt, character=character, dilemma=dilemma_now)
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
) -> tuple[mx.array, mx.array]:
    """Advance one token and fold the phase back onto the circle."""
    tokens = mx.array([token])
    if model.latch:
        hold, theta_b = model.commit_sentence(hold, theta[None, :], tokens)
        theta = theta_b[0]
    else:
        theta_b = theta[None, :]
    nxt = theta_b + model.step_scale(tokens) * model.velocity(
        theta_b,
        tokens,
        hold,
        character=mx.array([character]),
        dilemma=mx.array([dilemma]),
    )
    nxt = mx.arctan2(mx.sin(nxt), mx.cos(nxt))[0]
    mx.eval(nxt, hold)
    return nxt, hold


if __name__ == "__main__":
    main()
