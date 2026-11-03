# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Next-piece Kuramoto model with a low-rank clutch matrix.

The image model stores a dense ``n`` by ``n`` clutch matrix. Here the same tug is
factored as ``K = U @ V.T`` with a thin rank, so the step never builds that
square array. A piece shifts the angles that enter that tug. One Euler kick
updates the bag. Sine and cosine of the phases are read out as scores for
the next piece.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

_MIN_OSCILLATORS = 2
# Read off a short story such as Mia and the birdcage. The name stays in
# focus the whole way. The trouble is announced and held lightly, then fades
# when the ending starts. The ending pulls hardest, so the door can open and
# the bird can leave. Per-story arc weights are a later logic.
CHARACTER_DRIVE_SCALE = 1.5
STICKY_DRIVE_SCALE = 0.2
RESOLUTION_DRIVE_SCALE = 3.0
# Catalog of phase patterns. Not the vocabulary, and not a wheel-by-wheel map.
# The live paper (arXiv 2607.18439) stores M patterns and a routing matrix.
N_MEMORIES = 32
# Sharpness of the overlap softmax. The paper's symbol is gamma.
RECOGNITION_GAMMA = 50.0
# A nudge clicks a slot when its mean cosine with that pose is at least this.
# Below it, the nudge did not land, and the catalog does not record it.
WRITE_ALIGN = 0.5
# Counts fade by half each pass, so the untrained start does not own the order.
ROUTE_COUNT_KEEP = 0.5


def clock_ranges(
    n: int,
    n_slow: int,
    n_stamp: int,
    n_arc: int = 0,
) -> tuple[range, ...]:
    """Split the end of the bag into the slow clock, the stamp clock, and the arc.

    The slow clock takes a short step on every word. The stamp clock takes a
    full step only when the piece ends a sentence, and holds still otherwise.
    ``n_arc`` wheels sit after the stamp. Each takes a full step only while the
    resolution drive is on for that row, and holds still otherwise. Omitting
    ``n_arc`` keeps the two-clock split: 256 with 8 and 8 is slow 240–247 and
    stamp 248–255. The same bag with ``n_arc=1`` is slow 239–246, stamp
    247–254, and arc wheel 255.

    More wheels are a wider bag, not a longer story. A longer story is already
    follow-on windows that carry the phase and stop the gradient at the window
    boundary. Either count may be 0. At least one wheel keeps the ordinary word step.
    """
    if n_slow < 0 or n_stamp < 0 or n_arc < 0 or n_slow + n_stamp + n_arc >= n:
        raise ValueError("n_slow, n_stamp, and n_arc must leave at least one word wheel.")
    arc_at = n - n_arc
    stamp_at = arc_at - n_stamp
    slow_at = stamp_at - n_slow
    slow = range(slow_at, stamp_at)
    stamp = range(stamp_at, arc_at)
    if n_arc == 0:
        return slow, stamp
    return slow, stamp, range(arc_at, n)


def _clock_split(
    n: int,
    n_slow: int,
    n_stamp: int,
    n_arc: int,
    *,
    latch: bool,
) -> tuple[range, range, range]:
    """Return slow, stamp, and arc ranges. An omitted arc is an empty range."""
    if latch and n_stamp:
        raise ValueError("The latch replaces the stamp clock.")
    if latch and n_arc:
        raise ValueError("The latch and the arc wheel both want the last wheel.")
    body = n - int(latch)
    split = clock_ranges(body, n_slow, n_stamp, n_arc)
    arc = split[2] if n_arc else range(body, body)
    return split[0], split[1], arc


class KuramotoWordLM(nn.Module):
    """Recurrent low-rank Kuramoto bag that predicts the next piece."""

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        *,
        n_oscillators: int = 64,
        rank: int = 8,
        n_roots: int = 4,
        readout_rank: int = 0,
        euler_steps: int = 1,
        dt: float = 0.2,
        n_slow: int = 0,
        n_stamp: int = 0,
        n_arc: int = 1,
        heavy_pull: float = 4.0,
        heavy_dt_scale: float = 0.1,
        boundary_ids: tuple[int, ...] = (),
        latch: bool = False,
        latch_blend: float = 0.25,
        eos_id: int | None = None,
        canon_ids: list[int] | None = None,
        beat_canon_ids: list[int] | None = None,
    ) -> None:
        """Build a bag of ``n_oscillators`` with rank-``rank`` clutches.

        ``n_arc`` defaults to 1, the same count as ``sidecar.train.N_ARC``.
        That wheel is stolen from the word wheels. It is not a 257th wheel.
        """
        super().__init__()
        if min(n_oscillators, rank, n_roots, euler_steps) < 1 or n_oscillators < _MIN_OSCILLATORS:
            raise ValueError("Need n_oscillators >= 2, rank >= 1, n_roots >= 1, euler_steps >= 1.")
        if not 0 < latch_blend <= 1:
            raise ValueError("latch_blend must be in (0, 1].")
        if heavy_pull <= 0 or heavy_dt_scale <= 0:
            raise ValueError("heavy_pull and heavy_dt_scale must be positive.")
        self.vocab_size = int(vocab_size)
        self.n = int(n_oscillators)
        self.rank = int(rank)
        self.n_roots = int(n_roots)
        self.readout_rank = int(readout_rank)
        self.euler_steps = int(euler_steps)
        self.dt = float(dt)
        self.n_slow = int(n_slow)
        self.n_stamp = int(n_stamp)
        self.n_arc = int(n_arc)
        self.slow, self.stamp, self.arc = _clock_split(
            self.n, self.n_slow, self.n_stamp, self.n_arc, latch=latch
        )
        self.heavy_pull = float(heavy_pull)
        self.heavy_dt_scale = float(heavy_dt_scale)
        self.boundary_ids = tuple(int(piece) for piece in boundary_ids)
        self.latch = bool(latch)
        self.latch_blend = float(latch_blend)
        self.latch_at = self.n - 1 if self.latch else -1
        self.eos_id = None if eos_id is None else int(eos_id)

        scale = self.n**-0.5
        self.omega = nn.Parameter(0.01 * torch.randn(self.n))
        self.U = nn.Parameter(scale * torch.randn(self.n, self.rank))
        self.V = nn.Parameter(scale * torch.randn(self.n, self.rank))
        self.root_phase = nn.Embedding(self.vocab_size, self.n_roots)
        nn.init.normal_(self.root_phase.weight, std=0.1)
        self.D = nn.Parameter(0.1 * torch.randn(self.n, self.n_roots))
        self.theta0 = nn.Parameter(torch.zeros(self.n))
        self._init_memory()
        if self.readout_rank > 0:
            self.readout_down = nn.Linear(2 * self.n, self.readout_rank, bias=False)
            self.readout_up = nn.Linear(self.readout_rank, self.vocab_size)
        else:
            self.readout = nn.Linear(2 * self.n, self.vocab_size)
        if canon_ids is None:
            canon = torch.arange(self.vocab_size)
        else:
            canon = torch.tensor(canon_ids, dtype=torch.long)
        self.register_buffer("canon", canon, persistent=False)
        if beat_canon_ids is None:
            beat_canon = torch.arange(self.vocab_size)
        else:
            beat_canon = torch.tensor(beat_canon_ids, dtype=torch.long)
        self.register_buffer("beat_canon", beat_canon, persistent=False)

    def velocity(  # noqa: PLR0913
        self,
        theta: Tensor,
        tokens: Tensor,
        latch: Tensor | None = None,
        *,
        character: Tensor | None = None,
        dilemma: Tensor | None = None,
        resolution: Tensor | None = None,
        recognition: Tensor | None = None,
    ) -> Tensor:
        """Phase velocity for one Euler evaluation.

        ``theta`` is ``(batch, n)``. The word enters as a phase offset inside
        the tug, ``shown + D @ phi(word)``, not as a second force after it.
        Character, dilemma, and resolution shift only the slow wheels when
        that band exists. A negative id adds nothing. With no slow wheels the
        specials fall back onto every wheel. Slow wheels couple only with
        each other. Peer tugs use ``U @ V.T`` without forming that square.
        ``recognition`` is the overlap of the pose just left. ``K`` transpose
        ``r`` nudges toward the next catalog pose and blends one coupling per
        stored pose. A missing recognition is uniform. Neither term snaps.
        """
        offset = self._phase_offset(tokens, character, dilemma, resolution)
        shown = self._shown(theta, latch) + offset
        sin_src, cos_src = self._outbound(shown.sin(), shown.cos())
        # The paper's blend is the tug. The pose spring stays behind ``route_extras``.
        if self.use_route:
            if recognition is None:
                recognition = self._uniform_recognition(theta.shape[0], theta.device)
            successor = self._from_recognition(recognition)
            sin_k, cos_k = self._memory_fields(sin_src, cos_src, successor)
            if self.route_extras:
                sin_k = sin_k + successor @ self.patterns.sin()
                cos_k = cos_k + successor @ self.patterns.cos()
        else:
            sin_k, cos_k = self._couple(sin_src, cos_src)
        return self.omega + theta.cos() * sin_k - theta.sin() * cos_k

    def _root_phi(self, ids: Tensor) -> Tensor:
        """``tanh(root_phase) * pi`` for each id. A negative id is zero."""
        active = (ids >= 0).to(dtype=self.D.dtype).unsqueeze(1)
        safe = ids.clamp(min=0)
        phi = self.root_phase(self.beat_canon[safe]).tanh() * math.pi
        return phi * active

    def _phase_offset(
        self,
        tokens: Tensor,
        character: Tensor | None,
        dilemma: Tensor | None,
        resolution: Tensor | None,
    ) -> Tensor:
        """Word offset on the word band. Specials offset the slow band."""
        phi = self.root_phase(self.beat_canon[tokens]).tanh() * math.pi
        spec = torch.zeros_like(phi)
        if character is not None:
            spec = spec + CHARACTER_DRIVE_SCALE * self._root_phi(character)
        if dilemma is not None:
            spec = spec + STICKY_DRIVE_SCALE * self._root_phi(dilemma)
        if resolution is not None:
            spec = spec + RESOLUTION_DRIVE_SCALE * self._root_phi(resolution)
        word = phi @ self.D.transpose(0, 1)
        extra = spec @ self.D.transpose(0, 1)
        if not self.n_slow:
            return word + extra
        slow = word.new_zeros(self.n)
        slow[self.slow.start : self.slow.stop] = 1
        return word * (1 - slow) + extra * slow

    def _project(self, src: Tensor) -> Tensor:
        """Apply ``K = U @ V.T`` to one sine or cosine row. ``K`` is not stored."""
        return (src @ self.V) @ self.U.transpose(0, 1)

    def _init_memory(self) -> None:
        """Start the catalog off. ``scatter_route`` turns the nudge on.

        A zero catalog is the angle 0, so the nudge stays off until training
        gives the patterns and ``K`` a real start.
        """
        # Off until ``scatter_route``. A zero catalog is angle 0, and the
        # nudge would pull every wheel there.
        self.use_route = False
        self.route_extras = False
        self.patterns = nn.Parameter(torch.zeros(N_MEMORIES, self.n))
        self.route_K = nn.Parameter(torch.zeros(N_MEMORIES, N_MEMORIES))
        self.mem_U = nn.Parameter(torch.zeros(N_MEMORIES, self.n, self.rank))
        self.mem_V = nn.Parameter(torch.zeros(N_MEMORIES, self.n, self.rank))

    def scatter_route(self) -> None:
        """Open every catalog slot on the rest pose and on ``U`` and ``V``."""
        self.use_route = True
        memories = self.patterns.shape[0]
        with torch.no_grad():
            self.patterns.copy_(self.theta0.detach().view(1, -1).expand(memories, -1))
            self.route_K.zero_()
            self.mem_U.copy_(self.U.detach().view(1, self.n, self.rank).expand(memories, -1, -1))
            self.mem_V.copy_(self.V.detach().view(1, self.n, self.rank).expand(memories, -1, -1))

    def _uniform_recognition(self, batch: int, device: torch.device) -> Tensor:
        """No pose has been left yet, so every catalog entry is equally likely."""
        memories = self.patterns.shape[0]
        return torch.ones(batch, memories, device=device) / memories

    def _overlap(self, pose: Tensor) -> Tensor:
        """Instantaneous recognition. ``r`` itself lags this by one kick."""
        sin_p = self.patterns.sin()
        cos_p = self.patterns.cos()
        align = pose.sin() @ sin_p.transpose(0, 1) + pose.cos() @ cos_p.transpose(0, 1)
        return torch.softmax(align * (2.0 * RECOGNITION_GAMMA), dim=-1)

    def _from_recognition(self, recognition: Tensor) -> Tensor:
        """``K`` transpose ``r``. Row ``k`` says what follows memory ``k``."""
        routing = torch.softmax(self.route_K, dim=-1)
        return recognition @ routing

    def _blend_coupling(self, src: Tensor, successor: Tensor) -> Tensor:
        """Blend one low-rank coupling per stored pose. No dense ``n`` by ``n``."""
        coeff = torch.einsum("bn,mnr->bmr", src, self.mem_V)
        field = torch.einsum("bmr,mnr->bmn", coeff, self.mem_U)
        return torch.einsum("bm,bmn->bn", successor, field)

    def _memory_fields(
        self,
        sin_src: Tensor,
        cos_src: Tensor,
        successor: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Per-pose couplings, kept inside each timescale."""

        def blend(src: Tensor) -> Tensor:
            if not self.n_slow:
                return self._blend_coupling(src, successor)
            slow = src.new_zeros(self.n)
            slow[self.slow.start : self.slow.stop] = 1
            fast = 1 - slow
            return (
                self._blend_coupling(src * fast, successor) * fast
                + self._blend_coupling(src * slow, successor) * slow
            )

        return blend(sin_src), blend(cos_src)

    def _couple(self, sin_src: Tensor, cos_src: Tensor) -> tuple[Tensor, Tensor]:
        """Tug inside a timescale. Slow wheels do not tug the word wheels.

        The split is the timescale. The catalog adds its own couplings later.
        """
        if not self.n_slow:
            return self._project(sin_src), self._project(cos_src)
        slow = sin_src.new_zeros(self.n)
        slow[self.slow.start : self.slow.stop] = 1
        fast = 1 - slow
        sin_k = self._project(sin_src * fast) * fast + self._project(sin_src * slow) * slow
        cos_k = self._project(cos_src * fast) * fast + self._project(cos_src * slow) * slow
        return sin_k, cos_k

    def logits_from(self, features: Tensor) -> Tensor:
        """Next-piece scores. Rank 0 keeps the full linear map."""
        if self.readout_rank > 0:
            return self.readout_up(self.readout_down(features))
        return self.readout(features)

    def _open_dilemma(self, seen: Tensor, tokens: Tensor, dilemma: Tensor) -> tuple[Tensor, Tensor]:
        """Turn the dilemma on at the piece that says it, and keep it on."""
        folded = self.beat_canon[tokens]
        seen = seen | ((dilemma >= 0) & (folded == dilemma))
        active = torch.where(seen, dilemma, torch.full_like(dilemma, -1))
        return seen, active

    def _outbound(self, sin_t: Tensor, cos_t: Tensor) -> tuple[Tensor, Tensor]:
        """Scale the tug that the set-aside wheels send to the word wheels."""
        if not self.n_slow and not self.n_stamp and not self.n_arc and not self.latch:
            return sin_t, cos_t
        gain = sin_t.new_ones(self.n)
        if self.n_slow or self.n_stamp or self.n_arc:
            gain[self.slow.start : self.n - int(self.latch)] *= self.heavy_pull
        if self.latch:
            gain[self.latch_at] = self.heavy_pull
        return sin_t * gain, cos_t * gain

    def _at_period(self, tokens: Tensor) -> Tensor:
        """Mark rows whose piece ends a sentence."""
        if not self.boundary_ids:
            return torch.zeros(tokens.shape[0], dtype=torch.bool, device=tokens.device)
        ends = torch.tensor(self.boundary_ids, device=tokens.device, dtype=tokens.dtype)
        return (tokens.unsqueeze(1) == ends.unsqueeze(0)).any(dim=1)

    def step_for(self, tokens: Tensor, resolution_on: Tensor | None = None) -> float | Tensor:
        """Per-wheel Euler step.

        The stamp clock moves only at a sentence end. The arc wheel takes a
        full step only while ``resolution_on`` is set for that row.
        """
        if not self.n_slow and not self.n_stamp and not self.n_arc:
            return self.dt
        step = self.omega.new_full((self.n,), self.dt)
        if self.n_slow:
            step[self.slow.start : self.slow.stop] *= self.heavy_dt_scale
        if self.latch:
            step[self.latch_at] = self.dt * self.heavy_dt_scale
        if self.n_stamp:
            step[self.stamp.start : self.stamp.stop] = 0
        if self.n_arc:
            step[self.arc.start : self.arc.stop] = 0
        if not self.n_stamp and not self.n_arc:
            return step
        base = step.unsqueeze(0).expand(tokens.shape[0], -1).clone()
        if self.n_stamp:
            at_period = self._at_period(tokens)
            base[at_period, self.stamp.start : self.stamp.stop] = self.dt
        if self.n_arc and resolution_on is not None:
            base[resolution_on, self.arc.start : self.arc.stop] = self.dt
        return base

    def _shown(self, theta: Tensor, latch: Tensor | None) -> Tensor:
        """Swap the live dilemma wheel for the held angle."""
        if not self.latch:
            return theta
        if latch is None:
            latch = torch.zeros(theta.shape[0], 1, device=theta.device, dtype=theta.dtype)
        return torch.cat((theta[:, :-1], latch), dim=1)

    def _commit(self, latch: Tensor, theta: Tensor, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Store the sentence in the latch, then clear the accumulator."""
        end = self._at_period(tokens).unsqueeze(1)
        live = theta[:, self.latch_at : self.latch_at + 1]
        mixed = (1.0 - self.latch_blend) * latch + self.latch_blend * live
        held = torch.where(end, mixed, latch)
        cleared = torch.where(end, torch.zeros_like(live), live)
        return held, torch.cat((theta[:, :-1], cleared), dim=1)

    def forward(
        self,
        token_ids: Tensor,
        character: Tensor | None = None,
        dilemma: Tensor | None = None,
    ) -> Tensor:
        """Return next-piece logits ``(batch, time - 1, vocab)``.

        The bag starts at ``theta0``. After the end piece is kicked, the next
        kick starts at ``theta0`` again and the latch is clear. A run of end
        pieces after that first one is padding and is not kicked. A sentence
        period still only latches; it does not rest the bag. ``character`` is
        on for every kick. ``dilemma`` turns on at the piece that folds to
        that id and stays on, at a quarter of a word kick. Omit either one
        and that drive stays off.
        """
        batch = token_ids.shape[0]
        theta = self.theta0.unsqueeze(0).expand(batch, -1).clone()
        latch = theta.new_zeros(batch, 1)
        # True until this row has kicked its end piece and not yet started another.
        in_story = torch.ones(batch, dtype=torch.bool, device=theta.device)
        seen = torch.zeros(batch, dtype=torch.bool, device=theta.device)
        recognition = self._uniform_recognition(batch, theta.device)
        logits: list[Tensor] = []
        for t in range(token_ids.shape[1] - 1):
            tokens = token_ids[:, t]
            dilemma_now = None
            if dilemma is not None:
                seen, dilemma_now = self._open_dilemma(seen, tokens, dilemma)
            if self.eos_id is None:
                is_end = torch.zeros(batch, dtype=torch.bool, device=theta.device)
            else:
                is_end = tokens == self.eos_id
            # Padding is an end piece that follows the end piece. A later real
            # piece opens the next story and is kicked from rest.
            padding = (~in_story) & is_end
            take = ~padding
            if self.latch:
                held, committed = self._commit(latch, theta, tokens)
                latch = torch.where(take.unsqueeze(1), held, latch)
                theta = torch.where(take.unsqueeze(1), committed, theta)
            left = self._overlap(self._shown(theta, latch))
            stepped = theta
            for _ in range(self.euler_steps):
                stepped = stepped + self.step_for(tokens) * self.velocity(
                    stepped,
                    tokens,
                    latch,
                    character=character,
                    dilemma=dilemma_now,
                    recognition=recognition,
                )
            stepped = torch.atan2(stepped.sin(), stepped.cos())
            theta = torch.where(take.unsqueeze(1), stepped, theta)
            recognition = torch.where(take.unsqueeze(1), left, recognition)
            shown = self._shown(theta, latch)
            features = torch.cat((shown.sin(), shown.cos()), dim=-1)
            logits.append(self.logits_from(features))
            if self.eos_id is not None:
                reset = take & is_end
                theta = torch.where(reset.unsqueeze(1), self.theta0.unsqueeze(0), theta)
                latch = torch.where(reset.unsqueeze(1), torch.zeros_like(latch), latch)
                recognition = torch.where(
                    reset.unsqueeze(1),
                    self._uniform_recognition(batch, theta.device),
                    recognition,
                )
                in_story = ~is_end
        return torch.stack(logits, dim=1)

    def outgoing_strength(self) -> Tensor:
        """How hard each wheel tugs the others, as the column sum of ``|K|``."""
        coupling = self.U.detach() @ self.V.detach().transpose(0, 1)
        gain = coupling.new_ones(self.n)
        if self.n_slow or self.n_stamp or self.n_arc:
            gain[self.slow.start : self.n - int(self.latch)] = self.heavy_pull
        if self.latch:
            gain[self.latch_at] = self.heavy_pull
        return (coupling.abs() * gain).sum(dim=0)

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list[int],
        *,
        steps: int,
        device: torch.device,
        eos_id: int | None = None,
    ) -> list[int]:
        """Continue a prompt one piece at a time. No dilemma is loaded. Stop on the end piece."""
        self.eval()
        theta = self.theta0.detach().clone().unsqueeze(0).to(device)
        hold = torch.zeros(1, 1, device=device)
        recognition = self._uniform_recognition(1, device)
        ids = list(prompt_ids)
        for char_id in ids:
            tokens = torch.tensor([char_id], device=device)
            if self.latch:
                hold, theta = self._commit(hold, theta, tokens)
            left = self._overlap(self._shown(theta, hold))
            for _ in range(self.euler_steps):
                theta = theta + self.step_for(tokens) * self.velocity(
                    theta, tokens, hold, recognition=recognition
                )
            theta = torch.atan2(theta.sin(), theta.cos())
            recognition = left
        for _ in range(steps):
            features = torch.cat(
                (self._shown(theta, hold).sin(), self._shown(theta, hold).cos()), dim=-1
            )
            nxt = int(self.logits_from(features).argmax(dim=-1).item())
            if eos_id is not None and nxt == eos_id:
                break
            ids.append(nxt)
            tokens = torch.tensor([nxt], device=device)
            if self.latch:
                hold, theta = self._commit(hold, theta, tokens)
            left = self._overlap(self._shown(theta, hold))
            for _ in range(self.euler_steps):
                theta = theta + self.step_for(tokens) * self.velocity(
                    theta, tokens, hold, recognition=recognition
                )
            theta = torch.atan2(theta.sin(), theta.cos())
            recognition = left
        return ids
