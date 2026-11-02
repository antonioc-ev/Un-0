# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Next-piece Kuramoto model with a low-rank clutch matrix.

The image model stores a dense ``n`` by ``n`` clutch matrix. Here the same tug is
factored as ``K = U @ V.T`` with a thin rank, so the step never builds that
square array. A piece sets a few root-wheel phases; those roots tug the bag
one way; one Euler kick updates the bag; sine and cosine of the phases are
read out as scores for the next piece.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

_MIN_OSCILLATORS = 2


def clock_ranges(n: int, n_slow: int, n_stamp: int) -> tuple[range, range]:
    """Split the end of the bag into two clocks.

    The slow clock takes a short step on every word. The stamp clock takes a
    full step only when the piece ends a sentence, and holds still otherwise.
    256 wheels with 8 and 8 is slow 240–247 and stamp 248–255. Either count
    may be 0. At least one wheel keeps the ordinary word step.
    """
    if n_slow < 0 or n_stamp < 0 or n_slow + n_stamp >= n:
        raise ValueError("n_slow and n_stamp must leave at least one word wheel.")
    start = n - n_slow - n_stamp
    return range(start, start + n_slow), range(start + n_slow, n)


class KuramotoWordLM(nn.Module):
    """Recurrent low-rank Kuramoto bag that predicts the next piece."""

    def __init__(  # noqa: PLR0913
        self,
        vocab_size: int,
        *,
        n_oscillators: int = 64,
        rank: int = 8,
        n_roots: int = 4,
        euler_steps: int = 1,
        dt: float = 0.2,
        n_slow: int = 0,
        n_stamp: int = 0,
        heavy_pull: float = 4.0,
        heavy_dt_scale: float = 0.1,
        boundary_ids: tuple[int, ...] = (),
        latch: bool = False,
        latch_blend: float = 0.25,
        eos_id: int | None = None,
        canon_ids: list[int] | None = None,
        beat_canon_ids: list[int] | None = None,
    ) -> None:
        """Build a bag of ``n_oscillators`` with rank-``rank`` clutches."""
        super().__init__()
        if min(n_oscillators, rank, n_roots, euler_steps) < 1 or n_oscillators < _MIN_OSCILLATORS:
            raise ValueError("Need n_oscillators >= 2, rank >= 1, n_roots >= 1, euler_steps >= 1.")
        if latch and n_stamp:
            raise ValueError("The latch replaces the stamp clock.")
        if not 0 < latch_blend <= 1:
            raise ValueError("latch_blend must be in (0, 1].")
        slow, stamp = clock_ranges(n_oscillators - int(latch), n_slow, n_stamp)
        if heavy_pull <= 0 or heavy_dt_scale <= 0:
            raise ValueError("heavy_pull and heavy_dt_scale must be positive.")
        self.vocab_size = int(vocab_size)
        self.n = int(n_oscillators)
        self.rank = int(rank)
        self.n_roots = int(n_roots)
        self.euler_steps = int(euler_steps)
        self.dt = float(dt)
        self.n_slow = int(n_slow)
        self.n_stamp = int(n_stamp)
        self.slow = slow
        self.stamp = stamp
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

    def velocity(
        self,
        theta: Tensor,
        tokens: Tensor,
        latch: Tensor | None = None,
        *,
        character: Tensor | None = None,
        dilemma: Tensor | None = None,
    ) -> Tensor:
        """Phase velocity for one Euler evaluation.

        ``theta`` is ``(batch, n)``. ``tokens`` is ``(batch,)`` and sets the
        root phases of the current piece. A non-negative ``character`` adds
        that piece's root drive on this kick. A non-negative ``dilemma`` does
        the same. A negative id adds nothing. Resolution has no drive. Peer
        tugs use ``K = U @ V.T`` without forming ``K``. The latch, when
        present, is what the dilemma wheel shows the others.
        """
        shown = self._shown(theta, latch)
        sin_src, cos_src = self._outbound(shown.sin(), shown.cos())
        sin_k = (sin_src @ self.V) @ self.U.transpose(0, 1)
        cos_k = (cos_src @ self.V) @ self.U.transpose(0, 1)
        vel = self.omega + theta.cos() * sin_k - theta.sin() * cos_k

        phi = self.root_phase(self.canon[tokens]).tanh() * math.pi
        sin_p = phi.sin()
        cos_p = phi.cos()
        if character is not None:
            char_sin, char_cos = self._root_trig(character)
            sin_p = sin_p + char_sin
            cos_p = cos_p + char_cos
        if dilemma is not None:
            dil_sin, dil_cos = self._root_trig(dilemma)
            sin_p = sin_p + dil_sin
            cos_p = cos_p + dil_cos
        drive_sin = sin_p @ self.D.transpose(0, 1)
        drive_cos = cos_p @ self.D.transpose(0, 1)
        return vel + theta.cos() * drive_sin - theta.sin() * drive_cos

    def _root_trig(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        """Sine and cosine of ``tanh(root_phase)`` for each id. Negatives are zero."""
        active = (ids >= 0).to(dtype=self.D.dtype).unsqueeze(1)
        safe = ids.clamp(min=0)
        phi = self.root_phase(self.canon[safe]).tanh() * math.pi
        return phi.sin() * active, phi.cos() * active

    def _open_dilemma(self, seen: Tensor, tokens: Tensor, dilemma: Tensor) -> tuple[Tensor, Tensor]:
        """Mark rows whose dilemma piece has appeared, including this kick."""
        folded = self.beat_canon[tokens]
        seen = seen | ((dilemma >= 0) & (folded == dilemma))
        active = torch.where(seen, dilemma, torch.full_like(dilemma, -1))
        return seen, active

    def _outbound(self, sin_t: Tensor, cos_t: Tensor) -> tuple[Tensor, Tensor]:
        """Scale the tug that both clocks send to the word wheels."""
        if not self.n_slow and not self.n_stamp:
            return sin_t, cos_t
        gain = sin_t.new_ones(self.n)
        if self.n_slow or self.n_stamp:
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

    def step_for(self, tokens: Tensor) -> float | Tensor:
        """Per-wheel Euler step. The stamp clock moves only at a sentence end."""
        if not self.n_slow and not self.n_stamp:
            return self.dt
        step = self.omega.new_full((self.n,), self.dt)
        if self.n_slow:
            step[self.slow.start : self.slow.stop] *= self.heavy_dt_scale
        if self.latch:
            step[self.latch_at] = self.dt * self.heavy_dt_scale
        if self.n_stamp:
            step[self.stamp.start : self.stamp.stop] = 0
        if not self.n_stamp:
            return step
        base = step.unsqueeze(0).expand(tokens.shape[0], -1).clone()
        at_period = self._at_period(tokens)
        base[at_period, self.stamp.start : self.stamp.stop] = self.dt
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
        on from the first kick. ``dilemma`` turns on at the first piece in
        this window that folds to that id. Omit either one and that drive
        stays off.
        """
        batch = token_ids.shape[0]
        theta = self.theta0.unsqueeze(0).expand(batch, -1).clone()
        latch = theta.new_zeros(batch, 1)
        # True until this row has kicked its end piece and not yet started another.
        in_story = torch.ones(batch, dtype=torch.bool, device=theta.device)
        seen = torch.zeros(batch, dtype=torch.bool, device=theta.device)
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
            stepped = theta
            for _ in range(self.euler_steps):
                stepped = stepped + self.step_for(tokens) * self.velocity(
                    stepped,
                    tokens,
                    latch,
                    character=character,
                    dilemma=dilemma_now,
                )
            stepped = torch.atan2(stepped.sin(), stepped.cos())
            theta = torch.where(take.unsqueeze(1), stepped, theta)
            shown = self._shown(theta, latch)
            features = torch.cat((shown.sin(), shown.cos()), dim=-1)
            logits.append(self.readout(features))
            if self.eos_id is not None:
                reset = take & is_end
                theta = torch.where(reset.unsqueeze(1), self.theta0.unsqueeze(0), theta)
                latch = torch.where(reset.unsqueeze(1), torch.zeros_like(latch), latch)
                in_story = ~is_end
        return torch.stack(logits, dim=1)

    def outgoing_strength(self) -> Tensor:
        """How hard each wheel tugs the others, as the column sum of ``|K|``."""
        coupling = self.U.detach() @ self.V.detach().transpose(0, 1)
        gain = coupling.new_ones(self.n)
        if self.n_slow or self.n_stamp:
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
        ids = list(prompt_ids)
        for char_id in ids:
            tokens = torch.tensor([char_id], device=device)
            if self.latch:
                hold, theta = self._commit(hold, theta, tokens)
            for _ in range(self.euler_steps):
                theta = theta + self.step_for(tokens) * self.velocity(theta, tokens, hold)
            theta = torch.atan2(theta.sin(), theta.cos())
        for _ in range(steps):
            features = torch.cat(
                (self._shown(theta, hold).sin(), self._shown(theta, hold).cos()), dim=-1
            )
            nxt = int(self.readout(features).argmax(dim=-1).item())
            if eos_id is not None and nxt == eos_id:
                break
            ids.append(nxt)
            tokens = torch.tensor([nxt], device=device)
            if self.latch:
                hold, theta = self._commit(hold, theta, tokens)
            for _ in range(self.euler_steps):
                theta = theta + self.step_for(tokens) * self.velocity(theta, tokens, hold)
            theta = torch.atan2(theta.sin(), theta.cos())
        return ids
