# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Story generator whose memory is a graph, not an oscillator bag.

Articles, conjunctions, prepositions, and pieces with no letters never become nodes. A capital letter
does not make a second node: ``She`` and ``she`` are one. Every other piece
becomes a node the first time it is said, and that node stays for the rest of
the story. Attention over those nodes is the edge.

The storyline is the prose. A reader ranks the nodes: the character is the
name the story returns to, the dilemma is the other word it returns to, and
the resolution is the last sentence. The lead-in is only a prefix. The beats
are read off the story after it is written.
"""

from __future__ import annotations

import contextlib
import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from sidecar.beats import PackedBeats, StoryBeats, story_beats
from sidecar.prepare import EOS, piece_canon_ids

# One backward window. Nodes carry into the next window.
TOKEN_WINDOW = 256


# Glue. Predicted, never stored, and not a story beat. Pronouns and verbs stay.
_GLUE = {
    "a", "an", "the",
    "and", "but", "or", "because",
    "to", "of", "in", "on", "at", "for", "with", "from",
}


def node_skip_ids(itos: list[str]) -> set[int]:
    """Glue and non-words. These are predicted, and they are not stored."""
    skip = set()
    for index, piece in enumerate(itos):
        if piece == EOS or not any(ch.isalpha() for ch in piece):
            skip.add(index)
            continue
        if piece.strip().lower() in _GLUE:
            skip.add(index)
    return skip


def node_canon_ids(itos: list[str]) -> list[int]:
    """Map each piece to one node. ``She`` and ``she`` share the lowercase id."""
    return piece_canon_ids(itos)


class GraphState:
    """Nodes seen so far in one story, and the attention edges that built them."""

    def __init__(self, dim: int, device: torch.device) -> None:
        """Start empty."""
        self.ids: list[int] = []
        self.index: dict[int, int] = {}
        self.h = torch.zeros(0, dim, device=device)
        self.s = torch.zeros(dim, device=device)
        self.links: list[tuple[int, int, float]] = []


def pack_stories(
    stories: list[list[int]],
    skip_ids: set[int],
    canon: list[int] | None = None,
) -> tuple[Tensor, Tensor, int, int]:
    """Pad one batch and count the nodes it will grow.

    The budget is the most distinct content words any story absorbs. The count
    is how many next-piece targets the batch scores.
    """
    if not stories:
        raise ValueError("A training batch needs at least one story.")
    width = max(len(story) for story in stories)
    rows = [story + [0] * (width - len(story)) for story in stories]
    tokens = torch.tensor(rows, dtype=torch.long)
    lengths = torch.tensor([len(story) for story in stories], dtype=torch.long)
    budget = 1
    count = 0
    for story in stories:
        seen: set[int] = set()
        for token in story[:-1]:
            if token not in skip_ids:
                seen.add(canon[token] if canon is not None else token)
        budget = max(budget, len(seen))
        count += max(0, len(story) - 1)
    return tokens, lengths, budget, count


def _half_context(device: torch.device, *, half: bool) -> contextlib.AbstractContextManager[object]:
    """Half precision on the GPU. The loss itself stays in fp32."""
    if not half or device.type == "cpu":
        return contextlib.nullcontext()
    dtype = torch.float16 if device.type == "mps" else torch.bfloat16
    return torch.autocast(device_type=device.type, dtype=dtype)


class StoryGraphLM(nn.Module):
    """Next-piece model with a persistent node per content word."""

    def __init__(
        self,
        vocab_size: int,
        *,
        dim: int = 64,
        skip_ids: set[int] | None = None,
        canon_ids: list[int] | None = None,
        boundary_ids: tuple[int, ...] | list[int] | None = None,
    ) -> None:
        """Build the embedding, the storyline, the reader, and the score."""
        super().__init__()
        self.vocab_size = int(vocab_size)
        self.dim = int(dim)
        self.skip_ids = set(skip_ids or ())
        self.attn_scale = math.sqrt(self.dim)
        self.embed = nn.Embedding(self.vocab_size, self.dim)
        self.gru = nn.GRUCell(self.dim, self.dim)
        self.storyline = nn.GRUCell(self.dim, self.dim)
        self.readout = nn.Linear(self.dim * 2, self.vocab_size)
        self.character_head = nn.Linear(4, 1)
        self.dilemma_head = nn.Linear(4, 1)
        self.character_query = nn.Parameter(torch.zeros(self.dim))
        self.dilemma_query = nn.Parameter(torch.zeros(self.dim))
        self._read_hits: Tensor | None = None
        skip_mask = torch.zeros(self.vocab_size, dtype=torch.bool)
        if self.skip_ids:
            skip_mask[list(self.skip_ids)] = True
        self.register_buffer("skip_mask", skip_mask, persistent=False)
        boundary = torch.zeros(self.vocab_size, dtype=torch.bool)
        if boundary_ids:
            boundary[list(boundary_ids)] = True
        self.register_buffer("boundary", boundary, persistent=False)
        if canon_ids is None:
            canon = torch.arange(self.vocab_size)
        else:
            canon = torch.tensor(canon_ids, dtype=torch.long)
        self.register_buffer("canon", canon, persistent=False)
        self.register_buffer("df", torch.ones(self.vocab_size))
        self.register_buffer("n_docs", torch.ones(()))

    def set_document_frequency(self, counts: list[int], n_docs: int) -> None:
        """Store how many training stories contain each piece."""
        if len(counts) != self.vocab_size:
            raise ValueError("Document counts must cover the vocabulary.")
        self.df.copy_(torch.tensor(counts, dtype=self.df.dtype, device=self.df.device))
        self.n_docs.fill_(float(max(n_docs, 1)))

    def logits(self, state: GraphState, token: int) -> Tensor:
        """Scores for the piece that follows ``token``. The storyline is updated first."""
        emb = self.embed.weight[token]
        self._advance_one(state, emb)
        return self.readout(torch.cat((state.s, emb), dim=0))

    def _advance_one(self, state: GraphState, emb: Tensor) -> None:
        """Pull the storyline toward the node this piece attends to."""
        if state.h.shape[0] == 0:
            return
        state.s = self.storyline(self._message_one(state.h, emb), state.s)

    def _message_one(self, h: Tensor, query: Tensor) -> Tensor:
        """Attention of one piece over the nodes already stored."""
        scores = h @ query / self.attn_scale
        weights = torch.softmax(scores, dim=0)
        return weights @ h

    def absorb(self, state: GraphState, token: int, *, record: bool) -> None:
        """Add ``token`` as a node. Capitals share the lowercase node."""
        if token in self.skip_ids:
            return
        key = int(self.canon[token])
        emb = self.embed.weight[key]
        if key not in state.index:
            state.index[key] = len(state.ids)
            state.ids.append(key)
            state.h = emb.unsqueeze(0) if state.h.shape[0] == 0 else torch.cat((state.h, emb.unsqueeze(0)))
        row = state.index[key]
        weights = torch.softmax(state.h @ state.h[row] / self.attn_scale, dim=0)
        msg = weights @ state.h
        updated = self.gru(msg, state.h[row])
        state.h = torch.cat((state.h[:row], updated.unsqueeze(0), state.h[row + 1 :]))
        if record and weights.shape[0] > 1:
            others = weights.detach().clone()
            others[row] = -1
            partner = int(others.argmax())
            state.links.append((key, state.ids[partner], float(weights[partner])))

    def story_loss(self, token_ids: list[int]) -> Tensor:
        """Mean next-piece loss for one story."""
        device = self.embed.weight.device
        tokens = torch.tensor([token_ids], dtype=torch.long, device=device)
        lengths = torch.tensor([len(token_ids)], dtype=torch.long, device=device)
        seen: set[int] = set()
        canon = self.canon.detach().cpu().tolist()
        for token in token_ids[:-1]:
            if token not in self.skip_ids:
                seen.add(int(canon[token]))
        return self.loss_on_tokens(
            tokens,
            lengths,
            budget=max(1, len(seen)),
            count=max(1, len(token_ids) - 1),
        )

    def batch_loss(self, stories: list[list[int]]) -> tuple[Tensor, int]:
        """Token-mean loss. One attention and one GRU cover the whole batch."""
        device = self.embed.weight.device
        tokens, lengths, budget, count = pack_stories(
            stories, self.skip_ids, self.canon.detach().cpu().tolist(),
        )
        loss = self.loss_on_tokens(
            tokens.to(device),
            lengths.to(device),
            budget=budget,
            count=count,
        )
        return loss, count

    def loss_on_tokens(  # noqa: PLR0913
        self,
        tokens: Tensor,
        lengths: Tensor,
        *,
        budget: int,
        count: int,
        window: int = TOKEN_WINDOW,
        half: bool = False,
    ) -> Tensor:
        """Next-piece loss for a padded batch already on the model device.

        Node state is ``(batch, nodes, dim)``. Each piece fans out across every
        story at once. A story that fits in one window backprops through every
        word. A longer story keeps those nodes and recomputes one window at a
        time.
        """
        batch, width = tokens.shape
        device = tokens.device
        h = torch.zeros(batch, budget, self.dim, device=device)
        counts = torch.zeros(batch, dtype=torch.long, device=device)
        slot = torch.full((batch, self.vocab_size), -1, dtype=torch.long, device=device)
        story = torch.zeros(batch, self.dim, device=device)
        embedded = self.embed(tokens)
        alive = torch.arange(width, device=device)[None, :] < (lengths[:, None] - 1)
        total = torch.zeros((), device=device)
        limit = width - 1
        if limit <= window:
            part, _, _, _, _ = self._roll(
                h, counts, slot, story, embedded, alive, tokens, 0, limit, half,
            )
            return (total + part) / max(count, 1)
        start = 0
        while start < limit:
            end = min(start + window, limit)
            part, h, counts, slot, story = checkpoint(
                self._roll,
                h,
                counts,
                slot,
                story,
                embedded,
                alive,
                tokens,
                start,
                end,
                half,
                use_reentrant=False,
                preserve_rng_state=False,
            )
            total = total + part
            start = end
        return total / max(count, 1)

    def _roll(  # noqa: PLR0913
        self,
        h: Tensor,
        counts: Tensor,
        slot: Tensor,
        story: Tensor,
        embedded: Tensor,
        alive: Tensor,
        tokens: Tensor,
        start: int,
        end: int,
        half: bool,  # noqa: FBT001
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Score and absorb one window. Nodes and the storyline carry out."""
        total = embedded.new_zeros(())
        with _half_context(embedded.device, half=half):
            for t in range(start, end):
                token = tokens[:, t]
                emb = embedded[:, t]
                story = self._advance_cell(h, counts, story, emb, self.storyline)
                logits = self.readout(torch.cat((story, emb), dim=-1))
                step = F.cross_entropy(logits, tokens[:, t + 1], reduction="none")
                total = total + (step * alive[:, t].to(step.dtype)).sum()
                content = alive[:, t] & ~self.skip_mask[token]
                h, counts, slot = self._fanout_absorb(h, counts, slot, token, content)
        return total, h, counts, slot, story

    def _fanout_absorb(
        self,
        h: Tensor,
        counts: Tensor,
        slot: Tensor,
        token: Tensor,
        content: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Update every story's current node in one matmul. Capitals share a node."""
        index = torch.arange(token.shape[0], device=token.device)
        key = self.canon[token]
        held = slot[index, key]
        is_new = content & (held < 0)
        rows = torch.where(is_new, counts, held.clamp(min=0))
        at_row = torch.arange(h.shape[1], device=token.device)[None, :] == rows[:, None]
        node_emb = self.embed(key)
        h = torch.where(is_new[:, None, None] & at_row.unsqueeze(-1), node_emb[:, None, :], h)
        counts = counts + is_new.to(counts.dtype)
        written = torch.where(is_new, rows, slot[index, key])
        slot = slot.scatter(1, key.unsqueeze(1), written.unsqueeze(1))
        query = h[index, rows]
        scores = torch.matmul(h, query.unsqueeze(-1)).squeeze(-1) / self.attn_scale
        visible = torch.arange(h.shape[1], device=token.device)[None, :] < counts[:, None]
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(~visible.any(dim=-1, keepdim=True), 0)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(content[:, None], weights, torch.zeros_like(weights))
        msg = torch.matmul(weights.unsqueeze(1), h).squeeze(1)
        updated = self.gru(msg, query)
        delta = torch.where(content[:, None], updated - query, torch.zeros_like(query))
        h = h + delta[:, None, :] * at_row.to(h.dtype).unsqueeze(-1)
        return h, counts, slot

    def _advance_cell(
        self,
        h: Tensor,
        counts: Tensor,
        state: Tensor,
        emb: Tensor,
        cell: nn.GRUCell,
    ) -> Tensor:
        """Move one bag toward the node its current piece attends to."""
        scores = torch.matmul(h, emb.unsqueeze(-1)).squeeze(-1) / self.attn_scale
        visible = torch.arange(h.shape[1], device=h.device)[None, :] < counts[:, None]
        scores = scores.masked_fill(~visible, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(~visible.any(dim=-1, keepdim=True), 0)
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(visible, weights, torch.zeros_like(weights))
        msg = torch.matmul(weights.unsqueeze(1), h).squeeze(1)
        updated = cell(msg, state)
        return torch.where((counts > 0).unsqueeze(-1), updated, state)

    def beat_loss(self, packed: PackedBeats) -> Tensor:
        """Teach the reader. The pack was built on the CPU. One matmul scores the batch."""
        device = self.embed.weight.device
        if packed.nodes.device != device:
            packed = packed.to(device)
        character, dilemma = self._rank_batch(packed)
        total = character.new_zeros(())
        parts = 0
        if packed.n_character:
            chosen = packed.character >= 0
            total = total + F.cross_entropy(character[chosen], packed.character[chosen])
            parts += 1
        if packed.n_dilemma:
            chosen = packed.dilemma >= 0
            total = total + F.cross_entropy(dilemma[chosen], packed.dilemma[chosen])
            parts += 1
            self._read_hits = (dilemma.argmax(dim=-1)[chosen] == packed.dilemma[chosen]).detach()
        else:
            self._read_hits = None
        return total / max(parts, 1)

    def read_hits(self) -> tuple[Tensor, int] | None:
        """Hit total and count for this step. The total stays on the device."""
        if self._read_hits is None or self._read_hits.numel() == 0:
            return None
        return self._read_hits.sum(), int(self._read_hits.numel())

    def read_rate(self) -> float:
        """Share of this step's dilemmas the reader named. Syncs the GPU."""
        if self._read_hits is None or self._read_hits.numel() == 0:
            return 0.0
        return float(self._read_hits.float().mean())

    def extract(self, token_ids: list[int], itos: list[str]) -> tuple[str, str, str]:
        """Read the beats off one story, using the trained ranker."""
        canon = [int(piece) for piece in self.canon.detach().cpu().tolist()]
        boundary = {int(piece) for piece in self.boundary.nonzero(as_tuple=False).flatten().tolist()}
        df = [float(count) for count in self.df.detach().cpu().tolist()]
        n_docs = int(self.n_docs.detach().cpu().item())
        beat = story_beats(token_ids, itos, self.skip_ids, canon, boundary, df, n_docs)
        if beat is None:
            return "", "", ""
        char_scores, dilemma_scores = self._rank(beat, self.embed.weight.device)
        character = itos[beat.nodes[int(char_scores.argmax())]].strip()
        dilemma = ""
        if beat.dilemma_row >= 0:
            dilemma = itos[beat.nodes[int(dilemma_scores.argmax())]].strip()
        resolution = "".join(itos[token] for token in beat.resolution).strip()
        return character, dilemma, resolution

    def _rank(self, beat: StoryBeats, device: torch.device) -> tuple[Tensor, Tensor]:
        """Score every content node as the character and as the dilemma."""
        node_ids = torch.tensor(beat.nodes, dtype=torch.long, device=device)
        counts = torch.tensor(beat.counts, dtype=torch.float, device=device)
        named = torch.tensor(beat.named, dtype=torch.float, device=device)
        in_ending = torch.tensor(beat.in_ending, dtype=torch.float, device=device)
        hidden = self.embed(node_ids)
        idf = torch.log((self.n_docs + 1) / (self.df[node_ids] + 1))
        features = torch.stack((counts.log1p(), named, idf, in_ending), dim=-1)
        character = self.character_head(features).squeeze(-1)
        character = character + hidden @ self.character_query / self.attn_scale
        dilemma = self.dilemma_head(features).squeeze(-1)
        dilemma = dilemma + hidden @ self.dilemma_query / self.attn_scale
        return character, dilemma

    def _rank_batch(self, packed: PackedBeats) -> tuple[Tensor, Tensor]:
        """Score every story's nodes in one matmul."""
        hidden = self.embed(packed.nodes)
        idf = torch.log((self.n_docs + 1) / (self.df[packed.nodes] + 1))
        features = torch.stack((packed.counts.log1p(), packed.named, idf, packed.in_ending), dim=-1)
        character = self.character_head(features).squeeze(-1)
        character = character + hidden @ self.character_query / self.attn_scale
        dilemma = self.dilemma_head(features).squeeze(-1)
        dilemma = dilemma + hidden @ self.dilemma_query / self.attn_scale
        missing = torch.finfo(character.dtype).min
        character = character.masked_fill(~packed.mask, missing)
        dilemma = dilemma.masked_fill(~packed.mask, missing)
        return character, dilemma

    def generate(
        self,
        prompt_ids: list[int],
        *,
        length: int,
        temperature: float,
        repetition_penalty: float,
        eos_id: int,
    ) -> tuple[list[int], GraphState]:
        """Continue a prompt. Stop on the end piece or at ``length``."""
        state = GraphState(self.dim, self.embed.weight.device)
        ids = list(prompt_ids)
        for token in ids:
            emb = self.embed.weight[token]
            self._advance_one(state, emb)
            self.absorb(state, token, record=True)
        for _ in range(length):
            emb = self.embed.weight[ids[-1]]
            scores = self.readout(torch.cat((state.s, emb), dim=0))
            scores = _penalize(scores, ids, repetition_penalty)
            if temperature <= 0:
                nxt = int(scores.argmax())
            else:
                nxt = int(torch.multinomial(torch.softmax(scores / temperature, dim=0), 1))
            if nxt == eos_id:
                break
            ids.append(nxt)
            emb = self.embed.weight[nxt]
            self._advance_one(state, emb)
            self.absorb(state, nxt, record=True)
        return ids, state


def _penalize(scores: Tensor, seen: list[int], penalty: float) -> Tensor:
    if penalty == 1.0 or not seen:
        return scores
    adjusted = scores.clone()
    for token in set(seen):
        if adjusted[token] < 0:
            adjusted[token] = adjusted[token] * penalty
        else:
            adjusted[token] = adjusted[token] / penalty
    return adjusted


def strongest_links(state: GraphState, itos: list[str], limit: int = 12) -> list[str]:
    """Return the heaviest attention edges, decoded, heaviest first."""
    ranked = sorted(state.links, key=lambda item: item[2], reverse=True)
    lines: list[str] = []
    seen: set[tuple[int, int]] = set()
    for src, dst, weight in ranked:
        pair = (src, dst)
        if pair in seen:
            continue
        seen.add(pair)
        lines.append(f"{itos[src].strip()} -> {itos[dst].strip()} {weight:.2f}")
        if len(lines) >= limit:
            break
    return lines
