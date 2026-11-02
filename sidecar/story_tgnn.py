# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Temporal graph for a story. Separate from the oscillator bag.

A sentence adds nodes and edges. Later sentences pass messages. They do not
replace the graph, so an early edge is still there when a new animal arrives.
The open want is the first see/want edge. It stays until the story ends.
"""

from __future__ import annotations

import re
import sys

import torch
from torch import nn

_SKIP = {
    "a", "an", "the", "and", "or", "to", "of", "in", "on", "at", "for", "with",
    "from", "into", "over", "by", "as", "was", "were", "is", "are", "be", "been",
    "being", "that", "this", "there", "then", "once", "upon", "very", "so", "but",
    "not", "just", "too", "would", "could", "should", "little", "big", "small",
    "one", "day", "next", "time", "middle",
}
_VERBS = {
    "named", "saw", "wanted", "loved", "love", "found", "said", "went", "go",
    "going", "gone", "find", "play", "played", "come", "came", "look", "looked",
    "see", "seen", "asked", "replied", "gave", "started", "start", "shake",
    "shook", "help", "helped", "get", "got", "hurt", "laugh", "laughed", "hug",
    "hugged", "kiss", "kissed", "need", "needed", "wish", "wished", "lose",
    "lost", "try", "tried", "remember", "remembered", "scare", "scared",
}
_WANT = {"saw", "wanted", "found", "needed", "wished", "lost", "tried"}
_PRONOUNS = {"she", "he", "it", "her", "him", "his", "they", "them", "their", "we"}
_STARTERS = {"once", "one", "when", "then", "after", "suddenly", "the", "a"}


def _bucket(text: str, n: int) -> int:
    total = 2166136261
    for byte in text.encode():
        total ^= byte
        total = (total * 16777619) & 0xFFFFFFFF
    return total % n


def sentences(text: str) -> list[str]:
    """Split on sentence marks. Keep the mark with its sentence."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part for part in parts if part.strip()]


class Edge:
    """One lasting link between two things in the story."""

    def __init__(self, src: str, dst: str, rel: str, at: int) -> None:
        """Record the two ends, the relation, and which sentence added it."""
        self.src = src
        self.dst = dst
        self.rel = rel
        self.at = at

    def __str__(self) -> str:
        """Render the edge as ``src -rel-> dst``."""
        return f"{self.src} -{self.rel}-> {self.dst}"


class StoryState:
    """Nodes and edges accumulated so far. Absent nodes are left untouched."""

    def __init__(self, dim: int) -> None:
        """Start with no nodes."""
        self.order: list[str] = []
        self.index: dict[str, int] = {}
        self.h = torch.zeros(0, dim)
        self.edges: list[Edge] = []
        self.last_name: str | None = None

    def want(self) -> Edge | None:
        """First see/want edge, if the story has opened one."""
        for edge in self.edges:
            if edge.rel in _WANT:
                return edge
        return None


def want_piece_ids(itos: list[str], state: StoryState) -> list[int]:
    """Vocab rows whose text is an end of the open want, or the current name."""
    edge = state.want()
    if edge is None:
        return []
    keys = {edge.src, edge.dst}
    if state.last_name is not None:
        keys.add(state.last_name)
    return [index for index, piece in enumerate(itos) if piece.strip().lower() in keys]


def observe(
    sentence: str,
    last_name: str | None,
) -> tuple[list[str], list[tuple[str, str, str]], str | None]:
    """Read a sentence into nodes, edges, and the latest name."""
    mentioned: list[str] = []
    edges: list[tuple[str, str, str]] = []
    prev: str | None = None
    pending: str | None = None
    name = last_name
    for raw in re.findall(r"[A-Za-z']+", sentence):
        low = raw.lower()
        if low in _PRONOUNS and name is not None:
            current = name
        elif low in _VERBS:
            pending = low
            continue
        elif low in _SKIP:
            continue
        else:
            current = low
            if raw[:1].isupper() and low not in _STARTERS:
                name = low
        if current not in mentioned:
            mentioned.append(current)
        if prev is not None and prev != current:
            edges.append((prev, current, pending or "with"))
            pending = None
        prev = current
    return mentioned, edges, name


class StoryTGNN(nn.Module):
    """Message passing over a story graph. Nodes survive the next sentence."""

    def __init__(self, dim: int = 32, buckets: int = 1024) -> None:
        """Build the embedding, the message, and the per-node update."""
        super().__init__()
        self.dim = int(dim)
        self.buckets = int(buckets)
        self.embed = nn.Embedding(self.buckets, self.dim)
        self.rel_embed = nn.Embedding(self.buckets, self.dim)
        self.msg = nn.Linear(self.dim * 3, self.dim)
        self.gru = nn.GRUCell(self.dim, self.dim)

    def new_state(self) -> StoryState:
        """Empty graph, before the first sentence."""
        return StoryState(self.dim)

    def read(self, text: str) -> StoryState:
        """Run every sentence. The graph at the end still holds the first edges."""
        state = self.new_state()
        for at, sentence in enumerate(sentences(text)):
            state = self.step(state, sentence, at)
        return state

    def step(self, state: StoryState, sentence: str, at: int) -> StoryState:
        """Add this sentence's nodes and edges, then pass messages."""
        mentioned, found, name = observe(sentence, state.last_name)
        state.last_name = name
        for word in mentioned:
            if word in state.index:
                continue
            state.index[word] = len(state.order)
            state.order.append(word)
            fresh = self.embed(torch.tensor([_bucket(word, self.buckets)]))
            state.h = fresh if state.h.numel() == 0 else torch.cat((state.h, fresh), dim=0)
        known = {(edge.src, edge.dst, edge.rel) for edge in state.edges}
        for src, dst, rel in found:
            if (src, dst, rel) in known:
                continue
            state.edges.append(Edge(src, dst, rel, at))
        if not mentioned or state.h.numel() == 0:
            return state
        agg = torch.zeros_like(state.h)
        for src, dst, rel in found:
            src_i = state.index[src]
            dst_i = state.index[dst]
            rel_h = self.rel_embed(torch.tensor([_bucket(rel, self.buckets)])).squeeze(0)
            packed = torch.cat((state.h[src_i], state.h[dst_i], rel_h), dim=0)
            message = torch.tanh(self.msg(packed))
            agg[dst_i] = agg[dst_i] + message
            agg[src_i] = agg[src_i] + message
        updated = state.h.clone()
        for word in mentioned:
            i = state.index[word]
            updated[i] = self.gru(agg[i], state.h[i])
        state.h = updated
        return state


def trace(text: str, model: StoryTGNN | None = None) -> str:
    """Print the graph after each sentence, including the open want."""
    net = model or StoryTGNN()
    state = net.new_state()
    lines: list[str] = []
    for at, sentence in enumerate(sentences(text)):
        before = len(state.edges)
        state = net.step(state, sentence, at)
        added = ", ".join(str(edge) for edge in state.edges[before:]) or "(none)"
        want = state.want()
        lines.append(f"[{at}] {sentence}")
        lines.append(f"    added: {added}")
        lines.append(f"    nodes: {', '.join(state.order)}")
        lines.append(f"    want: {want if want is not None else '(not yet)'}")
    return "\n".join(lines)


def main() -> None:
    """Read a story from the command line and print the lasting graph."""
    text = " ".join(sys.argv[1:]).strip()
    if not text:
        text = sys.stdin.read()
    print(trace(text), flush=True)


if __name__ == "__main__":
    main()
