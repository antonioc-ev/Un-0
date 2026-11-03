# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Read character, dilemma, and resolution off a story.

The character is the capitalized name the story returns to. The dilemma is
the other repeated content word, weighted so a rare word outranks a common
one. The resolution is the last sentence. The arc reads whether that sentence
names the character, the dilemma, both, or neither. Nothing here calls
another model.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

_REPEAT = 2
_ENDING_MARKS = 2
# Same glue the node list skips. These pieces are not a name in the ending.
_GLUE = frozenset({
    "a", "an", "the",
    "and", "but", "or", "because",
    "to", "of", "in", "on", "at", "for", "with", "from",
})
EndingNames = Literal["both", "who", "trouble", "neither"]


@dataclass(frozen=True)
class StoryBeats:
    """One reading of one story. Rows index ``nodes``."""

    nodes: tuple[int, ...]
    counts: tuple[int, ...]
    named: tuple[bool, ...]
    in_ending: tuple[bool, ...]
    character_row: int
    dilemma_row: int
    resolution: tuple[int, ...]


def document_frequency(
    stories: list[list[int]],
    *,
    skip_ids: set[int],
    canon: list[int],
    itos: list[str],
) -> list[int]:
    """Count how many stories contain each piece. Skipped pieces stay at zero."""
    counts = [0] * len(itos)
    for story in stories:
        seen: set[int] = set()
        for token in story:
            if token in skip_ids:
                continue
            seen.add(canon[token])
        for key in seen:
            counts[key] += 1
    return counts


def story_beats(  # noqa: PLR0913
    story: list[int],
    itos: list[str],
    skip_ids: set[int],
    canon: list[int],
    boundary_ids: set[int],
    document_counts: list[float],
    n_docs: int,
) -> StoryBeats | None:
    """Point at the character, the dilemma, and the last sentence.

    A capitalized piece outranks a common one. Among the rest, a word this
    story repeats and other stories rarely use is the dilemma. The resolution
    is the last sentence. A one-sentence story has no separate resolution. A
    story with no content word has no reading.
    """
    marks = [index for index, token in enumerate(story) if token in boundary_ids]
    ending = marks[-_ENDING_MARKS] + 1 if len(marks) >= _ENDING_MARKS else -1
    if ending >= len(story):
        ending = -1
    counts: dict[int, int] = {}
    named: dict[int, bool] = {}
    before: set[int] = set()
    after: set[int] = set()
    order: list[int] = []
    at_boundary = True
    for index, token in enumerate(story):
        is_boundary = token in boundary_ids
        if token not in skip_ids:
            key = canon[token]
            if key not in counts:
                named[key] = any(ch.isupper() for ch in itos[token]) and not at_boundary
                order.append(key)
                counts[key] = 0
            counts[key] += 1
            if ending >= 0 and index >= ending:
                after.add(key)
            else:
                before.add(key)
        if is_boundary:
            at_boundary = True
        elif token not in skip_ids:
            at_boundary = False
    if not order:
        return None
    character = max(order, key=lambda key: (named[key], counts[key]))
    pool = [key for key in order if key != character and counts[key] >= _REPEAT and key in before]
    if not pool:
        pool = [key for key in order if key != character and key in before]
    dilemma = (
        max(pool, key=lambda key: _rarity(key, counts, document_counts, n_docs)) if pool else -1
    )
    resolution = tuple(story[ending:-1] if ending >= 0 else ())
    return StoryBeats(
        nodes=tuple(order),
        counts=tuple(counts[key] for key in order),
        named=tuple(named[key] for key in order),
        in_ending=tuple(key in after for key in order),
        character_row=order.index(character),
        dilemma_row=order.index(dilemma) if dilemma >= 0 else -1,
        resolution=resolution,
    )


def category_ids(  # noqa: PLR0913
    story: list[int],
    itos: list[str],
    skip_ids: set[int],
    canon: list[int],
    boundary_ids: set[int],
    document_counts: list[float],
    n_docs: int,
) -> tuple[int, int, tuple[int, ...]]:
    """Return the character id, the dilemma id, and the resolution tokens.

    A missing character or dilemma is ``-1``. A story with no separate last
    sentence has an empty resolution. The ids are pieces, already folded
    through ``canon``.
    """
    beat = story_beats(story, itos, skip_ids, canon, boundary_ids, document_counts, n_docs)
    if beat is None:
        return -1, -1, ()
    dilemma = -1 if beat.dilemma_row < 0 else beat.nodes[beat.dilemma_row]
    return beat.nodes[beat.character_row], dilemma, beat.resolution


@dataclass(frozen=True)
class StoryArc:
    """Whether the last sentence names the character, the dilemma, or both."""

    who: str
    trouble: str
    ending_names: EndingNames


def arc_score(
    character: int,
    dilemma: int,
    resolution: tuple[int, ...],
    itos: list[str],
) -> StoryArc:
    """Read the arc off the character, the dilemma, and the last sentence.

    ``who`` and ``trouble`` are those words. A missing id leaves that side
    empty. ``ending_names`` is ``both``, ``who``, ``trouble``, or ``neither``,
    from a casefold match of whole resolution pieces. Glue and pieces with no
    letters do not count. A missing character or dilemma names neither.
    """
    who = "" if character < 0 else itos[character].strip()
    trouble = "" if dilemma < 0 else itos[dilemma].strip()
    if character < 0 or dilemma < 0:
        return StoryArc(who, trouble, "neither")
    named = {_content_word(itos[piece]) for piece in resolution}
    named.discard("")
    who_word = _content_word(who)
    trouble_word = _content_word(trouble)
    has_who = who_word in named
    has_trouble = trouble_word in named
    if has_who and has_trouble:
        ending: EndingNames = "both"
    elif has_who:
        ending = "who"
    elif has_trouble:
        ending = "trouble"
    else:
        ending = "neither"
    return StoryArc(who, trouble, ending)


def resolution_span(
    ids: list[int],
    boundary_ids: set[int],
    eos_id: int,
) -> tuple[int, ...]:
    """Return the last sentence. One sentence, or none, is an empty ending."""
    body: list[int] = []
    for token in ids:
        if token == eos_id:
            break
        body.append(token)
    marks = [index for index, token in enumerate(body) if token in boundary_ids]
    if len(marks) < _ENDING_MARKS:
        return ()
    return tuple(body[marks[-_ENDING_MARKS] + 1 :])


def holds_cohesion(arc: StoryArc) -> bool:
    """Return whether the ending names the character, the dilemma, or both."""
    return arc.ending_names != "neither"


def _content_word(piece: str) -> str:
    """Casefold one piece. Glue and pieces with no letters count as empty."""
    word = piece.strip().casefold()
    if word in _GLUE or not any(ch.isalpha() for ch in word):
        return ""
    return word


@dataclass
class PackedBeats:
    """One batch of readings, padded to the widest story."""

    nodes: Tensor
    counts: Tensor
    named: Tensor
    in_ending: Tensor
    mask: Tensor
    character: Tensor
    dilemma: Tensor
    n_character: int
    n_dilemma: int

    def to(self, device: torch.device) -> PackedBeats:
        """Copy the pack onto ``device``. The copy is blocking."""
        return PackedBeats(
            self.nodes.to(device),
            self.counts.to(device),
            self.named.to(device),
            self.in_ending.to(device),
            self.mask.to(device),
            self.character.to(device),
            self.dilemma.to(device),
            self.n_character,
            self.n_dilemma,
        )


def pack_beats(  # noqa: PLR0913
    stories: list[list[int]],
    itos: list[str],
    skip_ids: set[int],
    canon: list[int],
    boundary_ids: set[int],
    document_counts: list[float],
    n_docs: int,
) -> PackedBeats:
    """Read every story in the batch on the CPU, before the GPU step."""
    readings = [
        story_beats(story, itos, skip_ids, canon, boundary_ids, document_counts, n_docs)
        for story in stories
    ]
    width = max((len(beat.nodes) for beat in readings if beat is not None), default=1)
    batch = len(stories)
    nodes = torch.zeros(batch, width, dtype=torch.long)
    counts = torch.zeros(batch, width)
    named = torch.zeros(batch, width)
    in_ending = torch.zeros(batch, width)
    mask = torch.zeros(batch, width, dtype=torch.bool)
    character = torch.full((batch,), -1, dtype=torch.long)
    dilemma = torch.full((batch,), -1, dtype=torch.long)
    for row, beat in enumerate(readings):
        if beat is None:
            continue
        size = len(beat.nodes)
        nodes[row, :size] = torch.tensor(beat.nodes)
        counts[row, :size] = torch.tensor(beat.counts, dtype=torch.float)
        named[row, :size] = torch.tensor(beat.named, dtype=torch.float)
        in_ending[row, :size] = torch.tensor(beat.in_ending, dtype=torch.float)
        mask[row, :size] = True
        character[row] = beat.character_row
        dilemma[row] = beat.dilemma_row
    return PackedBeats(
        nodes,
        counts,
        named,
        in_ending,
        mask,
        character,
        dilemma,
        int((character >= 0).sum()),
        int((dilemma >= 0).sum()),
    )


def _rarity(key: int, counts: dict[int, int], document_counts: list[float], n_docs: int) -> float:
    """Prefer a word this story repeats and other stories rarely use."""
    seen = document_counts[key] if key < len(document_counts) else 0
    return counts[key] * math.log((n_docs + 1) / (seen + 1))
