# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""BM25 over the shelf the trainer just loaded.

Each window ends on a piece. The next piece is what followed that window in a
kept story. A rare word in the words so far outranks a window that only shares
glue, so the list is the stored continuation, not a handwritten word list.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path


class ShelfRoute:
    """Successor list for the pieces just said. Built once when the shelf loads."""

    def __init__(self, stories: list[list[int]], *, window: int = 24) -> None:
        """Index every window that has a following piece."""
        self.stories = stories
        self.window = window
        self.k1 = 1.2
        self.b = 0.75
        self.occ: dict[int, list[tuple[int, int]]] = defaultdict(list)
        df: Counter[int] = Counter()
        lengths: list[int] = []
        for story_index, story in enumerate(stories):
            for pos in range(len(story) - 1):
                self.occ[story[pos]].append((story_index, pos))
                start = max(0, pos + 1 - window)
                doc = story[start : pos + 1]
                lengths.append(len(doc))
                for term in set(doc):
                    df[term] += 1
        self.windows = len(lengths)
        self.avgdl = (sum(lengths) / len(lengths)) if lengths else 1.0
        self.idf = {
            term: math.log(1.0 + (self.windows - count + 0.5) / (count + 0.5))
            for term, count in df.items()
        }
        self.stories_kept = len(stories)
        self.tokens = sum(len(story) for story in stories)

    def save(self, path: Path) -> None:
        """Write the index next to the shelf so a test can skip the rebuild."""
        payload = {
            "window": self.window,
            "windows": self.windows,
            "avgdl": self.avgdl,
            "k1": self.k1,
            "b": self.b,
            "stories_kept": self.stories_kept,
            "tokens": self.tokens,
            "idf": [[term, value] for term, value in self.idf.items()],
            "occ": [[term, rows] for term, rows in self.occ.items()],
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path, stories: list[list[int]]) -> ShelfRoute:
        """Read an index saved for this same shelf. The stories stay the text."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        route = cls.__new__(cls)
        route.stories = stories
        route.window = int(payload["window"])
        route.windows = int(payload["windows"])
        route.k1 = float(payload["k1"])
        route.b = float(payload["b"])
        route.avgdl = float(payload["avgdl"])
        route.stories_kept = int(payload["stories_kept"])
        route.tokens = int(payload["tokens"])
        route.idf = {int(term): float(value) for term, value in payload["idf"]}
        route.occ = {int(term): [tuple(pair) for pair in rows] for term, rows in payload["occ"]}
        kept = len(stories)
        tokens = sum(len(story) for story in stories)
        if kept != route.stories_kept or tokens != route.tokens:
            raise ValueError(
                f"Saved BM25 is {route.stories_kept} stories, {route.tokens} pieces; "
                f"this shelf is {kept} stories, {tokens} pieces."
            )
        return route

    def successors(self, ids: list[int], limit: int = 8) -> list[tuple[int, float]]:
        """Rank the pieces that followed a window matching ``ids``.

        The window has to end on the last piece. A next piece keeps its best
        window, so one exact story beats several partial overlaps.
        """
        if not ids:
            return []
        anchor = ids[-1]
        query = Counter(ids[-self.window :])
        best: dict[int, float] = {}
        for story_index, pos in self.occ.get(anchor, ()):
            story = self.stories[story_index]
            nxt = pos + 1
            if nxt >= len(story):
                continue
            start = max(0, nxt - self.window)
            score = self._bm25(query, story[start:nxt])
            tok = story[nxt]
            if score > best.get(tok, -1.0):
                best[tok] = score
        ranked = sorted(best.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]

    def locked_next(self, ids: list[int], *, minimum: int = 8) -> int | None:
        """Return the next piece when one shelf story owns this ending.

        The match is an exact suffix, longer than a shared opening. Several
        stories that stop at the same short ending do not lock.
        """
        if len(ids) < minimum:
            return None
        anchor = ids[-1]
        best_len = 0
        winners: list[tuple[int, int]] = []
        for story_index, pos in self.occ.get(anchor, ()):
            story = self.stories[story_index]
            if pos + 1 >= len(story):
                continue
            length = 0
            while (
                length < len(ids)
                and pos - length >= 0
                and story[pos - length] == ids[-1 - length]
            ):
                length += 1
            if length < minimum or length < best_len:
                continue
            if length > best_len:
                best_len = length
                winners = [(story_index, pos)]
            else:
                winners.append((story_index, pos))
        if len(winners) != 1:
            return None
        story_index, pos = winners[0]
        return self.stories[story_index][pos + 1]

    def next_sentence(self, ids: list[int], boundary: set[int], *, limit: int = 24) -> list[int]:
        """Return the best matching window's next sentence, as parts to tell."""
        if not ids:
            return []
        anchor = ids[-1]
        query = Counter(ids[-self.window :])
        best_score = -1.0
        best_at: tuple[int, int] | None = None
        for story_index, pos in self.occ.get(anchor, ()):
            story = self.stories[story_index]
            if pos + 1 >= len(story):
                continue
            start = max(0, pos + 1 - self.window)
            score = self._bm25(query, story[start : pos + 1])
            if score > best_score:
                best_score = score
                best_at = (story_index, pos)
        if best_at is None:
            return []
        story = self.stories[best_at[0]]
        parts: list[int] = []
        for tok in story[best_at[1] + 1 :]:
            parts.append(tok)
            if tok in boundary or len(parts) >= limit:
                break
        return parts

    def _bm25(self, query: Counter[int], doc: list[int]) -> float:
        tf = Counter(doc)
        norm = self.k1 * (1.0 - self.b + self.b * len(doc) / self.avgdl)
        score = 0.0
        for term in query:
            freq = tf.get(term, 0)
            if not freq:
                continue
            idf = self.idf.get(term, 0.0)
            score += idf * (freq * (self.k1 + 1.0)) / (freq + norm)
        return score
