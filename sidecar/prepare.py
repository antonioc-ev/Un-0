# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Pack TinyStories into the kicks the bag trains on.

The kick grew in three steps. Characters came first: one letter per kick, and
the samples collapsed. A fixed grid came next, 2 characters, then 3, then 4.
Four was the grid that could be trained once a rare piece was spelled as
letters instead of an unknown token. That grid still cut inside words
(``upo`` / ``n``). Whole words are the packing now. The splitter already keeps
a period, a quote, and a newline off the word, so turning the grid off makes
each word one kick. ``chunk`` 0 is that packing. A positive ``chunk`` is the
old grid. A word seen fewer than ``min_count`` times is still spelled letter
by letter. Each story ends with ``<|endoftext|>``, the piece the query treats
as a stop. The full TinyStories shelf is about 2.1 million stories.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
import json
from pathlib import Path

from datasets import load_dataset

_MIN_STORY_CHARS = 40
EOS = "<|endoftext|>"
_DEFAULT_COUNT = 1000


def piece_canon_ids(itos: list[str]) -> list[int]:
    """Map each piece onto its lowercase piece when that piece exists.

    ``She`` and ``she`` are one kick. Playback still writes the capital when
    the sentence calls for it.
    """
    lookup = {piece: index for index, piece in enumerate(itos)}
    return [lookup.get(piece.casefold(), index) for index, piece in enumerate(itos)]


def case_groups(itos: list[str]) -> list[list[int]]:
    """Surface pieces that casefold together, in first-seen order."""
    groups: dict[str, list[int]] = {}
    order: list[str] = []
    for index, piece in enumerate(itos):
        key = piece.casefold()
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(index)
    return [groups[key] for key in order]


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--count",
        type=int,
        default=None,
        help="How many stories to keep. Defaults to 1000 on download, or the whole file on reuse.",
    )
    parser.add_argument(
        "--chunk",
        type=int,
        default=0,
        help="0 keeps each word whole. A positive value is the old fixed grid.",
    )
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument(
        "--reuse-stories",
        action="store_true",
        help="Retokenize sidecar/data/stories.jsonl instead of downloading again.",
    )
    return parser


def load_stories(path: Path) -> list[str]:
    """Read the raw story texts."""
    with path.open(encoding="utf-8") as handle:
        return [str(json.loads(line)["text"]) for line in handle]


_CONTRACTIONS = ("'re", "'ve", "'ll", "'s", "'t", "'m", "'d")


def _contraction_at(text: str, i: int) -> str:
    """Return a contraction piece at ``i``, or an empty string."""
    if text[i] != "'":
        return ""
    return next((item for item in _CONTRACTIONS if text.startswith(item, i)), "")


def _whitespace_atom(text: str, i: int) -> tuple[str, int] | None:
    """Take a whitespace run. A single space before a word is left for that word."""
    if not text[i].isspace():
        return None
    j = i + 1
    while j < len(text) and text[j].isspace():
        j += 1
    attached = j == i + 1 and text[i] == " " and j < len(text) and text[j].isalnum()
    if attached:
        return None
    return text[i:j], j


def _marked_run(text: str, i: int, pred: Callable[[str], bool]) -> tuple[str, int] | None:
    """Take an optional leading space plus a run accepted by ``pred``."""
    start = i
    if text[i] == " " and i + 1 < len(text) and pred(text[i + 1]):
        i += 1
    if i >= len(text) or not pred(text[i]):
        return None
    i += 1
    while i < len(text) and pred(text[i]):
        i += 1
    return text[start:i], i


def atoms(text: str) -> list[str]:
    """Split ``text`` into words, numbers, punctuation, and whitespace runs.

    A single space before a word stays on that word. A period, quote, or
    newline is never glued onto the word it follows.
    """
    out: list[str] = []
    i = 0
    while i < len(text):
        contraction = _contraction_at(text, i)
        if contraction:
            out.append(contraction)
            i += len(contraction)
            continue
        whitespace = _whitespace_atom(text, i)
        if whitespace is not None:
            piece, i = whitespace
            out.append(piece)
            continue
        word = _marked_run(text, i, str.isalpha)
        if word is None:
            word = _marked_run(text, i, str.isdigit)
        if word is None:
            word = _marked_run(text, i, lambda ch: not ch.isspace() and not ch.isalnum())
        if word is None:
            out.append(text[i])
            i += 1
            continue
        piece, i = word
        out.append(piece)
    return out


def pieces(text: str, chunk: int) -> list[str]:
    """Return one kick per word when ``chunk`` is 0.

    A positive ``chunk`` is the earlier fixed grid. It slices a long word
    every ``chunk`` characters and leaves punctuation whole.
    """
    if chunk < 0:
        raise ValueError("chunk must be >= 0")
    found = atoms(text)
    if chunk == 0:
        return found
    out: list[str] = []
    for atom in found:
        if len(atom) <= chunk:
            out.append(atom)
            continue
        out.extend(atom[i : i + chunk] for i in range(0, len(atom), chunk))
    return out


def chunk_ids(text: str, *, chunk: int, stoi: dict[str, int]) -> list[int]:
    """Map text to ids. A piece that was not kept is spelled letter by letter."""
    ids: list[int] = []
    for gram in pieces(text, chunk):
        piece_id = stoi.get(gram)
        if piece_id is not None:
            ids.append(piece_id)
            continue
        for letter in gram:
            letter_id = stoi.get(letter)
            if letter_id is None:
                raise KeyError(f"No piece for {gram!r}; {letter!r} is not in the vocabulary.")
            ids.append(letter_id)
    return ids


def document_ids(text: str, *, chunk: int, stoi: dict[str, int]) -> list[int]:
    """Pack one story and append the shared end piece."""
    end = stoi[EOS]
    return [*chunk_ids(text, chunk=chunk, stoi=stoi), end]


def build_vocab(
    texts: list[str], *, chunk: int, min_count: int
) -> tuple[list[str], dict[str, int]]:
    """Keep pieces seen at least ``min_count`` times, and every letter."""
    counts: Counter[str] = Counter()
    letters: set[str] = set()
    for text in texts:
        counts.update(pieces(text, chunk))
        letters.update(text)
    kept = {gram for gram, count in counts.items() if count >= min_count}
    kept.update(letters)
    kept.discard(EOS)
    itos = [EOS, *sorted(kept)]
    return itos, {gram: index for index, gram in enumerate(itos)}


def main() -> None:
    """Write the story list and the chunked training ids."""
    args = build_parser().parse_args()
    if int(args.chunk) < 0 or int(args.min_count) < 1:
        raise ValueError("chunk must be >= 0 and min-count must be >= 1.")
    if args.count is not None and int(args.count) < 1:
        raise ValueError("count must be >= 1.")
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    stories_path = output_dir / "stories.jsonl"
    vocab_path = output_dir / "vocab.json"
    tokens_path = output_dir / "tokens.jsonl"

    if args.reuse_stories:
        if not stories_path.exists():
            raise FileNotFoundError(f"No story list at {stories_path}.")
        texts = load_stories(stories_path)
        if args.count is not None:
            texts = texts[: int(args.count)]
    else:
        stream = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
        texts = []
        count = _DEFAULT_COUNT if args.count is None else int(args.count)
        with stories_path.open("w", encoding="utf-8") as handle:
            for row in stream:
                text = str(row["text"]).strip()
                if len(text) < _MIN_STORY_CHARS:
                    continue
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                texts.append(text)
                if len(texts) >= count:
                    break

    chunk = int(args.chunk)
    itos, stoi = build_vocab(texts, chunk=chunk, min_count=int(args.min_count))
    token_rows = [document_ids(text, chunk=chunk, stoi=stoi) for text in texts]
    with tokens_path.open("w", encoding="utf-8") as handle:
        for row in token_rows:
            handle.write(json.dumps(row) + "\n")
    vocab_path.write_text(
        json.dumps(
            {
                "itos": itos,
                "chunk": chunk,
                "min_count": int(args.min_count),
                "eos": EOS,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    n_tokens = sum(len(row) for row in token_rows)
    grid = [gram for text in texts for gram in pieces(text, chunk)]
    n_spelled = sum(gram not in stoi for gram in grid)
    print(
        f"stories {len(texts)} chunk {chunk} tokens {n_tokens} "
        f"vocab {len(itos)} spelled {n_spelled / max(len(grid), 1):.1%} -> {tokens_path}"
    )


if __name__ == "__main__":
    main()
