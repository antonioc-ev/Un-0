# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Train the sidecar Kuramoto model on one story per row.

Each story is read in full first and named: character, dilemma, and
resolution. The arc reads whether the last sentence names them. One epoch then
shuffles the stories and walks that order. A long story keeps its opening
and still ends on the end piece; a short story is padded with the end piece
after its own end. The loss is next-kick cross-entropy and skips that
padding. One Euler step is one kick, and a kick is one whole word. The
character drive is on from the first kick. The dilemma drive starts when
that piece appears in the window. The packing is defined in ``sidecar.prepare``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import queue
import random
import threading

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from sidecar.beats import StoryArc, arc_score, category_ids, document_frequency
from sidecar.model import KuramotoWordLM
from sidecar.prepare import EOS, chunk_ids, piece_canon_ids
from sidecar.tgnn_lm import node_skip_ids

# A whole-word story is about 210 kicks. The window is one story, not a crop
# through a joined stream. A long story keeps the opening and the end piece.
STORY_WINDOW = 256
# 128 wheels, rank 16, and 8 roots flattened near loss 3.1. A new sentence
# replaced Lily in that one angle. The 128-wheel passes are kept in
# sidecar/data/checkpoints_n128_rank16_roots8. The even-kick 256-wheel passes
# are in sidecar/data/checkpoints_n256_even_kick (epochs 1, 10, 16, 20, 22).
# The tug rank and the piece-drive grow with the wheel count so the new wheels
# are actually driven.
N_OSCILLATORS = 256
RANK = 32
N_ROOTS = 16
# Eight wheels take a short step on every word, so a repeated name can pile up.
# One more angle accumulates the sentence, and a period copies it into a latch
# the next sentence is not allowed to erase. The latch is the dilemma try.
N_SLOW = 8
N_STAMP = 0
LATCH = True
LATCH_BLEND = 0.25
HEAVY_PULL = 4.0
HEAVY_DT_SCALE = 0.1


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=STORY_WINDOW)
    parser.add_argument("--lr", type=float, default=8e-3)
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
        "--stories",
        type=int,
        default=None,
        help="Use only the first N packed stories. Default is the whole shelf.",
    )
    parser.add_argument("--euler-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--sample-length", type=int, default=60)
    return parser


def load_corpus(data_dir: Path) -> tuple[list[str], list[list[int]], int]:
    """Load the chunk vocab and the pre-packed token rows."""
    vocab = json.loads((data_dir / "vocab.json").read_text(encoding="utf-8"))
    itos = list(vocab["itos"])
    chunk = int(vocab.get("chunk", 1))
    with (data_dir / "tokens.jsonl").open(encoding="utf-8") as handle:
        stories = [[int(token) for token in json.loads(line)] for line in handle]
    return itos, stories, chunk


def limit_stories(stories: list[list[int]], count: int | None) -> list[list[int]]:
    """Keep the first ``count`` stories. ``None`` keeps the shelf."""
    if count is None:
        return stories
    if count < 1:
        raise ValueError("stories must be >= 1")
    return stories[:count]


def sentence_end_ids(itos: list[str]) -> list[int]:
    """Pieces that end a sentence: punctuation containing ``.``, ``!``, or ``?``."""
    found: list[int] = []
    for index, piece in enumerate(itos):
        if any(ch.isalpha() for ch in piece):
            continue
        if any(ch in ".!?" for ch in piece):
            found.append(index)
    return found


@dataclass(frozen=True)
class StoryLabels:
    """Categories for one full story, and whether the ending names them."""

    character: int
    dilemma: int
    resolution: tuple[int, ...]
    arc: StoryArc


@dataclass(frozen=True)
class StoryBatch:
    """Windowed rows plus the categories read before the cut."""

    tokens: Tensor
    character: Tensor
    dilemma: Tensor
    resolution: tuple[tuple[int, ...], ...]
    arc: tuple[StoryArc, ...]

    def to(self, device: torch.device) -> StoryBatch:
        """Copy the tokens and the drive ids. The copy blocks. The arc stays."""
        return StoryBatch(
            self.tokens.to(device),
            self.character.to(device),
            self.dilemma.to(device),
            self.resolution,
            self.arc,
        )


def label_shelf(
    stories: list[list[int]],
    itos: list[str],
) -> tuple[list[StoryLabels], list[int]]:
    """Name every full story once, before any row is cut to the window.

    Document frequency is one pass over this shelf. The only stop list is
    glue: the end piece, punctuation, articles, conjunctions, and
    prepositions.
    """
    canon = piece_canon_ids(itos)
    skip = node_skip_ids(itos)
    boundary = set(sentence_end_ids(itos))
    freq = document_frequency(stories, skip_ids=skip, canon=canon, itos=itos)
    counts = [float(count) for count in freq]
    n_docs = len(stories)
    labels: list[StoryLabels] = []
    for story in stories:
        character, dilemma, resolution = category_ids(
            story,
            itos,
            skip,
            canon,
            boundary,
            counts,
            n_docs,
        )
        arc = arc_score(character, dilemma, resolution, itos)
        labels.append(StoryLabels(character, dilemma, resolution, arc))
    return labels, canon


def _story_batch(rows: list[list[int]], labels: list[StoryLabels]) -> StoryBatch:
    """Attach the full-story categories, including the arc, to the windowed rows."""
    return StoryBatch(
        tokens=torch.tensor(rows, dtype=torch.long),
        character=torch.tensor([label.character for label in labels], dtype=torch.long),
        dilemma=torch.tensor([label.dilemma for label in labels], dtype=torch.long),
        resolution=tuple(label.resolution for label in labels),
        arc=tuple(label.arc for label in labels),
    )


def fit_story(story: list[int], seq_len: int, eos_id: int) -> list[int]:
    """Fit one story into ``seq_len`` without borrowing the next story.

    A longer story keeps its opening. The tail is dropped, then the end piece
    is written on the last kick so the window still ends. Beat labels are
    read from the full story before this cut, so a dilemma word in the tail
    keeps its id and the drive stays off while that piece is outside the
    window. A shorter story is padded with the end piece after the real end.
    Those extra end pieces are padding, not more words.
    """
    if seq_len < 2:  # noqa: PLR2004
        raise ValueError("seq_len must be at least 2.")
    if len(story) > seq_len:
        if story[-1] == eos_id:
            return [*story[: seq_len - 1], eos_id]
        return list(story[:seq_len])
    if len(story) == seq_len:
        return list(story)
    return [*story, *([eos_id] * (seq_len - len(story)))]


def pack_stream(stories: list[list[int]]) -> list[int]:
    """Join documents into one stream. Each row already ends with the end piece."""
    return [token for story in stories for token in story]


def crop_batch(
    stream: list[int],
    *,
    batch_size: int,
    seq_len: int,
    generator: torch.Generator,
) -> Tensor:
    """Draw random crops from the packed stream, shaped ``(batch_size, seq_len)``."""
    limit = len(stream) - seq_len
    if limit < 1:
        raise ValueError("The packed stream is shorter than one training window.")
    rows = []
    for _ in range(batch_size):
        start = int(torch.randint(limit, (), generator=generator).item())
        rows.append(stream[start : start + seq_len])
    return torch.tensor(rows, dtype=torch.long)


def prefetch_stream(
    stream: list[int],
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    seed: int,
) -> tuple[queue.Queue[Tensor], threading.Thread]:
    """Fill a short queue of stream crops. This is the pre-story trainer."""
    batches: queue.Queue[Tensor] = queue.Queue(maxsize=2)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def produce() -> None:
        for _ in range(steps):
            batches.put(
                crop_batch(stream, batch_size=batch_size, seq_len=seq_len, generator=generator)
            )

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    return batches, worker


def story_epoch_steps(n_stories: int, batch_size: int) -> int:
    """Batches in one shuffled pass. A short tail that cannot fill a batch is dropped."""
    if n_stories < 1:
        raise ValueError("Need at least one story.")
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    return max(1, n_stories // batch_size)


def story_target_mask(tokens: Tensor, eos_id: int) -> Tensor:
    """Mark next-piece targets up through the real end piece.

    ``tokens`` is ``(batch, seq)``. The mask lines up with ``tokens[:, 1:]``.
    The target that is the first end piece stays true. Later padding is false.
    A row with no end piece stays true on every target.
    """
    is_end = tokens == eos_id
    has_end = is_end.any(dim=1)
    first = is_end.long().argmax(dim=1)
    positions = torch.arange(tokens.shape[1] - 1, device=tokens.device)
    keep = positions.unsqueeze(0) < first.unsqueeze(1)
    return torch.where(has_end.unsqueeze(1), keep, torch.ones_like(keep))


def next_piece_loss(logits: Tensor, tokens: Tensor, eos_id: int) -> Tensor:
    """Mean next-piece cross-entropy. Padding after the end piece adds nothing."""
    targets = tokens[:, 1:]
    mask = story_target_mask(tokens, eos_id).to(logits.dtype)
    per = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        targets.reshape(-1),
        reduction="none",
    ).reshape(targets.shape)
    return (per * mask).sum() / mask.sum().clamp(min=1)


def prefetch_batches(  # noqa: PLR0913
    stories: list[list[int]],
    *,
    steps: int,
    batch_size: int,
    seq_len: int,
    eos_id: int,
    seed: int,
    labels: list[StoryLabels] | None = None,
) -> tuple[queue.Queue[Tensor | StoryBatch], threading.Thread]:
    """Fill a short queue of story rows ahead of the training step.

    The worker shuffles once, then packs whole stories. When ``labels`` is
    set, each label was read from the full story and the row is cut after
    that. The queue holds two batches. Nothing is pinned, and the caller
    copies onto the device itself.
    """
    if not stories:
        raise ValueError("Need at least one story.")
    if steps < 1 or batch_size < 1:
        raise ValueError("steps and batch_size must be at least 1.")
    if labels is not None and len(labels) != len(stories):
        raise ValueError("Each story needs one label.")
    batches: queue.Queue[Tensor | StoryBatch] = queue.Queue(maxsize=2)
    generator = torch.Generator()
    generator.manual_seed(seed)

    def produce() -> None:
        order = torch.randperm(len(stories), generator=generator).tolist()
        cursor = 0
        span = len(order)
        for _ in range(steps):
            rows: list[list[int]] = []
            chosen: list[StoryLabels] = []
            for _row in range(batch_size):
                # A shelf smaller than the batch repeats so every step has the
                # same width. A normal epoch does not wrap: the tail was dropped.
                index = order[cursor % span]
                rows.append(fit_story(stories[index], seq_len, eos_id))
                if labels is not None:
                    chosen.append(labels[index])
                cursor += 1
            if labels is None:
                batches.put(torch.tensor(rows, dtype=torch.long))
            else:
                batches.put(_story_batch(rows, chosen))

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    return batches, worker


def wheel_report(model: KuramotoWordLM) -> str:
    """Summarize how uneven the learned wheel-to-wheel tugs are."""
    strength = model.outgoing_strength()
    median = float(strength.median())
    quiet = int((strength < 0.1 * max(median, 1e-8)).sum())
    return (
        f"wheel tug min {float(strength.min()):.3f} "
        f"median {median:.3f} max {float(strength.max()):.3f} "
        f"quiet {quiet}/{model.n}"
    )


def main() -> None:
    """Train for a few epochs and print a short sample."""
    args = build_parser().parse_args()
    random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    itos, stories, chunk = load_corpus(args.data_dir)
    stories = limit_stories(stories, args.stories)
    labels, canon = label_shelf(stories, itos)
    seq_len = int(args.seq_len)
    batch_size = int(args.batch_size)
    eos_id = itos.index(EOS)
    total_tokens = sum(len(story) for story in stories)
    steps_per_epoch = story_epoch_steps(len(stories), batch_size)

    model = KuramotoWordLM(
        len(itos),
        n_oscillators=int(args.n_oscillators),
        rank=int(args.rank),
        n_roots=int(args.n_roots),
        euler_steps=int(args.euler_steps),
        n_slow=int(args.n_slow),
        n_stamp=int(args.n_stamp),
        heavy_pull=float(args.heavy_pull),
        heavy_dt_scale=float(args.heavy_dt_scale),
        boundary_ids=tuple(sentence_end_ids(itos)),
        latch=bool(args.latch),
        latch_blend=float(args.latch_blend),
        eos_id=eos_id,
        beat_canon_ids=canon,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"device {device} stories {len(stories)} tokens {total_tokens} "
        f"chunk {chunk} vocab {len(itos)} params {n_params} "
        f"euler {args.euler_steps} batch {batch_size} steps/epoch {steps_per_epoch}",
        flush=True,
    )

    for epoch in range(int(args.epochs)):
        model.train()
        running = torch.zeros((), device=device)
        batches, worker = prefetch_batches(
            stories,
            steps=steps_per_epoch,
            batch_size=batch_size,
            seq_len=seq_len,
            eos_id=eos_id,
            seed=int(args.seed) + epoch,
            labels=labels,
        )
        for _ in range(steps_per_epoch):
            batch = batches.get().to(device)
            logits = model(batch.tokens, batch.character, batch.dilemma)
            loss = next_piece_loss(logits, batch.tokens, eos_id)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running = running + loss.detach()
        worker.join()
        mean_loss = float(running) / steps_per_epoch
        print(
            f"epoch {epoch + 1}/{args.epochs} loss {mean_loss:.3f} {wheel_report(model)}",
            flush=True,
        )

    stoi = {gram: index for index, gram in enumerate(itos)}
    prompt_ids = chunk_ids(args.prompt, chunk=chunk, stoi=stoi)
    continuation = model.generate(
        prompt_ids,
        steps=int(args.sample_length),
        device=device,
        eos_id=stoi.get(EOS),
    )
    text = "".join(itos[index] for index in continuation if itos[index] != EOS)
    checkpoint = args.data_dir / "kuramoto_chunk.pt"
    torch.save(
        {"model": model.state_dict(), "itos": itos, "args": vars(args)},
        checkpoint,
    )
    print(f"sample:\n{text}", flush=True)
    print(f"saved {checkpoint}", flush=True)


if __name__ == "__main__":
    main()
