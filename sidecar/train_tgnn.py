# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Train the story-graph generator.

Each story is one graph. The next piece trains the storyline. A reader learns
to point at the character and the dilemma, and the last sentence is the
resolution. Training stops when neither the loss nor the reading improves.
A finished epoch is written under ``sidecar/data/tgnn/``.

The fast path is written down in ``sidecar/EFFICIENCY.md``. Do not replace the
batch matmul or the prefetch thread with a Python walk on the GPU.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import queue
import shutil
import threading
import time

import torch

from sidecar.beats import PackedBeats, document_frequency, pack_beats
from sidecar.prepare import chunk_ids
from sidecar.tgnn_lm import (
    TOKEN_WINDOW,
    StoryGraphLM,
    node_canon_ids,
    node_skip_ids,
    pack_stories,
    strongest_links,
)
from sidecar.train import limit_stories, load_corpus, sentence_end_ids

_PEAK_LR = 1e-3
_MIN_TOKENS = 2
_LOG_SECONDS = 15.0


def training_device() -> torch.device:
    """Use the GPU. Never train on CPU when one is available."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    raise SystemExit("No GPU is available. Refusing to train on CPU.")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument("--stories", type=int, default=10000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=_PEAK_LR)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--plateau-patience",
        type=int,
        default=3,
        help="Stop after this many epochs fail to beat the best loss by 0.01.",
    )
    return parser


def prefetch_batches(  # noqa: PLR0913
    stories: list[list[int]],
    *,
    batch_size: int,
    skip_ids: set[int],
    canon: list[int],
    itos: list[str],
    boundary_ids: set[int],
    document_counts: list[int],
    n_docs: int,
) -> tuple[queue.Queue[tuple[torch.Tensor, torch.Tensor, int, int, PackedBeats]], threading.Thread]:
    """Pack the next batches on a side thread while the GPU step runs.

    Stories are ordered by length so a short story is not padded out to a long
    neighbor. The beats are read here too, so the GPU step is one matmul.
    The queue holds two batches, matching the oscillator prefetch.
    """
    ordered = sorted(stories, key=len)
    counts = [float(count) for count in document_counts]
    batches: queue.Queue[tuple[torch.Tensor, torch.Tensor, int, int, PackedBeats]] = queue.Queue(
        maxsize=2,
    )

    def produce() -> None:
        for start in range(0, len(ordered), batch_size):
            chunk = ordered[start : start + batch_size]
            tokens, lengths, budget, count = pack_stories(chunk, skip_ids, canon)
            beats = pack_beats(chunk, itos, skip_ids, canon, boundary_ids, counts, n_docs)
            batches.put((tokens, lengths, budget, count, beats))

    worker = threading.Thread(target=produce, daemon=True)
    worker.start()
    return batches, worker


def save_epoch(model: StoryGraphLM, path: Path, *, dim: int, skip_ids: set[int]) -> None:
    """Write one queryable checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": model.state_dict(), "dim": dim, "skip_ids": sorted(skip_ids)},
        path,
    )


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Train until the epoch budget or the plateau."""
    args = build_parser().parse_args()
    torch.manual_seed(int(args.seed))
    device = training_device()
    itos, stories, chunk = load_corpus(args.data_dir)
    stories = limit_stories(stories, int(args.stories))
    skip = node_skip_ids(itos)
    canon = node_canon_ids(itos)
    boundary = tuple(sentence_end_ids(itos))
    folded = sum(index != shared for index, shared in enumerate(canon))
    model = StoryGraphLM(
        len(itos),
        dim=int(args.dim),
        skip_ids=skip,
        canon_ids=canon,
        boundary_ids=boundary,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=0.01)
    out = args.data_dir / "tgnn"
    usable = [story for story in stories if len(story) >= _MIN_TOKENS]
    freq = document_frequency(usable, skip_ids=skip, canon=canon, itos=itos)
    model.set_document_frequency(freq, len(usable))
    batch_size = int(args.batch_size)
    steps_per_epoch = max(1, math.ceil(len(usable) / batch_size))
    print(
        f"device {device} stories {len(stories)} vocab {len(itos)} "
        f"dim {int(args.dim)} skip {len(skip)} casefold {folded} "
        f"batch {batch_size} window {TOKEN_WINDOW} fp16 steps/epoch {steps_per_epoch} "
        f"reader",
        flush=True,
    )
    best = math.inf
    best_read = 0.0
    quiet = 0
    start_epoch = 0
    out.mkdir(parents=True, exist_ok=True)
    if args.resume is None:
        (out / "index.jsonl").write_text("", encoding="utf-8")
    else:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        start_epoch = int(args.resume.stem.split("_")[-1])
        if (out / "index.jsonl").exists():
            for line in (out / "index.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                loss = float(row["loss"])
                if loss < best - 0.01:
                    best = loss
                    quiet = 0
                else:
                    quiet += 1
        print(f"resuming after epoch {start_epoch} best {best:.3f}", flush=True)
    started = time.perf_counter()
    next_log = started + _LOG_SECONDS
    lead_in = chunk_ids(
        "Once upon a time",
        chunk=chunk,
        stoi={piece: index for index, piece in enumerate(itos)},
    )
    for epoch in range(start_epoch + 1, int(args.epochs) + 1):
        total = torch.zeros((), device=device)
        seen = 0
        batches, worker = prefetch_batches(
            usable,
            batch_size=batch_size,
            skip_ids=skip,
            canon=canon,
            itos=itos,
            boundary_ids=set(boundary),
            document_counts=freq,
            n_docs=len(usable),
        )
        read_hits = torch.zeros((), device=device)
        read_count = 0
        for step in range(1, steps_per_epoch + 1):
            tokens, lengths, budget, count, beats = batches.get()
            tokens = tokens.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = model.loss_on_tokens(
                tokens, lengths, budget=budget, count=count, half=True,
            )
            loss = loss + model.beat_loss(beats)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total = total + loss.detach() * count
            seen += count
            step_hits = model.read_hits()
            if step_hits is not None:
                hit_total, hit_count = step_hits
                read_hits = read_hits + hit_total
                read_count += hit_count
            now = time.perf_counter()
            if now >= next_log:
                print(
                    f"epoch {epoch} step {step}/{steps_per_epoch} "
                    f"loss {float(loss.detach()):.3f} read {model.read_rate():.2f} "
                    f"{now - started:.0f}s",
                    flush=True,
                )
                next_log = now + _LOG_SECONDS
        worker.join()
        mean = float(total) / max(1, seen)
        read_acc = float(read_hits / max(read_count, 1))
        path = out / f"epoch_{epoch:04d}.pt"
        save_epoch(model, path, dim=int(args.dim), skip_ids=skip)
        loss_improved = mean < best - 0.01
        read_improved = read_acc > best_read + 0.01
        if loss_improved:
            best = mean
        if read_improved:
            best_read = read_acc
        if loss_improved or read_improved:
            quiet = 0
            save_epoch(model, out / "best.pt", dim=int(args.dim), skip_ids=skip)
            save_epoch(model, args.data_dir / "tgnn.pt", dim=int(args.dim), skip_ids=skip)
        else:
            quiet += 1
        elapsed = time.perf_counter() - started
        print(
            f"epoch {epoch} loss {mean:.3f} read {read_acc:.2f} best {best:.3f} "
            f"quiet {quiet} {elapsed:.0f}s saved {path.name}",
            flush=True,
        )
        next_log = time.perf_counter() + _LOG_SECONDS
        model.eval()
        with torch.no_grad():
            ids, state = model.generate(
                lead_in, length=80, temperature=0.2, repetition_penalty=1.1, eos_id=0,
            )
            character, dilemma, resolution = model.extract(ids, itos)
            read_character, read_dilemma, read_resolution = model.extract(usable[0], itos)
        model.train()
        wrote = "".join(itos[index] for index in ids[len(lead_in) :])
        print("lead-in: Once upon a time", flush=True)
        print(f"wrote: {wrote}", flush=True)
        print(
            f"character: {character or '(none)'} dilemma: {dilemma or '(none)'}",
            flush=True,
        )
        print(f"resolution: {resolution or '(none)'}", flush=True)
        print(
            f"read: character {read_character or '(none)'} dilemma {read_dilemma or '(none)'}",
            flush=True,
        )
        print(f"read resolution: {read_resolution or '(none)'}", flush=True)
        links = strongest_links(state, itos, limit=6)
        print("links: " + (", ".join(links) if links else "(none)"), flush=True)
        record = {"epoch": epoch, "loss": round(mean, 4), "read": round(read_acc, 4)}
        with (out / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
        if quiet >= int(args.plateau_patience):
            print(
                f"stopping: loss stayed above {best:.3f} and reading stayed at "
                f"{best_read:.2f} for {quiet} epochs",
                flush=True,
            )
            break
    best_path = out / "best.pt"
    if best_path.exists():
        shutil.copyfile(best_path, args.data_dir / "tgnn.pt")


if __name__ == "__main__":
    main()
