# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Continue a start string from the saved Metal bag.

The wheels and the piece list have to be the same packing. A bag saved before
the vocabulary changed will not load. Pass a ``.safetensors`` path as the first
argument, or pass ``--weights``. Generation stops on ``<|endoftext|>`` or at
``--length``, whichever comes first.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from sidecar.beats import arc_score, holds_cohesion, resolution_span
from sidecar.prepare import EOS
from sidecar.shelf_route import ShelfRoute
from sidecar.train import (
    STORY_WINDOW,
    keep_short_stories,
    kept_story_at,
    label_shelf,
    limit_stories,
    load_corpus,
    sentence_end_ids,
)

if TYPE_CHECKING:
    from sidecar.train_mlx import KuramotoMLX


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Once upon a time",
        help="Words to continue. A path ending in .safetensors is the bag, same as --weights.",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="A checkpoint file. Defaults to the best bag saved in the data dir.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=400,
        help="Most pieces to add. Stories that reach the end piece stop sooner.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Stay near the top piece. 0 repeats one cycle. Higher values wander off the story.",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.3,
        help="Down-weight content words and a blank line already used. "
        "Glue and sentence marks keep their score. 1.0 leaves every piece.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--shelf",
        action="store_true",
        help="Also print the kept training story and its character, dilemma, and ending.",
    )
    parser.add_argument(
        "--from-story",
        type=int,
        default=None,
        help="Continue a kept training story from the word where its dilemma appears. "
        "Index 0 is the first story that fits the trainer window. "
        "The lead-in prompt is not used.",
    )
    parser.add_argument(
        "--cohesion",
        type=int,
        default=0,
        help="Continue this many training stories and read whether the ending still names them.",
    )
    return parser


def use_bag_path(args: argparse.Namespace) -> argparse.Namespace:
    """Treat a ``.safetensors`` first argument as the bag.

    ``--weights`` still wins when both are set. The start string falls back to
    the usual opening so the filename is not read as the story.
    """
    prompt = args.prompt
    if isinstance(prompt, str) and prompt.endswith(".safetensors"):
        if args.weights is None:
            args.weights = Path(prompt)
        args.prompt = "Once upon a time"
    return args


def _through_dilemma(story: list[int], dilemma: int, canon: list[int], eos_id: int) -> list[int]:
    """Keep the opening through the first time the dilemma is said."""
    prefix: list[int] = []
    for token in story:
        if token == eos_id:
            break
        prefix.append(token)
        if dilemma >= 0 and canon[token] == dilemma:
            return prefix
    return prefix


def report_cohesion(  # noqa: PLR0913
    model: KuramotoMLX,
    itos: list[str],
    stories: list[list[int]],
    *,
    chunk: int,
    count: int,
    length: int,
    temperature: float,
    repetition_penalty: float,
    route: ShelfRoute | None = None,
) -> int:
    """Continue each story from its dilemma and print whether the ending holds.

    Returns how many endings name the character, the dilemma, or both.
    """
    from sidecar.train_mlx import continue_ids  # noqa: PLC0415

    shelf = stories[:count]
    labels, canon = label_shelf(shelf, itos)
    eos_id = itos.index(EOS)
    boundary = set(sentence_end_ids(itos))
    held = 0
    print(f"cohesion {len(labels)}", flush=True)
    for index, (story, label) in enumerate(zip(shelf, labels, strict=True)):
        prefix = _through_dilemma(story, label.dilemma, canon, eos_id)
        written = continue_ids(
            model,
            itos,
            chunk=chunk,
            prompt="",
            length=length,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            token_ids=prefix,
            character=label.character,
            dilemma=label.dilemma,
            canon=canon,
            route=route,
        )
        ending = resolution_span(written, boundary, eos_id)
        arc = arc_score(label.character, label.dilemma, ending, itos)
        kept = holds_cohesion(arc)
        held += int(kept)
        text = "".join(itos[token] for token in ending)
        print(
            f"{index} shelf {label.arc.ending_names} written {arc.ending_names} "
            f"{'holds' if kept else 'slips'} | {text}",
            flush=True,
        )
    print(f"cohesion held {held}/{len(labels)}", flush=True)
    return held


def main() -> None:  # noqa: PLR0915
    """Load the saved bag and print one continuation."""
    import mlx.core as mx  # noqa: PLC0415

    from sidecar.train_mlx import (  # noqa: PLC0415
        KuramotoMLX,
        load_bag_weights,
        read_bag_meta,
        readout_spec,
        sample,
        widths,
    )

    args = use_bag_path(build_parser().parse_args())
    data_dir: Path = args.data_dir
    weights_path = args.weights or (data_dir / "kuramoto_mlx.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"No saved bag at {weights_path}.")
    itos, stories, chunk = load_corpus(data_dir)
    saved = mx.load(str(weights_path))
    saved_vocab, readout_rank = readout_spec(saved)
    if saved_vocab != len(itos):
        raise ValueError(
            f"Saved bag has {saved_vocab} pieces; the vocabulary on disk has {len(itos)}. "
            "Train again so the wheels match this packing."
        )
    n, rank, n_roots = widths(saved)
    meta = read_bag_meta(weights_path)
    limited = limit_stories(stories, 10000)
    shelf = keep_short_stories(limited, STORY_WINDOW)
    labels, canon = label_shelf(shelf, itos)
    route_path = data_dir / "bm25_shelf.json"
    if route_path.is_file():
        try:
            route = ShelfRoute.load(route_path, shelf)
        except ValueError:
            route = ShelfRoute(shelf)
            route.save(route_path)
    else:
        route = ShelfRoute(shelf)
        route.save(route_path)
    mx.random.seed(int(args.seed))
    model = KuramotoMLX(
        len(itos),
        n=n,
        rank=rank,
        n_roots=n_roots,
        n_slow=int(meta["n_slow"]),
        n_stamp=int(meta["n_stamp"]),
        n_arc=int(meta.get("n_arc", 0)),
        heavy_pull=float(meta["heavy_pull"]),
        heavy_dt_scale=float(meta["heavy_dt_scale"]),
        boundary_ids=tuple(sentence_end_ids(itos)),
        latch=bool(meta.get("latch", 0)),
        latch_blend=float(meta.get("latch_blend", 0.25)),
        beat_canon=canon,
        readout_rank=readout_rank,
    )
    load_bag_weights(model, weights_path)
    if int(args.cohesion) > 0:
        report_cohesion(
            model,
            itos,
            shelf,
            chunk=chunk,
            count=int(args.cohesion),
            length=int(args.length),
            temperature=float(args.temperature),
            repetition_penalty=float(args.repetition_penalty),
            route=route,
        )
        return
    eos_id = itos.index(EOS)
    token_ids = None
    character = -1
    dilemma = -1
    if args.from_story is not None:
        index = int(args.from_story)
        try:
            story = kept_story_at(limited, index)[1]
        except ValueError as exc:
            raise SystemExit(str(exc)) from None
        label = labels[index]
        character = label.character
        dilemma = label.dilemma
        token_ids = _through_dilemma(story, dilemma, canon, eos_id)
        if args.shelf:
            who = "" if character < 0 else itos[character].strip()
            trouble = "" if dilemma < 0 else itos[dilemma].strip()
            kept_text = "".join(itos[token] for token in story if token != eos_id)
            print(
                f"story {index} character {who or '(none)'} dilemma {trouble or '(none)'} "
                f"ending_names {label.arc.ending_names}",
                flush=True,
            )
            print(f"shelf: {kept_text}", flush=True)
    text = sample(
        model,
        itos,
        chunk=chunk,
        prompt=str(args.prompt),
        length=int(args.length),
        temperature=float(args.temperature),
        repetition_penalty=float(args.repetition_penalty),
        token_ids=token_ids,
        character=character,
        dilemma=dilemma,
        canon=canon,
        route=route,
    )
    print(text, flush=True)


if __name__ == "__main__":
    main()
