# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Continue a start string from the saved Metal bag.

The wheels and the piece list have to be the same packing. A bag saved before
the vocabulary changed will not load. Each training pass is written to
``sidecar/data/checkpoints/`` and can be loaded with ``--weights``. Generation
stops on ``<|endoftext|>`` or at ``--length``, whichever comes first.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from sidecar.prepare import EOS
from sidecar.train import label_shelf, limit_stories, load_corpus, sentence_end_ids
from sidecar.train_mlx import KuramotoMLX, read_bag_meta, sample, widths


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
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
        default=1.1,
        help="Down-weight pieces already used. 1.0 lets a repeated piece take over.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--from-story",
        type=int,
        default=None,
        help="Continue a training story from the word where its dilemma appears. "
        "The lead-in prompt is not used.",
    )
    return parser


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


def main() -> None:
    """Load the saved bag and print one continuation."""
    args = build_parser().parse_args()
    data_dir: Path = args.data_dir
    weights_path = args.weights or (data_dir / "kuramoto_mlx.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"No saved bag at {weights_path}.")
    itos, stories, chunk = load_corpus(data_dir)
    saved = mx.load(str(weights_path))
    saved_vocab = int(saved["readout.bias"].shape[0])
    if saved_vocab != len(itos):
        raise ValueError(
            f"Saved bag has {saved_vocab} pieces; the vocabulary on disk has {len(itos)}. "
            "Train again so the wheels match this packing."
        )
    n, rank, n_roots = widths(saved)
    meta = read_bag_meta(weights_path)
    mx.random.seed(int(args.seed))
    model = KuramotoMLX(
        len(itos),
        n=n,
        rank=rank,
        n_roots=n_roots,
        n_slow=int(meta["n_slow"]),
        n_stamp=int(meta["n_stamp"]),
        heavy_pull=float(meta["heavy_pull"]),
        heavy_dt_scale=float(meta["heavy_dt_scale"]),
        boundary_ids=tuple(sentence_end_ids(itos)),
        latch=bool(meta.get("latch", 0)),
        latch_blend=float(meta.get("latch_blend", 0.25)),
    )
    model.load_weights(str(weights_path))
    shelf = limit_stories(stories, 10000)
    labels, canon = label_shelf(shelf, itos)
    eos_id = itos.index(EOS)
    token_ids = None
    character = -1
    dilemma = -1
    if args.from_story is not None:
        index = int(args.from_story)
        if index < 0 or index >= len(labels):
            raise SystemExit(f"Story {index} is outside the shelf of {len(labels)}.")
        label = labels[index]
        character = label.character
        dilemma = label.dilemma
        token_ids = _through_dilemma(shelf[index], dilemma, canon, eos_id)
        who = "" if character < 0 else itos[character].strip()
        trouble = "" if dilemma < 0 else itos[dilemma].strip()
        print(
            f"story {index} character {who or '(none)'} dilemma {trouble or '(none)'} "
            f"ending_names {label.arc.ending_names}",
            flush=True,
        )
        print("content: " + "".join(itos[token] for token in token_ids), flush=True)
    text = sample(
        model,
        itos,
        chunk=chunk,
        prompt=str(args.prompt),
        length=int(args.length),
        temperature=float(args.temperature),
        repetition_penalty=float(args.repetition_penalty),
        token_ids=token_ids,
    )
    print(text, flush=True)


if __name__ == "__main__":
    main()
