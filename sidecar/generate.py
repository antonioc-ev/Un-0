# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Generate a story with the oscillator bag and the story graph.

The bag picks the next piece. After each piece the graph is rebuilt from the
words already said. Once a see/want edge exists, those words are pulled up in
the bag's scores so the next piece can still say them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlx.core as mx

from sidecar.prepare import EOS, chunk_ids
from sidecar.story_tgnn import StoryTGNN, want_piece_ids
from sidecar.train import load_corpus, sentence_end_ids
from sidecar.train_mlx import (
    KuramotoMLX,
    apply_repetition_penalty,
    read_bag_meta,
    spin,
    widths,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
    parser.add_argument("--data-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="Oscillator checkpoint. Defaults to the bag saved in the data dir.",
    )
    parser.add_argument("--length", type=int, default=160, help="Most pieces to add.")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument(
        "--graph-pull",
        type=float,
        default=2.0,
        help="How hard the open want pulls its words up in the bag's scores. 0 is the bag alone.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser


def pull_want(
    logits: mx.array,
    itos: list[str],
    text: str,
    strength: float,
    graph: StoryTGNN,
) -> mx.array:
    """Add ``strength`` to every vocab row that names an end of the open want."""
    if strength == 0.0 or not text.strip():
        return logits
    picks = want_piece_ids(itos, graph.read(text))
    if not picks:
        return logits
    bonus = mx.zeros((logits.shape[0],))
    bonus = bonus.at[mx.array(picks)].add(strength)
    return logits + bonus


def generate(  # noqa: PLR0913
    model: KuramotoMLX,
    itos: list[str],
    *,
    chunk: int,
    prompt: str,
    length: int,
    temperature: float,
    repetition_penalty: float,
    graph_pull: float,
) -> tuple[str, str]:
    """Continue ``prompt``. Return the story and the open want, if one was opened."""
    stoi = {gram: index for index, gram in enumerate(itos)}
    eos_id = stoi.get(EOS)
    ids = chunk_ids(prompt, chunk=chunk, stoi=stoi)
    graph = StoryTGNN()
    theta = model.theta0
    hold = mx.zeros((1, 1))
    for token in ids:
        theta, hold = spin(model, theta, hold, token)
    for _ in range(length):
        shown = model.shown_state(theta[None, :], hold)[0]
        features = mx.concatenate([mx.sin(shown), mx.cos(shown)], axis=-1)
        logits = apply_repetition_penalty(
            model.readout(features[None, :])[0],
            ids,
            repetition_penalty,
        )
        logits = pull_want(logits, itos, "".join(itos[index] for index in ids), graph_pull, graph)
        if temperature <= 0:
            nxt = int(mx.argmax(logits).item())
        else:
            nxt = int(mx.random.categorical(logits / temperature).item())
        if eos_id is not None and nxt == eos_id:
            break
        ids.append(nxt)
        theta, hold = spin(model, theta, hold, nxt)
    story = "".join(itos[index] for index in ids)
    want = graph.read(story).want()
    return story, "(not yet)" if want is None else str(want)


def main() -> None:
    """Load the bag and print one story, then the want the graph kept."""
    args = build_parser().parse_args()
    data_dir: Path = args.data_dir
    weights_path = args.weights or (data_dir / "kuramoto_mlx.safetensors")
    if not weights_path.exists():
        raise FileNotFoundError(f"No saved bag at {weights_path}.")
    itos, _, chunk = load_corpus(data_dir)
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
    story, want = generate(
        model,
        itos,
        chunk=chunk,
        prompt=str(args.prompt),
        length=int(args.length),
        temperature=float(args.temperature),
        repetition_penalty=float(args.repetition_penalty),
        graph_pull=float(args.graph_pull),
    )
    print(story, flush=True)
    print(f"want: {want}", flush=True)


if __name__ == "__main__":
    main()
