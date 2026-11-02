# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Generate a story from a trained graph model, then print the edges it used.

The oscillator query is ``python -m sidecar.query``. This command is the graph.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sidecar.prepare import EOS, chunk_ids
from sidecar.tgnn_lm import StoryGraphLM, node_canon_ids, strongest_links
from sidecar.train import load_corpus, sentence_end_ids


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", default="Once upon a time")
    parser.add_argument("--data-dir", type=Path, default=Path("sidecar/data"))
    parser.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="A graph checkpoint. Defaults to sidecar/data/tgnn.pt.",
    )
    parser.add_argument("--length", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--repetition-penalty", type=float, default=1.1)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def load_model(path: Path, itos: list[str], device: torch.device) -> StoryGraphLM:
    """Rebuild the graph model from a checkpoint of this same shape."""
    saved = torch.load(path, map_location=device, weights_only=False)
    model = StoryGraphLM(
        len(itos),
        dim=int(saved["dim"]),
        skip_ids=set(saved["skip_ids"]),
        canon_ids=node_canon_ids(itos),
        boundary_ids=tuple(sentence_end_ids(itos)),
    ).to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model


def main() -> None:
    """Print one story and the attention edges stored while it was written."""
    args = build_parser().parse_args()
    weights = args.weights or (args.data_dir / "tgnn.pt")
    if not weights.exists():
        raise FileNotFoundError(
            f"No graph checkpoint at {weights}. Train with python -m sidecar.train_tgnn."
        )
    itos, _, chunk = load_corpus(args.data_dir)
    device = torch.device("cpu")
    torch.manual_seed(int(args.seed))
    model = load_model(weights, itos, device)
    stoi = {piece: index for index, piece in enumerate(itos)}
    prompt = chunk_ids(str(args.prompt), chunk=chunk, stoi=stoi)
    eos_id = stoi[EOS]
    with torch.no_grad():
        ids, state = model.generate(
            prompt,
            length=int(args.length),
            temperature=float(args.temperature),
            repetition_penalty=float(args.repetition_penalty),
            eos_id=eos_id,
        )
        character, dilemma, resolution = model.extract(ids, itos)
    lead = "".join(itos[index] for index in prompt)
    wrote = "".join(itos[index] for index in ids[len(prompt) :])
    print(f"lead-in: {lead}", flush=True)
    print(f"wrote: {wrote}", flush=True)
    print(f"character: {character or '(none)'} dilemma: {dilemma or '(none)'}", flush=True)
    print(f"resolution: {resolution or '(none)'}", flush=True)
    links = strongest_links(state, itos)
    print("graph:", flush=True)
    for line in links or ["(none)"]:
        print(f"  {line}", flush=True)


if __name__ == "__main__":
    main()
