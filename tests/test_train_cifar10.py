from __future__ import annotations


def test_dino_batch_size_default_and_override() -> None:
    """The DINO extraction microbatch is configurable without changing its default."""
    from un0.train_cifar10 import build_parser

    parser = build_parser()
    assert parser.parse_args([]).dino_batch_size == 64
    assert parser.parse_args(["--dino-batch-size", "1024"]).dino_batch_size == 1024
