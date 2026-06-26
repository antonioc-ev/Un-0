from __future__ import annotations

import os
from argparse import Namespace
from unittest.mock import patch

import pytest
import torch

from un0.common import resolve_device


def test_resolve_device_auto_prefers_cuda_over_mps(monkeypatch) -> None:
    """`auto` keeps preferring CUDA when both GPU backends are reported ready."""
    def fake_current_accelerator(*, check_available):
        assert check_available is True
        return torch.device("cuda")

    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        fake_current_accelerator,
    )

    assert resolve_device("auto").type == "cuda"


def test_resolve_device_auto_falls_back_to_mps(monkeypatch) -> None:
    """`auto` selects MPS when CUDA is unavailable on Apple Silicon."""
    def fake_current_accelerator(*, check_available):
        assert check_available is True
        return torch.device("mps")

    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        fake_current_accelerator,
    )

    assert resolve_device("auto").type == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    """`auto` stays usable when neither CUDA nor MPS is available."""
    def fake_current_accelerator(*, check_available):
        assert check_available is True
        return None

    monkeypatch.setattr(
        torch.accelerator,
        "current_accelerator",
        fake_current_accelerator,
    )

    assert resolve_device("auto").type == "cpu"


def test_train_cifar10_distributed_path_keeps_cuda_ddp(monkeypatch) -> None:
    """DDP setup still binds the local CUDA device instead of using `auto`."""
    import un0.train_cifar10 as train_cifar10

    args = Namespace(
        seed=42,
        precision="bf16",
        precomputed_dino_features=None,
        fid_every_epochs=0,
        dataset="uoft-cs/cifar10",
        batch_size=2,
        num_workers=0,
    )

    monkeypatch.setattr(train_cifar10.torch.cuda, "set_device", lambda *_args: None)
    init_calls: list[tuple[str, torch.device]] = []
    monkeypatch.setattr(
        train_cifar10.dist,
        "init_process_group",
        lambda *, backend, device_id: init_calls.append((backend, device_id)),
    )
    monkeypatch.setattr(train_cifar10, "disable_cudnn_sdp_on_blackwell", lambda: None)
    monkeypatch.setattr(train_cifar10, "disable_torchscript_gpu_fuser_on_blackwell", lambda: None)
    monkeypatch.setattr(train_cifar10, "seed_everything", lambda *_args: None)
    monkeypatch.setattr(
        train_cifar10,
        "build_cifar10_dataloader",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after setup")),
    )

    with (
        patch.object(train_cifar10, "resolve_device", side_effect=AssertionError("auto not used")),
        patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "3"}, clear=False),
        pytest.raises(RuntimeError, match="stop after setup"),
    ):
        train_cifar10.train(args)

    assert init_calls == [("nccl", torch.device("cuda:3"))]


def test_train_imagenet_distributed_path_keeps_cuda_ddp(monkeypatch) -> None:
    """ImageNet DDP setup still binds the local CUDA device instead of using `auto`."""
    import un0.train_imagenet as train_imagenet

    args = Namespace(
        seed=42,
        precision="bf16",
        fid_every_epochs=0,
        val_root=None,
        data_root="/tmp/imagenet64",  # noqa: S108
        batch_size=2,
        num_workers=0,
    )

    monkeypatch.setattr(train_imagenet.torch.cuda, "set_device", lambda *_args: None)
    init_calls: list[tuple[str, torch.device]] = []
    monkeypatch.setattr(
        train_imagenet.dist,
        "init_process_group",
        lambda *, backend, device_id: init_calls.append((backend, device_id)),
    )
    monkeypatch.setattr(train_imagenet, "disable_cudnn_sdp_on_blackwell", lambda: None)
    monkeypatch.setattr(train_imagenet, "disable_torchscript_gpu_fuser_on_blackwell", lambda: None)
    monkeypatch.setattr(train_imagenet, "seed_everything", lambda *_args: None)
    monkeypatch.setattr(
        train_imagenet,
        "build_imagenet64_dataloader",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("stop after setup")),
    )

    with (
        patch.object(train_imagenet, "resolve_device", side_effect=AssertionError("auto not used")),
        patch.dict(os.environ, {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "4"}, clear=False),
        pytest.raises(RuntimeError, match="stop after setup"),
    ):
        train_imagenet.train(args)

    assert init_calls == [("nccl", torch.device("cuda:4"))]
