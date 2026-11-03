# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Two clocks share the last wheels. One creeps every word. One moves at a period."""

import inspect
import math
from pathlib import Path

import pytest
import torch

from sidecar.beats import StoryArc, arc_score
from sidecar.model import KuramotoWordLM, clock_ranges
from sidecar.prepare import EOS
from sidecar.train import (
    STORY_WINDOW,
    fit_story,
    keep_short_stories,
    kept_story_at,
    label_shelf,
    next_piece_loss,
    prefetch_batches,
    stack_story_windows,
    story_epoch_steps,
    story_parts,
    story_target_mask,
)


def test_a_safetensors_path_selects_the_bag() -> None:
    """A bare checkpoint path is the bag. ``--weights`` still names the bag."""
    from sidecar.query import build_parser, use_bag_path

    bare = use_bag_path(
        build_parser().parse_args(
            [
                "sidecar/data/checkpoints_n512_fullstory/epoch_0146.safetensors",
                "--temperature",
                "0.23",
                "--length",
                "160",
            ]
        )
    )
    assert bare.weights == Path("sidecar/data/checkpoints_n512_fullstory/epoch_0146.safetensors")
    assert bare.prompt == "Once upon a time"
    assert bare.temperature == 0.23
    assert bare.length == 160
    assert bare.from_story is None

    flagged = use_bag_path(
        build_parser().parse_args(
            ["--weights", "sidecar/data/kuramoto_mlx.safetensors", "Once upon a time"]
        )
    )
    assert flagged.weights == Path("sidecar/data/kuramoto_mlx.safetensors")
    assert flagged.prompt == "Once upon a time"

    both = use_bag_path(
        build_parser().parse_args(["epoch_0001.safetensors", "--weights", "best.safetensors"])
    )
    assert both.weights == Path("best.safetensors")
    assert both.prompt == "Once upon a time"


def test_clock_ranges_split_the_end_of_the_bag() -> None:
    """The two clocks occupy the last wheels. 256 with 8 and 8 is 240–247 and 248–255."""
    slow, stamp = clock_ranges(256, 8, 8)
    assert list(slow) == list(range(240, 248))
    assert list(stamp) == list(range(248, 256))
    assert clock_ranges(8, 0, 0) == (range(8, 8), range(8, 8))


def test_arc_wheel_steps_only_after_the_resolution_starts() -> None:
    """One word wheel is the arc. A period still moves only the stamp."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train import N_ARC
    from sidecar.train_mlx import KuramotoMLX

    slow, stamp, arc = clock_ranges(256, 8, 8, n_arc=1)
    assert N_ARC == 1
    assert list(slow) == list(range(239, 247))
    assert list(stamp) == list(range(247, 255))
    assert list(arc) == [255]
    torch_bag = KuramotoWordLM(
        4,
        n_oscillators=256,
        rank=1,
        n_roots=1,
        n_slow=8,
        n_stamp=8,
    )
    assert torch_bag.n_arc == 1
    assert torch_bag.arc == range(255, 256)

    # words 0–1, slow 2, stamp 3, arc 4. Piece 1 is a sentence end.
    model = KuramotoMLX(
        5,
        n=5,
        rank=1,
        n_roots=1,
        n_slow=1,
        n_stamp=1,
        heavy_pull=4.0,
        heavy_dt_scale=0.1,
        boundary_ids=(1,),
        eos_id=2,
    )
    assert model.n_arc == 1
    assert (model.slow, model.stamp, model.arc) == (range(2, 3), range(3, 4), range(4, 5))
    model.omega = mx.ones((model.n,))
    model.U = mx.zeros((model.n, 1))
    model.V = mx.zeros((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.theta0 = mx.zeros((model.n,))
    off = model.step_scale(mx.array([0]), mx.array([False]))
    on = model.step_scale(mx.array([0]), mx.array([True]))
    ended = model.step_scale(mx.array([1]), mx.array([False]))
    mx.eval(off, on, ended)
    assert math.isclose(float(off[0, 0]), 0.2, abs_tol=1e-5)
    assert math.isclose(float(off[0, 2]), 0.02, abs_tol=1e-5)
    assert math.isclose(float(off[0, 3]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(off[0, 4]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(on[0, 4]), model.dt, abs_tol=1e-5)
    assert math.isclose(float(on[0, 3]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(ended[0, 3]), model.dt, abs_tol=1e-5)
    assert math.isclose(float(ended[0, 2]), 0.02, abs_tol=1e-5)
    assert math.isclose(float(ended[0, 4]), 0.0, abs_tol=1e-5)

    resolution = mx.array([3])
    before = mx.array([2])
    _logits, held, *_rest = model.roll(mx.array([[0, 0, 0]]), None, None, resolution, before)
    _logits, opened, *_rest = model.roll(mx.array([[0, 0, 3, 0]]), None, None, resolution, before)
    _logits, stamped, *_rest = model.roll(mx.array([[1, 0]]), None, None, resolution, mx.array([9]))
    _logits, rested, *_rest = model.roll(mx.array([[2, 0]]), None, None, resolution, mx.array([0]))
    mx.eval(held, opened, stamped, rested)
    assert math.isclose(float(held[0, 4]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(opened[0, 4]), model.dt, abs_tol=1e-5)
    assert math.isclose(float(opened[0, 3]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(stamped[0, 3]), model.dt, abs_tol=1e-5)
    assert math.isclose(float(stamped[0, 4]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(mx.max(mx.abs(rested))), 0.0, abs_tol=1e-5)

    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.root_phase.weight = mx.zeros((model.vocab_size, 1))
    arc_phase = mx.zeros((1, model.n)).at[0, 4].add(math.pi / 2)
    stamp_phase = mx.zeros((1, model.n)).at[0, 3].add(math.pi / 2)
    from_arc = model.velocity(arc_phase, mx.array([0]))
    from_stamp = model.velocity(stamp_phase, mx.array([0]))
    mx.eval(from_arc, from_stamp)
    assert math.isclose(float(from_arc[0, 0]), 4.0, abs_tol=1e-5)
    assert math.isclose(float(from_stamp[0, 0]), 4.0, abs_tol=1e-5)


def test_stamp_clock_moves_only_at_a_sentence_end() -> None:
    """A period gives the stamp wheel a full step. Any other piece leaves it still."""
    model = KuramotoWordLM(
        5,
        n_oscillators=4,
        rank=1,
        n_roots=1,
        n_slow=1,
        n_stamp=1,
        n_arc=0,
        heavy_pull=4.0,
        heavy_dt_scale=0.1,
        boundary_ids=(1,),
    )
    assert (model.slow, model.stamp) == (range(2, 3), range(3, 4))
    with torch.no_grad():
        model.omega.zero_()
        model.D.zero_()
        model.U.fill_(1.0)
        model.V.fill_(1.0)
        model.root_phase.weight.zero_()
        held = torch.zeros(1, 4)
        held[0, 3] = math.pi / 2
        from_stamp = model.velocity(held, torch.zeros(1, dtype=torch.long))
        model.U.zero_()
        model.V.zero_()
        model.omega.fill_(1.0)
        ordinary = torch.zeros(1, dtype=torch.long)
        period = torch.ones(1, dtype=torch.long)
        rest = model.step_for(ordinary) * model.velocity(torch.zeros(1, 4), ordinary)
        ended = model.step_for(period) * model.velocity(torch.zeros(1, 4), period)
    assert torch.allclose(from_stamp[0, 0], torch.tensor(4.0))
    assert torch.allclose(rest[0, 0], torch.tensor(0.2))
    assert torch.allclose(rest[0, 2], torch.tensor(0.02))
    assert torch.allclose(rest[0, 3], torch.tensor(0.0))
    assert torch.allclose(ended[0, 2], torch.tensor(0.02))
    assert torch.allclose(ended[0, 3], torch.tensor(0.2))


def test_latch_holds_the_sentence_and_pulls_from_that_hold() -> None:
    """A period stores the accumulator. Later tugs use the store, not the live wheel."""
    model = KuramotoWordLM(
        5,
        n_oscillators=4,
        rank=1,
        n_roots=1,
        n_slow=1,
        n_arc=0,
        latch=True,
        latch_blend=0.5,
        heavy_pull=4.0,
        heavy_dt_scale=0.1,
        boundary_ids=(1,),
    )
    assert model.latch_at == 3
    assert model.slow == range(2, 3)
    theta = torch.zeros(1, 4)
    theta[0, 3] = 1.0
    latch = torch.zeros(1, 1)
    kept, still = model._commit(latch, theta, torch.zeros(1, dtype=torch.long))
    stored, cleared = model._commit(latch, theta, torch.ones(1, dtype=torch.long))
    with torch.no_grad():
        model.omega.zero_()
        model.D.zero_()
        model.U.fill_(1.0)
        model.V.fill_(1.0)
        model.root_phase.weight.zero_()
        pull = model.velocity(
            torch.zeros(1, 4),
            torch.zeros(1, dtype=torch.long),
            torch.tensor([[math.pi / 2]]),
        )
    assert torch.allclose(kept, torch.zeros(1, 1))
    assert torch.allclose(still[0, 3], torch.tensor(1.0))
    assert torch.allclose(stored, torch.tensor([[0.5]]))
    assert torch.allclose(cleared[0, 3], torch.tensor(0.0))
    assert torch.allclose(pull[0, 0], torch.tensor(4.0), atol=1e-5)


def test_story_parts_keep_the_ending_of_a_long_story() -> None:
    """A story past the window continues. The real ending stays in the last part."""
    eos = 9
    story = [1, 2, 3, 4, 5, eos]
    assert story_parts(story, 4, eos) == [[1, 2, 3, 4], [5, eos, eos, eos]]
    assert story_parts([1, 2, eos], 6, eos) == [[1, 2, eos, eos, eos, eos]]
    windows, active = stack_story_windows([[1, eos], story], 4, eos)
    assert windows[0].tolist() == [[1, eos, eos, eos], [1, 2, 3, 4]]
    assert windows[1].tolist() == [[eos, eos, eos, eos], [5, eos, eos, eos]]
    assert active[0].tolist() == [True, True]
    assert active[1].tolist() == [False, True]


def test_short_shelf_skips_a_story_that_does_not_fit() -> None:
    """A story past the cap is dropped whole. The tail is not cut and closed."""
    short = [1, 9]
    fits = [1, 2, 3, 9]
    longer = [1, 2, 3, 4, 9]
    assert keep_short_stories([short, fits, longer], 4) == [short, fits]
    assert keep_short_stories([longer], 4) == []


def test_from_story_zero_is_the_first_story_that_fits() -> None:
    """Continuation index 0 is the first story the trainer keeps."""
    longer = [7] * (STORY_WINDOW + 1)
    first = [1, 2]
    later = [3, 4, 5]
    shelf, chosen = kept_story_at([longer, first, later], 0)
    assert chosen == first
    assert shelf == [first, later]
    with pytest.raises(ValueError, match="2 stories fit"):
        kept_story_at([longer, first, later], 2)


def test_story_window_keeps_one_story_and_masks_the_padding() -> None:
    """A row is one story. The loss keeps the real end piece and drops the pad."""
    eos = 9
    assert fit_story([1, 2, eos], 6, eos) == [1, 2, eos, eos, eos, eos]
    assert fit_story([1, 2, 3, 4, 5, eos], 4, eos) == [1, 2, 3, eos]
    assert fit_story([1, 2, 3, eos], 4, eos) == [1, 2, 3, eos]
    assert fit_story([1, 2, 3, 4, 5], 3, eos) == [1, 2, 3]
    assert story_epoch_steps(5, 2) == 2
    assert story_epoch_steps(1, 8) == 1
    tokens = torch.tensor(
        [
            [1, 2, eos, eos, eos],
            [1, 2, 3, 4, eos],
            [1, 2, 3, eos, 4],
        ]
    )
    mask = story_target_mask(tokens, eos)
    assert mask[0].tolist() == [True, True, False, False]
    assert mask[1].tolist() == [True, True, True, True]
    assert mask[2].tolist() == [True, True, True, False]
    assert story_target_mask(torch.tensor([[1, 2, 3, 4]]), eos).tolist() == [[True, True, True]]


def test_prefetch_shuffles_whole_stories() -> None:
    """An epoch deals each story once. The end piece closes the row."""
    eos = 9
    stories = [[1, eos], [2, 3, eos], [4, 5, 6, eos], [7, eos]]
    batches, worker = prefetch_batches(
        stories,
        steps=2,
        batch_size=2,
        seq_len=5,
        eos_id=eos,
        seed=1,
    )
    assert batches.maxsize == 2
    rows: list[list[int]] = []
    for _ in range(2):
        rows.extend(batches.get().tolist())
    worker.join()
    assert sorted(rows) == sorted(fit_story(story, 5, eos) for story in stories)
    for row in rows:
        first = row.index(eos)
        assert row[first:] == [eos] * (len(row) - first)


def test_prefetch_repeats_only_to_fill_a_short_shelf() -> None:
    """A shelf smaller than the batch still yields the fixed batch width."""
    eos = 9
    batches, worker = prefetch_batches(
        [[1, 2, eos]],
        steps=1,
        batch_size=2,
        seq_len=4,
        eos_id=eos,
        seed=0,
    )
    batch = batches.get()
    worker.join()
    assert batch.tolist() == [[1, 2, eos, eos], [1, 2, eos, eos]]


def _resting_bag(*, latch: bool, eos: int, period: int) -> KuramotoWordLM:
    """A bag whose phases advance by ``dt`` and whose score is sine of the phase."""
    model = KuramotoWordLM(
        6,
        n_oscillators=4,
        rank=1,
        n_roots=1,
        n_arc=0 if latch else 1,
        latch=latch,
        latch_blend=1.0,
        boundary_ids=(period,),
        eos_id=eos,
    )
    with torch.no_grad():
        model.omega.fill_(1.0)
        model.U.zero_()
        model.V.zero_()
        model.D.zero_()
        model.theta0.zero_()
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        model.readout.weight[0, 0] = 1.0
        if latch:
            model.readout.weight[0, model.latch_at] = 1.0
    return model


def test_end_piece_rests_the_wheels_and_a_period_does_not() -> None:
    """Tokens after the end piece match a fresh story. A period keeps spinning."""
    eos, period, word = 4, 3, 1
    model = _resting_bag(latch=False, eos=eos, period=period)
    carried = torch.tensor(
        [
            [word, word, eos, word, word, word],
            [word, eos, eos, eos, word, word],
        ]
    )
    period_row = torch.tensor([[word, period, word, word]])
    with torch.no_grad():
        logits = model(carried)
        period_logits = model(period_row)
    step = math.sin(0.2)
    two = math.sin(0.4)
    three = math.sin(0.6)
    assert torch.allclose(logits[0, 2, 0], torch.tensor(three), atol=1e-5)
    assert torch.allclose(logits[0, 3, 0], torch.tensor(step), atol=1e-5)
    assert torch.allclose(logits[0, 4, 0], torch.tensor(two), atol=1e-5)
    assert torch.allclose(logits[1, 1, 0], torch.tensor(two), atol=1e-5)
    assert torch.allclose(logits[1, 2, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(logits[1, 3, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(logits[1, 4, 0], torch.tensor(step), atol=1e-5)
    assert torch.allclose(period_logits[0, -1, 0], torch.tensor(three), atol=1e-5)


def test_end_piece_clears_the_latch() -> None:
    """The piece after the end piece does not inherit the held sentence angle."""
    eos, period, word = 4, 3, 1
    model = _resting_bag(latch=True, eos=eos, period=period)
    full = torch.tensor([[word, period, word, eos, word, word, word]])
    fresh = torch.tensor([[word, word, word]])
    with torch.no_grad():
        full_logits = model(full)
        fresh_logits = model(fresh)
    assert torch.allclose(full_logits[:, 4:], fresh_logits, atol=1e-5)
    assert torch.allclose(full_logits[0, 4, 0], torch.tensor(math.sin(0.2)), atol=1e-5)
    assert not torch.allclose(full_logits[:, 3], fresh_logits[:, 0], atol=1e-4)


def test_padding_after_the_end_piece_does_not_train() -> None:
    """Changing the pad, or kicking it, does not move the loss or the end-piece drive."""
    eos = 4
    model = KuramotoWordLM(6, n_oscillators=4, rank=1, n_roots=1, eos_id=eos)
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        model.omega.zero_()
        model.U.fill_(1.0)
        model.V.fill_(1.0)
        model.theta0.zero_()
        model.D.fill_(0.5)
        model.root_phase.weight.copy_(
            torch.randn(model.root_phase.weight.shape, generator=generator)
        )
        model.readout.weight.copy_(torch.randn(model.readout.weight.shape, generator=generator))
        model.readout.bias.zero_()
    tokens = torch.tensor([[1, 2, 3, eos, eos, eos]])
    changed = torch.tensor([[1, 2, 3, eos, 5, 1]])
    with torch.no_grad():
        same = next_piece_loss(model(tokens), tokens, eos)
        other = next_piece_loss(model(changed), changed, eos)
    assert torch.allclose(same, other)
    loss = next_piece_loss(model(tokens), tokens, eos)
    loss.backward()
    assert model.root_phase.weight.grad is not None
    assert torch.allclose(
        model.root_phase.weight.grad[eos], torch.zeros_like(model.root_phase.weight.grad[eos])
    )
    assert torch.count_nonzero(model.root_phase.weight.grad[1]) > 0


def test_mlx_follow_on_window_keeps_the_wheels() -> None:
    """The second window of one story starts from the wheels the first left."""
    mlx = pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    del mlx
    eos, word = 4, 1
    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False, eos_id=eos)
    model.omega = mx.ones((model.n,))
    model.U = mx.zeros((model.n, 1))
    model.V = mx.zeros((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.theta0 = mx.zeros((model.n,))
    weight = mx.zeros((6, 2 * model.n))
    model.readout.weight = weight.at[0, 0].add(1.0)
    model.readout.bias = mx.zeros((6,))
    first = mx.array([[word, word, word, word]])
    _logits, theta, latch, in_story, seen = model.roll(first)
    mx.eval(theta)
    assert float(mx.max(mx.abs(theta))) > 0.0
    continued, _theta, _latch, _in_story, _seen = model.roll(
        mx.array([[word, eos, eos, eos]]),
        theta=theta,
        latch=latch,
        in_story=in_story,
        seen=seen,
    )
    fresh = model(mx.array([[word, eos, eos, eos]]))
    mx.eval(continued, fresh)
    assert not math.isclose(float(continued[0, 0, 0]), float(fresh[0, 0, 0]), abs_tol=1e-5)


def test_mlx_word_kick_uses_the_casefold() -> None:
    """``She`` and ``she`` are one kick. The next-piece target stays the surface id."""
    mlx = pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    del mlx
    model = KuramotoMLX(
        4,
        n=4,
        rank=1,
        n_roots=1,
        n_slow=0,
        n_stamp=0,
        latch=False,
        beat_canon=[0, 1, 1, 3],
    )
    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.D = mx.ones((model.n, 1))
    model.theta0 = mx.zeros((model.n,))
    phase = mx.zeros((4, 1))
    phase = phase.at[1, 0].add(1.0)
    phase = phase.at[2, 0].add(-1.0)
    model.root_phase.weight = phase
    folded = model.velocity(mx.zeros((1, model.n)), mx.array([2]))
    lower = model.velocity(mx.zeros((1, model.n)), mx.array([1]))
    other = model.velocity(mx.zeros((1, model.n)), mx.array([3]))
    mx.eval(folded, lower, other)
    assert mx.allclose(folded, lower)
    assert not mx.allclose(folded, other)


def test_mlx_credit_stages_match_a_full_backward() -> None:
    """Staged credit matches one full walk on a chain short enough to fit."""
    mlx = pytest.importorskip("mlx")
    from mlx import nn
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX, window_credit

    del mlx
    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False, eos_id=4)
    tokens = mx.array([[1, 2, 1, 3, 1, 2, 1, 5, 1]])
    character = mx.array([-1])
    dilemma = mx.array([-1])
    resolution = mx.array([-1])
    resolution_at = mx.array([tokens.shape[1]])
    theta = mx.zeros((1, model.n))
    latch = mx.zeros((1, 1))
    in_story = mx.ones((1,), dtype=mx.bool_)
    seen = mx.zeros((1,), dtype=mx.bool_)

    def full_loss(token_ids: mx.array) -> mx.array:
        logits = model(
            token_ids,
            character,
            dilemma,
            resolution,
            resolution_at,
        )
        targets = token_ids[:, 1:]
        losses = nn.losses.cross_entropy(
            logits.reshape(-1, model.vocab_size),
            targets.reshape(-1),
            reduction="none",
        ).reshape(targets.shape)
        return losses.mean()

    _full_value, full_grad = nn.value_and_grad(model, full_loss)(tokens)
    loss, staged_grad, *_rest = window_credit(
        model,
        tokens,
        character,
        dilemma,
        resolution,
        resolution_at,
        theta,
        latch,
        in_story,
        seen,
        span=4,
    )
    mx.eval(loss, full_grad["omega"], staged_grad["omega"])
    # The staged walk clips the joined gradient to norm 1. Compare direction.
    full_norm = mx.sqrt(mx.sum(full_grad["omega"] ** 2))
    staged_norm = mx.sqrt(mx.sum(staged_grad["omega"] ** 2))
    assert float(full_norm) > 0.0
    assert float(staged_norm) > 0.0
    full_dir = full_grad["omega"] / full_norm
    staged_dir = staged_grad["omega"] / staged_norm
    assert mx.allclose(full_dir, staged_dir, atol=1e-3)


def test_mlx_compiled_window_is_finite_and_an_empty_window_is_not_an_error() -> None:
    """The default step is one compiled graph. A pad-only window does not crash."""
    mlx = pytest.importorskip("mlx")
    import mlx.core as mx
    import mlx.optimizers as optim

    from sidecar.train_mlx import KuramotoMLX, compile_window_step, run_compiled_window

    del mlx
    eos = 4
    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False, eos_id=eos)
    mx.eval(model.parameters())
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    compiled = compile_window_step(model, optimizer)
    theta = mx.zeros((1, model.n))
    latch = mx.zeros((1, 1))
    in_story = mx.ones((1,), dtype=mx.bool_)
    seen = mx.zeros((1,), dtype=mx.bool_)
    empty = mx.full((1, 6), eos)
    loss, theta, *_rest = run_compiled_window(
        model,
        compiled,
        optimizer,
        empty,
        mx.array([-1]),
        mx.array([-1]),
        mx.array([-1]),
        mx.array([empty.shape[1]]),
        theta,
        latch,
        in_story,
        seen,
    )
    mx.eval(loss, theta)
    assert float(loss) == 0.0
    assert float(mx.max(mx.abs(theta))) == 0.0
    tokens = mx.array([[1, 2, 3, eos, eos, eos]])
    loss, theta, *_rest = run_compiled_window(
        model,
        compiled,
        optimizer,
        tokens,
        mx.array([-1]),
        mx.array([-1]),
        mx.array([-1]),
        mx.array([tokens.shape[1]]),
        mx.zeros((1, model.n)),
        mx.zeros((1, 1)),
        mx.ones((1,), dtype=mx.bool_),
        mx.zeros((1,), dtype=mx.bool_),
    )
    mx.eval(loss, theta)
    assert math.isfinite(float(loss))
    assert float(loss) > 0.0


def test_mlx_empty_window_is_not_an_error() -> None:
    """A window that is only the end token has nothing to teach. The story is over."""
    mlx = pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX, window_credit

    del mlx
    eos = 4
    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False, eos_id=eos)
    tokens = mx.full((1, 8), eos)
    loss, grads, theta, *_rest = window_credit(
        model,
        tokens,
        mx.array([-1]),
        mx.array([-1]),
        mx.array([-1]),
        mx.array([tokens.shape[1]]),
        mx.zeros((1, model.n)),
        mx.zeros((1, 1)),
        mx.ones((1,), dtype=mx.bool_),
        mx.zeros((1,), dtype=mx.bool_),
        span=4,
    )
    mx.eval(loss, theta)
    assert grads is None
    assert float(loss) == 0.0
    assert float(mx.max(mx.abs(theta))) == 0.0


def test_mlx_end_piece_rests_the_wheels() -> None:
    """The Metal bag rests after the end piece and ignores the pad that follows."""
    mlx = pytest.importorskip("mlx")
    from mlx import nn
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX, loss_fn

    del mlx
    eos, period, word = 4, 3, 1
    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False, eos_id=eos)
    model.omega = mx.ones((model.n,))
    model.U = mx.zeros((model.n, 1))
    model.V = mx.zeros((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.theta0 = mx.zeros((model.n,))
    weight = mx.zeros((6, 2 * model.n))
    model.readout.weight = weight.at[0, 0].add(1.0)
    model.readout.bias = mx.zeros((6,))
    carried = mx.array(
        [
            [word, word, eos, word, word, word],
            [word, eos, eos, eos, word, word],
        ]
    )
    period_row = mx.array([[word, period, word, word]])
    logits = model(carried)
    period_logits = model(period_row)
    mx.eval(logits, period_logits)
    step = math.sin(0.2)
    two = math.sin(0.4)
    three = math.sin(0.6)
    assert math.isclose(float(logits[0, 2, 0]), three, abs_tol=1e-5)
    assert math.isclose(float(logits[0, 3, 0]), step, abs_tol=1e-5)
    assert math.isclose(float(logits[0, 4, 0]), two, abs_tol=1e-5)
    assert math.isclose(float(logits[1, 2, 0]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(logits[1, 4, 0]), step, abs_tol=1e-5)
    assert math.isclose(float(period_logits[0, -1, 0]), three, abs_tol=1e-5)

    scored = KuramotoMLX(
        6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=True, eos_id=eos
    )
    tokens = mx.array([[1, 2, 3, eos, eos, eos]])
    other = mx.array([[1, 2, 3, eos, 5, 0]])
    mx.eval(scored.parameters())
    assert math.isclose(float(loss_fn(scored, tokens)), float(loss_fn(scored, other)), abs_tol=1e-5)
    compiled = mx.compile(lambda batch: loss_fn(scored, batch))
    compiled_loss = compiled(tokens)
    mx.eval(compiled_loss)
    assert math.isfinite(float(compiled_loss))
    _loss, grads = nn.value_and_grad(scored, lambda batch: loss_fn(scored, batch))(tokens)
    mx.eval(grads)
    assert math.isfinite(float(_loss))


def _drive_model(vocab: int = 8, *, beat_canon: list[int] | None = None) -> KuramotoWordLM:
    """A bag whose only push is the root drive, scored as the sine of wheel 0."""
    model = KuramotoWordLM(
        vocab,
        n_oscillators=4,
        rank=1,
        n_roots=1,
        eos_id=0,
        beat_canon_ids=beat_canon,
    )
    with torch.no_grad():
        model.omega.zero_()
        model.U.fill_(1.0)
        model.V.fill_(1.0)
        model.D.fill_(1.0)
        model.theta0.zero_()
        model.root_phase.weight.zero_()
        for index in range(2, vocab):
            model.root_phase.weight[index, 0] = 0.3 + 0.1 * index
        model.readout.weight.zero_()
        model.readout.bias.zero_()
        model.readout.weight[0, 0] = 1.0
    return model


def test_character_drive_is_on_first_and_dilemma_waits_for_its_piece() -> None:
    """The name moves the first kick. The dilemma waits until its piece is said."""
    model = _drive_model()
    tokens = torch.tensor([[1, 1, 3, 1]])
    missing = torch.tensor([-1])
    character = torch.tensor([2])
    dilemma = torch.tensor([3])
    other = torch.tensor([4])
    with torch.no_grad():
        plain = model(tokens)
        absent = model(tokens, missing, missing)
        named = model(tokens, character, missing)
        on = model(tokens, missing, dilemma)
        alt = model(tokens, missing, other)
        theta = model.theta0.view(1, -1)
        word = torch.tensor([1])
        base = model.velocity(theta, word)
        with_character = model.velocity(theta, word, character=character)
        with_missing = model.velocity(theta, word, dilemma=missing)
        active_a = model.velocity(theta, torch.tensor([3]), dilemma=dilemma)
        active_b = model.velocity(theta, torch.tensor([3]), dilemma=other)
    assert torch.allclose(plain, absent)
    assert not torch.allclose(named[:, 0], plain[:, 0])
    assert not torch.allclose(with_character, base)
    assert torch.allclose(with_missing, base)
    assert torch.allclose(on[:, :2], plain[:, :2])
    assert not torch.allclose(on[:, 2], plain[:, 2])
    assert not torch.allclose(on[:, 2], alt[:, 2])
    assert not torch.allclose(active_a, active_b)
    seen = torch.zeros(1, dtype=torch.bool)
    seen, hit = model._open_dilemma(seen, torch.tensor([3]), dilemma)
    _, later = model._open_dilemma(seen, torch.tensor([1]), dilemma)
    assert int(hit) == 3
    assert int(later) == 3
    forward = inspect.signature(KuramotoWordLM.forward).parameters
    assert forward["dilemma"].default is None
    assert "resolution" not in forward
    assert "arc" not in forward
    assert "dilemma" not in inspect.signature(KuramotoWordLM.generate).parameters


def test_dilemma_drive_follows_the_casefold_id() -> None:
    """A cased piece turns the dilemma on when it folds to the stored id."""
    model = _drive_model(6, beat_canon=[0, 1, 2, 3, 4, 3])
    tokens = torch.tensor([[1, 5, 1]])
    missing = torch.tensor([-1])
    with torch.no_grad():
        plain = model(tokens)
        folded = model(tokens, missing, torch.tensor([3]))
    assert torch.allclose(folded[:, :1], plain[:, :1])
    assert not torch.allclose(folded[:, 1], plain[:, 1])


def test_tail_dilemma_keeps_its_id_and_the_window_drive_stays_off() -> None:
    """The full story names the tail word. The first window has not reached it yet."""
    itos = [EOS, ".", " bird", " Lily", " pond", " needle"]
    story = [2, 3, 1, 3, 3, 1, 5, 5, 3, 1, 4, 1, 0]
    labels, _canon = label_shelf([story], itos)
    assert labels[0].arc == StoryArc("Lily", "needle", "neither")
    assert itos[labels[0].character].strip() == "Lily"
    assert itos[labels[0].dilemma].strip() == "needle"
    assert labels[0].resolution == (4, 1)
    batches, worker = prefetch_batches(
        [story],
        steps=1,
        batch_size=1,
        seq_len=6,
        eos_id=0,
        seed=0,
        labels=labels,
    )
    assert batches.maxsize == 2
    batch = batches.get()
    worker.join()
    assert batch.arc == (StoryArc("Lily", "needle", "neither"),)
    assert batch.dilemma.tolist() == [5]
    assert 5 not in batch.tokens[0].tolist()
    assert 5 in batch.windows[1][0].tolist()
    assert 4 in batch.windows[1][0].tolist()
    assert batch.resolution == ((4, 1),)
    model = _drive_model(len(itos))
    window = batch.tokens
    quiet_id = torch.full_like(batch.dilemma, -1)
    with torch.no_grad():
        held = model(window, batch.character, batch.dilemma)
        quiet = model(window, batch.character, quiet_id)
        live = window.clone()
        live[0, 1] = int(batch.dilemma)
        woken = model(live, batch.character, batch.dilemma)
        still = model(live, batch.character, quiet_id)
    assert torch.allclose(held, quiet)
    assert not torch.allclose(woken, still)


def test_repeated_glue_does_not_become_the_dilemma() -> None:
    """An article is glue. A repeated rare word is the dilemma."""
    itos = [EOS, ".", " bird", " Lily", " the", " needle", " pond"]
    story = [2, 4, 4, 4, 3, 1, 3, 5, 5, 1, 6, 1, 0]
    labels, _canon = label_shelf([story], itos)
    assert itos[labels[0].character].strip() == "Lily"
    assert itos[labels[0].dilemma].strip() == "needle"
    assert labels[0].arc == StoryArc("Lily", "needle", "neither")


def test_cohesion_holds_when_the_ending_names_them() -> None:
    """The last sentence holds when it names the character, the dilemma, or both."""
    from sidecar.beats import holds_cohesion, resolution_span

    itos = [EOS, ".", " Fin", " crab", " happily", " ever", " after"]
    story = [2, 3, 1, 2, 3, 4, 5, 6, 1, 0]
    ending = resolution_span(story, {1}, 0)
    assert ending == (2, 3, 4, 5, 6, 1)
    assert holds_cohesion(arc_score(2, 3, ending, itos))
    assert not holds_cohesion(arc_score(2, 3, (4, 5, 6, 1), itos))
    assert resolution_span([2, 3, 1, 0], {1}, 0) == ()


def test_arc_matches_whole_pieces_in_the_ending() -> None:
    """The ending names a category only when that whole word is in the last sentence."""
    itos = [EOS, ".", " Fin", " crab", " happily", " ever", " after", " kayak", " the", " FIN"]
    assert arc_score(2, 3, (2, 3, 1), itos) == StoryArc("Fin", "crab", "both")
    assert arc_score(2, 7, (9, 1), itos) == StoryArc("Fin", "kayak", "who")
    assert arc_score(2, 3, (8, 3, 1), itos) == StoryArc("Fin", "crab", "trouble")
    assert arc_score(2, 3, (4, 5, 6, 1), itos) == StoryArc("Fin", "crab", "neither")
    assert arc_score(-1, 3, (3, 1), itos) == StoryArc("", "crab", "neither")
    assert arc_score(2, -1, (2, 1), itos) == StoryArc("Fin", "", "neither")


def test_resolution_drive_starts_at_the_last_sentence() -> None:
    """The resolution word kicks on the piece that says it, not on the words after."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX, resolution_columns

    itos = [EOS, ".", " Lily", " needle", " shared"]
    tokens = torch.tensor([[2, 3, 4, 1, 0]])
    piece, start = resolution_columns(tokens, ((4, 1),), itos, list(range(5)))
    assert piece.tolist() == [4]
    assert start.tolist() == [2]
    missing_piece, missing_start = resolution_columns(
        tokens, ((4, 2, 1),), itos, list(range(5)),
    )
    assert missing_piece.tolist() == [-1]
    assert missing_start.tolist() == [5]

    model = KuramotoMLX(5, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.D = mx.ones((model.n, 1))
    model.theta0 = mx.zeros((model.n,))
    phases = mx.zeros((5, 1)).at[4, 0].add(0.7)
    model.root_phase.weight = phases
    batch = mx.array(tokens.numpy())
    plain = model(batch)
    late = model(batch, None, None, mx.array(piece.numpy()), mx.array(start.numpy()))
    mx.eval(plain, late)
    assert mx.allclose(late[:, :2], plain[:, :2], atol=1e-5)
    assert not bool(mx.allclose(late[:, 2], plain[:, 2], atol=1e-5).item())


def test_mlx_dilemma_fades_when_the_resolution_starts() -> None:
    """The trouble stays, lightly, until the ending starts. Then that push fades."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.D = mx.ones((model.n, 1))
    model.theta0 = mx.zeros((model.n,))
    phases = mx.zeros((6, 1)).at[3, 0].add(0.6)
    model.root_phase.weight = phases
    tokens = mx.array([[1, 3, 1, 1, 1]])
    dilemma = mx.array([3])
    held, *_rest = model.roll(tokens, None, dilemma, None, None)
    faded, *_rest = model.roll(tokens, None, dilemma, None, mx.array([3]))
    mx.eval(held, faded)
    assert mx.allclose(held[:, :3], faded[:, :3], atol=1e-5)
    assert not bool(mx.allclose(held[:, 3], faded[:, 3], atol=1e-5).item())


def test_mlx_dilemma_drive_waits_and_a_bare_sample_adds_none() -> None:
    """The dilemma kicks on its piece only. The next piece is on its own."""
    pytest.importorskip("mlx")
    from mlx import nn
    import mlx.core as mx

    from sidecar.generate import generate
    from sidecar.query import main as query_main
    from sidecar.train_mlx import KuramotoMLX, loss_fn, sample, spin

    assert inspect.signature(sample).parameters["dilemma"].default == -1
    assert inspect.signature(spin).parameters["dilemma"].default == -1
    assert "dilemma" not in inspect.signature(generate).parameters
    assert "dilemma" not in inspect.signature(query_main).parameters

    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.D = mx.ones((model.n, 1))
    model.theta0 = mx.zeros((model.n,))
    phases = mx.zeros((6, 1))
    phases = phases.at[3, 0].add(0.6)
    model.root_phase.weight = phases
    tokens = mx.array([[1, 1, 3, 1]])
    missing = mx.array([-1])
    plain = model(tokens)
    late = model(tokens, missing, mx.array([3]))
    mx.eval(plain, late)
    assert mx.allclose(late[:, :2], plain[:, :2], atol=1e-5)
    assert not bool(mx.allclose(late[:, 2], plain[:, 2], atol=1e-5).item())
    seen = mx.zeros((1,), dtype=mx.bool_)
    seen, hit = model._open_dilemma(seen, mx.array([3]), mx.array([3]))
    _, later = model._open_dilemma(seen, mx.array([1]), mx.array([3]))
    assert int(hit) == 3
    assert int(later) == 3
    compiled = mx.compile(lambda batch: loss_fn(model, batch))
    compiled_loss = compiled(tokens)
    mx.eval(compiled_loss)
    assert math.isfinite(float(compiled_loss))
    _loss, grads = nn.value_and_grad(model, lambda batch: loss_fn(model, batch))(tokens)
    mx.eval(grads)
    assert math.isfinite(float(_loss))


def test_specials_drive_the_slow_wheels() -> None:
    """Character moves a slow wheel. A word wheel keeps the word kick only."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    model = KuramotoMLX(4, n=6, rank=1, n_roots=1, n_slow=2, n_stamp=0, n_arc=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.D = mx.ones((model.n, 1))
    phases = mx.zeros((4, 1))
    phases = phases.at[1, 0].add(0.8)
    model.root_phase.weight = phases
    theta = mx.zeros((1, model.n))
    word = mx.array([0])
    plain = model.velocity(theta, word)
    named = model.velocity(theta, word, character=mx.array([1]))
    mx.eval(plain, named)
    assert mx.allclose(named[0, : model.slow.start], plain[0, : model.slow.start])
    assert not bool(mx.allclose(named[0, model.slow.start], plain[0, model.slow.start]).item())


def test_playback_keeps_sentence_case() -> None:
    """The word is chosen without case. The written piece keeps the capital."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import _case_table, choose_piece

    itos = [" the", " The", " inside", " Inside", "."]
    groups, table = _case_table(itos)
    inside = mx.array([0.0, 0.0, 2.0, 1.0, 0.0])
    after_word = choose_piece(inside, itos, groups, table, previous=" bird", temperature=0)
    after_period = choose_piece(inside, itos, groups, table, previous=".", temperature=0)
    assert itos[after_word] == " inside"
    assert itos[after_period] == " Inside"
    capital = mx.array([1.0, 4.0, 0.0, 0.0, 0.0])
    assert itos[choose_piece(capital, itos, groups, table, previous=".", temperature=0)] == " The"
    assert itos[choose_piece(capital, itos, groups, table, previous=" bird", temperature=0)] == " the"


def _route_logits(width: int, source: int, dest: int, *, torch_bag: bool) -> object:
    """Row-stochastic logits. Row ``source`` sends its mass to ``dest``."""
    if torch_bag:
        logits = torch.zeros(width, width)
        logits[source, dest] = 40.0
        return logits
    import mlx.core as mx

    logits = mx.zeros((width, width))
    return logits.at[source, dest].add(40.0)


def test_catalog_opens_on_the_plain_tug() -> None:
    """Every slot starts as the rest pose and as ``U`` and ``V``.

    The blend of those copies is the plain tug. A drawn pose is not the start.
    """
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    model = KuramotoMLX(6, n=4, rank=2, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=False)
    model.scatter_route()
    pose = mx.random.normal((2, model.n))
    word = mx.array([0, 1], dtype=mx.int32)
    model.use_route = False
    plain = model.velocity(pose, word)
    model.use_route = True
    routed = model.velocity(pose, word)
    mx.eval(plain, routed, model.patterns)
    assert bool(mx.allclose(plain, routed, atol=1e-5).item())
    assert bool(mx.allclose(model.patterns, mx.zeros_like(model.patterns)).item())


def test_write_stores_a_pose_the_wheels_reached() -> None:
    """A kick that leaves rest opens the next slot and counts where it came from.

    A nudge that stays in its slot writes nothing. Landing again from rest
    counts the origin. Adam does not move the written pose.
    """
    pytest.importorskip("mlx")
    import mlx.core as mx
    import mlx.optimizers as optim

    from sidecar.train_mlx import KuramotoMLX, _store_pose, compile_window_step, run_compiled_window

    model = KuramotoMLX(4, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.zeros((model.n, 1))
    model.V = mx.zeros((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.root_phase.weight = mx.zeros((model.vocab_size, 1))
    model.scatter_route()
    rest = mx.array([[0, 1]])
    model.write_catalog(rest)
    assert model.catalog_filled == 1
    assert float(mx.max(mx.abs(model.route_K))) == 0.0

    model.omega = mx.full((model.n,), 0.2)
    model.write_catalog(rest)
    assert model.catalog_filled == 1
    assert float(mx.max(mx.abs(model.route_K))) == 0.0

    model.omega = mx.full((model.n,), 8.0)
    moved = mx.array([[0, 1], [0, 1]])
    model.write_catalog(moved)
    assert model.catalog_filled == 2
    assert float(model.route_K[0, 1]) == 2.0
    assert float(mx.max(mx.abs(model.patterns[0]))) == 0.0
    reached = model.patterns[1]
    model.write_catalog(moved)
    assert model.catalog_filled == 2
    assert math.isclose(float(model.route_K[0, 1]), 0.5 * 2.0 + 2.0, abs_tol=1e-5)
    assert bool(mx.allclose(model.patterns[1], reached).item())
    stayed = mx.broadcast_to(model.patterns[1], (2, model.n))
    _patterns, order, filled, prev = _store_pose(
        model.patterns,
        model.route_K,
        model.catalog_filled,
        mx.ones((2,), dtype=mx.int32),
        stayed,
        mx.ones((2,), dtype=mx.bool_),
    )
    assert filled == model.catalog_filled
    assert float(mx.max(mx.abs(order - model.route_K))) == 0.0
    assert bool(mx.all(prev == 1).item())

    kept = mx.array(model.patterns)
    kept_k = mx.array(model.route_K)
    optimizer = optim.AdamW(learning_rate=1e-3, weight_decay=0.01)
    compiled = compile_window_step(model, optimizer)
    run_compiled_window(
        model,
        compiled,
        optimizer,
        mx.array([[0, 1, 2, 3]]),
        mx.array([-1]),
        mx.array([-1]),
        mx.array([-1]),
        mx.array([4]),
        mx.zeros((1, model.n)),
        mx.zeros((1, 1)),
        mx.ones((1,), dtype=mx.bool_),
        mx.zeros((1,), dtype=mx.bool_),
    )
    mx.eval(model.parameters())
    assert bool(mx.allclose(model.patterns, kept).item())
    assert bool(mx.allclose(model.route_K, kept_k).item())


def test_k_transpose_r_changes_a_fixed_pose() -> None:
    """The pose stays put. Changing ``K`` changes the velocity.

    One ``K`` sends the recognized pattern to one stored phase, and another
    ``K`` sends it to a different phase. With the route absent, that term
    adds nothing. A period is still the only time the stamp wheel steps.
    """
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    model = KuramotoMLX(
        4, n=4, rank=1, n_roots=1, n_slow=1, n_stamp=1, n_arc=0, latch=False, boundary_ids=(1,)
    )
    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.root_phase.weight = mx.zeros((model.vocab_size, 1))
    pose = mx.zeros((1, model.n)).at[0, 0].add(math.pi / 2)
    word = mx.zeros((1,), dtype=mx.int32)
    model.use_route = False
    absent = model.velocity(pose, word)
    patterns = mx.zeros((model.patterns.shape[0], model.n))
    patterns = patterns.at[0, 0].add(math.pi / 2)
    patterns = patterns.at[1].add(-math.pi / 2)
    model.patterns = patterns
    remembered = mx.zeros((1, patterns.shape[0])).at[0, 0].add(1.0)
    model.mem_V = mx.ones_like(model.mem_V)
    model.mem_U = mx.zeros_like(model.mem_U)
    model.mem_U = model.mem_U.at[0].add(1.0)
    model.mem_U = model.mem_U.at[1].add(-1.0)
    model.use_route = True
    model.route_K = _route_logits(patterns.shape[0], 0, 0, torch_bag=False)
    to_first = model.velocity(pose, word, recognition=remembered)
    mx.eval(to_first)
    model.route_K = _route_logits(patterns.shape[0], 0, 1, torch_bag=False)
    to_second = model.velocity(pose, word, recognition=remembered)
    mx.eval(to_second)
    model.use_route = False
    cleared = model.velocity(pose, word)
    ordinary = model.step_scale(word)
    ended = model.step_scale(mx.ones((1,), dtype=mx.int32))
    mx.eval(absent, cleared, ordinary, ended)
    # ``K`` picks which stored coupling tugs. The plain tug returns when the route is off.
    assert not bool(mx.allclose(to_first, to_second).item())
    assert bool(mx.allclose(absent, cleared).item())
    assert math.isclose(float(ordinary[0, model.stamp.start]), 0.0, abs_tol=1e-5)
    assert math.isclose(float(ended[0, model.stamp.start]), model.dt, abs_tol=1e-5)

    torch_bag = KuramotoWordLM(
        4,
        n_oscillators=4,
        rank=1,
        n_roots=1,
        n_slow=1,
        n_stamp=1,
        n_arc=0,
        latch=False,
        boundary_ids=(1,),
    )
    with torch.no_grad():
        torch_bag.omega.zero_()
        torch_bag.U.fill_(1.0)
        torch_bag.V.fill_(1.0)
        torch_bag.D.zero_()
        torch_bag.root_phase.weight.zero_()
        torch_bag.use_route = False
        tpose = torch.zeros(1, 4)
        tpose[0, 0] = math.pi / 2
        token = torch.zeros(1, dtype=torch.long)
        torch_absent = torch_bag.velocity(tpose, token)
        phases = torch.zeros_like(torch_bag.patterns)
        phases[0, 0] = math.pi / 2
        phases[1] = -math.pi / 2
        torch_bag.patterns.copy_(phases)
        torch_bag.mem_V.fill_(1.0)
        torch_bag.mem_U.zero_()
        torch_bag.mem_U[0].fill_(1.0)
        torch_bag.mem_U[1].fill_(-1.0)
        torch_bag.use_route = True
        remembered_t = torch.zeros(1, phases.shape[0])
        remembered_t[0, 0] = 1.0
        torch_bag.route_K.copy_(_route_logits(phases.shape[0], 0, 0, torch_bag=True))
        torch_first = torch_bag.velocity(tpose, token, recognition=remembered_t)
        torch_bag.route_K.copy_(_route_logits(phases.shape[0], 0, 1, torch_bag=True))
        torch_second = torch_bag.velocity(tpose, token, recognition=remembered_t)
    assert torch.allclose(torch_absent, torch.tensor(absent.tolist()), atol=1e-5)
    assert torch.allclose(torch_first, torch.tensor(to_first.tolist()), atol=1e-5)
    assert torch.allclose(torch_second, torch.tensor(to_second.tolist()), atol=1e-5)


def test_recognition_lags_the_pose_just_left() -> None:
    """The kick uses the overlap of the pose already left, not the one it is in."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    model = KuramotoMLX(4, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.zeros((model.n, 1))
    model.V = mx.zeros((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.root_phase.weight = mx.zeros((model.vocab_size, 1))
    patterns = mx.zeros((model.patterns.shape[0], model.n))
    patterns = patterns.at[0, 0].add(math.pi / 2)
    patterns = patterns.at[1, 0].add(1.0)
    model.patterns = patterns
    model.mem_V = mx.ones_like(model.mem_V)
    model.mem_U = mx.zeros_like(model.mem_U)
    model.mem_U = model.mem_U.at[1].add(1.0)
    model.use_route = True
    model.route_K = _route_logits(patterns.shape[0], 0, 1, torch_bag=False)
    pose = mx.zeros((1, model.n)).at[0, 0].add(math.pi / 2)
    word = mx.zeros((1,), dtype=mx.int32)
    fresh = model.velocity(pose, word)
    lagged = model._overlap(pose)
    aimed = model.velocity(pose, word, recognition=lagged)
    mx.eval(fresh, aimed, lagged)
    assert int(mx.argmax(lagged).item()) == 0
    assert not bool(mx.allclose(fresh, aimed).item())


def test_aim_nudges_toward_the_next_pose_without_landing() -> None:
    """One kick moves toward the catalog pose and does not arrive there."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    model = KuramotoMLX(4, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.zeros((model.n, 1))
    model.V = mx.zeros((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.root_phase.weight = mx.zeros((model.vocab_size, 1))
    target = 1.0
    patterns = mx.zeros((model.patterns.shape[0], model.n))
    patterns = patterns.at[1, 0].add(target)
    model.patterns = patterns
    model.use_route = True
    model.route_K = _route_logits(patterns.shape[0], 0, 1, torch_bag=False)
    remembered = mx.zeros((1, patterns.shape[0])).at[0, 0].add(1.0)
    pose = mx.zeros((1, model.n))
    word = mx.zeros((1,), dtype=mx.int32)
    stepped = pose + model.dt * model.velocity(pose, word, recognition=remembered)
    mx.eval(stepped)
    moved = float(stepped[0, 0])
    assert abs(moved) < 1e-5
    assert moved < target - 0.2


def test_each_stored_pose_has_its_own_coupling() -> None:
    """The successor picks which stored coupling tugs, with the aim held at zero."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX

    model = KuramotoMLX(4, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.zeros((model.n, 1))
    model.V = mx.zeros((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.root_phase.weight = mx.zeros((model.vocab_size, 1))
    model.patterns = mx.zeros_like(model.patterns)
    model.mem_V = mx.ones_like(model.mem_V)
    model.mem_U = mx.zeros_like(model.mem_U)
    model.mem_U = model.mem_U.at[0].add(1.0)
    model.use_route = True
    remembered = mx.zeros((1, model.patterns.shape[0])).at[0, 0].add(1.0)
    pose = mx.zeros((1, model.n)).at[0, 0].add(math.pi / 2)
    word = mx.zeros((1,), dtype=mx.int32)
    model.route_K = _route_logits(model.patterns.shape[0], 0, 0, torch_bag=False)
    through_first = model.velocity(pose, word, recognition=remembered)
    mx.eval(through_first)
    model.route_K = _route_logits(model.patterns.shape[0], 0, 1, torch_bag=False)
    through_second = model.velocity(pose, word, recognition=remembered)
    mx.eval(through_second)
    assert not bool(mx.allclose(through_first, through_second).item())


def test_an_old_bag_without_k_adds_nothing(tmp_path: Path) -> None:
    """A checkpoint with no ``K`` still opens, and the route term stays off."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import KuramotoMLX, load_bag_weights

    model = KuramotoMLX(4, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=False)
    model.omega = mx.zeros((model.n,))
    model.U = mx.ones((model.n, 1))
    model.V = mx.ones((model.n, 1))
    model.D = mx.zeros((model.n, 1))
    model.root_phase.weight = mx.zeros((model.vocab_size, 1))
    path = tmp_path / "old_bag.safetensors"
    model.save_weights(str(path))
    saved = dict(mx.load(str(path)))
    del saved["route_K"]
    del saved["patterns"]
    mx.save_safetensors(str(path), saved)
    fresh = KuramotoMLX(4, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, n_arc=0, latch=False)
    fresh.patterns = mx.ones_like(fresh.patterns)
    fresh.route_K = mx.ones_like(fresh.route_K)
    load_bag_weights(fresh, path)
    assert fresh.use_route is False
    pose = mx.zeros((1, fresh.n)).at[0, 0].add(math.pi / 2)
    word = mx.zeros((1,), dtype=mx.int32)
    loaded = fresh.velocity(pose, word)
    fresh.use_route = True
    fresh.patterns = mx.zeros_like(fresh.patterns).at[0, 0].add(math.pi / 2)
    fresh.route_K = _route_logits(fresh.patterns.shape[0], 0, 0, torch_bag=False)
    routed = fresh.velocity(pose, word)
    fresh.use_route = False
    again = fresh.velocity(pose, word)
    mx.eval(loaded, routed, again)
    # An empty catalog coupling is not the plain tug. Turning the route back off restores it.
    assert not bool(mx.allclose(loaded, routed).item())
    assert bool(mx.allclose(loaded, again).item())


def test_repetition_penalty_follows_english_weights() -> None:
    """A used content word and a blank line are cut. Glue and marks stay."""
    pytest.importorskip("mlx")
    import mlx.core as mx

    from sidecar.train_mlx import apply_repetition_penalty

    itos = [" bird", ".", ",", " and", "\n\n"]
    logits = mx.array([4.0, 4.0, -2.0, 4.0, 4.0])
    out = apply_repetition_penalty(logits, [0, 1, 2, 3, 4], 2.0, itos)
    mx.eval(out)
    assert float(out[0]) == pytest.approx(2.0)
    assert float(out[1]) == pytest.approx(4.0)
    assert float(out[2]) == pytest.approx(-2.0)
    assert float(out[3]) == pytest.approx(4.0)
    assert float(out[4]) == pytest.approx(2.0)
