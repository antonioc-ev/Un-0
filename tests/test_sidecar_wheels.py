# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Two clocks share the last wheels. One creeps every word. One moves at a period."""

import inspect
import math

import pytest
import torch

from sidecar.beats import StoryArc, arc_score
from sidecar.model import KuramotoWordLM, clock_ranges
from sidecar.prepare import EOS
from sidecar.train import (
    fit_story,
    label_shelf,
    next_piece_loss,
    prefetch_batches,
    story_epoch_steps,
    story_target_mask,
)


def test_clock_ranges_split_the_end_of_the_bag() -> None:
    """256 wheels with 8 and 8 are 240–247 and 248–255."""
    slow, stamp = clock_ranges(256, 8, 8)
    assert list(slow) == list(range(240, 248))
    assert list(stamp) == list(range(248, 256))
    assert clock_ranges(8, 0, 0) == (range(8, 8), range(8, 8))


def test_stamp_clock_moves_only_at_a_sentence_end() -> None:
    """A period gives the stamp wheel a full step. Any other piece leaves it still."""
    model = KuramotoWordLM(
        5,
        n_oscillators=4,
        rank=1,
        n_roots=1,
        n_slow=1,
        n_stamp=1,
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
        model.U.zero_()
        model.V.zero_()
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


def test_mlx_end_piece_does_not_rest_the_wheels() -> None:
    """The Metal bag keeps spinning through the end piece. Every target is scored."""
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
    four = math.sin(0.8)
    five = math.sin(1.0)
    assert math.isclose(float(logits[0, 2, 0]), three, abs_tol=1e-5)
    assert math.isclose(float(logits[0, 3, 0]), four, abs_tol=1e-5)
    assert math.isclose(float(logits[0, 4, 0]), five, abs_tol=1e-5)
    assert math.isclose(float(logits[1, 2, 0]), three, abs_tol=1e-5)
    assert math.isclose(float(logits[1, 4, 0]), five, abs_tol=1e-5)
    assert math.isclose(float(period_logits[0, -1, 0]), three, abs_tol=1e-5)

    scored = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=True, eos_id=eos)
    tokens = mx.array([[1, 2, 3, eos, eos, eos]])
    other = mx.array([[1, 2, 3, eos, 5, 0]])
    mx.eval(scored.parameters())
    assert not math.isclose(float(loss_fn(scored, tokens)), float(loss_fn(scored, other)), abs_tol=1e-5)
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
        model.U.zero_()
        model.V.zero_()
        model.D.fill_(0.7)
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
    """The full story names the tail word. The cut row never turns that drive on."""
    itos = [EOS, ".", " bird", " Lily", " pond", " needle"]
    story = [2, 3, 1, 3, 3, 5, 5, 3, 1, 4, 1, 0]
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


def test_arc_matches_whole_pieces_in_the_ending() -> None:
    """The ending names a category only when that whole word is in the last sentence."""
    itos = [EOS, ".", " Fin", " crab", " happily", " ever", " after", " kayak", " the", " FIN"]
    assert arc_score(2, 3, (2, 3, 1), itos) == StoryArc("Fin", "crab", "both")
    assert arc_score(2, 7, (9, 1), itos) == StoryArc("Fin", "kayak", "who")
    assert arc_score(2, 3, (8, 3, 1), itos) == StoryArc("Fin", "crab", "trouble")
    assert arc_score(2, 3, (4, 5, 6, 1), itos) == StoryArc("Fin", "crab", "neither")
    assert arc_score(-1, 3, (3, 1), itos) == StoryArc("", "crab", "neither")
    assert arc_score(2, -1, (2, 1), itos) == StoryArc("Fin", "", "neither")


def test_mlx_generation_takes_no_dilemma_and_the_step_stays_finite() -> None:
    """Sampling has no dilemma argument. A plain window still trains."""
    pytest.importorskip("mlx")
    from mlx import nn
    import mlx.core as mx

    from sidecar.generate import generate
    from sidecar.query import main as query_main
    from sidecar.train_mlx import KuramotoMLX, loss_fn, sample, spin

    assert "dilemma" not in inspect.signature(sample).parameters
    assert "dilemma" not in inspect.signature(spin).parameters
    assert "dilemma" not in inspect.signature(generate).parameters
    assert "dilemma" not in inspect.signature(query_main).parameters

    model = KuramotoMLX(6, n=4, rank=1, n_roots=1, n_slow=0, n_stamp=0, latch=False)
    tokens = mx.array([[1, 1, 3, 1]])
    compiled = mx.compile(lambda batch: loss_fn(model, batch))
    compiled_loss = compiled(tokens)
    mx.eval(compiled_loss)
    assert math.isfinite(float(compiled_loss))
    _loss, grads = nn.value_and_grad(model, lambda batch: loss_fn(model, batch))(tokens)
    mx.eval(grads)
    assert math.isfinite(float(_loss))
