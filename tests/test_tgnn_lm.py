# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""The graph keeps a content word and does not store an article."""

import torch

from sidecar.beats import pack_beats, story_beats
from sidecar.prepare import EOS
from sidecar.tgnn_lm import StoryGraphLM, node_canon_ids, node_skip_ids, pack_stories


def test_articles_are_not_nodes_and_a_word_stays() -> None:
    """``the`` adds nothing. ``tree`` is still stored after later pieces."""
    itos = [EOS, " the", " tree", " bird", "."]
    skip = node_skip_ids(itos)
    assert 1 in skip
    assert 4 in skip
    assert 2 not in skip
    glue = node_skip_ids([EOS, " and", " to", " she", " tree"])
    assert glue == {0, 1, 2}
    model = StoryGraphLM(len(itos), dim=8, skip_ids=skip)
    state_loss = model.story_loss([2, 1, 3, 4, 0])
    state_loss.backward()
    assert torch.isfinite(state_loss)
    model.zero_grad()
    with torch.no_grad():
        ids, state = model.generate(
            [2], length=1, temperature=0.0, repetition_penalty=1.0, eos_id=0
        )
    assert 2 in state.index
    assert 1 not in state.index
    assert ids[0] == 2


def test_capital_and_lower_are_one_node() -> None:
    """``She`` and ``she`` share a node. The storyline is part of the loss."""
    itos = [EOS, " She", " she", " girl"]
    canon = node_canon_ids(itos)
    assert canon[1] == 2
    model = StoryGraphLM(len(itos), dim=8, skip_ids=node_skip_ids(itos), canon_ids=canon)
    loss = model.story_loss([1, 2, 3])
    loss.backward()
    assert torch.isfinite(loss)
    assert model.storyline.weight_ih.grad is not None
    assert float(model.storyline.weight_ih.grad.abs().sum()) > 0
    with torch.no_grad():
        _, state = model.generate(
            [1, 2], length=1, temperature=0.0, repetition_penalty=1.0, eos_id=0
        )
    assert state.ids.count(2) == 1
    assert 1 not in state.ids


def test_reader_names_the_needle_and_not_the_first_sentence_only() -> None:
    """Lily is the character. Needle, said after the opening, is the dilemma."""
    itos = [EOS, ".", " Lily", " needle", " sharp", " mom", " happy", " shared"]
    story = [4, 2, 3, 1, 2, 3, 3, 1, 6, 7, 1, 0]
    beat = story_beats(story, itos, {0, 1}, list(range(len(itos))), {1}, [0.0] * len(itos), 8)
    assert beat is not None
    assert itos[beat.nodes[beat.character_row]].strip() == "Lily"
    assert itos[beat.nodes[beat.dilemma_row]].strip() == "needle"
    model = StoryGraphLM(len(itos), dim=8, skip_ids={0, 1}, boundary_ids=(1,))
    packed = pack_beats([story], itos, {0, 1}, list(range(len(itos))), {1}, [0.0] * len(itos), 8)
    loss = model.beat_loss(packed)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.character_head.weight.grad is not None
    assert float(model.character_head.weight.grad.abs().sum()) > 0


def test_a_repeated_common_word_loses_to_a_rare_capitalized_name() -> None:
    """Capitalization picks the name. Rarity picks the dilemma. No word list."""
    itos = [EOS, ".", " Lily", " she", " needle", " bird"]
    story = [3, 3, 2, 4, 1, 3, 4, 1, 5, 1, 0]
    document_counts = [0.0, 0.0, 1.0, 8.0, 0.0, 0.0]
    beat = story_beats(story, itos, {0, 1}, list(range(len(itos))), {1}, document_counts, 8)
    assert beat is not None
    assert itos[beat.nodes[beat.character_row]].strip() == "Lily"
    assert itos[beat.nodes[beat.dilemma_row]].strip() == "needle"


def test_batch_loss_trains() -> None:
    """Several stories share one score step and the loss stays finite."""
    model = StoryGraphLM(6, dim=8, skip_ids={1})
    loss, count = model.batch_loss([[2, 3, 4, 0], [3, 2, 5, 4, 0]])
    loss.backward()
    assert count == 7
    assert torch.isfinite(loss)


def test_windows_carry_the_same_scores() -> None:
    """Splitting a story keeps the same next-piece scores. The nodes carry over."""
    torch.manual_seed(0)
    model = StoryGraphLM(8, dim=8, skip_ids={1, 0})
    stories = [[2, 1, 2, 3, 4, 5, 2, 6, 3], [3, 3, 1, 5, 2, 6, 4, 2]]
    tokens, lengths, budget, count = pack_stories(stories, model.skip_ids)
    whole = model.loss_on_tokens(tokens, lengths, budget=budget, count=count, window=100)
    windowed = model.loss_on_tokens(tokens, lengths, budget=budget, count=count, window=3)
    assert torch.allclose(whole, windowed, atol=1e-5, rtol=1e-5)


def test_batched_step_matches_one_story_at_a_time() -> None:
    """The fan-out step scores the same loss and gradient as the story walk."""
    torch.manual_seed(0)
    model = StoryGraphLM(8, dim=8, skip_ids={1, 0})
    stories = [[2, 1, 2, 3, 4], [3, 3, 1, 5, 2, 6], [4, 2, 5]]
    batched, count = model.batch_loss(stories)
    batched.backward()
    batched_grads = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }
    model.zero_grad(set_to_none=True)
    total = torch.zeros(())
    for story in stories:
        total = total + model.story_loss(story) * (len(story) - 1)
    (total / count).backward()
    walked = {
        name: param.grad for name, param in model.named_parameters() if param.grad is not None
    }
    assert count == sum(len(story) - 1 for story in stories)
    assert torch.allclose(batched.detach(), (total / count).detach(), atol=1e-5, rtol=1e-5)
    assert batched_grads.keys() == walked.keys()
    for name, left in batched_grads.items():
        assert torch.allclose(left, walked[name], atol=1e-5, rtol=1e-5)
