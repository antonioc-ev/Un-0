# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""A later sentence adds nodes. It does not delete the open want."""

from sidecar.story_tgnn import StoryTGNN, want_piece_ids

_STORY = (
    "Once upon a time, there was a little girl named Lily. "
    "She loved to play with her toys and she would go to the park. "
    "One day, she saw a big tree in the middle of the forest. "
    "One day, the bird was very fast and wanted to find it. "
    "The dog was so happy. The cat started to shake the nut."
)


def test_tree_edge_survives_the_bird() -> None:
    """Lily seeing the tree is still an edge after the bird, dog, and cat arrive."""
    state = StoryTGNN().read(_STORY)
    want = state.want()
    assert want is not None
    assert {want.src, want.dst} == {"lily", "tree"}
    assert "bird" in state.index
    assert "dog" in state.index
    assert "cat" in state.index
    bird_edges = [edge for edge in state.edges if "bird" in {edge.src, edge.dst}]
    assert bird_edges
    assert all(edge.at > want.at for edge in bird_edges)


def test_open_want_names_the_vocab_pieces() -> None:
    """The tree edge points at the Lily and tree rows, not at the bird."""
    state = StoryTGNN().read(_STORY)
    itos = ["<|endoftext|>", " Lily", " tree", " bird", " she"]
    picks = want_piece_ids(itos, state)
    assert picks == [1, 2]
