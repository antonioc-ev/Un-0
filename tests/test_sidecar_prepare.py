# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Sentence boundaries stay intact when stories are packed."""

from sidecar.prepare import EOS, atoms, build_vocab, document_ids, pieces


def test_pieces_roundtrip_and_keep_the_period() -> None:
    """A period is its own piece, and joining the pieces restores the text."""
    text = 'Once upon a time, Lily said, "Let\'s play!" They ran outside.'
    packed = pieces(text, 4)
    assert "".join(packed) == text
    assert "." in packed
    assert "!" in packed or '!"' in packed
    assert not any("." in piece and any(ch.isalpha() for ch in piece) for piece in packed)


def test_long_word_is_cut_without_swallowing_punctuation() -> None:
    """The old grid stays inside the word."""
    assert pieces("outside.", 4) == ["outs", "ide", "."]


def test_whole_word_kick_keeps_the_word() -> None:
    """Chunk 0 is one kick per word, and the period stays its own kick."""
    assert pieces("outside.", 0) == ["outside", "."]
    assert pieces("Once upon a time, Lily.", 0) == [
        "Once",
        " upon",
        " a",
        " time",
        ",",
        " Lily",
        ".",
    ]


def test_document_ends_with_the_end_piece() -> None:
    """The end id is one piece after the story, not characters of its name."""
    text = "Tom had a red ball."
    _, stoi = build_vocab([text], chunk=4, min_count=1)
    ids = document_ids(text, chunk=4, stoi=stoi)
    assert ids[-1] == stoi[EOS]
    assert EOS not in pieces(text, 4)
