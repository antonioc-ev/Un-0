# © 2024–2026 Evolver Transformation, Inc. All rights reserved.
# This software and all associated documentation (the "Software") are the exclusive, confidential, and proprietary property of Evolver Transformation, Inc.
# Any unauthorized use—including but not limited to copying, distribution, modification, reverse engineering, or repurposing of this Software, in whole or in part—is a direct violation of intellectual property law and will be pursued to the fullest extent permitted by law.
# This restriction applies unequivocally to all individuals and entities, including former employees, contractors, partners, and affiliates.
# Continued possession or use of this Software after termination of any relationship with Evolver Transformation, Inc. is strictly forbidden and may result in legal action.

"""Laptop-sized Kuramoto language sidecar.

The image model stays in ``un0``. This package is a separate, small next-piece
model: a low-rank oscillator bag driven by story text. A kick is a whole word.
The path from characters, through the 4-character grid, to that word is
written in ``sidecar.prepare``.
"""
