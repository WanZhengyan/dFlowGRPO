"""
Shared constants and configuration for discrete flow GRPO training and evaluation.

All machine-specific paths should be passed via CLI arguments or shell scripts.
This module only contains model/architecture constants that are intrinsic to FUDOKI.
"""

# FUDOKI model constants
VOCABULARY_SIZE_TXT = 102400
VOCABULARY_SIZE_IMG = 16384
IMG_LEN = 576
CFG_SCALE = 5.0  # default CFG scale for inference

# GenEval tag categories
GENEVAL_TAGS = [
    "single_object", "two_object", "counting",
    "colors", "color_attr", "position",
]
