"""
Checkpoint discovery utilities for batch evaluation.
"""

import os
import re
import glob


def extract_step(name):
    """Extract step number from a checkpoint directory name."""
    m = re.search(r"step(\d+)", name)
    return int(m.group(1)) if m else None


def discover_checkpoints(grpo_dir, step_interval=None,
                         step_start=None, step_end=None):
    """
    Discover GRPO checkpoints in a training output directory.

    Filters (all optional, applied conjunctively):
      - step >= step_start
      - step <= step_end
      - (step - (step_start or 0)) % step_interval == 0

    Returns: [(name, path, step)] sorted by step number.
    """
    dirs = set()
    for pat in ["checkpoint_*epoch*_step*", "checkpoint_*step*"]:
        dirs.update(glob.glob(os.path.join(grpo_dir, pat)))
    base = step_start if step_start is not None else 0
    out = []
    for d in dirs:
        nm = os.path.basename(d)
        st = extract_step(nm)
        if st is None:
            continue
        if step_start is not None and st < step_start:
            continue
        if step_end is not None and st > step_end:
            continue
        if step_interval and ((st - base) % step_interval) != 0:
            continue
        if not os.path.exists(os.path.join(d, "lora_adapter")):
            continue
        out.append((nm, d, st))
    out.sort(key=lambda x: x[2])
    return out
