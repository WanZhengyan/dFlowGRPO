"""
ScienceQA multimodal understanding dataset for GRPO.

Expected directory layout (already present at ``dataset_understanding/ScienceQA``):
    ScienceQA/
      data/scienceqa/problems.json       # dict: qid -> {question, choices, answer, image, split, ...}
      train/<qid>/image.png              # train split images
      val/<qid>/image.png                # val split images
      test/<qid>/image.png               # test split images

``__getitem__`` returns a dict with:
    {
        "qid": str,
        "prompt": str,       # formatted question + options text
        "image": PIL.Image,  # 384x384 RGB (or None if question has no image)
        "answer_letter": str,   # e.g. "A", "B", ...
        "answer_text": str,     # full text of the correct choice
        "choices_letters": list[str],   # e.g. ["A","B","C"]
        "choices": list[str],           # full choice texts
        "split": str,
    }
"""

import json
import os
import string
from typing import Optional

from PIL import Image
from torch.utils.data import Dataset


_LETTERS = list(string.ascii_uppercase)


def _resolve_image_path(root, qid, split, img_field):
    if img_field is None:
        return None
    for cand_split in (split, "train", "val", "test"):
        p = os.path.join(root, cand_split, qid, img_field)
        if os.path.isfile(p):
            return p
    return None


def _format_prompt(question, choices, hint=None):
    """Build a MCQ-style prompt similar to VLMEvalKit.build_prompt."""
    opt_lines = []
    for i, c in enumerate(choices):
        opt_lines.append(f"{_LETTERS[i]}. {c}")
    parts = []
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {question}")
    if choices:
        parts.append("Options:\n" + "\n".join(opt_lines))
        parts.append("Please select the correct answer from the options above. "
                     "Answer with the letter only.")
    return "\n".join(parts)


class ScienceQADataset(Dataset):
    """ScienceQA MCQ dataset for understanding GRPO."""

    def __init__(self, dataset_root: str, split: str = "train",
                 require_image: bool = True,
                 problems_json: Optional[str] = None,
                 max_prompt_chars: Optional[int] = 1800):
        """
        Args:
            dataset_root: path to ``.../dataset_understanding/ScienceQA``
            split: one of "train", "val", "test".
            require_image: if True, drop questions without an image.
            max_prompt_chars: if not None, drop items whose formatted prompt
                exceeds this many characters. A rough tokenizer-free filter to
                keep prompts under ``txt_max_length`` after encoding.
                ~1800 chars ≈ 450 tokens at ~4 chars/token.
        """
        self.root = dataset_root
        self.split = split
        problems_json = problems_json or os.path.join(
            dataset_root, "data", "scienceqa", "problems.json")
        with open(problems_json, "r") as f:
            problems = json.load(f)

        self.items = []
        for qid, prob in problems.items():
            if prob.get("split") != split:
                continue
            img_field = prob.get("image")
            if require_image and img_field is None:
                continue
            img_path = _resolve_image_path(self.root, qid, split, img_field)
            if require_image and img_path is None:
                continue
            choices = prob.get("choices", []) or []
            ans_idx = prob.get("answer", -1)
            if not (0 <= ans_idx < len(choices)):
                continue
            prompt = _format_prompt(prob["question"], choices,
                                    hint=prob.get("hint") or None)
            if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
                continue
            self.items.append({
                "qid": qid,
                "image_path": img_path,
                "prompt": prompt,
                "answer_letter": _LETTERS[ans_idx],
                "answer_text": choices[ans_idx],
                "choices_letters": _LETTERS[:len(choices)],
                "choices": choices,
                "split": split,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = Image.open(it["image_path"]).convert("RGB") if it["image_path"] else None
        return {
            "qid": it["qid"],
            "prompt": it["prompt"],
            "image": img,
            "answer_letter": it["answer_letter"],
            "answer_text": it["answer_text"],
            "choices_letters": it["choices_letters"],
            "choices": it["choices"],
            "split": it["split"],
        }

    @staticmethod
    def collate_fn(examples):
        """Return a list of dicts (no tensor stacking — handled downstream)."""
        return examples
