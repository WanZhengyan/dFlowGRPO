"""
Dataset classes for GRPO training prompts.

Supports:
  - TextPromptDataset: simple text file, one prompt per line (e.g. pickscore)
  - GenevalPromptDataset: JSONL with per-sample metadata (GenEval benchmark)
"""

import json
import os
from torch.utils.data import Dataset


class TextPromptDataset(Dataset):
    """Plain text prompt dataset — one prompt per line."""

    def __init__(self, dataset_dir, split="train"):
        fpath = os.path.join(dataset_dir, f"{split}.txt")
        with open(fpath) as f:
            self.prompts = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx]}

    @staticmethod
    def collate_fn(examples):
        return [ex["prompt"] for ex in examples]


class GenevalPromptDataset(Dataset):
    """GenEval JSONL dataset with per-sample metadata (tag, objects, etc.)."""

    def __init__(self, dataset_dir, split="train"):
        fpath = os.path.join(dataset_dir, f"{split}_metadata.jsonl")
        with open(fpath, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas
