"""
HPSv3 (Human Preference Score v3) scorer.

HPSv3 is a Qwen2-VL-based reward model that outputs a Gaussian reward
distribution (mu, sigma); we use `mu` as the scalar preference score.

Install:
    pip install hpsv3

Weights (MizzenAI/HPSv3) are auto-downloaded from HuggingFace on first use.
"""
from __future__ import annotations

import os
import tempfile
from typing import List

import torch
from PIL import Image


class HPSv3Scorer(torch.nn.Module):
    def __init__(self, device: str = "cuda", dtype=torch.float32,
                 micro_batch_size: int | None = None):
        super().__init__()
        self.device = device
        self.dtype = dtype
        # Per-forward cap on (image, prompt) pairs. HPSv3 uses Qwen2-VL which
        # packs all images in a batch into one long sequence; without
        # FlashAttention, SDPA falls back to dense O(N^2) attention and can
        # OOM with 256 GiB allocations on H100 80GB for batches >=~16.
        # Env override: HPSV3_MICRO_BATCH (default 4 when flash_attn is
        # unavailable, 16 otherwise).
        if micro_batch_size is None:
            env = os.environ.get("HPSV3_MICRO_BATCH")
            if env:
                micro_batch_size = int(env)
            else:
                try:
                    import flash_attn  # noqa: F401
                    micro_batch_size = 16
                except Exception:
                    micro_batch_size = 4
        self.micro_batch_size = max(1, int(micro_batch_size))

        try:
            from hpsv3 import HPSv3RewardInferencer
        except ImportError as e:
            raise ImportError(
                "HPSv3 requires the `hpsv3` package. "
                "Install with `pip install hpsv3`."
            ) from e

        # HPSv3 internally instantiates `transformers.TrainingArguments`
        # (via its TrainingConfig). `TrainingArguments.__post_init__` calls
        # `AcceleratorState._reset_state(reset_partial_state=True)`, which
        # invalidates any `Accelerator` already created by the caller. We
        # snapshot / rebuild the accelerate state around the HPSv3 init so
        # subsequent uses of the outer `Accelerator` keep working.
        try:
            from accelerate.state import AcceleratorState, PartialState
        except Exception:
            AcceleratorState = PartialState = None

        had_state = False
        if AcceleratorState is not None:
            try:
                _ = AcceleratorState().num_processes
                had_state = True
            except Exception:
                had_state = False

        self._inferencer = HPSv3RewardInferencer(device=device)

        if had_state and PartialState is not None:
            # HPSv3's TrainingArguments may have reset AcceleratorState.
            # Re-build the singletons so the outer Accelerator keeps working.
            try:
                _ = AcceleratorState().num_processes
            except Exception:
                try:
                    PartialState()
                    AcceleratorState()
                except Exception:
                    pass

    @torch.no_grad()
    def __call__(self, prompts: List[str], images: List[Image.Image]) -> List[float]:
        # HPSv3's public API expects image paths. Dump PILs to a temp dir.
        scores: List[float] = []
        with tempfile.TemporaryDirectory() as tmpdir:
            paths: List[str] = []
            for i, img in enumerate(images):
                p = os.path.join(tmpdir, f"img_{i:04d}.png")
                img.save(p)
                paths.append(p)

            # HPSv3's HPSv3RewardInferencer.reward(paths, prompts) returns a
            # tensor of shape [N, 2] where [:, 0] = mu (the score we want) and
            # [:, 1] = sigma. Official usage: `rewards[i][0].item()`.
            # Chunk to avoid OOM in Qwen2-VL SDPA attention (quadratic in
            # packed sequence length) when flash_attn is unavailable.
            prompts_list = list(prompts)
            mb = self.micro_batch_size
            for start in range(0, len(paths), mb):
                end = start + mb
                chunk_paths = paths[start:end]
                chunk_prompts = prompts_list[start:end]
                chunk_res = self._inferencer.reward(chunk_paths, chunk_prompts)

                # Normalise to a list of per-sample mu scalars.
                if isinstance(chunk_res, torch.Tensor):
                    # Shape [N, 2] -> take mu column. Also handles [N] just in case.
                    if chunk_res.ndim == 2:
                        mus = chunk_res[:, 0].detach().float().cpu().tolist()
                    else:
                        mus = chunk_res.detach().float().cpu().tolist()
                    scores.extend(float(m) for m in mus)
                else:
                    # Fallback for list/tuple/dict variants across hpsv3 versions.
                    for r in chunk_res:
                        if isinstance(r, torch.Tensor):
                            mu = float(r.flatten()[0].item())
                        elif isinstance(r, (list, tuple)):
                            mu = float(r[0])
                        elif isinstance(r, dict):
                            mu = float(r.get("mu", r.get("reward", 0.0)))
                        else:
                            mu = float(r)
                        scores.append(mu)

                # Free any cached activations between chunks.
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        return scores


def hpsv3_score(device):
    """Factory matching the signature of other reward functions in flow_grpo.rewards."""
    scorer = HPSv3Scorer(device=device, dtype=torch.float32)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(im) for im in images]
        scores = scorer(list(prompts), list(images))
        return scores, {}

    return _fn
