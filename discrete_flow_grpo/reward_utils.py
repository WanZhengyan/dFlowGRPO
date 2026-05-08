"""
Reward function construction for GRPO training and GenEval evaluation.

Training rewards:
  build_training_reward_fn() — flexible multi-reward via dict, used during training.

Evaluation rewards:
  build_geneval_reward_fn() — HTTP-based GenEval reward server client.
"""

import importlib
import pickle
import torch
import numpy as np
import requests
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
from PIL import Image
from collections import defaultdict


# ---------------------------------------------------------------------------
# Training reward function (multi-reward, local or GPU-based)
# ---------------------------------------------------------------------------

# Registry: (module_path, function_name, needs_device)
_REWARD_REGISTRY = {
    "aesthetic":     ("flow_grpo.aesthetic_scorer", "AestheticScorer", True),
    "clip":          ("flow_grpo.clip_scorer",      "ClipScorer",      True),
    "pickscore":     ("flow_grpo.rewards",          "pickscore_score", True),
    "imagereward":   ("flow_grpo.rewards",          "imagereward_score", True),
    "jpeg":          ("flow_grpo.rewards",          "jpeg_compressibility", False),
    "deqa":          ("flow_grpo.rewards",          "deqa_score_remote", True),
    "unifiedreward": ("flow_grpo.rewards",          "unifiedreward_score_sglang", True),
    "qwenvl":        ("flow_grpo.rewards",          "qwenvl_score",    True),
    "geneval":       ("flow_grpo.rewards",          "geneval_score",   True),
    "hpsv3":         ("flow_grpo.hpsv3_scorer",     "hpsv3_score",     True),
}


def build_training_reward_fn(score_dict, reward_device="cpu"):
    """
    Build a multi-reward function on the given device.

    Args:
        score_dict: {"reward_name": weight, ...}
            E.g. {"aesthetic": 0.5, "clip": 0.5} or {"geneval": 1.0}
        reward_device: "cpu" or "cuda" / "cuda:0"

    Returns:
        reward_fn(images, prompts, metadata) -> (scores_tensor, details_dict)
    """
    dev = reward_device
    score_fns = {}

    for name, weight in score_dict.items():
        if name not in _REWARD_REGISTRY:
            raise ValueError(
                f"Unknown reward '{name}'. Available: {list(_REWARD_REGISTRY.keys())}")
        mod_path, func_name, needs_device = _REWARD_REGISTRY[name]

        if name == "aesthetic":
            mod = importlib.import_module(mod_path)
            scorer = mod.AestheticScorer(dtype=torch.float32).to(dev)

            def _aes_fn(images, prompts, metadata, _s=scorer, _d=dev):
                if isinstance(images, torch.Tensor):
                    images = (images * 255).round().clamp(0, 255).to(torch.uint8).to(_d)
                return _s(images), {}
            score_fns[name] = _aes_fn

        elif name == "clip":
            mod = importlib.import_module(mod_path)
            scorer = mod.ClipScorer(device=dev)

            def _clip_fn(images, prompts, metadata, _s=scorer, _d=dev):
                if isinstance(images, torch.Tensor):
                    images = images.to(_d)
                return _s(images, prompts), {}
            score_fns[name] = _clip_fn

        elif name == "geneval":
            mod = importlib.import_module(mod_path)
            constructor = getattr(mod, func_name)
            raw_fn = constructor(dev)

            def _geneval_fn(images, prompts, metadata, _raw=raw_fn):
                if isinstance(metadata, dict):
                    n = len(images) if hasattr(images, '__len__') else images.shape[0]
                    metadata_list = ([metadata] * n if metadata
                                     else [{} for _ in range(n)])
                else:
                    metadata_list = metadata
                scores, rewards, strict_rewards, group_rewards, group_strict_rewards = \
                    _raw(images, prompts, metadata_list, only_strict=True)
                details = {
                    "accuracy": rewards,
                    "strict_accuracy": strict_rewards,
                }
                return scores, details
            score_fns[name] = _geneval_fn

        else:
            mod = importlib.import_module(mod_path)
            constructor = getattr(mod, func_name)
            if needs_device:
                raw = constructor(dev)
            else:
                raw = constructor()

            # For the sglang UnifiedReward client, upstream fires all images in
            # a single asyncio.gather. The real fix is on the server side
            # (REQUEST_TIMEOUT=300 in start_unifiedreward_*.sh). Here we only
            # keep a thin chunking layer as a safety valve. Default = 64 means
            # we pass the whole reward-batch straight through (no artificial
            # throttling), which lets sglang's continuous batching fill the GPU.
            # Lower it (e.g. export UNIFIEDREWARD_CLIENT_CHUNK=8) only if you
            # see timeouts again.
            if name == "unifiedreward":
                import os
                _chunk = int(os.environ.get("UNIFIEDREWARD_CLIENT_CHUNK", "64"))

                # ----------------------------------------------------------
                # Workaround for `RuntimeError: Event loop is closed` during
                # eval (e.g. evaluate_drawbench.py).
                #
                # flow_grpo/rewards.py::unifiedreward_score_sglang creates a
                # single AsyncOpenAI client at closure-construction time. Its
                # underlying httpx.AsyncClient connection pool is bound to the
                # event loop that first uses it. The reward closure then calls
                # `asyncio.run(...)` per batch, which spins up a NEW loop and
                # tears it down each time. On the 2nd call, httpx tries to
                # clean up sockets via the previous (now-closed) loop -->
                #     RuntimeError: Event loop is closed
                # which the OpenAI SDK re-raises as `APIConnectionError`.
                #
                # Training rarely hits this because reward calls are sparse
                # and interleaved with GPU work; DrawBench eval calls the
                # reward back-to-back many times and reliably trips it.
                #
                # Fix without touching flow_grpo/: rebuild the upstream
                # closure (and therefore a fresh AsyncOpenAI client) for
                # every chunk. The constructor itself only defines a few
                # async functions and instantiates an HTTP client, so the
                # overhead is negligible compared to the LLM inference call.
                # ----------------------------------------------------------
                _constructor = constructor  # capture
                _dev = dev

                def _unified_chunked(images, prompts, metadata,
                                     _ctor=_constructor, _d=_dev, _c=_chunk):
                    import gc as _gc
                    n = len(images) if hasattr(images, "__len__") else images.shape[0]
                    all_scores = []
                    for s in range(0, n, _c):
                        e = min(s + _c, n)
                        img_slice = images[s:e]
                        pr_slice = prompts[s:e] if prompts is not None else prompts
                        # Fresh closure -> fresh AsyncOpenAI client -> fresh
                        # httpx pool -> never carries a dead event loop.
                        fresh = _ctor(_d)
                        try:
                            sc, _ = fresh(img_slice, pr_slice, metadata)
                            all_scores.extend(sc)
                        finally:
                            # Drop the closure (and its AsyncOpenAI client)
                            # right now, OUTSIDE any running event loop, and
                            # force GC so httpx's __del__ -> aclose() runs
                            # synchronously instead of being scheduled on the
                            # already-closed loop from `asyncio.run`. This
                            # silences the harmless but noisy
                            #   "Task exception was never retrieved ...
                            #    RuntimeError: Event loop is closed"
                            # warnings emitted between chunks / checkpoints.
                            del fresh
                            _gc.collect()
                    return all_scores, {}
                score_fns[name] = _unified_chunked
            else:
                score_fns[name] = raw

    @torch.no_grad()
    def reward_fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images_dev = images.to(dev)
        else:
            images_dev = images

        total_scores = None
        details = {}

        for name, weight in score_dict.items():
            fn = score_fns[name]
            scores, meta = fn(images_dev, prompts, metadata)

            if isinstance(scores, torch.Tensor):
                scores = scores.cpu().float()
            elif isinstance(scores, np.ndarray):
                scores = torch.from_numpy(scores).float()
            elif isinstance(scores, list):
                scores = torch.tensor(scores, dtype=torch.float32)

            details[name] = scores
            weighted = weight * scores

            if total_scores is None:
                total_scores = weighted
            else:
                total_scores = total_scores + weighted

        details["combined"] = total_scores
        return total_scores, details

    return reward_fn


def resolve_training_reward(args, device):
    """
    Resolve reward function from CLI args (--reward_dict / --reward_type).

    Returns: (reward_fn, score_dict, description_string)
    """
    reward_device = str(device) if args.reward_on_gpu else "cpu"

    if args.reward_dict is not None:
        import json
        score_dict = json.loads(args.reward_dict)
        reward_fn = build_training_reward_fn(score_dict, reward_device=reward_device)
        desc = f"Reward (multi): {score_dict} on {reward_device}"
    elif args.reward_type is not None:
        legacy_map = {
            "aesthetic": {"aesthetic": 1.0},
            "clip": {"clip": 1.0},
            "aesthetic_clip": {
                "aesthetic": args.aesthetic_weight,
                "clip": args.clip_weight,
            },
            "pickscore": {"pickscore": 1.0},
            "jpeg": {"jpeg": 1.0},
        }
        score_dict = legacy_map[args.reward_type]
        reward_fn = build_training_reward_fn(score_dict, reward_device=reward_device)
        desc = f"Reward (legacy --reward_type={args.reward_type}): {score_dict} on {reward_device}"
    else:
        score_dict = {"aesthetic": 0.5, "clip": 0.5}
        reward_fn = build_training_reward_fn(score_dict, reward_device=reward_device)
        desc = f"Reward (default): {score_dict} on {reward_device}"

    return reward_fn, score_dict, desc


# ---------------------------------------------------------------------------
# GenEval reward server client (for evaluation script)
# ---------------------------------------------------------------------------
def build_geneval_reward_fn(server_url, batch_size=48):
    """
    Return a callable that sends images to a running GenEval reward server.

    The server should be launched separately:
        cd reward-server/
        gunicorn --config gunicorn.conf.py 'app_geneval:create_app()'

    Args:
        server_url: e.g. "http://127.0.0.1:18085"
        batch_size: max images per HTTP request

    Returns:
        reward_fn(images, metadatas, only_strict) -> dict with keys:
            scores, rewards, strict_rewards, group_rewards, group_strict_rewards
    """
    sess = requests.Session()
    retries = Retry(total=10, backoff_factor=1,
                    status_forcelist=[500, 502, 503, 504],
                    allowed_methods=False)
    sess.mount("http://", HTTPAdapter(max_retries=retries))

    @torch.no_grad()
    def reward_fn(images, metadatas, only_strict=False):
        # tensor -> PIL
        if isinstance(images, torch.Tensor):
            arr = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            arr = arr.transpose(0, 2, 3, 1)
            pil_images = [Image.fromarray(a) for a in arr]
        elif isinstance(images, list) and len(images) and isinstance(images[0], Image.Image):
            pil_images = images
        else:
            pil_images = [Image.fromarray(a) for a in images]

        all_scores, all_rewards, all_strict = [], [], []
        all_gs, all_gsr = defaultdict(list), defaultdict(list)

        for s in range(0, len(pil_images), batch_size):
            e = min(s + batch_size, len(pil_images))
            jpegs = []
            for img in pil_images[s:e]:
                buf = BytesIO()
                img.save(buf, format="JPEG")
                jpegs.append(buf.getvalue())
            data = pickle.dumps({
                "images": jpegs,
                "meta_datas": metadatas[s:e],
                "only_strict": only_strict,
            })
            try:
                r = sess.post(server_url, data=data, timeout=180)
                r.raise_for_status()
                rd = pickle.loads(r.content)
                all_scores += rd["scores"]
                all_rewards += rd["rewards"]
                all_strict += rd["strict_rewards"]
                for k, v in rd["group_strict_rewards"].items():
                    all_gsr[k].extend(v)
                for k, v in rd["group_rewards"].items():
                    all_gs[k].extend(v)
            except Exception as exc:
                n = e - s
                print(f"[geneval] WARNING: request failed ({exc}), "
                      f"0.0 for {n} images")
                all_scores += [0.0] * n
                all_rewards += [0.0] * n
                all_strict += [0.0] * n

        return {
            "scores": all_scores,
            "rewards": all_rewards,
            "strict_rewards": all_strict,
            "group_rewards": dict(all_gs),
            "group_strict_rewards": dict(all_gsr),
        }

    return reward_fn
