"""
DrawBench evaluator with a shared image cache.

Why a separate script?
  evaluate_checkpoint.py is shared with run_eval.sh (PickScore / GenEval) and
  must keep its existing output layout. For DrawBench we score the same set
  of images with up to 6 reward models, so we want to:

    1. Generate each (checkpoint, suffix) ONCE into a shared image cache.
    2. On subsequent reward runs, skip generation and only re-score from the
       cached PNGs.

Output layout (compatible with the existing run_eval_drawbench.sh layout —
plus a sibling cache directory):

    <base_out_dir>/
      _image_cache/
        <suffix>/
          manifest.json
          images/{idx:06d}.png
      <eval_reward>/
        <suffix>/
          results.json          # {tag, reward, mean, std, ..., per_image_results}
        summary.json            # batch mode

Where <suffix> is e.g. "step3200_ema_16nfe_cfg" or "baseline_16nfe_cfg",
matching what run_eval_drawbench.sh already names its subdirs.

Usage:
  # Single checkpoint
  accelerate launch --num_processes 7 evaluate_drawbench.py \
      --checkpoint_path $CKPT \
      --image_embedding_path $CKPT/image_embedding.pt \
      --test_prompts_file .../drawbench/test.txt \
      --lora_path .../checkpoint_..._step3200/lora_adapter_ema \
      --output_dir <base>/<reward>/step3200_ema_16nfe_cfg \
      --reward_dict '{"pickscore": 1}'

  # Batch (delegates to checkpoint_utils.discover_checkpoints)
  accelerate launch --num_processes 7 evaluate_drawbench.py \
      --checkpoint_path $CKPT \
      --image_embedding_path $CKPT/image_embedding.pt \
      --test_prompts_file .../drawbench/test.txt \
      --grpo_output_dir .../output_grpo_xxx \
      --output_dir <base>/<reward> \
      --reward_dict '{"pickscore": 1}' \
      --ema_only --skip_existing

  # Cache-only (no scoring): pre-build images for many checkpoints, then
  # later run with each reward to score them without regenerating.
  accelerate launch ... evaluate_drawbench.py ... --image_cache_only
"""

import torch
import argparse
import os
import json
import glob
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

# Patch torch.load for older checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from fudoki.janus.models import VLChatProcessor
from fudoki.eval_loop import CFGScaledModel

from config import VOCABULARY_SIZE_IMG, CFG_SCALE
from model_utils import load_model, decode_image_tokens, build_data_info, NoCFGModel
from reward_utils import build_training_reward_fn
from checkpoint_utils import discover_checkpoints
from sampling import sample_only


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser(description="DrawBench evaluator with image cache")
    p.add_argument("--seed", type=int, default=42)

    # Paths
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--image_embedding_path", type=str, required=True)
    p.add_argument("--test_prompts_file", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True,
                   help="Single mode: <base>/<reward>/<suffix>. "
                        "Batch mode: <base>/<reward>.")

    # LoRA / EMA / gen_modules (single mode)
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--ema_path", type=str, default=None)
    p.add_argument("--gen_modules_path", type=str, default=None)

    # Batch mode
    p.add_argument("--grpo_output_dir", type=str, default=None)
    p.add_argument("--step_interval", type=int, default=None)
    p.add_argument("--step_start", type=int, default=None)
    p.add_argument("--step_end", type=int, default=None)
    p.add_argument("--ema_only", action="store_true")
    p.add_argument("--skip_baseline", action="store_true")
    p.add_argument("--skip_existing", action="store_true")
    p.add_argument("--eval_steps", type=str, default=None,
                   help="Comma-separated steps (e.g. '1000,2000'). "
                        "If set, only those steps are evaluated.")

    # Reward
    p.add_argument("--reward_dict", type=str, default='{"pickscore": 1}')
    p.add_argument("--reward_on_gpu", action="store_true")

    # Image cache
    p.add_argument("--image_cache_dir", type=str, default=None,
                   help="Cache root. Default: <base>/_image_cache where "
                        "<base> is the parent dir of the per-reward output dir.")
    p.add_argument("--cache_suffix", type=str, default=None,
                   help="Cache subdir name (single mode only). "
                        "Default: basename(--output_dir).")
    p.add_argument("--image_cache_only", action="store_true",
                   help="Only build the image cache, do not score.")

    # Sampling / eval settings
    p.add_argument("--discrete_fm_steps", type=int, default=16)
    p.add_argument("--txt_max_length", type=int, default=500)
    p.add_argument("--num_prompts", type=int, default=0,
                   help="Number of test prompts (0 = all).")
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--reward_batch_size", type=int, default=64)

    # CFG
    p.add_argument("--no_cfg", action="store_true")
    p.add_argument("--cfg_scale", type=float, default=None)

    return p.parse_args()


# ---------------------------------------------------------------------------
# Image cache helpers
# ---------------------------------------------------------------------------
def _resolve_cache_root(args, per_reward_output_dir):
    """
    Cache root defaults to ``<base>/_image_cache`` where ``<base>`` is the
    PARENT of ``per_reward_output_dir`` — i.e. the dir that already contains
    sibling ``<reward>/`` subdirs.

    For single mode, ``per_reward_output_dir`` is the parent of the suffix
    dir (i.e. dirname(--output_dir)).
    For batch mode, ``per_reward_output_dir`` is --output_dir itself.
    """
    if args.image_cache_dir:
        return args.image_cache_dir
    base = os.path.dirname(per_reward_output_dir.rstrip("/"))
    return os.path.join(base, "_image_cache")


def _is_cache_complete(cache_dir, expected_count):
    """Return (ok, manifest)."""
    mpath = os.path.join(cache_dir, "manifest.json")
    if not os.path.exists(mpath):
        return False, None
    try:
        with open(mpath) as f:
            manifest = json.load(f)
    except Exception:
        return False, None
    images = manifest.get("images", [])
    if len(images) != expected_count:
        return False, manifest
    for e in images:
        if not os.path.exists(os.path.join(cache_dir, e["path"])):
            return False, manifest
    return True, manifest


@torch.no_grad()
def _generate_to_cache(model, eval_prompts, vl_chat_processor, path_img,
                      step_size, n_samples_per_prompt, accelerator, args,
                      cache_dir):
    """Generate images for this rank's prompt shard and write them to
    ``cache_dir/images/{idx:06d}.png``. Rank 0 writes ``manifest.json``.
    Returns the manifest dict (rank 0 only; others get None and read it
    back at the synchronization barrier below)."""
    model.eval()
    device = accelerator.device
    images_dir = os.path.join(cache_dir, "images")
    if accelerator.is_main_process:
        os.makedirs(images_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    use_cfg = not getattr(args, "no_cfg", False)
    if use_cfg:
        cfg_model = CFGScaledModel(model=model, g_or_u='generation')
        effective_cfg = args.cfg_scale if args.cfg_scale is not None else CFG_SCALE
    else:
        cfg_model = NoCFGModel(model)
        effective_cfg = 0.0
    accelerator.print(
        f"  CFG: {'ON (scale={:.1f})'.format(effective_cfg) if use_cfg else 'OFF'}")

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    n_total = len(eval_prompts)
    per_rank = (n_total + world_size - 1) // world_size
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n_total)
    my_prompts = eval_prompts[my_start:my_end]

    accelerator.print(
        f"  GENERATE: {n_total} prompts / {world_size} GPUs (~{per_rank}/GPU), "
        f"{n_samples_per_prompt} imgs/prompt -> {cache_dir}")

    local_entries = []
    for pi, prompt in enumerate(tqdm(my_prompts, desc=f"[GPU{rank} gen]",
                                     disable=(rank != 0))):
        gpi = my_start + pi
        torch.manual_seed(args.seed + gpi)
        torch.cuda.manual_seed(args.seed + gpi)

        x_init, data_info = build_data_info(
            prompt, n_samples_per_prompt, vl_chat_processor, path_img,
            args.txt_max_length, device)
        final_tokens, image_mask = sample_only(
            model=cfg_model, path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
            x_init=x_init, data_info=data_info,
            step_size=step_size, cfg_scale=effective_cfg,
            n_mc=1, device=device)
        itk = final_tokens[image_mask == 1].reshape(n_samples_per_prompt, -1)
        pix = decode_image_tokens(model, itk)  # [B,3,H,W] float 0..1
        pix_u8 = (pix * 255).round().clamp(0, 255).to(torch.uint8)
        pix_u8 = pix_u8.permute(0, 2, 3, 1).contiguous().cpu().numpy()  # [B,H,W,C]

        for b in range(pix_u8.shape[0]):
            global_idx = gpi * n_samples_per_prompt + b
            rel = os.path.join("images", f"{global_idx:06d}.png")
            Image.fromarray(pix_u8[b]).save(os.path.join(cache_dir, rel))
            local_entries.append({"idx": global_idx, "prompt": prompt, "path": rel})
        del x_init, data_info, final_tokens, itk, pix
    torch.cuda.empty_cache()

    accelerator.wait_for_everyone()
    all_entries = gather_object(local_entries)
    if accelerator.is_main_process:
        all_entries = sorted(all_entries, key=lambda e: e["idx"])
        manifest = {
            "cache_suffix": os.path.basename(cache_dir.rstrip("/")),
            "test_prompts_file": args.test_prompts_file,
            "num_prompts": n_total,
            "num_images_per_prompt": n_samples_per_prompt,
            "expected_count": n_total * n_samples_per_prompt,
            "discrete_fm_steps": args.discrete_fm_steps,
            "use_cfg": use_cfg,
            "cfg_scale": float(effective_cfg),
            "seed": args.seed,
            "images": all_entries,
        }
        with open(os.path.join(cache_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"  Wrote manifest: {cache_dir}/manifest.json "
              f"({len(all_entries)} images)")
    accelerator.wait_for_everyone()


def _read_manifest(cache_dir):
    with open(os.path.join(cache_dir, "manifest.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _score_from_cache(reward_fn, reward_name, manifest, cache_dir,
                     accelerator, save_dir, tag, reward_batch_size):
    """Load cached PNGs, score with ``reward_fn``, and write
    ``save_dir/results.json``. Returns (mean, std)."""
    rank = accelerator.process_index
    world_size = accelerator.num_processes
    device = accelerator.device

    entries = manifest["images"]
    n_total = len(entries)
    per_rank = (n_total + world_size - 1) // world_size
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n_total)
    my_entries = entries[my_start:my_end]

    accelerator.print(
        f"  SCORE[{reward_name}]: {n_total} imgs / {world_size} GPUs "
        f"(~{per_rank}/GPU, batch={reward_batch_size})")

    local_results = []
    n_local = len(my_entries)
    for s in tqdm(range(0, n_local, reward_batch_size),
                  desc=f"[GPU{rank} {reward_name}]", disable=(rank != 0)):
        e = min(s + reward_batch_size, n_local)
        batch = my_entries[s:e]
        imgs = []
        for ent in batch:
            arr = np.array(Image.open(os.path.join(cache_dir, ent["path"])
                                      ).convert("RGB"))
            imgs.append(torch.from_numpy(arr))
        batch_u8 = torch.stack(imgs, dim=0)                         # [B,H,W,C]
        batch_float = batch_u8.permute(0, 3, 1, 2).float() / 255.0  # [B,3,H,W]
        prompts = [ent["prompt"] for ent in batch]

        scores_b, _ = reward_fn(batch_float.to(device), prompts, {})
        if isinstance(scores_b, torch.Tensor):
            scores_np = scores_b.cpu().numpy()
        elif isinstance(scores_b, np.ndarray):
            scores_np = scores_b
        else:
            scores_np = np.array(scores_b)
        for i, ent in enumerate(batch):
            local_results.append({
                "idx": ent["idx"],
                "prompt": ent["prompt"],
                "score": float(scores_np[i]),
            })
        del batch_u8, batch_float

    accelerator.wait_for_everyone()
    all_results = gather_object(local_results)

    mean_r, std_r = 0.0, 0.0
    if accelerator.is_main_process:
        all_results = sorted(all_results, key=lambda r: r["idx"])
        scores = np.array([r["score"] for r in all_results])
        mean_r = float(scores.mean()) if len(scores) else 0.0
        std_r = float(scores.std()) if len(scores) else 0.0
        print(f"  [{tag}|{reward_name}] mean={mean_r:.4f} "
              f"std={std_r:.4f} (n={len(all_results)})")
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump({
                "tag": tag,
                "reward": reward_name,
                "mean": mean_r,
                "std": std_r,
                "num_images": len(all_results),
                "image_cache_dir": cache_dir,
                "per_image_results": all_results,
            }, f, indent=2)
        print(f"  Saved -> {save_dir}/results.json")
    accelerator.wait_for_everyone()
    return mean_r, std_r


# ---------------------------------------------------------------------------
# Driver: ensure cache, then optionally score
# ---------------------------------------------------------------------------
def _run_one(*, build_model_fn, reward_fn, reward_name, eval_prompts,
             vl_chat_processor, path_img, step_size, args, accelerator,
             cache_dir, save_dir, tag, expected_count):
    """Ensure cache at ``cache_dir`` exists (calling ``build_model_fn()`` only
    if (re)generation is needed), then score into ``save_dir``.

    ``build_model_fn`` is a callable that returns a freshly loaded FUDOKI
    model. We avoid loading it when the cache is complete.
    """
    # Rank 0 decides; broadcast via gather_object.
    decision = [None]
    if accelerator.is_main_process:
        ok, manifest = _is_cache_complete(cache_dir, expected_count)
        if ok:
            print(f"  [cache hit] {cache_dir} ({len(manifest['images'])} imgs)")
            decision[0] = False
        else:
            if manifest is not None:
                print(f"  [cache stale] {cache_dir} -> regenerating")
            else:
                print(f"  [cache miss] {cache_dir} -> generating")
            decision[0] = True
    gathered = gather_object(decision)
    need_build = next((x for x in gathered if x is not None), False)

    if need_build:
        os.makedirs(cache_dir, exist_ok=True) if accelerator.is_main_process else None
        accelerator.wait_for_everyone()
        model = build_model_fn()
        _generate_to_cache(
            model, eval_prompts, vl_chat_processor, path_img,
            step_size, args.num_images_per_prompt, accelerator, args, cache_dir,
        )
        del model
        torch.cuda.empty_cache()

    if args.image_cache_only:
        accelerator.print("  --image_cache_only: skipping reward scoring.")
        return 0.0, 0.0

    # All ranks read the manifest from disk to score (it's small).
    accelerator.wait_for_everyone()
    manifest = _read_manifest(cache_dir)
    return _score_from_cache(
        reward_fn, reward_name, manifest, cache_dir,
        accelerator, save_dir, tag, args.reward_batch_size,
    )


# ---------------------------------------------------------------------------
# Batch-mode helper: find existing results.json (alt suffixes)
# ---------------------------------------------------------------------------
def _find_existing_results(reward_output_dir, base_tag, step,
                           nfe_suffix, cfg_suffix):
    """Return path to an existing results.json that matches ``base_tag``
    (e.g. 'baseline' or 'checkpoint_..._step3200_ema'), with reasonable
    fallbacks for older naming conventions. Same logic as
    evaluate_checkpoint.py's _find_existing_json."""
    candidates = [base_tag]
    if base_tag == "baseline":
        candidates.append(f"baseline{nfe_suffix}{cfg_suffix}")
    elif step is not None:
        if base_tag.endswith("_ema"):
            candidates.append(f"step{step}_ema{nfe_suffix}{cfg_suffix}")
        else:
            candidates.append(f"step{step}{nfe_suffix}{cfg_suffix}")
    for sub in candidates:
        p = os.path.join(reward_output_dir, sub, "results.json")
        if os.path.exists(p):
            return p
    # Glob fallback
    if base_tag == "baseline":
        pat = os.path.join(reward_output_dir, "baseline*", "results.json")
        want_step = None
    elif step is not None:
        sfx = "_ema" if base_tag.endswith("_ema") else ""
        pat = os.path.join(reward_output_dir, f"*step{step}{sfx}*", "results.json")
        want_step = step
    else:
        return None
    for m in sorted(glob.glob(pat)):
        if want_step is not None:
            sub = os.path.basename(os.path.dirname(m))
            mm = re.search(r"step(\d+)", sub)
            if not mm or int(mm.group(1)) != want_step:
                continue
        return m
    return None


def _suffix_for(base_tag, step, nfe_suffix, cfg_suffix):
    """Canonical suffix used both for results subdir and cache subdir."""
    if base_tag == "baseline":
        return f"baseline{nfe_suffix}{cfg_suffix}"
    if base_tag.endswith("_ema"):
        return f"step{step}_ema{nfe_suffix}{cfg_suffix}"
    return f"step{step}{nfe_suffix}{cfg_suffix}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()
    accelerator = Accelerator()
    device = accelerator.device

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Prompts
    with open(args.test_prompts_file) as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    eval_prompts = (all_prompts[:args.num_prompts]
                    if args.num_prompts > 0 else all_prompts)
    accelerator.print(
        f"Loaded {len(eval_prompts)} test prompts (from {len(all_prompts)} total)")

    # Reward
    score_dict = json.loads(args.reward_dict)
    reward_device = str(device) if args.reward_on_gpu else "cpu"
    if args.image_cache_only:
        reward_fn = None
        reward_name = "(cache-only)"
        accelerator.print("image_cache_only mode -- reward construction skipped")
    else:
        reward_fn = build_training_reward_fn(score_dict, reward_device=reward_device)
        reward_name = next(iter(score_dict.keys())) if score_dict else "reward"
        accelerator.print(f"Reward: {score_dict} on {reward_device}")
    accelerator.print(f"GPUs: {accelerator.num_processes}")

    vl_chat_processor = VLChatProcessor.from_pretrained(args.checkpoint_path)
    path_img = MixtureDiscreteSoftmaxProbPath(
        mode='image', embedding_path=args.image_embedding_path)
    path_img.embedding = path_img.embedding.to(device)
    step_size = 1.0 / args.discrete_fm_steps

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    expected_count = len(eval_prompts) * args.num_images_per_prompt

    # =====================================================================
    # BATCH MODE
    # =====================================================================
    if args.grpo_output_dir is not None:
        reward_output_dir = args.output_dir          # <base>/<reward>
        cache_root = _resolve_cache_root(args, reward_output_dir)
        accelerator.print(f"Image cache root: {cache_root}")

        nfe_suffix = f"_{args.discrete_fm_steps}nfe"
        cfg_suffix = "" if args.no_cfg else "_cfg"
        results_summary = {}

        def _eval_one(base_tag, step, model_loader):
            """Run the cache+score pipeline for a single (tag, model)."""
            suffix = _suffix_for(base_tag, step, nfe_suffix, cfg_suffix)
            save_dir = os.path.join(reward_output_dir, suffix)
            cache_dir = os.path.join(cache_root, suffix)

            # SKIP_EXISTING: if results.json already exists for this reward,
            # do nothing (cache may or may not exist; we don't need it).
            if args.skip_existing:
                hit = _find_existing_results(
                    reward_output_dir, base_tag, step, nfe_suffix, cfg_suffix)
                if hit is not None:
                    rel = os.path.relpath(hit, reward_output_dir)
                    if accelerator.is_main_process:
                        try:
                            with open(hit) as f:
                                d = json.load(f)
                            results_summary[base_tag] = {
                                "mean": float(d["mean"]),
                                "std": float(d.get("std", 0.0)),
                            }
                            accelerator.print(
                                f"\n  [SKIP] {base_tag} (results exist at {rel}; "
                                f"mean={results_summary[base_tag]['mean']:.4f})")
                        except Exception as e:
                            accelerator.print(
                                f"\n  [SKIP] {base_tag} (results exist at {rel}; "
                                f"warn: {e})")
                    accelerator.wait_for_everyone()
                    return

            accelerator.print("\n" + "=" * 60)
            accelerator.print(f"Evaluating {base_tag}  (suffix={suffix})")
            accelerator.print("=" * 60)
            torch.manual_seed(args.seed)
            torch.cuda.manual_seed(args.seed)
            mean_r, std_r = _run_one(
                build_model_fn=model_loader,
                reward_fn=reward_fn, reward_name=reward_name,
                eval_prompts=eval_prompts, vl_chat_processor=vl_chat_processor,
                path_img=path_img, step_size=step_size, args=args,
                accelerator=accelerator,
                cache_dir=cache_dir, save_dir=save_dir,
                tag=base_tag, expected_count=expected_count,
            )
            if accelerator.is_main_process and not args.image_cache_only:
                results_summary[base_tag] = {"mean": mean_r, "std": std_r}

        # ---- 1) Baseline ----
        if not args.skip_baseline:
            _eval_one("baseline", None,
                      lambda: load_model(args.checkpoint_path, device=device))

        # ---- 2) Discover checkpoints ----
        ckpts = discover_checkpoints(
            args.grpo_output_dir, args.step_interval,
            step_start=args.step_start, step_end=args.step_end)

        # Optional EVAL_STEPS filter
        if args.eval_steps:
            wanted = {int(s.strip()) for s in args.eval_steps.split(",") if s.strip()}
            ckpts = [c for c in ckpts if c[2] in wanted]
            accelerator.print(f"Filtered to EVAL_STEPS={sorted(wanted)} -> "
                              f"{len(ckpts)} checkpoints")

        accelerator.print(f"\n{len(ckpts)} checkpoints to evaluate")
        for n, _, s in ckpts:
            accelerator.print(f"  {n}  (step {s})")

        for ckpt_name, ckpt_dir, step in ckpts:
            lora_p = os.path.join(ckpt_dir, "lora_adapter")
            ema_p = os.path.join(ckpt_dir, "ema_state.pt")
            gen_p = os.path.join(ckpt_dir, "gen_modules_state.pt")
            gen_p = gen_p if os.path.exists(gen_p) else None
            ema_lora_p = os.path.join(ckpt_dir, "lora_adapter_ema")
            has_ema_lora = (os.path.exists(ema_lora_p)
                            and os.listdir(ema_lora_p))
            has_ema = has_ema_lora or os.path.exists(ema_p)

            # LoRA only
            if not args.ema_only and os.path.exists(lora_p):
                _eval_one(
                    ckpt_name, step,
                    lambda lp=lora_p: load_model(
                        args.checkpoint_path, lora_path=lp,
                        gen_modules_path=gen_p, device=device),
                )

            # LoRA + EMA
            if has_ema:
                ema_tag = f"{ckpt_name}_ema"
                if has_ema_lora:
                    loader = (lambda elp=ema_lora_p: load_model(
                        args.checkpoint_path, lora_path=elp,
                        gen_modules_path=gen_p, device=device))
                else:
                    loader = (lambda lp=lora_p, ep=ema_p: load_model(
                        args.checkpoint_path, lora_path=lp,
                        ema_path=ep, gen_modules_path=gen_p,
                        device=device))
                _eval_one(ema_tag, step, loader)

        # ---- Summary ----
        if accelerator.is_main_process and not args.image_cache_only:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"{'Checkpoint':<55} {'Mean':>8} {'Std':>8}")
            print("-" * 73)
            for name, res in results_summary.items():
                print(f"{name:<55} {res['mean']:>8.4f} {res['std']:>8.4f}")
            sp = os.path.join(reward_output_dir, "summary.json")
            os.makedirs(os.path.dirname(sp), exist_ok=True)
            with open(sp, "w") as f:
                json.dump(results_summary, f, indent=2)
            print(f"\nSummary -> {sp}")

    # =====================================================================
    # SINGLE MODE  (--output_dir already includes the suffix)
    # =====================================================================
    else:
        save_dir = args.output_dir
        suffix = args.cache_suffix or os.path.basename(save_dir.rstrip("/"))
        # Per-reward dir is the parent of save_dir; cache root is its sibling.
        per_reward_dir = os.path.dirname(save_dir.rstrip("/"))
        cache_root = _resolve_cache_root(args, per_reward_dir)
        cache_dir = os.path.join(cache_root, suffix)
        accelerator.print(f"Image cache: {cache_dir}")

        # tag for results.json
        tag = "baseline"
        if args.lora_path:
            parent = os.path.basename(os.path.dirname(args.lora_path))
            tag = (parent if parent not in ("lora_adapter", "lora_adapter_ema")
                   else os.path.basename(
                       os.path.dirname(os.path.dirname(args.lora_path))))
            if args.ema_path or "ema" in (args.lora_path or ""):
                tag += "_ema"

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        def _loader():
            return load_model(
                args.checkpoint_path,
                lora_path=args.lora_path,
                ema_path=args.ema_path,
                gen_modules_path=args.gen_modules_path,
                device=device,
            )

        _run_one(
            build_model_fn=_loader,
            reward_fn=reward_fn, reward_name=reward_name,
            eval_prompts=eval_prompts, vl_chat_processor=vl_chat_processor,
            path_img=path_img, step_size=step_size, args=args,
            accelerator=accelerator,
            cache_dir=cache_dir, save_dir=save_dir,
            tag=tag, expected_count=expected_count,
        )
