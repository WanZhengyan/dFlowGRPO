"""
Evaluate saved GRPO checkpoints on the GenEval benchmark (multi-GPU).

Thin entry point — all logic lives in the shared modules:
  config.py, model_utils.py, reward_utils.py, evaluation.py, checkpoint_utils.py

Usage:
  # Multi-GPU, single checkpoint:
  accelerate launch -            if has_ema:
                ema_tag = f"{ckpt_name}_ema"
                ema_dir = os.path.join(args.output_dir, ema_tag)
                existing_json = (_find_existing_json(ema_tag, step)
                                 if args.skip_existing else None)
                ema_done = existing_json is not None
                if ema_done:
                    loaded = _load_existing_geneval(existing_json)
                    rel = os.path.relpath(existing_json, args.output_dir)
                    if loaded is not None and accelerator.is_main_process:
                        summary[ema_tag] = loaded
                        accelerator.print(
                            f"\n  [SKIP] {ema_tag} (results exist at {rel}; "
                            f"overall={loaded['overall']['avg_score']:.4f})")
                    else:
                        accelerator.print(
                            f"\n  [SKIP] {ema_tag} (results exist at {rel})")
                else: evaluate_geneval.py \
      --checkpoint_path .../checkpoints \
      --image_embedding_path .../image_embedding.pt \
      --geneval_metadata .../test_metadata.jsonl \
      --lora_path .../checkpoint_epoch1_step1700/lora_adapter \
      --output_dir eval_geneval_results/step1700

  # Multi-GPU, batch mode:
  accelerate launch --num_processes 8 evaluate_geneval.py \
      --checkpoint_path .../checkpoints \
      --image_embedding_path .../image_embedding.pt \
      --geneval_metadata .../test_metadata.jsonl \
      --grpo_output_dir output_lora_geneval \
      --output_dir eval_geneval_results \
      --step_interval 20

  # Multi-GPU, baseline only:
  accelerate launch --num_processes 8 evaluate_geneval.py \
      --checkpoint_path .../checkpoints \
      --image_embedding_path .../image_embedding.pt \
      --geneval_metadata .../test_metadata.jsonl \
      --output_dir eval_geneval_results/baseline
"""

import torch
import argparse
import os
import json
from collections import defaultdict
from accelerate import Accelerator

# Patch torch.load for older checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from fudoki.janus.models import VLChatProcessor

# Import from refactored modules
from config import GENEVAL_TAGS
from model_utils import load_model
from reward_utils import build_geneval_reward_fn
from evaluation import evaluate_geneval
from checkpoint_utils import discover_checkpoints


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------
def parse_arguments():
    p = argparse.ArgumentParser(
        description="Evaluate GRPO checkpoints on GenEval (multi-GPU)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Base FUDOKI model checkpoint directory")
    p.add_argument("--image_embedding_path", type=str, required=True)
    p.add_argument("--discrete_fm_steps", type=int, default=8)
    p.add_argument("--txt_max_length", type=int, default=500)
    p.add_argument("--output_dir", type=str, required=True)

    # single-checkpoint mode
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--ema_path", type=str, default=None)
    p.add_argument("--gen_modules_path", type=str, default=None)

    # batch mode
    p.add_argument("--grpo_output_dir", type=str, default=None,
                   help="GRPO output dir with checkpoint_epoch*_step* sub-dirs")
    p.add_argument("--step_interval", type=int, default=None,
                   help="Only evaluate checkpoints whose step satisfies "
                        "(step - step_start) %% step_interval == 0")
    p.add_argument("--step_start", type=int, default=None,
                   help="Lower bound (inclusive) on checkpoint step to evaluate.")
    p.add_argument("--step_end", type=int, default=None,
                   help="Upper bound (inclusive) on checkpoint step to evaluate.")

    # GenEval
    p.add_argument("--geneval_metadata", type=str, required=True,
                   help="Path to GenEval test_metadata.jsonl file")
    p.add_argument("--geneval_server_url", type=str,
                   default="http://127.0.0.1:18085")
    p.add_argument("--num_images_per_prompt", type=int, default=1)
    p.add_argument("--reward_batch_size", type=int, default=12)
    p.add_argument("--num_prompts", type=int, default=None,
                   help="Limit number of prompts (default: all)")
    p.add_argument("--no_save_images", action="store_true",
                   help="Skip saving generated images to disk")
    p.add_argument("--no_cfg", action="store_true",
                   help="Disable classifier-free guidance (single forward pass)")
    p.add_argument("--cfg_scale", type=float, default=None,
                   help="Override CFG scale (default: 5.0). Ignored if --no_cfg.")
    p.add_argument("--ema_only", action="store_true",
                   help="In batch mode, only evaluate LoRA+EMA (skip LoRA-only).")
    p.add_argument("--skip_baseline", action="store_true",
                   help="In batch mode, skip the baseline evaluation.")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip checkpoints whose output dir already contains "
                        "geneval_results.json.")
    return p.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
if __name__ == "__main__":
    args = parse_arguments()
    save_images = not args.no_save_images

    accelerator = Accelerator()
    device = accelerator.device

    # Load metadata
    accelerator.print(f"Loading metadata: {args.geneval_metadata}")
    with open(args.geneval_metadata) as f:
        all_meta = [json.loads(l) for l in f if l.strip()]
    if args.num_prompts is not None:
        all_meta = all_meta[: args.num_prompts]

    if accelerator.is_main_process:
        tc = defaultdict(int)
        for m in all_meta:
            tc[m["tag"]] += 1
        print(f"Prompts: {len(all_meta)}")
        for t in GENEVAL_TAGS:
            print(f"  {t}: {tc.get(t, 0)}")

    reward_fn = build_geneval_reward_fn(
        args.geneval_server_url, args.reward_batch_size)
    accelerator.print(f"Reward server: {args.geneval_server_url}")
    accelerator.print(f"GPUs: {accelerator.num_processes}")

    vl_proc = VLChatProcessor.from_pretrained(args.checkpoint_path)
    path_img = MixtureDiscreteSoftmaxProbPath(
        mode="image", embedding_path=args.image_embedding_path)
    path_img.embedding = path_img.embedding.to(device)
    step_size = 1.0 / args.discrete_fm_steps

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ================================================================
    # Batch mode
    # ================================================================
    if args.grpo_output_dir is not None:
        summary = {}

        # Bash run_eval.sh writes to e.g. step3100_ema_8nfe_cfg/.
        # Probe both naming styles when --skip_existing.
        _nfe_suffix = f"_{args.discrete_fm_steps}nfe"
        _cfg_suffix = "" if getattr(args, "no_cfg", False) else "_cfg"

        def _candidate_dirs(base_tag, step):
            cands = [base_tag]
            if base_tag == "baseline":
                cands.append(f"baseline{_nfe_suffix}{_cfg_suffix}")
            elif step is not None:
                if base_tag.endswith("_ema"):
                    cands.append(f"step{step}_ema{_nfe_suffix}{_cfg_suffix}")
                else:
                    cands.append(f"step{step}{_nfe_suffix}{_cfg_suffix}")
            return cands

        def _find_existing_json(base_tag, step, fname="geneval_results.json"):
            for sub in _candidate_dirs(base_tag, step):
                p = os.path.join(args.output_dir, sub, fname)
                if os.path.exists(p):
                    return p
            # Glob fallback — handles older result dirs whose suffix
            # differs from the current --discrete_fm_steps / cfg setting
            # (e.g. baseline_8nfe vs current 16nfe → "baseline_16nfe"
            # miss). Any matching geneval_results.json on disk counts
            # as "already evaluated", because we only need the metrics
            # for the summary curve.
            import glob as _g, re as _re
            if base_tag == "baseline":
                pat = os.path.join(args.output_dir, "baseline*", fname)
                want_step = None
            elif step is not None:
                suffix = "_ema" if base_tag.endswith("_ema") else ""
                pat = os.path.join(args.output_dir,
                                   f"*step{step}{suffix}*", fname)
                want_step = step
            else:
                return None
            for m in sorted(_g.glob(pat)):
                if want_step is not None:
                    sub = os.path.basename(os.path.dirname(m))
                    mm = _re.search(r"step(\d+)", sub)
                    if not mm or int(mm.group(1)) != want_step:
                        continue
                return m
            return None

        def _load_existing_geneval(path):
            """Load a previously written geneval_results.json; return metrics dict or None."""
            try:
                with open(path) as f:
                    d = json.load(f)
                m = d.get("metrics")
                if not isinstance(m, dict) or "overall" not in m:
                    return None
                return m
            except Exception as e:
                accelerator.print(f"  [WARN] failed to load {path}: {e}")
                return None

        # -- baseline --
        if not args.skip_baseline:
            baseline_dir = os.path.join(args.output_dir, "baseline")
            existing_json = (_find_existing_json("baseline", None)
                             if args.skip_existing else None)
            baseline_done = existing_json is not None
            if baseline_done:
                loaded = _load_existing_geneval(existing_json)
                rel = os.path.relpath(existing_json, args.output_dir)
                if loaded is not None and accelerator.is_main_process:
                    summary["baseline"] = loaded
                    accelerator.print(
                        f"\n  [SKIP] baseline (results exist at {rel}; "
                        f"overall={loaded['overall']['avg_score']:.4f})")
                else:
                    accelerator.print(
                        f"\n  [SKIP] baseline (results exist at {rel})")
            else:
                accelerator.print("\n" + "=" * 60)
                accelerator.print("Evaluating BASELINE (no LoRA)")
                accelerator.print("=" * 60)
                mdl = load_model(args.checkpoint_path, device=device)
                met = evaluate_geneval(
                    mdl, reward_fn, all_meta, vl_proc, path_img,
                    step_size, args.num_images_per_prompt, accelerator,
                    baseline_dir, args,
                    tag="baseline", save_images=save_images)
                if accelerator.is_main_process:
                    summary["baseline"] = met
                del mdl
                torch.cuda.empty_cache()

        # -- discover checkpoints --
        ckpts = discover_checkpoints(
            args.grpo_output_dir, args.step_interval,
            step_start=args.step_start, step_end=args.step_end)
        accelerator.print(
            f"\n{len(ckpts)} checkpoints"
            + (f" (every {args.step_interval} steps)"
               if args.step_interval else ""))
        for n, _, s in ckpts:
            accelerator.print(f"  {n}  (step {s})")

        # -- evaluate each --
        for ckpt_name, ckpt_dir, step in ckpts:
            lora_p = os.path.join(ckpt_dir, "lora_adapter")
            ema_p = os.path.join(ckpt_dir, "ema_state.pt")
            gen_p = os.path.join(ckpt_dir, "gen_modules_state.pt")
            gen_p = gen_p if os.path.exists(gen_p) else None

            # LoRA only (skip if --ema_only)
            if not args.ema_only:
                lora_dir = os.path.join(args.output_dir, ckpt_name)
                existing_json = (_find_existing_json(ckpt_name, step)
                                 if args.skip_existing else None)
                lora_done = existing_json is not None
                if lora_done:
                    loaded = _load_existing_geneval(existing_json)
                    rel = os.path.relpath(existing_json, args.output_dir)
                    if loaded is not None and accelerator.is_main_process:
                        summary[ckpt_name] = loaded
                        accelerator.print(
                            f"\n  [SKIP] {ckpt_name} (results exist at {rel}; "
                            f"overall={loaded['overall']['avg_score']:.4f})")
                    else:
                        accelerator.print(
                            f"\n  [SKIP] {ckpt_name} (results exist at {rel})")
                else:
                    accelerator.print("\n" + "=" * 60)
                    accelerator.print(
                        f"Evaluating {ckpt_name} "
                        f"(LoRA{' + gen_modules' if gen_p else ''})")
                    accelerator.print("=" * 60)
                    mdl = load_model(args.checkpoint_path, lora_path=lora_p,
                                     gen_modules_path=gen_p, device=device)
                    met = evaluate_geneval(
                        mdl, reward_fn, all_meta, vl_proc, path_img,
                        step_size, args.num_images_per_prompt, accelerator,
                        lora_dir, args,
                        tag=ckpt_name, save_images=save_images)
                    if accelerator.is_main_process:
                        summary[ckpt_name] = met
                    del mdl
                    torch.cuda.empty_cache()

            # LoRA + EMA
            ema_lora_p = os.path.join(ckpt_dir, "lora_adapter_ema")
            has_ema_lora = os.path.exists(ema_lora_p) and os.listdir(ema_lora_p)
            has_ema = has_ema_lora or os.path.exists(ema_p)
            if has_ema:
                ema_tag = f"{ckpt_name}_ema"
                ema_dir = os.path.join(args.output_dir, ema_tag)
                existing_json = (_find_existing_json(ema_tag, step)
                                 if args.skip_existing else None)
                ema_done = existing_json is not None
                if ema_done:
                    loaded = _load_existing_geneval(existing_json)
                    rel = os.path.relpath(existing_json, args.output_dir)
                    if loaded is not None and accelerator.is_main_process:
                        summary[ema_tag] = loaded
                        accelerator.print(
                            f"\n  [SKIP] {ema_tag} (results exist at {rel}; "
                            f"overall={loaded['overall']['avg_score']:.4f})")
                    else:
                        accelerator.print(
                            f"\n  [SKIP] {ema_tag} (results exist at {rel})")
                else:
                    accelerator.print("\n" + "=" * 60)
                    accelerator.print(f"Evaluating {ema_tag}")
                    accelerator.print("=" * 60)
                    if has_ema_lora:
                        mdl = load_model(args.checkpoint_path,
                                         lora_path=ema_lora_p,
                                         gen_modules_path=gen_p, device=device)
                    else:
                        mdl = load_model(args.checkpoint_path, lora_path=lora_p,
                                         ema_path=ema_p, gen_modules_path=gen_p,
                                         device=device)
                    met = evaluate_geneval(
                        mdl, reward_fn, all_meta, vl_proc, path_img,
                        step_size, args.num_images_per_prompt, accelerator,
                        ema_dir, args,
                        tag=ema_tag, save_images=save_images)
                    if accelerator.is_main_process:
                        summary[ema_tag] = met
                    del mdl
                    torch.cuda.empty_cache()

        # -- summary table --
        if accelerator.is_main_process:
            print("\n" + "=" * 80)
            print("GENEVAL SUMMARY")
            print("=" * 80)
            hdr = f"{'Checkpoint':<45}"
            for t in GENEVAL_TAGS:
                hdr += f" {t[:8]:>8}"
            hdr += f" {'overall':>8}"
            print(hdr)
            print("-" * len(hdr))
            for nm, mt in summary.items():
                row = f"{nm:<45}"
                for t in GENEVAL_TAGS:
                    row += (f" {mt[t]['avg_score']:>8.4f}"
                            if t in mt else f" {'N/A':>8}")
                row += f" {mt['overall']['avg_score']:>8.4f}"
                print(row)
            sp = os.path.join(args.output_dir, "geneval_summary.json")
            with open(sp, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary -> {sp}")

    # ================================================================
    # Single-checkpoint mode
    # ================================================================
    else:
        mdl = load_model(
            args.checkpoint_path,
            lora_path=args.lora_path,
            ema_path=args.ema_path,
            gen_modules_path=args.gen_modules_path,
            device=device)

        tag = "baseline"
        if args.lora_path:
            parent = os.path.basename(os.path.dirname(args.lora_path))
            tag = (parent if parent != "lora_adapter"
                   else os.path.basename(
                       os.path.dirname(os.path.dirname(args.lora_path))))
            if args.ema_path:
                tag += "_ema"

        evaluate_geneval(
            mdl, reward_fn, all_meta, vl_proc, path_img,
            step_size, args.num_images_per_prompt, accelerator,
            args.output_dir, args, tag=tag, save_images=save_images)
