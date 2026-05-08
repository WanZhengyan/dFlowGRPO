"""
Evaluate saved GRPO checkpoints (LoRA + optional EMA) on test prompts.

Supports multi-GPU via ``accelerate launch`` (each GPU generates + scores its
own shard; results are gathered on rank 0) as well as single-GPU mode
(``python evaluate_checkpoint.py``).

Supports single-checkpoint and batch-checkpoint modes.

Usage:
  # Single checkpoint, multi-GPU:
  accelerate launch --num_processes 8 evaluate_checkpoint.py \
      --checkpoint_path $CKPT_PATH \
      --image_embedding_path $CKPT_PATH/image_embedding.pt \
      --test_prompts_file /path/to/pickscore/test.txt \
      --lora_path /path/to/lora_adapter \
      --output_dir /path/to/eval_results/step200 \
      --reward_dict '{"pickscore": 1}'

  # Baseline (no LoRA):
  accelerate launch --num_processes 8 evaluate_checkpoint.py \
      --checkpoint_path $CKPT_PATH \
      --image_embedding_path $CKPT_PATH/image_embedding.pt \
      --test_prompts_file /path/to/pickscore/test.txt \
      --output_dir /path/to/eval_results/baseline \
      --reward_dict '{"pickscore": 1}'

  # Batch mode (evaluate baseline + all checkpoints in a directory):
  accelerate launch --num_processes 8 evaluate_checkpoint.py \
      --checkpoint_path $CKPT_PATH \
      --image_embedding_path $CKPT_PATH/image_embedding.pt \
      --test_prompts_file /path/to/pickscore/test.txt \
      --grpo_output_dir /path/to/output_lora_aesthetic \
      --output_dir /path/to/eval_results \
      --reward_dict '{"pickscore": 1}'

  # Single-GPU fallback (works the same way, 1 process):
  python evaluate_checkpoint.py \
      --checkpoint_path $CKPT_PATH \
      --image_embedding_path $CKPT_PATH/image_embedding.pt \
      --test_prompts_file /path/to/pickscore/test.txt \
      --output_dir /path/to/eval_results/baseline \
      --reward_dict '{"pickscore": 1}'
"""

import torch
import argparse
import os
import json
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
    parser = argparse.ArgumentParser(description="Evaluate GRPO checkpoints (multi-GPU)")
    parser.add_argument("--seed", type=int, default=42)

    # ---- Paths ----
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the base FUDOKI model checkpoint directory")
    parser.add_argument("--image_embedding_path", type=str, required=True)
    parser.add_argument("--test_prompts_file", type=str, required=True,
                        help="Path to test prompts text file (one prompt per line)")
    parser.add_argument("--output_dir", type=str, required=True)

    # ---- LoRA / EMA / gen_modules (single checkpoint mode) ----
    parser.add_argument("--lora_path", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--ema_path", type=str, default=None,
                        help="Path to EMA state dict (.pt file)")
    parser.add_argument("--gen_modules_path", type=str, default=None,
                        help="Path to trained gen_modules state dict (.pt file)")

    # ---- Batch mode ----
    parser.add_argument("--grpo_output_dir", type=str, default=None,
                        help="Path to GRPO output dir with checkpoint_epoch*_step* subdirs. "
                             "If set, evaluates baseline + all checkpoints automatically.")
    parser.add_argument("--step_interval", type=int, default=None,
                        help="Only evaluate checkpoints whose step satisfies "
                             "(step - step_start) %% step_interval == 0")
    parser.add_argument("--step_start", type=int, default=None,
                        help="Lower bound (inclusive) on checkpoint step.")
    parser.add_argument("--step_end", type=int, default=None,
                        help="Upper bound (inclusive) on checkpoint step.")
    parser.add_argument("--ema_only", action="store_true",
                        help="In batch mode, only evaluate LoRA+EMA (skip LoRA-only).")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="In batch mode, skip the baseline evaluation.")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip checkpoints whose output dir already contains results.json.")

    # ---- Reward ----
    parser.add_argument("--reward_dict", type=str, default='{"pickscore": 1}')
    parser.add_argument("--reward_on_gpu", action="store_true")

    # ---- Multi-reward mode (generate once, score with many rewards) ----
    parser.add_argument("--multi_rewards", type=str, default=None,
                        help="Comma-separated reward names. When set, ignores "
                             "--reward_dict and scores with each reward. "
                             "Per-reward results go to "
                             "{output_root}/{reward}/{suffix}/results.json.")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Required with --multi_rewards: top-level results dir.")
    parser.add_argument("--suffix", type=str, default=None,
                        help="Required with --multi_rewards: run-tag subdir "
                             "(e.g. 'step2600_ema_16nfe', 'baseline_16nfe').")

    # ---- Sampling / eval settings ----
    parser.add_argument("--discrete_fm_steps", type=int, default=40)
    parser.add_argument("--txt_max_length", type=int, default=500)
    parser.add_argument("--num_prompts", type=int, default=50,
                        help="Number of test prompts to evaluate")
    parser.add_argument("--num_images_per_prompt", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for sampling (unused, kept for compat)")
    parser.add_argument("--reward_batch_size", type=int, default=12,
                        help="Batch size for reward computation")

    # ---- CFG ----
    parser.add_argument("--no_cfg", action="store_true",
                        help="Disable classifier-free guidance")
    parser.add_argument("--cfg_scale", type=float, default=None,
                        help="Override CFG scale (default: 5.0). Ignored if --no_cfg.")

    # ---- Save images ----
    parser.add_argument("--no_save_images", action="store_true",
                        help="Skip saving generated images to disk")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Generation-only pass (shared by single-reward and multi-reward modes)
# ---------------------------------------------------------------------------
@torch.no_grad()
def _generate_images(
    model, eval_prompts, vl_chat_processor, path_img,
    step_size, n_samples_per_prompt, accelerator, args,
):
    """
    Generate images for this rank's shard of prompts. Returns:
        local_imgs:    list of torch.uint8 CPU tensors  (H,W,C)
        local_prompts: list of str  (len == len(local_imgs))

    Each prompt produces `n_samples_per_prompt` consecutive images.
    Images are returned as 0-255 uint8 to keep RAM low when scoring across
    many reward models.
    """
    model.eval()
    device = accelerator.device

    # Resolve CFG
    use_cfg = not getattr(args, "no_cfg", False)
    if use_cfg:
        cfg_model = CFGScaledModel(model=model, g_or_u='generation')
        effective_cfg = args.cfg_scale if args.cfg_scale is not None else CFG_SCALE
    else:
        cfg_model = NoCFGModel(model)
        effective_cfg = 0.0
    accelerator.print(
        f"  CFG: {'ON (scale={:.1f})'.format(effective_cfg) if use_cfg else 'OFF (no_cfg)'}")

    # Shard prompts across GPUs
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    n_total = len(eval_prompts)
    per_rank = (n_total + world_size - 1) // world_size
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n_total)
    my_prompts = eval_prompts[my_start:my_end]

    accelerator.print(
        f"  {n_total} prompts / {world_size} GPUs = ~{per_rank}/GPU, "
        f"{n_samples_per_prompt} imgs/prompt")

    local_imgs = []
    local_prompts = []
    for pi, prompt in enumerate(tqdm(
            my_prompts, desc=f"[GPU{rank} gen]", disable=(rank != 0))):
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
        pix = decode_image_tokens(model, itk)  # [B,3,H,W], float 0..1
        # Convert to uint8 HWC on CPU
        pix_u8 = (pix * 255).round().clamp(0, 255).to(torch.uint8)
        pix_u8 = pix_u8.permute(0, 2, 3, 1).contiguous().cpu()
        for b in range(pix_u8.shape[0]):
            local_imgs.append(pix_u8[b])
            local_prompts.append(prompt)
        del x_init, data_info, final_tokens, itk, pix, pix_u8

    torch.cuda.empty_cache()
    return local_imgs, local_prompts


def _score_and_save(
    reward_fn, reward_name, local_imgs_u8, local_prompts,
    accelerator, save_dir, tag, reward_batch_size, save_images,
):
    """
    Run `reward_fn` on this rank's images, gather, and save results.json on rank 0.
    `local_imgs_u8`: list of uint8 HWC CPU tensors.
    Returns (mean, std) on all ranks.
    """
    rank = accelerator.process_index
    device = accelerator.device
    img_counter = 0
    local_results = []

    if save_images:
        rank_save_dir = os.path.join(save_dir, f"gpu{rank}")
        os.makedirs(rank_save_dir, exist_ok=True)

    n_local = len(local_imgs_u8)
    for s in tqdm(range(0, n_local, reward_batch_size),
                  desc=f"[GPU{rank} {reward_name}]", disable=(rank != 0)):
        e = min(s + reward_batch_size, n_local)
        # Stack -> float [B,3,H,W] in 0..1, matching what reward_fn expects.
        batch_u8 = torch.stack(local_imgs_u8[s:e], dim=0)          # [B,H,W,C] uint8
        batch_float = batch_u8.permute(0, 3, 1, 2).float() / 255.0 # [B,3,H,W]
        batch_prompts = local_prompts[s:e]

        scores_b, _ = reward_fn(batch_float.to(device), batch_prompts, {})

        if isinstance(scores_b, torch.Tensor):
            scores_np = scores_b.cpu().numpy()
        elif isinstance(scores_b, np.ndarray):
            scores_np = scores_b
        else:
            scores_np = np.array(scores_b)

        for i in range(len(batch_prompts)):
            sc = float(scores_np[i])
            local_results.append({"prompt": batch_prompts[i], "score": sc})
            if save_images:
                arr = batch_u8[i].numpy()
                pil = Image.fromarray(arr)
                sp = batch_prompts[i][:80].replace("/", "_").replace(" ", "_")
                fn = f"{img_counter:04d}_r{sc:.3f}_{sp}.png"
                pil.save(os.path.join(rank_save_dir, fn))
            img_counter += 1
        del batch_u8, batch_float

    accelerator.wait_for_everyone()
    all_results_gathered = gather_object(local_results)

    mean_reward, std_reward = 0.0, 0.0
    if accelerator.is_main_process:
        scores_arr = np.array([r["score"] for r in all_results_gathered])
        mean_reward = float(scores_arr.mean()) if len(scores_arr) else 0.0
        std_reward = float(scores_arr.std()) if len(scores_arr) else 0.0
        print(f"  [{tag}|{reward_name}] mean={mean_reward:.4f}, "
              f"std={std_reward:.4f}  (n={len(all_results_gathered)})")

        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump({
                "tag": tag,
                "reward": reward_name,
                "mean": mean_reward,
                "std": std_reward,
                "num_images": len(all_results_gathered),
                "per_image_results": all_results_gathered,
            }, f, indent=2)
        print(f"  Saved -> {save_dir}/results.json")

    accelerator.wait_for_everyone()
    return mean_reward, std_reward


# ---------------------------------------------------------------------------
# Core multi-GPU evaluation loop
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_model(
    model, reward_fn, eval_prompts, vl_chat_processor, path_img,
    step_size, n_samples_per_prompt, accelerator, save_dir, args,
    tag="",
):
    """
    Multi-GPU evaluation: each GPU generates images for its shard of prompts,
    computes rewards locally, then results are gathered on rank 0.

    Returns (mean_reward, std_reward) on all ranks.
    """
    model.eval()
    device = accelerator.device
    save_images = not getattr(args, "no_save_images", False)

    # Resolve CFG
    use_cfg = not getattr(args, "no_cfg", False)
    if use_cfg:
        cfg_model = CFGScaledModel(model=model, g_or_u='generation')
        effective_cfg = args.cfg_scale if args.cfg_scale is not None else CFG_SCALE
    else:
        cfg_model = NoCFGModel(model)
        effective_cfg = 0.0
    accelerator.print(
        f"  CFG: {'ON (scale={:.1f})'.format(effective_cfg) if use_cfg else 'OFF (no_cfg)'}")

    # Shard prompts across GPUs
    world_size = accelerator.num_processes
    rank = accelerator.process_index
    n_total = len(eval_prompts)
    per_rank = (n_total + world_size - 1) // world_size
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n_total)
    my_prompts = eval_prompts[my_start:my_end]

    reward_batch_size = getattr(args, "reward_batch_size", 64)

    accelerator.print(
        f"  {n_total} prompts / {world_size} GPUs = ~{per_rank}/GPU, "
        f"{n_samples_per_prompt} imgs/prompt, "
        f"reward_batch={reward_batch_size}")

    if save_images:
        rank_save_dir = os.path.join(save_dir, f"gpu{rank}")
        os.makedirs(rank_save_dir, exist_ok=True)

    # ---- Each GPU: generate + score ----
    pending_imgs = []
    pending_prompts = []
    local_results = []      # per-image dicts for gathering
    img_counter = 0

    def _flush_pending():
        nonlocal img_counter
        if not pending_imgs:
            return
        batch_imgs = torch.cat(pending_imgs, dim=0)
        batch_prompts = list(pending_prompts)
        pending_imgs.clear()
        pending_prompts.clear()

        scores_b, details_b = reward_fn(batch_imgs, batch_prompts, {})

        if isinstance(scores_b, torch.Tensor):
            scores_np = scores_b.cpu().numpy()
        elif isinstance(scores_b, np.ndarray):
            scores_np = scores_b
        else:
            scores_np = np.array(scores_b)

        for i in range(len(batch_prompts)):
            sc = float(scores_np[i])
            local_results.append({
                "prompt": batch_prompts[i],
                "score": sc,
            })
            if save_images:
                arr = (batch_imgs[i].permute(1, 2, 0).cpu().numpy() * 255
                       ).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                sp = batch_prompts[i][:80].replace("/", "_").replace(" ", "_")
                fn = f"{img_counter:04d}_r{sc:.3f}_{sp}.png"
                pil.save(os.path.join(rank_save_dir, fn))
            img_counter += 1
        del batch_imgs

    for pi, prompt in enumerate(tqdm(
            my_prompts, desc=f"[GPU{rank}]", disable=(rank != 0))):
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
        pix = decode_image_tokens(model, itk)
        pending_imgs.append(pix.cpu())
        pending_prompts.extend([prompt] * n_samples_per_prompt)
        del x_init, data_info, final_tokens, itk, pix

        n_pending = sum(t.shape[0] for t in pending_imgs)
        if n_pending >= reward_batch_size:
            _flush_pending()

    _flush_pending()
    torch.cuda.empty_cache()

    print(f"  [GPU{rank}] scored {len(local_results)} images locally")

    # ---- Gather results to rank 0 ----
    accelerator.wait_for_everyone()
    all_results_gathered = gather_object(local_results)

    mean_reward, std_reward = 0.0, 0.0
    if accelerator.is_main_process:
        all_scores_flat = [r["score"] for r in all_results_gathered]
        scores_arr = np.array(all_scores_flat)
        mean_reward = float(scores_arr.mean())
        std_reward = float(scores_arr.std())

        print(f"\n  Total scored: {len(all_results_gathered)} images")
        print(f"  [{tag}] reward_mean={mean_reward:.4f}, reward_std={std_reward:.4f}")

        os.makedirs(save_dir, exist_ok=True)

        # Save results JSON
        results_log = {
            "tag": tag,
            "mean": mean_reward,
            "std": std_reward,
            "num_prompts": n_total,
            "num_images": len(all_results_gathered),
            "per_image_results": all_results_gathered,
        }
        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results_log, f, indent=2)
        print(f"  Saved -> {save_dir}/results.json")

    accelerator.wait_for_everyone()
    return mean_reward, std_reward


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

    # Load test prompts
    with open(args.test_prompts_file) as f:
        all_prompts = [line.strip() for line in f if line.strip()]
    eval_prompts = all_prompts[:args.num_prompts] if args.num_prompts > 0 else all_prompts
    accelerator.print(f"Loaded {len(eval_prompts)} test prompts (from {len(all_prompts)} total)")

    # Build reward function (skip if multi-reward: built per-reward below)
    score_dict = json.loads(args.reward_dict)
    reward_device = str(device) if args.reward_on_gpu else "cpu"
    if args.multi_rewards:
        reward_fn = None
        accelerator.print(f"Multi-reward mode; rewards={args.multi_rewards}")
    else:
        reward_fn = build_training_reward_fn(score_dict, reward_device=reward_device)
        accelerator.print(f"Reward: {score_dict} on {reward_device}")
    accelerator.print(f"GPUs: {accelerator.num_processes}")

    # Load processor and path
    vl_chat_processor = VLChatProcessor.from_pretrained(args.checkpoint_path)
    path_img = MixtureDiscreteSoftmaxProbPath(
        mode='image', embedding_path=args.image_embedding_path)
    path_img.embedding = path_img.embedding.to(device)

    step_size = 1.0 / args.discrete_fm_steps

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # ---- Batch mode: evaluate baseline + all checkpoints ----
    if args.grpo_output_dir is not None:
        results_summary = {}

        # Bash run_eval.sh names dirs as e.g. step3100_ema_16nfe_cfg.
        # Build alt candidate names so SKIP can also pick those up.
        _nfe_suffix = f"_{args.discrete_fm_steps}nfe"
        _cfg_suffix = "" if getattr(args, "no_cfg", False) else "_cfg"

        def _candidate_dirs(base_tag, step):
            """Return list of subdir names to probe, in priority order."""
            cands = [base_tag]
            if base_tag == "baseline":
                cands.append(f"baseline{_nfe_suffix}{_cfg_suffix}")
            elif step is not None:
                if base_tag.endswith("_ema"):
                    cands.append(f"step{step}_ema{_nfe_suffix}{_cfg_suffix}")
                else:
                    cands.append(f"step{step}{_nfe_suffix}{_cfg_suffix}")
            return cands

        def _find_existing_json(base_tag, step, fname="results.json"):
            for sub in _candidate_dirs(base_tag, step):
                p = os.path.join(args.output_dir, sub, fname)
                if os.path.exists(p):
                    return p
            # Glob fallback — handles older runs whose dir name has a
            # different NFE/CFG suffix (e.g. baseline_8nfe vs current
            # EVAL_FM_STEPS=16 → probe "baseline_16nfe" miss). The first
            # match found is good enough — they all evaluate the same
            # checkpoint, just with different denoise settings.
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
                # When matching by step, ensure the matched dir's step
                # number is exactly `step` (avoid "step100" ⊂ "step1000").
                if want_step is not None:
                    sub = os.path.basename(os.path.dirname(m))
                    mm = _re.search(r"step(\d+)", sub)
                    if not mm or int(mm.group(1)) != want_step:
                        continue
                return m
            return None

        def _load_existing_results(path):
            """Load a previously written results.json; return (mean,std) or None."""
            try:
                with open(path) as f:
                    d = json.load(f)
                return float(d["mean"]), float(d.get("std", 0.0))
            except Exception as e:
                accelerator.print(f"  [WARN] failed to load {path}: {e}")
                return None

        # 1) Baseline (no LoRA)
        if not args.skip_baseline:
            baseline_dir = os.path.join(args.output_dir, "baseline")
            existing_json = (_find_existing_json("baseline", None)
                             if args.skip_existing else None)
            baseline_done = existing_json is not None
            if baseline_done:
                loaded = _load_existing_results(existing_json)
                if loaded is not None and accelerator.is_main_process:
                    results_summary["baseline"] = {
                        "mean": loaded[0], "std": loaded[1]}
                    accelerator.print(
                        f"\n  [SKIP] baseline (results exist at "
                        f"{os.path.relpath(existing_json, args.output_dir)}; "
                        f"mean={loaded[0]:.4f})")
                else:
                    accelerator.print(
                        f"\n  [SKIP] baseline (results exist at "
                        f"{os.path.relpath(existing_json, args.output_dir)})")
            else:
                accelerator.print("\n" + "=" * 60)
                accelerator.print("Evaluating BASELINE (no LoRA)")
                accelerator.print("=" * 60)
                torch.manual_seed(args.seed)
                torch.cuda.manual_seed(args.seed)
                model = load_model(args.checkpoint_path, device=device)
                mean_r, std_r = evaluate_model(
                    model, reward_fn, eval_prompts, vl_chat_processor, path_img,
                    step_size, args.num_images_per_prompt, accelerator,
                    baseline_dir, args, tag="baseline",
                )
                if accelerator.is_main_process:
                    results_summary["baseline"] = {"mean": mean_r, "std": std_r}
                del model
                torch.cuda.empty_cache()

        # 2) Discover and evaluate each checkpoint
        ckpts = discover_checkpoints(
            args.grpo_output_dir, args.step_interval,
            step_start=args.step_start, step_end=args.step_end)
        accelerator.print(
            f"\n{len(ckpts)} checkpoints"
            + (f" (every {args.step_interval} steps)"
               if args.step_interval else ""))
        for n, _, s in ckpts:
            accelerator.print(f"  {n}  (step {s})")

        for ckpt_name, ckpt_dir, step in ckpts:
            lora_p = os.path.join(ckpt_dir, "lora_adapter")
            ema_p = os.path.join(ckpt_dir, "ema_state.pt")
            gen_p = os.path.join(ckpt_dir, "gen_modules_state.pt")
            gen_p = gen_p if os.path.exists(gen_p) else None

            # LoRA only (skip if --ema_only)
            if not args.ema_only and os.path.exists(lora_p):
                lora_dir = os.path.join(args.output_dir, ckpt_name)
                existing_json = (_find_existing_json(ckpt_name, step)
                                 if args.skip_existing else None)
                lora_done = existing_json is not None
                if lora_done:
                    loaded = _load_existing_results(existing_json)
                    rel = os.path.relpath(existing_json, args.output_dir)
                    if loaded is not None and accelerator.is_main_process:
                        results_summary[ckpt_name] = {
                            "mean": loaded[0], "std": loaded[1]}
                        accelerator.print(
                            f"\n  [SKIP] {ckpt_name} (results exist at "
                            f"{rel}; mean={loaded[0]:.4f})")
                    else:
                        accelerator.print(
                            f"\n  [SKIP] {ckpt_name} (results exist at {rel})")
                else:
                    accelerator.print("\n" + "=" * 60)
                    accelerator.print(
                        f"Evaluating {ckpt_name} "
                        f"(LoRA{' + gen_modules' if gen_p else ''})")
                    accelerator.print("=" * 60)
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed(args.seed)
                    model = load_model(
                        args.checkpoint_path, lora_path=lora_p,
                        gen_modules_path=gen_p, device=device)
                    mean_r, std_r = evaluate_model(
                        model, reward_fn, eval_prompts, vl_chat_processor, path_img,
                        step_size, args.num_images_per_prompt, accelerator,
                        lora_dir, args, tag=ckpt_name,
                    )
                    if accelerator.is_main_process:
                        results_summary[ckpt_name] = {"mean": mean_r, "std": std_r}
                    del model
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
                    loaded = _load_existing_results(existing_json)
                    rel = os.path.relpath(existing_json, args.output_dir)
                    if loaded is not None and accelerator.is_main_process:
                        results_summary[ema_tag] = {
                            "mean": loaded[0], "std": loaded[1]}
                        accelerator.print(
                            f"\n  [SKIP] {ema_tag} (results exist at "
                            f"{rel}; mean={loaded[0]:.4f})")
                    else:
                        accelerator.print(
                            f"\n  [SKIP] {ema_tag} (results exist at {rel})")
                else:
                    accelerator.print("\n" + "=" * 60)
                    accelerator.print(f"Evaluating {ema_tag}")
                    accelerator.print("=" * 60)
                    torch.manual_seed(args.seed)
                    torch.cuda.manual_seed(args.seed)
                    if has_ema_lora:
                        model = load_model(args.checkpoint_path,
                                           lora_path=ema_lora_p,
                                           gen_modules_path=gen_p, device=device)
                    else:
                        model = load_model(args.checkpoint_path, lora_path=lora_p,
                                           ema_path=ema_p, gen_modules_path=gen_p,
                                           device=device)
                    mean_r, std_r = evaluate_model(
                        model, reward_fn, eval_prompts, vl_chat_processor, path_img,
                        step_size, args.num_images_per_prompt, accelerator,
                        ema_dir, args, tag=ema_tag,
                    )
                    if accelerator.is_main_process:
                        results_summary[ema_tag] = {"mean": mean_r, "std": std_r}
                    del model
                    torch.cuda.empty_cache()

        # Print summary
        if accelerator.is_main_process:
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"{'Checkpoint':<45} {'Mean':>8} {'Std':>8}")
            print("-" * 63)
            for name, res in results_summary.items():
                print(f"{name:<45} {res['mean']:>8.4f} {res['std']:>8.4f}")

            summary_path = os.path.join(args.output_dir, "summary.json")
            os.makedirs(os.path.dirname(summary_path), exist_ok=True)
            with open(summary_path, "w") as f:
                json.dump(results_summary, f, indent=2)
            print(f"\nSummary saved to {summary_path}")

    # ---- Single checkpoint mode ----
    else:
        model = load_model(
            args.checkpoint_path,
            lora_path=args.lora_path,
            ema_path=args.ema_path,
            gen_modules_path=args.gen_modules_path,
            device=device,
        )

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

        # -------------------------------------------------------------------
        # Multi-reward mode: generate ONCE, score with multiple reward models.
        # Output goes to {output_root}/{reward}/{suffix}/results.json
        # (matches plots/plot_drawbench_hexagon.py's expected layout).
        # -------------------------------------------------------------------
        if args.multi_rewards:
            assert args.output_root and args.suffix, \
                "--multi_rewards requires --output_root and --suffix"
            reward_names = [r.strip() for r in args.multi_rewards.split(",") if r.strip()]
            accelerator.print(f"Multi-reward: {reward_names}")

            # 1) Generate once
            step_size_local = 1.0 / args.discrete_fm_steps
            local_imgs, local_prompts = _generate_images(
                model, eval_prompts, vl_chat_processor, path_img,
                step_size_local, args.num_images_per_prompt, accelerator, args,
            )
            accelerator.print(f"  [GPU{accelerator.process_index}] "
                              f"generated {len(local_imgs)} images locally")

            # Free FUDOKI before loading reward models
            del model
            torch.cuda.empty_cache()

            reward_device = str(device) if args.reward_on_gpu else "cpu"
            save_images_once = not getattr(args, "no_save_images", False)

            for ri, rname in enumerate(reward_names):
                accelerator.print("\n" + "=" * 60)
                accelerator.print(f"Scoring with reward: {rname}")
                accelerator.print("=" * 60)
                try:
                    one_reward_fn = build_training_reward_fn(
                        {rname: 1.0}, reward_device=reward_device)
                except Exception as e:
                    accelerator.print(f"  [WARN] failed to build '{rname}': {e}")
                    continue

                save_dir = os.path.join(args.output_root, rname, args.suffix)
                # Only save images on the FIRST reward — same images, don't duplicate.
                save_imgs_this = save_images_once and (ri == 0)
                try:
                    _score_and_save(
                        one_reward_fn, rname, local_imgs, local_prompts,
                        accelerator, save_dir, tag=args.suffix,
                        reward_batch_size=args.reward_batch_size,
                        save_images=save_imgs_this,
                    )
                except Exception as e:
                    accelerator.print(f"  [ERROR] '{rname}' scoring failed: {e}")
                # Release scorer GPU memory between rewards
                del one_reward_fn
                torch.cuda.empty_cache()
        else:
            evaluate_model(
                model, reward_fn, eval_prompts, vl_chat_processor, path_img,
                step_size, args.num_images_per_prompt, accelerator,
                args.output_dir, args, tag=tag,
            )
