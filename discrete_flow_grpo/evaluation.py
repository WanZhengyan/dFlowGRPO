"""
Evaluation functions shared between training-time eval and standalone GenEval eval.

Contains:
  - evaluate()              — training-time eval (sample + reward on GPU)
  - evaluate_geneval()      — standalone GenEval eval (interleaved generate + score)
  - compute_geneval_metrics() / print_metrics() — GenEval metric aggregation
"""

import os
import json
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from accelerate.utils import gather_object

from config import VOCABULARY_SIZE_IMG, GENEVAL_TAGS
from model_utils import decode_image_tokens, build_data_info, NoCFGModel
from sampling import sample_only
from fudoki.eval_loop import CFGScaledModel
from config import CFG_SCALE


# ---------------------------------------------------------------------------
# GenEval metrics
# ---------------------------------------------------------------------------
def compute_geneval_metrics(all_results):
    """Compute per-tag and overall GenEval metrics from a list of result dicts."""
    per_tag = defaultdict(lambda: {"strict": [], "correct": [], "score": []})
    for r in all_results:
        per_tag[r["tag"]]["strict"].append(r["strict_correct"])
        per_tag[r["tag"]]["correct"].append(r["correct"])
        per_tag[r["tag"]]["score"].append(r["score"])

    metrics = {}
    ov_s, ov_c, ov_sc = [], [], []
    for tag in GENEVAL_TAGS:
        if tag not in per_tag:
            continue
        d = per_tag[tag]
        n = len(d["strict"])
        metrics[tag] = {
            "count": n,
            "strict_accuracy": sum(d["strict"]) / n if n else 0,
            "accuracy": sum(d["correct"]) / n if n else 0,
            "avg_score": sum(d["score"]) / n if n else 0,
        }
        ov_s += d["strict"]
        ov_c += d["correct"]
        ov_sc += d["score"]
    nt = len(ov_s)
    metrics["overall"] = {
        "count": nt,
        "strict_accuracy": sum(ov_s) / nt if nt else 0,
        "accuracy": sum(ov_c) / nt if nt else 0,
        "avg_score": sum(ov_sc) / nt if nt else 0,
    }
    return metrics


def print_metrics(metrics, tag=""):
    """Pretty-print GenEval metrics table."""
    print(f"\n  [{tag}] GenEval Results:")
    print(f"  {'Category':<20} {'Count':>6} {'Strict':>10} "
          f"{'Accuracy':>10} {'AvgScore':>10}")
    print(f"  {'-' * 58}")
    for cat in GENEVAL_TAGS + ["overall"]:
        if cat not in metrics:
            continue
        m = metrics[cat]
        print(f"  {cat:<20} {m['count']:>6} "
              f"{m['strict_accuracy']:>10.4f} "
              f"{m['accuracy']:>10.4f} "
              f"{m['avg_score']:>10.4f}")


# ---------------------------------------------------------------------------
# Standalone GenEval evaluation (multi-GPU, interleaved generate + score)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_geneval(
    model, reward_fn, eval_metadatas, vl_chat_processor, path_img,
    step_size, n_images_per_prompt, accelerator, save_dir, args,
    tag="", save_images=True,
):
    """
    Multi-GPU GenEval evaluation — each GPU generates AND scores independently.

    Each GPU:
      1. Generates images for its shard of prompts
      2. Sends images to reward server for scoring as soon as a batch is ready
    Finally, rank 0 gathers results and computes aggregate metrics.
    """
    model.eval()
    device = accelerator.device

    # Resolve CFG
    use_cfg = not getattr(args, "no_cfg", False)
    if use_cfg:
        cfg_model = CFGScaledModel(model=model, g_or_u="generation")
        effective_cfg = args.cfg_scale if args.cfg_scale is not None else CFG_SCALE
    else:
        cfg_model = NoCFGModel(model)
        effective_cfg = 0.0
    accelerator.print(
        f"  CFG: {'ON (scale={:.1f})'.format(effective_cfg) if use_cfg else 'OFF (no_cfg)'}")

    world_size = accelerator.num_processes
    rank = accelerator.process_index
    n_total = len(eval_metadatas)
    per_rank = (n_total + world_size - 1) // world_size
    my_start = rank * per_rank
    my_end = min(my_start + per_rank, n_total)
    my_metas = eval_metadatas[my_start:my_end]

    reward_batch_size = getattr(args, "reward_batch_size", 12)

    accelerator.print(
        f"  {n_total} prompts / {world_size} GPUs = ~{per_rank}/GPU, "
        f"{n_images_per_prompt} imgs/prompt, "
        f"reward_batch={reward_batch_size}")

    if save_images:
        rank_save_dir = os.path.join(save_dir, f"gpu{rank}")
        os.makedirs(rank_save_dir, exist_ok=True)

    # ---- Each GPU: generate + score interleaved ----
    pending_imgs = []
    pending_metas = []
    local_results = []
    img_counter = 0

    def _flush_pending():
        nonlocal img_counter
        if not pending_imgs:
            return
        batch_imgs = torch.cat(pending_imgs, dim=0)
        batch_metas = list(pending_metas)
        pending_imgs.clear()
        pending_metas.clear()

        result = reward_fn(batch_imgs, batch_metas, only_strict=False)

        for i in range(len(batch_metas)):
            m = batch_metas[i]
            sc = result["scores"][i]
            stk = result["strict_rewards"][i]
            crk = result["rewards"][i]
            local_results.append({
                "tag": m["tag"], "prompt": m["prompt"],
                "strict_correct": stk, "correct": crk, "score": sc,
            })
            if save_images:
                arr = (batch_imgs[i].permute(1, 2, 0).numpy() * 255
                       ).clip(0, 255).astype(np.uint8)
                pil = Image.fromarray(arr)
                sp = m["prompt"][:60].replace("/", "_").replace(" ", "_")
                fn = (f"{img_counter:05d}_{m['tag']}_"
                      f"{'T' if stk else 'F'}_s{sc:.2f}_{sp}.png")
                pil.save(os.path.join(rank_save_dir, fn))
            img_counter += 1
        del batch_imgs

    for pi, meta in enumerate(tqdm(
            my_metas, desc=f"[GPU{rank}]", disable=(rank != 0))):
        gpi = my_start + pi
        torch.manual_seed(args.seed + gpi)
        torch.cuda.manual_seed(args.seed + gpi)

        x_init, data_info = build_data_info(
            meta["prompt"], n_images_per_prompt, vl_chat_processor,
            path_img, args.txt_max_length, device)
        final_tokens, image_mask = sample_only(
            model=cfg_model, path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
            x_init=x_init, data_info=data_info,
            step_size=step_size, cfg_scale=effective_cfg,
            n_mc=1, device=device)
        itk = final_tokens[image_mask == 1].reshape(n_images_per_prompt, -1)
        pix = decode_image_tokens(model, itk)
        pending_imgs.append(pix.cpu())
        pending_metas.extend([meta] * n_images_per_prompt)
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

    metrics = {}
    if accelerator.is_main_process:
        print(f"\n  Total scored: {len(all_results_gathered)} images")
        os.makedirs(save_dir, exist_ok=True)

        metrics = compute_geneval_metrics(all_results_gathered)
        print_metrics(metrics, tag=tag)

        with open(os.path.join(save_dir, "geneval_results.json"), "w") as f:
            json.dump({"tag": tag, "metrics": metrics,
                       "per_image_results": all_results_gathered}, f, indent=2)
        print(f"  Saved -> {save_dir}/geneval_results.json")

    accelerator.wait_for_everyone()
    return metrics


# ---------------------------------------------------------------------------
# Training-time evaluation (sample + reward, all local)
# ---------------------------------------------------------------------------
@torch.no_grad()
def evaluate_training(
    model, cfg_model, raw_model, reward_fn,
    eval_prompts, vl_chat_processor, path_img,
    step_size, n_samples_per_prompt, device, accelerator,
    global_step, save_dir=None, eval_metadatas=None,
    txt_max_length=500, cfg_scale=5.0, reward_on_gpu=False,
):
    """
    Training-time evaluation: sample images and compute rewards.

    Each GPU evaluates its own subset of prompts, then rewards are gathered.

    Returns: (eval_mean, eval_std)
    """
    model.eval()

    # Fix random seed for reproducible eval
    rng_state_cpu = torch.random.get_rng_state()
    rng_state_cuda = torch.cuda.get_rng_state(device)
    torch.manual_seed(42 + accelerator.process_index)
    torch.cuda.manual_seed(42 + accelerator.process_index)

    # Phase 1: GPU sampling + decoding
    all_pixel_imgs = []
    all_prompts_flat = []

    for prompt in eval_prompts:
        x_init, data_info = build_data_info(
            prompt, n_samples_per_prompt, vl_chat_processor, path_img,
            txt_max_length, device)
        final_tokens, image_mask = sample_only(
            model=cfg_model, path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
            x_init=x_init, data_info=data_info,
            step_size=step_size, cfg_scale=cfg_scale,
            n_mc=1, device=device)
        img_tokens = final_tokens[image_mask == 1].reshape(n_samples_per_prompt, -1)
        pixel_imgs = decode_image_tokens(raw_model, img_tokens)
        all_pixel_imgs.append(pixel_imgs if reward_on_gpu else pixel_imgs.cpu())
        all_prompts_flat.extend([prompt] * n_samples_per_prompt)
        del x_init, data_info, final_tokens, img_tokens
        if not reward_on_gpu:
            del pixel_imgs

    # Phase 2: Reward computation
    all_pixel_imgs_cat = torch.cat(all_pixel_imgs, dim=0)
    del all_pixel_imgs
    if eval_metadatas is not None:
        eval_meta_flat = []
        for m in eval_metadatas:
            eval_meta_flat.extend([m] * n_samples_per_prompt)
    else:
        eval_meta_flat = {}
    scores, _ = reward_fn(all_pixel_imgs_cat, all_prompts_flat, eval_meta_flat)

    if isinstance(scores, torch.Tensor):
        all_rewards = scores.cpu().float()
    elif isinstance(scores, np.ndarray):
        all_rewards = torch.from_numpy(scores).float()
    elif isinstance(scores, list):
        all_rewards = torch.tensor(scores, dtype=torch.float32)
    else:
        all_rewards = torch.tensor(scores, dtype=torch.float32)

    # Phase 3: Save eval images
    all_pixel_imgs_cpu = all_pixel_imgs_cat.cpu()
    del all_pixel_imgs_cat
    torch.cuda.empty_cache()

    if save_dir is not None and accelerator.is_main_process:
        step_dir = os.path.join(save_dir, f"step_{global_step}")
        os.makedirs(step_dir, exist_ok=True)
        for i in range(all_pixel_imgs_cpu.shape[0]):
            img_np = (all_pixel_imgs_cpu[i].permute(1, 2, 0).numpy() * 255
                      ).clip(0, 255).astype(np.uint8)
            pil_img = Image.fromarray(img_np)
            safe_prompt = all_prompts_flat[i][:80].replace("/", "_").replace(" ", "_")
            reward_val = all_rewards[i].item()
            fname = f"{i:03d}_r{reward_val:.3f}_{safe_prompt}.png"
            pil_img.save(os.path.join(step_dir, fname))
        accelerator.print(
            f"  Saved {all_pixel_imgs_cpu.shape[0]} eval images to {step_dir}")

    del all_pixel_imgs_cpu

    # Gather across GPUs
    all_rewards_gpu = all_rewards.to(device)
    gathered = accelerator.gather(all_rewards_gpu)

    eval_mean = gathered.float().mean().item()
    eval_std = gathered.float().std().item()

    # Restore RNG state
    torch.random.set_rng_state(rng_state_cpu)
    torch.cuda.set_rng_state(rng_state_cuda, device)

    model.train()
    return eval_mean, eval_std
