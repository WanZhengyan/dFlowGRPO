"""
Discrete Flow on-policy DPO training for FUDOKI — main entry point.

Mirrors train_GRPO.py but replaces the PPO surrogate loss with the
on-policy DPO objective:

    L = -log σ( (β' / (K-1)) · Σ_k [ log π*(x⁺_{t_k}|x⁺_{t_{k-1}})/π_old(...)
                                    - log π*(x⁻_{t_k}|x⁻_{t_{k-1}})/π_old(...) ] )

Sampling, EMA, old-policy snapshot, LoRA, CFG, dataset selection,
checkpoint resume etc. are reused unchanged from the GRPO setup.
"""

import os
import json
import time as _time
import torch
import wandb
from math import ceil as _ceil
from tqdm import tqdm
from accelerate import Accelerator
from collections import defaultdict as _defaultdict

# Patch torch.load for older checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from fudoki.janus.models import VLChatProcessor

from config import VOCABULARY_SIZE_IMG, IMG_LEN
from model_utils import decode_image_tokens, build_data_info, build_data_info_multi
from reward_utils import resolve_training_reward
from evaluation import evaluate_training
from sampling import sample_with_log_prob
from utils import prompts_to_tensor

from train_args_dpo import parse_arguments
from train_setup import (
    resolve_train_cfg_scale, setup_model, make_disable_adapter_ctx,
    setup_optimizer, setup_ema, validate_ema_args, setup_old_policy,
    resume_from_checkpoint, setup_cfg_wrapper, setup_datasets, setup_eval_split,
)
from training_dpo import (
    parse_train_steps, merge_sampling_results, collect_prompts,
    build_pairs, train_on_prompt,
)


# ===================================================================
# Initialization
# ===================================================================
args = parse_arguments()
TRAIN_CFG_SCALE = resolve_train_cfg_scale(args)

accelerator = Accelerator()
device = accelerator.device
torch.manual_seed(args.seed)

# ---- Model ----
model, _gen_ref_state = setup_model(args, device, accelerator)
disable_adapter_and_gen_modules = make_disable_adapter_ctx(_gen_ref_state)

# ---- Tokenizer & path ----
vl_chat_processor = VLChatProcessor.from_pretrained(args.checkpoint_path)
path_img = MixtureDiscreteSoftmaxProbPath(mode='image', embedding_path=args.image_embedding_path)
path_img.embedding = path_img.embedding.to(device)

# ---- Reward ----
reward_fn, _score_dict, _reward_desc = resolve_training_reward(args, device)
accelerator.print(_reward_desc)

# ---- Datasets ----
dataloader, eval_dataset, _use_geneval_metadata, _prompt_dataset_name = setup_datasets(args, accelerator)

# ---- Optimizer ----
optimizer = setup_optimizer(model, args, accelerator)
model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)

# ---- EMA ----
trainable_params = [p for p in accelerator.unwrap_model(model).parameters() if p.requires_grad]
ema = setup_ema(args, trainable_params, device, accelerator)
validate_ema_args(args, ema, accelerator)

# ---- Resume ----
_resumed_global_step = resume_from_checkpoint(args, accelerator, model, ema)

# ---- Old-policy snapshot ----
_old_policy_params = setup_old_policy(args, trainable_params, accelerator)
_old_policy_temp = None

# ---- CFG wrapper ----
diff_cfg_model, raw_model = setup_cfg_wrapper(model, accelerator, device)

# ---- Eval split ----
eval_prompts_list, eval_metadatas_list, n_prompts_per_gpu = setup_eval_split(
    eval_dataset, _use_geneval_metadata, accelerator, args)

# ---- Output dir ----
save_path = args.output_dir
os.makedirs(save_path, exist_ok=True)
run_tag = args.wandb_name if args.wandb_name else "dpo"

if accelerator.is_main_process:
    wandb.init(project="discrete_flow_dpo", name=run_tag,
               settings=wandb.Settings(init_timeout=300), config=vars(args))

# ---- Derived constants ----
G = args.group_size
P = args.prompts_per_sample_batch
sample_batch_size = args.sample_batch_size if args.sample_batch_size is not None else G
n_mc = args.n_mc
step_size = 1.0 / args.discrete_fm_steps
kl_beta = args.kl_beta
beta_dpo = args.beta_dpo
train_batch_size = args.train_batch_size
grad_accum_steps = args.gradient_accumulation_steps
num_inner_updates = args.num_inner_updates
num_epochs = args.num_epochs

K_total = args.discrete_fm_steps
_max_train_step = K_total - 1 if args.include_last_step else K_total - 2
_n_total_trainable = _max_train_step + 1

train_step_set = set(parse_train_steps(args.train_steps, K_total, _max_train_step))
n_train_steps = len(train_step_set)

global_step = _resumed_global_step
_total_prompts_consumed = _resumed_global_step * grad_accum_steps * P


# ===================================================================
# Print configuration summary
# ===================================================================
if sample_batch_size < G:
    n_rounds = _ceil(G / sample_batch_size)
    accelerator.print(f"Sampling: group_size={G}, sample_batch_size={sample_batch_size}, "
                      f"{n_rounds} rounds per step")
else:
    accelerator.print(f"Sampling: group_size={G}, single round")

accelerator.print(f"DPO: β'={beta_dpo}, num_pairs={'auto(G/2)' if args.dpo_num_pairs<=0 else args.dpo_num_pairs}")
accelerator.print(f"CFG: train={TRAIN_CFG_SCALE}{' (--no_cfg halves forwards)' if args.no_cfg else ''}")
accelerator.print(f"Gradient accumulation: {grad_accum_steps} sampling calls per optimizer step "
                  f"(each call = {P} prompt(s) × {G} images = {P*G} images per GPU)")
if kl_beta > 0:
    if args.use_lora and not args.kl_old_policy:
        accelerator.print(f"KL penalty: β={kl_beta} (KL vs π_ref via disable_adapter)")
    else:
        accelerator.print(f"KL penalty: β={kl_beta} (log-ratio magnitude vs π_old)")
else:
    accelerator.print("KL penalty: disabled (β=0)")
if args.include_last_step:
    accelerator.print(f"Include last step: ENABLED (steps 0..{K_total-1})")
accelerator.print(f"Train steps: {sorted(train_step_set)} ({n_train_steps}/{_n_total_trainable})")


# ===================================================================
# Step-0 baseline evaluation
# ===================================================================
if _resumed_global_step == 0:
    accelerator.print("Running step-0 baseline evaluation...")
    eval_mean_0, eval_std_0 = evaluate_training(
        model=model, cfg_model=diff_cfg_model,
        raw_model=accelerator.unwrap_model(model),
        reward_fn=reward_fn, eval_prompts=eval_prompts_list,
        vl_chat_processor=vl_chat_processor, path_img=path_img,
        step_size=step_size, n_samples_per_prompt=args.eval_samples_per_prompt,
        device=device, accelerator=accelerator, global_step=0,
        save_dir=os.path.join(save_path, f"eval_images_{run_tag}"),
        eval_metadatas=eval_metadatas_list,
        txt_max_length=args.txt_max_length, cfg_scale=TRAIN_CFG_SCALE,
        reward_on_gpu=args.reward_on_gpu,
    )
    if accelerator.is_main_process:
        wandb.log({"eval/reward_mean": eval_mean_0, "eval/reward_std": eval_std_0}, step=0)
        accelerator.print(f"  [Baseline step 0] reward_mean={eval_mean_0:.4f}, reward_std={eval_std_0:.4f}")
else:
    accelerator.print(f"Skipping step-0 baseline eval (resuming from step {_resumed_global_step})")


# ===================================================================
# Training loop
# ===================================================================
for epoch in range(num_epochs):
    _items_per_epoch = len(dataloader)

    if _total_prompts_consumed > 0:
        _start_epoch = _total_prompts_consumed // _items_per_epoch
        _start_epoch_skip = _total_prompts_consumed % _items_per_epoch
    else:
        _start_epoch, _start_epoch_skip = 0, 0

    if epoch < _start_epoch:
        accelerator.print(f"Epoch {epoch+1}/{num_epochs} — skipped (already completed)")
        continue

    _skip_steps = _start_epoch_skip if epoch == _start_epoch else 0

    accelerator.print(f"Epoch {epoch+1}/{num_epochs}"
                      + (f" (skipping {_skip_steps}/{_items_per_epoch} items)" if _skip_steps > 0 else ""))
    dl_iter = iter(dataloader)
    pbar = tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}",
                disable=not accelerator.is_main_process)

    prompts_since_step = 0
    accum_metrics = _defaultdict(float)
    accum_reward_log = {}
    accum_rewards_list = []
    dl_consumed = 0

    while True:
        prompt_list, metadata_list = collect_prompts(dl_iter, P, use_geneval_metadata=_use_geneval_metadata)
        if prompt_list is None:
            break
        cur_P = len(prompt_list)
        dl_consumed += cur_P
        pbar.update(cur_P)

        if dl_consumed <= _skip_steps:
            continue

        total_G = cur_P * G
        _t0 = _time.time()

        # ============ Phase 1: Sampling (no grad) ============
        model.eval()
        raw_model = accelerator.unwrap_model(model)
        if prompts_since_step == 0:
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        _t_sample_start = _time.time()

        _using_ema_for_sampling = args.sample_with_ema and ema is not None
        if _using_ema_for_sampling:
            ema.copy_ema_to(trainable_params, store_temp=True)

        _using_old_policy_for_sampling = (
            _old_policy_params is not None and not _using_ema_for_sampling
        )
        if _using_old_policy_for_sampling:
            _old_policy_temp = [p.data.clone() for p in trainable_params]
            for p, old_p in zip(trainable_params, _old_policy_params):
                p.data.copy_(old_p)

        # Build multi-prompt batch (identical to GRPO sampling logic)
        if cur_P == 1:
            prompt = prompt_list[0]
            if sample_batch_size >= G:
                x_init, data_info = build_data_info(
                    prompt, G, vl_chat_processor, path_img, args.txt_max_length, device)
                sr = sample_with_log_prob(
                    model=diff_cfg_model, path_img=path_img,
                    vocabulary_size_img=VOCABULARY_SIZE_IMG,
                    x_init=x_init, data_info=data_info,
                    step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
                    n_mc=n_mc, device=device, offload_to_cpu=False,
                )
                del x_init
            else:
                sr_list = []
                remaining = G
                while remaining > 0:
                    bs = min(sample_batch_size, remaining)
                    x_init_r, data_info_r = build_data_info(
                        prompt, bs, vl_chat_processor, path_img, args.txt_max_length, device)
                    sr_r = sample_with_log_prob(
                        model=diff_cfg_model, path_img=path_img,
                        vocabulary_size_img=VOCABULARY_SIZE_IMG,
                        x_init=x_init_r, data_info=data_info_r,
                        step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
                        n_mc=n_mc, device=device, offload_to_cpu=False,
                    )
                    sr_list.append(sr_r)
                    del x_init_r, data_info_r
                    remaining -= bs
                sr = merge_sampling_results(sr_list)
                del sr_list
                _, data_info = build_data_info(
                    prompt, G, vl_chat_processor, path_img, args.txt_max_length, device)
        else:
            if sample_batch_size >= total_G:
                x_init, data_info = build_data_info_multi(
                    prompt_list, G, vl_chat_processor, path_img, args.txt_max_length, device)
                sr = sample_with_log_prob(
                    model=diff_cfg_model, path_img=path_img,
                    vocabulary_size_img=VOCABULARY_SIZE_IMG,
                    x_init=x_init, data_info=data_info,
                    step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
                    n_mc=n_mc, device=device, offload_to_cpu=False,
                )
                del x_init
            else:
                sr_list = []
                remaining = total_G
                offset = 0
                while remaining > 0:
                    bs = min(sample_batch_size, remaining)
                    sub_prompts = []
                    for idx in range(offset, offset + bs):
                        sub_prompts.append(prompt_list[idx // G])
                    dummy_img_tokens_r = torch.zeros(bs, IMG_LEN, dtype=torch.long, device=device)
                    t_r = torch.zeros(bs, device=device)
                    input_ids_r, _, data_info_r = prompts_to_tensor(
                        prompts=sub_prompts, vl_chat_processor=vl_chat_processor,
                        path=path_img, t=t_r, g_or_u="generation",
                        txt_max_length=args.txt_max_length, IMG_LEN=IMG_LEN,
                        img_tokens=dummy_img_tokens_r, device=device,
                    )
                    img_mask_r = data_info_r["image_token_mask"]
                    x_init_r = input_ids_r.clone()
                    x_0_r = torch.randint(0, VOCABULARY_SIZE_IMG, (bs, IMG_LEN),
                                          dtype=torch.long, device=device)
                    x_init_r[img_mask_r == 1] = x_0_r.flatten()
                    sr_r = sample_with_log_prob(
                        model=diff_cfg_model, path_img=path_img,
                        vocabulary_size_img=VOCABULARY_SIZE_IMG,
                        x_init=x_init_r, data_info=data_info_r,
                        step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
                        n_mc=n_mc, device=device, offload_to_cpu=False,
                    )
                    sr_list.append(sr_r)
                    del x_init_r, data_info_r
                    remaining -= bs
                    offset += bs
                sr = merge_sampling_results(sr_list)
                del sr_list
                _, data_info = build_data_info_multi(
                    prompt_list, G, vl_chat_processor, path_img, args.txt_max_length, device)

        torch.cuda.synchronize()
        _t_sample_end = _time.time()

        # ============ Phase 1.5: Compute rewards ============
        image_mask = sr["image_mask"]
        img_tokens = sr["final_tokens"][image_mask == 1].reshape(total_G, -1)
        flat_prompts = []
        for p in prompt_list:
            flat_prompts.extend([p] * G)
        if metadata_list is not None:
            flat_metadata = []
            for m in metadata_list:
                flat_metadata.extend([m] * G)
        else:
            flat_metadata = {}
        with torch.no_grad():
            pixel_imgs = decode_image_tokens(raw_model, img_tokens)
            scores, _ = reward_fn(pixel_imgs, flat_prompts, flat_metadata)
            del pixel_imgs
            if isinstance(scores, torch.Tensor):
                rewards = scores.to(device).float()
            else:
                rewards = torch.tensor(scores, device=device, dtype=torch.float32)
        del img_tokens
        del sr["final_tokens"]
        torch.cuda.empty_cache()

        K = len(sr["all_x1_samples"])
        _t_reward_end = _time.time()

        # Restore live θ after sampling
        if _using_ema_for_sampling:
            ema.copy_temp_to(trainable_params)
        if _using_old_policy_for_sampling and _old_policy_temp is not None:
            for p, live_p in zip(trainable_params, _old_policy_temp):
                p.data.copy_(live_p)
            _old_policy_temp = None

        # ============ Phase 2: Build (winner, loser) pairs ============
        pos_idx, neg_idx, _n_filtered_prompts = build_pairs(
            rewards=rewards, cur_P=cur_P, G=G,
            num_pairs=args.dpo_num_pairs,
            filter_zero_std=args.filter_zero_std,
        )
        n_pairs = pos_idx.shape[0]
        _all_filtered = (n_pairs == 0)

        # ============ Phase 3: Train ============
        model.train()
        _t_train_start = _time.time()

        if not _all_filtered:
            prompts_since_step += 1
            is_last_in_accum = (prompts_since_step >= grad_accum_steps)

            chunk_size_per_update = max(1, n_pairs // num_inner_updates)
            perm = torch.randperm(n_pairs, device=device)
            pos_perm = pos_idx[perm]
            neg_perm = neg_idx[perm]

            for inner_update in range(num_inner_updates):
                start = inner_update * chunk_size_per_update
                end = n_pairs if (inner_update == num_inner_updates - 1) else start + chunk_size_per_update
                pos_chunk = pos_perm[start:end]
                neg_chunk = neg_perm[start:end]

                metrics = train_on_prompt(
                    sr=sr, data_info=data_info,
                    pos_idx=pos_chunk, neg_idx=neg_chunk, K=K,
                    is_last_prompt_in_accum=(is_last_in_accum and inner_update == num_inner_updates - 1),
                    diff_cfg_model=diff_cfg_model, model=model,
                    accelerator=accelerator, device=device,
                    args=args, train_step_set=train_step_set,
                    n_train_steps=n_train_steps, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, num_inner_updates=num_inner_updates,
                    beta_dpo=beta_dpo, kl_beta=kl_beta, TRAIN_CFG_SCALE=TRAIN_CFG_SCALE,
                    disable_adapter_and_gen_modules=disable_adapter_and_gen_modules,
                    raw_model=raw_model,
                )
                for key in ("loss", "kl_loss", "dpo_acc_count", "dpo_total",
                             "delta_sum", "delta_sq_sum", "abs_delta_max",
                             "logits_sum", "abs_logits_max", "sat_count", "win_prob_sum",
                             "log_ratio_pos_sum", "log_ratio_pos_sq_sum",
                             "log_ratio_neg_sum", "log_ratio_neg_sq_sum",
                             "clamp_frac_sum", "clamp_frac_count", "ref_kl_sum"):
                    if key == "abs_delta_max" or key == "abs_logits_max":
                        accum_metrics[key] = max(accum_metrics[key], metrics[key])
                    else:
                        accum_metrics[key] += metrics[key]
        else:
            if accelerator.is_main_process:
                accelerator.print(f"  [filter] No valid pairs (filtered={_n_filtered_prompts}/{cur_P}) — skipping training")

        torch.cuda.synchronize()
        _t_train_end = _time.time()
        if accelerator.is_main_process and dl_consumed <= _skip_steps + 10 * P:
            accelerator.print(f"  [Timing P={cur_P}] total={_t_train_end - _t0:.1f}s  "
                              f"sample={_t_sample_end - _t_sample_start:.1f}s  "
                              f"reward={_t_reward_end - _t_sample_end:.1f}s  "
                              f"train={_t_train_end - _t_train_start:.1f}s  "
                              f"pairs={n_pairs}")

        accum_rewards_list.append(rewards.detach().cpu())

        if accelerator.is_main_process:
            all_rew = torch.cat(accum_rewards_list)
            per_prompt_stds = []
            for rew_cpu in accum_rewards_list:
                n_p = rew_cpu.shape[0] // G
                for pi in range(n_p):
                    per_prompt_stds.append(rew_cpu[pi * G : (pi + 1) * G].std())
            all_stds_t = torch.stack(per_prompt_stds) if per_prompt_stds else torch.tensor([0.0])
            accum_reward_log = {
                "reward_mean": all_rew.mean().item(),
                "reward_std": all_rew.std().item(),
                "reward/per_prompt_std_mean": all_stds_t.mean().item(),
                "reward/filtered_prompt_count": _n_filtered_prompts,
                "reward/n_pairs_per_call": n_pairs,
            }

        del sr, data_info, rewards, pos_idx, neg_idx
        torch.cuda.empty_cache()

        _t1 = _time.time()
        if accelerator.is_main_process:
            wandb.log({
                "timing/sampling": _t_sample_end - _t_sample_start,
                "timing/reward": _t_reward_end - _t_sample_end,
                "timing/training": _t1 - _t_reward_end,
                "timing/total": _t1 - _t0,
            }, step=global_step)

        pbar.set_postfix(
            reward=f"{accum_rewards_list[-1].mean().item():.4f}" if accum_rewards_list else "?",
            accum=f"{prompts_since_step}/{grad_accum_steps}",
            step=global_step,
        )

        # ============ Phase 4: Optimizer step ============
        if prompts_since_step >= grad_accum_steps:
            if args.max_grad_norm > 0:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            else:
                grad_norm = torch.tensor(0.0)
            optimizer.step()

            if ema is not None:
                ema.step(trainable_params, global_step)

            global_step += 1

            if (_old_policy_params is not None
                    and args.old_policy_update_interval > 0
                    and global_step % args.old_policy_update_interval == 0):
                for i, p in enumerate(trainable_params):
                    _old_policy_params[i].copy_(p.data)

            if accelerator.is_main_process:
                tot = max(accum_metrics["dpo_total"], 1)
                cfc = max(accum_metrics["clamp_frac_count"], 1)
                # std(δ) over all observed (pair, step) units
                delta_mean = accum_metrics["delta_sum"] / tot
                delta_var = max(accum_metrics["delta_sq_sum"] / tot - delta_mean ** 2, 0.0)
                lr_pos_mean = accum_metrics["log_ratio_pos_sum"] / tot
                lr_pos_var = max(accum_metrics["log_ratio_pos_sq_sum"] / tot - lr_pos_mean ** 2, 0.0)
                lr_neg_mean = accum_metrics["log_ratio_neg_sum"] / tot
                lr_neg_var = max(accum_metrics["log_ratio_neg_sq_sum"] / tot - lr_neg_mean ** 2, 0.0)
                log_dict = {
                    "loss": accum_metrics["loss"],
                    "kl_loss": accum_metrics["kl_loss"],
                    "total_loss": accum_metrics["loss"] + accum_metrics["kl_loss"],
                    "grad/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
                    "dpo/accuracy": accum_metrics["dpo_acc_count"] / tot,
                    "dpo/delta_mean": delta_mean,
                    "dpo/delta_std": delta_var ** 0.5,
                    "dpo/abs_delta_max": accum_metrics["abs_delta_max"],
                    "dpo/logits_mean": accum_metrics["logits_sum"] / tot,
                    "dpo/abs_logits_max": accum_metrics["abs_logits_max"],
                    "dpo/sat_frac": accum_metrics["sat_count"] / tot,
                    "dpo/win_prob_mean": accum_metrics["win_prob_sum"] / tot,
                    "dpo/log_ratio_pos_mean": lr_pos_mean,
                    "dpo/log_ratio_pos_std": lr_pos_var ** 0.5,
                    "dpo/log_ratio_neg_mean": lr_neg_mean,
                    "dpo/log_ratio_neg_std": lr_neg_var ** 0.5,
                    "dpo/log_ratio_clamp_frac": accum_metrics["clamp_frac_sum"] / cfc,
                    "dpo/ref_kl_mean": accum_metrics["ref_kl_sum"] / tot,
                    "dpo/n_pairs_total": tot,
                }
                log_dict.update(accum_reward_log)
                wandb.log(log_dict, step=global_step)

            prompts_since_step = 0
            accum_metrics = _defaultdict(float)
            accum_reward_log = {}
            accum_rewards_list = []
            optimizer.zero_grad(set_to_none=True)

            # ---- Periodic Evaluation ----
            if global_step % args.eval_every == 0:
                eval_mean, eval_std = evaluate_training(
                    model=model, cfg_model=diff_cfg_model,
                    raw_model=accelerator.unwrap_model(model),
                    reward_fn=reward_fn, eval_prompts=eval_prompts_list,
                    vl_chat_processor=vl_chat_processor, path_img=path_img,
                    step_size=step_size, n_samples_per_prompt=args.eval_samples_per_prompt,
                    device=device, accelerator=accelerator, global_step=global_step,
                    save_dir=os.path.join(save_path, f"eval_images_{run_tag}"),
                    eval_metadatas=eval_metadatas_list,
                    txt_max_length=args.txt_max_length, cfg_scale=TRAIN_CFG_SCALE,
                    reward_on_gpu=args.reward_on_gpu,
                )
                if accelerator.is_main_process:
                    wandb.log({"eval/reward_mean": eval_mean, "eval/reward_std": eval_std}, step=global_step)
                    accelerator.print(f"  [Eval step {global_step}] reward_mean={eval_mean:.4f}, reward_std={eval_std:.4f}")
                torch.cuda.empty_cache()

            # ---- Periodic EMA Evaluation ----
            if ema is not None and args.eval_ema_every > 0 and global_step % args.eval_ema_every == 0:
                accelerator.print(f"  [EMA Eval step {global_step}] Switching to EMA params...")
                try:
                    ema.copy_ema_to(trainable_params, store_temp=True)
                    ema_eval_mean, ema_eval_std = evaluate_training(
                        model=model, cfg_model=diff_cfg_model,
                        raw_model=accelerator.unwrap_model(model),
                        reward_fn=reward_fn, eval_prompts=eval_prompts_list,
                        vl_chat_processor=vl_chat_processor, path_img=path_img,
                        step_size=step_size, n_samples_per_prompt=args.eval_samples_per_prompt,
                        device=device, accelerator=accelerator, global_step=global_step,
                        save_dir=os.path.join(save_path, f"eval_images_ema_{run_tag}"),
                        eval_metadatas=eval_metadatas_list,
                        txt_max_length=args.txt_max_length, cfg_scale=TRAIN_CFG_SCALE,
                        reward_on_gpu=args.reward_on_gpu,
                    )
                    if accelerator.is_main_process:
                        wandb.log({"eval_ema/reward_mean": ema_eval_mean, "eval_ema/reward_std": ema_eval_std},
                                  step=global_step)
                        accelerator.print(f"  [EMA Eval step {global_step}] "
                                          f"reward_mean={ema_eval_mean:.4f}, reward_std={ema_eval_std:.4f}")
                finally:
                    ema.copy_temp_to(trainable_params)
                torch.cuda.empty_cache()

            # ---- Periodic Checkpoint ----
            if global_step % args.save_every == 0:
                checkpoint_dir = os.path.join(save_path,
                                              f"checkpoint_{run_tag}_epoch{epoch+1}_step{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                if args.use_lora and accelerator.is_main_process:
                    lora_save_dir = os.path.join(checkpoint_dir, "lora_adapter")
                    accelerator.unwrap_model(model).language_model.save_pretrained(lora_save_dir)
                    accelerator.print(f"  LoRA adapter (current) saved to {lora_save_dir}")
                if args.train_gen_modules and accelerator.is_main_process:
                    _raw = accelerator.unwrap_model(model)
                    gen_state = {n: p.data for n, p in _raw.named_parameters()
                                 if n.startswith(("gen_aligner.", "gen_head."))}
                    torch.save(gen_state, os.path.join(checkpoint_dir, "gen_modules_state.pt"))
                accelerator.save_state(checkpoint_dir)
                if accelerator.is_main_process:
                    _meta = {"global_step": global_step, "epoch": epoch + 1}
                    with open(os.path.join(checkpoint_dir, "training_meta.json"), "w") as f:
                        json.dump(_meta, f)
                if ema is not None and accelerator.is_main_process:
                    torch.save(ema.state_dict(), os.path.join(checkpoint_dir, "ema_state.pt"))
                    ema.copy_ema_to(trainable_params, store_temp=True)
                    if args.use_lora:
                        ema_lora_dir = os.path.join(checkpoint_dir, "lora_adapter_ema")
                        accelerator.unwrap_model(model).language_model.save_pretrained(ema_lora_dir)
                    if args.train_gen_modules:
                        _raw = accelerator.unwrap_model(model)
                        gen_state_ema = {n: p.data for n, p in _raw.named_parameters()
                                         if n.startswith(("gen_aligner.", "gen_head."))}
                        torch.save(gen_state_ema, os.path.join(checkpoint_dir, "gen_modules_state_ema.pt"))
                    ema.copy_temp_to(trainable_params)

    pbar.close()

    # ---- End of epoch checkpoint ----
    epoch_dir = os.path.join(save_path, f"checkpoint_{run_tag}_epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    if args.use_lora and accelerator.is_main_process:
        accelerator.unwrap_model(model).language_model.save_pretrained(
            os.path.join(epoch_dir, "lora_adapter"))
    accelerator.save_state(epoch_dir)
    if accelerator.is_main_process:
        with open(os.path.join(epoch_dir, "training_meta.json"), "w") as f:
            json.dump({"global_step": global_step, "epoch": epoch + 1}, f)
    accelerator.print(f"  Epoch {epoch+1} saved.")
