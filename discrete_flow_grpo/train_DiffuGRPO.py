"""
Discrete Flow DiffuGRPO training for FUDOKI — main entry point.

DiffuGRPO variant: per-sample geometric-mean ratio over D image-token
dimensions, evaluated by a single forward of pi_theta at x_0 (the initial
noise).  Only x_0, x_1 and log p_{theta_old}(x_1^d | x_0) are kept from the
sampling phase — much cheaper memory than (G)SPO which stores the whole
trajectory.

Modules used:
  - train_args.py             — CLI argument parsing
  - train_setup.py            — model / optimizer / EMA / dataset init
  - training_grpo.py          — collect_prompts helper
  - training_diffugrpo.py     — DiffuGRPO loss
  - sampling_diffugrpo.py     — minimal sampler with log_p_old at x_0
  - config.py / data.py / model_utils.py / reward_utils.py / evaluation.py
  - cfg_model.py              — differentiable CFG wrapper

This file contains ONLY the training loop.
"""

import os
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
from utils import prompts_to_tensor

from train_args import parse_arguments
from train_setup import (
    resolve_train_cfg_scale, setup_model, make_disable_adapter_ctx,
    setup_optimizer, setup_ema, validate_ema_args, setup_old_policy,
    resume_from_checkpoint, setup_cfg_wrapper, setup_datasets, setup_eval_split,
)
from training_grpo import collect_prompts
from sampling_diffugrpo import (
    sample_with_log_prob_diffugrpo, merge_sampling_results_diffugrpo,
)
from training_diffugrpo import train_on_prompt_diffugrpo


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

# ---- Resume (BEFORE old-policy snapshot) ----
_resumed_global_step = resume_from_checkpoint(args, accelerator, model, ema)

# ---- Old-policy snapshot ----
_old_policy_params = setup_old_policy(args, trainable_params, accelerator)
_old_policy_temp = None

# ---- CFG wrapper ----
diff_cfg_model, raw_model = setup_cfg_wrapper(model, accelerator, device)

# ---- Eval split ----
eval_prompts_list, eval_metadatas_list, n_prompts_per_gpu = setup_eval_split(
    eval_dataset, _use_geneval_metadata, accelerator, args)

# ---- Output dir / wandb ----
save_path = args.output_dir
os.makedirs(save_path, exist_ok=True)

_tag_parts = []
if args.train_gen_modules:
    _tag_parts.append("gen_module")
run_tag = "_".join(_tag_parts) if _tag_parts else "base"

if args.wandb_name:
    run_tag = args.wandb_name
else:
    run_tag = f"diffugrpo_{run_tag}"

if accelerator.is_main_process:
    wandb.init(project="discrete_flow_grpo", name=run_tag,
               settings=wandb.Settings(init_timeout=300), config=vars(args))

# ---- Derived constants ----
G = args.group_size
P = args.prompts_per_sample_batch
sample_batch_size = args.sample_batch_size if args.sample_batch_size is not None else G
n_mc = args.n_mc
step_size = 1.0 / args.discrete_fm_steps
epsilon = args.epsilon
epsilon_low = args.epsilon_low if args.epsilon_low is not None else epsilon
epsilon_high = args.epsilon_high if args.epsilon_high is not None else epsilon
kl_beta = args.kl_beta
train_batch_size = args.train_batch_size
grad_accum_steps = args.gradient_accumulation_steps
num_inner_updates = args.num_inner_updates
num_epochs = args.num_epochs

global_step = _resumed_global_step
_total_prompts_consumed = _resumed_global_step * grad_accum_steps * P


# ===================================================================
# Print configuration summary
# ===================================================================
accelerator.print("=" * 60)
accelerator.print("DIFFUGRPO TRAINING (geo-mean ratio over D image dims, single forward at x_0)")
accelerator.print("=" * 60)
if sample_batch_size < G:
    n_rounds = _ceil(G / sample_batch_size)
    accelerator.print(f"Sampling: group_size={G}, sample_batch_size={sample_batch_size}, "
                      f"{n_rounds} rounds per step")
else:
    accelerator.print(f"Sampling: group_size={G}, single round")

n_train_batches = _ceil(G / train_batch_size)
# DiffuGRPO does ONE forward per training mini-batch (at x_0), independent of K.
accelerator.print(f"Training: train_batch_size={train_batch_size}, "
                  f"forwards per prompt: {n_train_batches} (one per mini-batch, at x_0)")
accelerator.print(f"CFG: train={TRAIN_CFG_SCALE}, eval={TRAIN_CFG_SCALE}"
                  f"{' (--no_cfg halves forwards)' if args.no_cfg else ''}")
accelerator.print(f"Gradient accumulation: {grad_accum_steps} sampling calls per optimizer step "
                  f"(each call = {P} prompt(s) x {G} images = {P*G} images, "
                  f"total {grad_accum_steps * P} prompts / {grad_accum_steps * P * G} images per step per GPU)")
if P > 1:
    accelerator.print(f"  Multi-prompt sampling: P={P} prompts batched together per sampling call")
if kl_beta > 0:
    if args.use_lora and not args.kl_old_policy:
        accelerator.print(f"KL penalty: beta={kl_beta} (approx KL vs pi_ref via disable_adapter)")
    else:
        _kl_target = "pi_old" if args.kl_old_policy else "pi_old (no LoRA)"
        accelerator.print(f"KL penalty: beta={kl_beta} (approx KL vs {_kl_target})")
else:
    accelerator.print("KL penalty: disabled (beta=0)")
if num_inner_updates > 1:
    accelerator.print(f"Inner updates: {num_inner_updates} passes over {P * G} samples")
if args.token_level_loss:
    accelerator.print(f"Token-level loss: ENABLED (eps_low={epsilon_low}, eps_high={epsilon_high})")
if epsilon_low != epsilon_high:
    accelerator.print(f"Asymmetric clip: ratio in [{1 - epsilon_low:.6f}, {1 + epsilon_high:.6f}]")
else:
    accelerator.print(f"Symmetric clip: eps={epsilon_low}")
if args.dim_level_clip:
    accelerator.print("Dim-level clip: ENABLED")
if args.changed_only:
    accelerator.print("Changed-only masking: ENABLED")
if args.top_bottom_k > 0:
    assert args.top_bottom_k <= G // 2
    accelerator.print(f"Top-bottom-K selection: k={args.top_bottom_k}")


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
# Sampling helper (multi-prompt, multi-round)
# ===================================================================
def _sample_for_prompts(prompt_list, cur_P):
    total_G = cur_P * G
    if cur_P == 1:
        prompt = prompt_list[0]
        if sample_batch_size >= G:
            x_init, di = build_data_info(
                prompt, G, vl_chat_processor, path_img, args.txt_max_length, device)
            sr_local = sample_with_log_prob_diffugrpo(
                model=diff_cfg_model, path_img=path_img,
                vocabulary_size_img=VOCABULARY_SIZE_IMG,
                x_init=x_init, data_info=di,
                step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
                n_mc=n_mc, device=device, offload_to_cpu=False,
            )
            return sr_local, di
        sr_list = []
        remaining = G
        while remaining > 0:
            bs = min(sample_batch_size, remaining)
            x_init_r, di_r = build_data_info(
                prompt, bs, vl_chat_processor, path_img, args.txt_max_length, device)
            sr_r = sample_with_log_prob_diffugrpo(
                model=diff_cfg_model, path_img=path_img,
                vocabulary_size_img=VOCABULARY_SIZE_IMG,
                x_init=x_init_r, data_info=di_r,
                step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
                n_mc=n_mc, device=device, offload_to_cpu=False,
            )
            sr_list.append(sr_r)
            del x_init_r, di_r
            remaining -= bs
        sr_local = merge_sampling_results_diffugrpo(sr_list)
        del sr_list
        _, di = build_data_info(prompt, G, vl_chat_processor, path_img,
                                args.txt_max_length, device)
        return sr_local, di
    # cur_P > 1
    if sample_batch_size >= total_G:
        x_init, di = build_data_info_multi(
            prompt_list, G, vl_chat_processor, path_img, args.txt_max_length, device)
        sr_local = sample_with_log_prob_diffugrpo(
            model=diff_cfg_model, path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
            x_init=x_init, data_info=di,
            step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
            n_mc=n_mc, device=device, offload_to_cpu=False,
        )
        return sr_local, di
    sr_list = []
    remaining = total_G
    offset = 0
    while remaining > 0:
        bs = min(sample_batch_size, remaining)
        sub_prompts = [prompt_list[idx // G] for idx in range(offset, offset + bs)]
        dummy_img_tokens_r = torch.zeros(bs, IMG_LEN, dtype=torch.long, device=device)
        t_r = torch.zeros(bs, device=device)
        input_ids_r, _, di_r = prompts_to_tensor(
            prompts=sub_prompts, vl_chat_processor=vl_chat_processor,
            path=path_img, t=t_r, g_or_u="generation",
            txt_max_length=args.txt_max_length, IMG_LEN=IMG_LEN,
            img_tokens=dummy_img_tokens_r, device=device,
        )
        img_mask_r = di_r["image_token_mask"]
        x_init_r = input_ids_r.clone()
        x_0_r = torch.randint(0, VOCABULARY_SIZE_IMG, (bs, IMG_LEN),
                              dtype=torch.long, device=device)
        x_init_r[img_mask_r == 1] = x_0_r.flatten()
        sr_r = sample_with_log_prob_diffugrpo(
            model=diff_cfg_model, path_img=path_img,
            vocabulary_size_img=VOCABULARY_SIZE_IMG,
            x_init=x_init_r, data_info=di_r,
            step_size=step_size, cfg_scale=TRAIN_CFG_SCALE,
            n_mc=n_mc, device=device, offload_to_cpu=False,
        )
        sr_list.append(sr_r)
        del x_init_r, di_r
        remaining -= bs
        offset += bs
    sr_local = merge_sampling_results_diffugrpo(sr_list)
    del sr_list
    _, di = build_data_info_multi(
        prompt_list, G, vl_chat_processor, path_img, args.txt_max_length, device)
    return sr_local, di


# ===================================================================
# Training loop
# ===================================================================
for epoch in range(num_epochs):
    _items_per_epoch = len(dataloader)

    if _total_prompts_consumed > 0:
        _start_epoch = _total_prompts_consumed // _items_per_epoch
        _start_epoch_skip = _total_prompts_consumed % _items_per_epoch
    else:
        _start_epoch = 0
        _start_epoch_skip = 0

    if epoch < _start_epoch:
        accelerator.print(f"Epoch {epoch+1}/{num_epochs} — skipped (already completed)")
        continue

    _skip_steps = _start_epoch_skip if epoch == _start_epoch else 0

    accelerator.print(f"Epoch {epoch+1}/{num_epochs}"
                      + (f" (skipping {_skip_steps}/{_items_per_epoch} items to resume)" if _skip_steps > 0 else ""))
    dl_iter = iter(dataloader)
    total_dl_len = len(dataloader)
    pbar = tqdm(total=total_dl_len, desc=f"Epoch {epoch+1}",
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
            if dl_consumed % (100 * P) < P:
                accelerator.print(f"  Skipping dl items {dl_consumed}/{_skip_steps} (resuming)...")
            continue

        total_G = cur_P * G
        _t0 = _time.time()

        # ============ Phase 1: Sampling (no grad, old policy) ============
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

        sr, data_info = _sample_for_prompts(prompt_list, cur_P)

        torch.cuda.synchronize()
        _t_sample_end = _time.time()

        # ============ Reward computation ============
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
        # final_tokens are not used in DiffuGRPO training — drop to save memory
        del sr["final_tokens"]
        torch.cuda.empty_cache()

        _t_reward_end = _time.time()

        # Restore live params after sampling
        if _using_ema_for_sampling:
            ema.copy_temp_to(trainable_params)
        if _using_old_policy_for_sampling and _old_policy_temp is not None:
            for p, live_p in zip(trainable_params, _old_policy_temp):
                p.data.copy_(live_p)
            _old_policy_temp = None

        # ============ Phase 2: Compute advantages ============
        _global_reward_std = rewards.std() if args.global_std else None

        advantages = torch.zeros_like(rewards)
        _zero_std_ratio = 0
        for pi in range(cur_P):
            r_group = rewards[pi * G : (pi + 1) * G]
            r_group_std = r_group.std()
            if r_group_std < 1e-8:
                _zero_std_ratio += 1
            if args.global_std:
                denom = _global_reward_std + 1e-8
                adv_group = (r_group - r_group.mean()) / denom if denom >= 1e-8 else torch.zeros_like(r_group)
            else:
                if r_group_std < 1e-8:
                    adv_group = torch.zeros_like(r_group)
                else:
                    adv_group = (r_group - r_group.mean()) / (r_group_std + 1e-8)
            if args.adv_clip_max > 0:
                adv_group = torch.clamp(adv_group, -args.adv_clip_max, args.adv_clip_max)
            advantages[pi * G : (pi + 1) * G] = adv_group
        _zero_std_ratio = _zero_std_ratio / max(cur_P, 1)

        # ============ Phase 2.5: Filter zero-std prompt groups ============
        _n_filtered_prompts = 0
        if args.filter_zero_std:
            valid_g_indices = []
            for pi in range(cur_P):
                g_start = pi * G
                g_end = (pi + 1) * G
                if advantages[g_start:g_end].abs().sum().item() < 1e-8:
                    _n_filtered_prompts += 1
                else:
                    valid_g_indices.extend(range(g_start, g_end))
            if _n_filtered_prompts > 0 and accelerator.is_main_process:
                accelerator.print(f"  [filter_zero_std] Filtered {_n_filtered_prompts}/{cur_P} "
                                  f"zero-std prompt groups ({len(valid_g_indices)} samples remain)")
        else:
            valid_g_indices = list(range(total_G))

        # ============ Phase 2.6: Top-bottom-K selection ============
        _n_topbottom_kept = len(valid_g_indices)
        _n_topbottom_total = len(valid_g_indices)
        if args.top_bottom_k > 0:
            tbk = args.top_bottom_k
            filtered_g_indices = []
            valid_set = set(valid_g_indices)
            for pi in range(cur_P):
                g_start = pi * G
                g_end = (pi + 1) * G
                group_valid = [gi for gi in range(g_start, g_end) if gi in valid_set]
                if len(group_valid) == 0:
                    continue
                group_valid_sorted = sorted(group_valid, key=lambda gi: rewards[gi].item(), reverse=True)
                n_valid = len(group_valid_sorted)
                if 2 * tbk >= n_valid:
                    filtered_g_indices.extend(group_valid_sorted)
                else:
                    filtered_g_indices.extend(group_valid_sorted[:tbk])
                    filtered_g_indices.extend(group_valid_sorted[-tbk:])
            _n_topbottom_total = len(valid_g_indices)
            _n_topbottom_kept = len(filtered_g_indices)
            valid_g_indices = filtered_g_indices
            if accelerator.is_main_process:
                accelerator.print(f"  [top_bottom_k={tbk}] Kept {_n_topbottom_kept}/{_n_topbottom_total} samples")

        _all_filtered = len(valid_g_indices) == 0

        # ============ Phase 3: Train (DiffuGRPO loss) ============
        model.train()
        _t_train_start = _time.time()

        if not _all_filtered:
            prompts_since_step += 1
            is_last_in_accum = (prompts_since_step >= grad_accum_steps)
            effective_G = len(valid_g_indices)
            chunk_size_per_update = max(1, effective_G // num_inner_updates)

            perm_indices = [valid_g_indices[i] for i in torch.randperm(effective_G).tolist()]
            for inner_update in range(num_inner_updates):
                start = inner_update * chunk_size_per_update
                end = effective_G if (inner_update == num_inner_updates - 1) else start + chunk_size_per_update
                g_chunk = perm_indices[start:end]

                metrics = train_on_prompt_diffugrpo(
                    sr=sr, data_info=data_info,
                    advantages=advantages,
                    g_indices=g_chunk,
                    is_last_prompt_in_accum=(is_last_in_accum and inner_update == num_inner_updates - 1),
                    diff_cfg_model=diff_cfg_model, model=model,
                    accelerator=accelerator, device=device,
                    args=args, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, num_inner_updates=num_inner_updates,
                    epsilon_low=epsilon_low, epsilon_high=epsilon_high,
                    kl_beta=kl_beta, TRAIN_CFG_SCALE=TRAIN_CFG_SCALE,
                    disable_adapter_and_gen_modules=disable_adapter_and_gen_modules,
                    raw_model=raw_model,
                )

                for key in ("loss", "kl_loss", "clip_count", "clip_high", "clip_low",
                             "total_count", "approx_kl", "ratio_sum",
                             "log_ratio_mean_sum", "log_ratio_sum_sum",
                             "token_clip_count", "token_total_count",
                             "log_ratio_mean_pos_sum", "log_ratio_mean_neg_sum",
                             "count_pos", "count_neg",
                             "dim_clip_count", "dim_clip_total"):
                    accum_metrics[key] += metrics[key]
            # GRPO-style changed_frac (sampling-side, intermediate steps only,
            # single-GPU local counts) — overrides the endpoint-based counts
            # produced by the DiffuGRPO loss so that wandb's
            # `policy/changed_frac` is comparable to GRPO.
            accum_metrics["changed_count_total"] += sr["sample_changed_count"]
            accum_metrics["total_dim_count"] += sr["sample_total_count"]
        else:
            if accelerator.is_main_process:
                accelerator.print(f"  [filter_zero_std] ALL {cur_P} prompts filtered — skipping training")

        torch.cuda.synchronize()
        _t_train_end = _time.time()
        if accelerator.is_main_process and dl_consumed <= _skip_steps + 10 * P:
            _dt_sample = _t_sample_end - _t_sample_start
            _dt_reward = _t_reward_end - _t_sample_end
            _dt_train = _t_train_end - _t_train_start
            _dt_total = _t_train_end - _t0
            accelerator.print(f"  [Timing P={cur_P}] total={_dt_total:.1f}s  "
                              f"sample={_dt_sample:.1f}s  reward={_dt_reward:.1f}s  "
                              f"train={_dt_train:.1f}s")

        accum_rewards_list.append(rewards.detach().cpu())

        # Reward logging
        if accelerator.is_main_process:
            all_stds = []
            for rew_cpu in accum_rewards_list:
                n_p = rew_cpu.shape[0] // G
                for pi in range(n_p):
                    all_stds.append(rew_cpu[pi * G : (pi + 1) * G].std())
            all_stds_t = torch.stack(all_stds)
            zero_std_count = (all_stds_t < 1e-8).sum().item()
            all_rew = torch.cat(accum_rewards_list)
            accum_reward_log = {
                "reward_mean": all_rew.mean().item(),
                "reward_std": all_rew.std().item(),
                "reward/per_prompt_std_mean": all_stds_t.mean().item(),
                "reward/per_prompt_std_min": all_stds_t.min().item(),
                "reward/per_prompt_std_max": all_stds_t.max().item(),
                "reward/zero_std_prompt_count": zero_std_count,
                "reward/filtered_prompt_count": _n_filtered_prompts,
                "reward/topbottom_kept": _n_topbottom_kept,
                "reward/topbottom_total": _n_topbottom_total,
                "advantage/mean": advantages.mean().item(),
                "advantage/std": advantages.std().item(),
                "advantage/max": advantages.max().item(),
                "advantage/min": advantages.min().item(),
                "reward/zero_std_ratio": _zero_std_ratio,
            }
            if args.global_std:
                accum_reward_log["reward/global_reward_std"] = (
                    _global_reward_std.item() if _global_reward_std is not None else 0.0)

        # Free this step's SR data
        del sr, data_info, rewards, advantages
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

            # Refresh old-policy snapshot
            if (_old_policy_params is not None
                    and args.old_policy_update_interval > 0
                    and global_step % args.old_policy_update_interval == 0):
                for i, p in enumerate(trainable_params):
                    _old_policy_params[i].copy_(p.data)

            # Logging
            if accelerator.is_main_process:
                tc = accum_metrics["total_count"]
                log_dict = {
                    "loss": accum_metrics["loss"],
                    "kl_loss": accum_metrics["kl_loss"],
                    "total_loss": accum_metrics["loss"] + accum_metrics["kl_loss"],
                    "grad/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
                }
                if tc > 0:
                    log_dict["policy/clip_fraction"] = accum_metrics["clip_count"] / tc
                    log_dict["policy/clip_high"] = accum_metrics["clip_high"] / tc
                    log_dict["policy/clip_low"] = accum_metrics["clip_low"] / tc
                    log_dict["policy/approx_kl"] = accum_metrics["approx_kl"] / tc
                    log_dict["policy/ratio_mean"] = accum_metrics["ratio_sum"] / tc
                    log_dict["policy/log_ratio_mean"] = accum_metrics["log_ratio_mean_sum"] / tc
                    log_dict["policy/log_ratio_sum"] = accum_metrics["log_ratio_sum_sum"] / tc
                    ttc = accum_metrics["token_total_count"]
                    if ttc > 0:
                        log_dict["policy/token_clip_fraction"] = accum_metrics["token_clip_count"] / ttc
                    cp = accum_metrics["count_pos"]
                    cn = accum_metrics["count_neg"]
                    if cp > 0:
                        log_dict["policy/log_ratio_mean_adv_pos"] = accum_metrics["log_ratio_mean_pos_sum"] / cp
                    if cn > 0:
                        log_dict["policy/log_ratio_mean_adv_neg"] = accum_metrics["log_ratio_mean_neg_sum"] / cn
                    log_dict["policy/count_adv_pos"] = cp
                    log_dict["policy/count_adv_neg"] = cn
                    tdc = accum_metrics["total_dim_count"]
                    if tdc > 0:
                        log_dict["policy/changed_frac"] = accum_metrics["changed_count_total"] / tdc
                    dct = accum_metrics["dim_clip_total"]
                    if dct > 0:
                        log_dict["policy/dim_clip_fraction"] = accum_metrics["dim_clip_count"] / dct
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
                        wandb.log({"eval_ema/reward_mean": ema_eval_mean, "eval_ema/reward_std": ema_eval_std}, step=global_step)
                        accelerator.print(f"  [EMA Eval step {global_step}] reward_mean={ema_eval_mean:.4f}, reward_std={ema_eval_std:.4f}")
                finally:
                    ema.copy_temp_to(trainable_params)
                torch.cuda.empty_cache()

            # ---- Periodic Checkpoint ----
            if global_step % args.save_every == 0:
                checkpoint_dir = os.path.join(save_path,
                                              f"checkpoint_{run_tag}_epoch{epoch+1}_step{global_step}")
                os.makedirs(checkpoint_dir, exist_ok=True)
                if args.use_lora:
                    if accelerator.is_main_process:
                        lora_save_dir = os.path.join(checkpoint_dir, "lora_adapter")
                        accelerator.unwrap_model(model).language_model.save_pretrained(lora_save_dir)
                        accelerator.print(f"  LoRA adapter (current) saved to {lora_save_dir}")
                if args.train_gen_modules and accelerator.is_main_process:
                    _raw = accelerator.unwrap_model(model)
                    gen_state = {n: p.data for n, p in _raw.named_parameters()
                                 if n.startswith(("gen_aligner.", "gen_head."))}
                    torch.save(gen_state, os.path.join(checkpoint_dir, "gen_modules_state.pt"))
                    accelerator.print(f"  gen_modules saved ({len(gen_state)} tensors)")
                accelerator.save_state(checkpoint_dir)
                if ema is not None and accelerator.is_main_process:
                    torch.save(ema.state_dict(), os.path.join(checkpoint_dir, "ema_state.pt"))
                    ema.copy_ema_to(trainable_params, store_temp=True)
                    if args.use_lora:
                        ema_lora_dir = os.path.join(checkpoint_dir, "lora_adapter_ema")
                        accelerator.unwrap_model(model).language_model.save_pretrained(ema_lora_dir)
                        accelerator.print(f"  LoRA adapter (EMA) saved to {ema_lora_dir}")
                    if args.train_gen_modules:
                        _raw = accelerator.unwrap_model(model)
                        gen_state_ema = {n: p.data for n, p in _raw.named_parameters()
                                         if n.startswith(("gen_aligner.", "gen_head."))}
                        torch.save(gen_state_ema, os.path.join(checkpoint_dir, "gen_modules_state_ema.pt"))
                        accelerator.print(f"  gen_modules (EMA) saved")
                    ema.copy_temp_to(trainable_params)

    pbar.close()

    # ---- End of epoch checkpoint ----
    epoch_dir = os.path.join(save_path, f"checkpoint_{run_tag}_epoch{epoch+1}")
    os.makedirs(epoch_dir, exist_ok=True)
    if args.use_lora:
        if accelerator.is_main_process:
            lora_save_dir = os.path.join(epoch_dir, "lora_adapter")
            accelerator.unwrap_model(model).language_model.save_pretrained(lora_save_dir)
            accelerator.print(f"  LoRA adapter (current) saved to {lora_save_dir}")
    if args.train_gen_modules and accelerator.is_main_process:
        _raw = accelerator.unwrap_model(model)
        gen_state = {n: p.data for n, p in _raw.named_parameters()
                     if n.startswith(("gen_aligner.", "gen_head."))}
        torch.save(gen_state, os.path.join(epoch_dir, "gen_modules_state.pt"))
        accelerator.print(f"  gen_modules saved ({len(gen_state)} tensors)")
    accelerator.save_state(epoch_dir)
    if ema is not None and accelerator.is_main_process:
        torch.save(ema.state_dict(), os.path.join(epoch_dir, "ema_state.pt"))
        ema.copy_ema_to(trainable_params, store_temp=True)
        if args.use_lora:
            ema_lora_dir = os.path.join(epoch_dir, "lora_adapter_ema")
            accelerator.unwrap_model(model).language_model.save_pretrained(ema_lora_dir)
            accelerator.print(f"  LoRA adapter (EMA) saved to {ema_lora_dir}")
        if args.train_gen_modules:
            _raw = accelerator.unwrap_model(model)
            gen_state_ema = {n: p.data for n, p in _raw.named_parameters()
                             if n.startswith(("gen_aligner.", "gen_head."))}
            torch.save(gen_state_ema, os.path.join(epoch_dir, "gen_modules_state_ema.pt"))
            accelerator.print(f"  gen_modules (EMA) saved")
        ema.copy_temp_to(trainable_params)
    accelerator.print(f"  Epoch {epoch+1} saved.")
