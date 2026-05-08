"""
Discrete Flow GRPO training for FUDOKI — main entry point.

All code is organized into modules:
  - train_args.py      — CLI argument parsing
  - train_setup.py     — model / optimizer / EMA / dataset initialization
  - training_grpo.py   — core PPO surrogate loss (train_on_prompt, helpers)
  - config.py          — shared constants
  - data.py            — dataset classes
  - model_utils.py     — model loading, image decoding, data_info building
  - reward_utils.py    — reward function construction
  - evaluation.py      — training-time and GenEval evaluation
  - sampling.py        — sampling with log-prob / sampling only
  - cfg_model.py       — differentiable CFG wrapper

This file contains ONLY the training loop.
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

from train_args import parse_arguments
from train_setup import (
    resolve_train_cfg_scale, setup_model, make_disable_adapter_ctx,
    setup_optimizer, setup_ema, validate_ema_args, setup_old_policy,
    resume_from_checkpoint, setup_cfg_wrapper, setup_datasets, setup_eval_split,
)
from training_grpo import (
    parse_train_steps, merge_sampling_results, collect_prompts, train_on_prompt,
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

# ---- Resume (must happen BEFORE old-policy snapshot so snapshot captures trained weights) ----
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

_tag_parts = []
if args.train_gen_modules:
    _tag_parts.append("gen_module")
if args.per_step_reward:
    _tag_parts.append("perstepreward")
run_tag = "_".join(_tag_parts) if _tag_parts else "base"

if args.wandb_name:
    run_tag = args.wandb_name

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

K_total = args.discrete_fm_steps
_max_train_step = K_total - 1 if args.include_last_step else K_total - 2
_n_total_trainable = _max_train_step + 1

train_step_set = set(parse_train_steps(args.train_steps, K_total, _max_train_step))
n_train_steps = len(train_step_set)

# ---- Flow-GRPO-Fast random window ----
# When enabled, train_step_set / n_train_steps are *re-sampled* per training
# iteration (just before train_on_prompt) to a contiguous window.
_fast_window_size = int(getattr(args, "fast_window_size", 0) or 0)
_fast_window_enabled = _fast_window_size > 0
if _fast_window_enabled:
    _fast_window_max_end = args.fast_window_max_end
    if _fast_window_max_end is None or _fast_window_max_end < 0:
        _fast_window_max_end = K_total // 2
    # Right boundary is bounded by both _max_train_step (so we never sample
    # beyond the last trainable step) and the user/auto cap.
    _fast_window_max_end = min(_fast_window_max_end, _max_train_step)
    if _fast_window_size > _fast_window_max_end + 1:
        raise ValueError(
            f"--fast_window_size={_fast_window_size} is larger than the available "
            f"step range [0, {_fast_window_max_end}] (size {_fast_window_max_end + 1}). "
            f"Either lower --fast_window_size or raise --fast_window_max_end / "
            f"--include_last_step / --discrete_fm_steps.")
    # Inclusive range of valid window-start indices.
    _fast_start_min = 0
    _fast_start_max = _fast_window_max_end - _fast_window_size + 1
    _fast_n_starts = _fast_start_max - _fast_start_min + 1
    # Sliding-window counter: every sampling iteration the window slides by 1
    # (mod _fast_n_starts), giving a deterministic round-robin coverage of
    # the valid start range. Lower variance than uniform random.
    # On resume, we offset by the number of iterations already consumed so
    # the schedule continues seamlessly across restarts.
    _fast_window_iter = _resumed_global_step * grad_accum_steps
    accelerator.print(
        f"Flow-GRPO-Fast (sliding): window size={_fast_window_size}, "
        f"start ∈ [{_fast_start_min}, {_fast_start_max}] "
        f"({_fast_n_starts} positions, round-robin), "
        f"right-edge cap={_fast_window_max_end} (K={K_total})")

global_step = _resumed_global_step

# Resume logic — purely based on global_step.
# Total prompts consumed = global_step * grad_accum_steps * P.
# We divide by items_per_epoch (= len(dataloader)) to find which epoch to
# resume from and how many items to skip within that epoch.
# _start_epoch / _start_epoch_skip are computed lazily once len(dataloader) is known.
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

n_train_batches = _ceil(G / train_batch_size)
old_fwd = G * (args.discrete_fm_steps - 1)
new_fwd = n_train_batches * (args.discrete_fm_steps - 1)
accelerator.print(f"Training: train_batch_size={train_batch_size}, "
                  f"model forwards per step: {old_fwd} -> {new_fwd} ")
accelerator.print(f"CFG: train={TRAIN_CFG_SCALE}, eval={TRAIN_CFG_SCALE}"
                  f"{' (--no_cfg halves forwards)' if args.no_cfg else ''}")
accelerator.print(f"Gradient accumulation: {grad_accum_steps} sampling calls per optimizer step "
                  f"(each call = {P} prompt(s) × {G} images = {P*G} images, "
                  f"total {grad_accum_steps * P} prompts / {grad_accum_steps * P * G} images per step per GPU)")
if P > 1:
    accelerator.print(f"  Multi-prompt sampling: P={P} prompts batched together per sampling call")
if kl_beta > 0:
    if args.use_lora and not args.kl_old_policy:
        accelerator.print(f"KL penalty: β={kl_beta} (approx KL vs π_ref via disable_adapter)")
    else:
        _kl_target = "π_old" if args.kl_old_policy else "π_old (no LoRA)"
        accelerator.print(f"KL penalty: β={kl_beta} (approx KL vs {_kl_target})")
else:
    accelerator.print("KL penalty: disabled (β=0)")
if num_inner_updates > 1:
    _total_g = P * G
    accelerator.print(f"Inner updates: {num_inner_updates} passes over {_total_g} samples")
if args.per_step_reward:
    accelerator.print(
        "Per-step reward: ENABLED (x1 forward-difference)")
if args.token_level_loss:
    accelerator.print(f"Token-level loss: ENABLED (epsilon_low={epsilon_low}, epsilon_high={epsilon_high})")
if epsilon_low != epsilon_high:
    accelerator.print(f"Asymmetric clip: ratio ∈ [{1 - epsilon_low:.6f}, {1 + epsilon_high:.6f}]")
else:
    accelerator.print(f"Symmetric clip: ε={epsilon_low} → ratio ∈ [{1 - epsilon_low:.6f}, {1 + epsilon_high:.6f}]")
if args.dim_level_clip:
    accelerator.print("Dim-level clip: ENABLED")
if args.changed_only:
    accelerator.print("Changed-only masking: ENABLED")
if args.changed_frac_reward_bonus > 0:
    accelerator.print(f"Changed-frac reward bonus: +{args.changed_frac_reward_bonus} "
                      f"(threshold={args.changed_frac_threshold})")
if args.top_bottom_k > 0:
    assert args.top_bottom_k <= G // 2
    accelerator.print(f"Top-bottom-K selection: k={args.top_bottom_k}")
if args.include_last_step:
    accelerator.print(f"Include last step: ENABLED (steps 0..{K_total-1})")
if n_train_steps < _n_total_trainable:
    accelerator.print(f"Train steps: {sorted(train_step_set)} ({n_train_steps}/{_n_total_trainable})")
else:
    accelerator.print(f"Train steps: all {_n_total_trainable} steps (0..{_max_train_step})")


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

    # Compute which epochs to skip and how many items to skip in the current epoch,
    # purely from global_step.
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
        # _skip_steps is computed per-epoch at the top of the loop,
        # so subsequent epochs automatically get _skip_steps=0.

        total_G = cur_P * G
        _t0 = _time.time()

        # ============ Phase 1: Sampling (no grad) ============
        model.eval()
        raw_model = accelerator.unwrap_model(model)
        if prompts_since_step == 0:
            optimizer.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()

        _t_sample_start = _time.time()

        # Swap in EMA params for old-policy sampling
        _using_ema_for_sampling = args.sample_with_ema and ema is not None
        if _using_ema_for_sampling:
            ema.copy_ema_to(trainable_params, store_temp=True)

        # Swap in old-policy snapshot for sampling
        _using_old_policy_for_sampling = (
            _old_policy_params is not None and not _using_ema_for_sampling
        )
        if _using_old_policy_for_sampling:
            _old_policy_temp = [p.data.clone() for p in trainable_params]
            for p, old_p in zip(trainable_params, _old_policy_params):
                p.data.copy_(old_p)

        # Build multi-prompt batch
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

        # Compute rewards
        torch.cuda.synchronize()
        _t_sample_end = _time.time()

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

        # ---- Flow-GRPO-Fast: decide this iteration's training window NOW
        # (before per-step reward) so we can restrict reward decodes to just
        # the steps we'll actually train on. Sliding round-robin schedule:
        # iter 0 -> start = _fast_start_min, iter 1 -> +1, ..., wrap modulo
        # _fast_n_starts. All ranks share the same counter.
        # When fast-window is disabled, _iter_window is None and we fall back
        # to scoring the full trajectory (original behavior).
        _iter_window = None  # (k_min, k_max) inclusive, train steps to optimize
        if _fast_window_enabled:
            _w_start = _fast_start_min + (_fast_window_iter % _fast_n_starts)
            _w_end = _w_start + _fast_window_size - 1
            _iter_window = (_w_start, _w_end)
            _fast_window_iter += 1
            if accelerator.is_main_process and dl_consumed <= _skip_steps + 10 * P:
                accelerator.print(
                    f"  [fast-window:sliding] training on steps "
                    f"[{_w_start}, {_w_end}] (iter={_fast_window_iter - 1})")

        # ============ Phase 1.5: Changed-frac reward bonus ============
        if args.changed_frac_reward_bonus > 0:
            D_img = sr["changed_masks"][0].shape[1]
            n_steps_mask = K - 1
            changed_sum_per_image = torch.zeros(total_G, device=device)
            for k in range(n_steps_mask):
                changed_sum_per_image += sr["changed_masks"][k].sum(dim=-1).float()
            changed_frac_per_image = changed_sum_per_image / (n_steps_mask * D_img)
            bonus_mask = changed_frac_per_image < args.changed_frac_threshold
            n_bonus = bonus_mask.sum().item()
            rewards[bonus_mask] += args.changed_frac_reward_bonus
            if accelerator.is_main_process:
                accelerator.print(
                    f"  [changed_frac_bonus] {n_bonus}/{total_G} images got "
                    f"+{args.changed_frac_reward_bonus} bonus "
                    f"(frac < {args.changed_frac_threshold}, "
                    f"mean_frac={changed_frac_per_image.mean().item():.3f})")
            del changed_sum_per_image, changed_frac_per_image, bonus_mask

        # Per-step rewards (forward-difference of x_1 estimates)
        rewards_per_step = None
        if args.per_step_reward:
            # ---------------------------------------------------------------
            # We only need R_k for k in the set of "boundary" indices that
            # appear in some A_k = R_{k+1} - R_k for k in the *trained*
            # window. With fast-window training [k_min, k_max], the trained
            # advantages are A_{k_min}..A_{k_max}, so we need
            #     needed_R = {k_min, k_min+1, ..., k_max+1}.
            # When fast-window is off, needed_R = {0, 1, ..., K-1} (full).
            # R_{K-1} is always reused from the terminal `rewards`.
            #
            # Cost: |needed_R| - (1 if K-1 ∈ needed_R else 0) reward decodes.
            # With fast-window W=2, this drops from K-1 = 7 down to ~2.
            # ---------------------------------------------------------------
            if _iter_window is not None:
                _need_lo, _need_hi = _iter_window[0], _iter_window[1] + 1
            else:
                _need_lo, _need_hi = 0, K - 1
            R_per_step = [None] * K  # length K
            with torch.no_grad():
                for k in range(_need_lo, _need_hi + 1):
                    if k == K - 1:
                        # x_hat_1 at the last denoising step is exactly the
                        # sampled final image — reuse the terminal reward.
                        R_per_step[k] = rewards.clone()
                        continue

                    x1_list_k = sr["all_x1_samples"][k]
                    n_mc_k = len(x1_list_k)

                    # Pick a single Monte-Carlo index for the whole batch.
                    # Since each MC sample x̂₁^{k,j,i} is i.i.d. across j and
                    # across images i, sharing one j across the batch is
                    # statistically equivalent to picking one per image, but
                    # avoids the stack+gather cost.
                    j_pick = int(torch.randint(0, n_mc_k, (1,)).item())
                    x1_img_k = x1_list_k[j_pick].to(device)        # (total_G, D)

                    pix_k = decode_image_tokens(raw_model, x1_img_k)
                    scores_k, _ = reward_fn(pix_k, flat_prompts, flat_metadata)
                    if isinstance(scores_k, torch.Tensor):
                        rewards_k = scores_k.to(device).float()
                    else:
                        rewards_k = torch.tensor(scores_k, device=device, dtype=torch.float32)
                    R_per_step[k] = rewards_k
                    del pix_k, x1_img_k

            # Forward-difference: rewards_per_step[k] = R_{k+1} - R_k for
            # k in the trained window; outside the window we fill zeros (the
            # training loop only reads advantages_per_step[k] for k in
            # the trained set).
            rewards_per_step = [torch.zeros_like(rewards) for _ in range(K)]
            _diff_lo, _diff_hi = _need_lo, _need_hi - 1  # diffs cover [_need_lo, _need_hi-1]
            for k in range(_diff_lo, _diff_hi + 1):
                rewards_per_step[k] = R_per_step[k + 1] - R_per_step[k]
            # Last step (K-1) always has zero advantage (no next state).
            del R_per_step
            torch.cuda.empty_cache()

        _t_reward_end = _time.time()

        # Restore live θ after sampling
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

        advantages_per_step = None
        if rewards_per_step is not None:
            advantages_per_step = []
            for rewards_k in rewards_per_step:
                _global_rk_std = rewards_k.std() if args.global_std else None
                adv_k = torch.zeros_like(rewards_k)
                for pi in range(cur_P):
                    rk_group = rewards_k[pi * G : (pi + 1) * G]
                    rk_group_std = rk_group.std()
                    if args.global_std:
                        denom_k = _global_rk_std + 1e-8
                        ak_group = (rk_group - rk_group.mean()) / denom_k if denom_k >= 1e-8 else torch.zeros_like(rk_group)
                    else:
                        if rk_group_std < 1e-8:
                            ak_group = torch.zeros_like(rk_group)
                        else:
                            ak_group = (rk_group - rk_group.mean()) / (rk_group_std + 1e-8)
                    if args.adv_clip_max > 0:
                        ak_group = torch.clamp(ak_group, -args.adv_clip_max, args.adv_clip_max)
                    adv_k[pi * G : (pi + 1) * G] = ak_group
                advantages_per_step.append(adv_k)

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

        # ============ Phase 3: Train ============
        model.train()
        _t_train_start = _time.time()

        if not _all_filtered:
            prompts_since_step += 1
            is_last_in_accum = (prompts_since_step >= grad_accum_steps)
            effective_G = len(valid_g_indices)
            chunk_size_per_update = max(1, effective_G // num_inner_updates)

            perm_indices = [valid_g_indices[i] for i in torch.randperm(effective_G).tolist()]

            # Flow-GRPO-Fast: use the sliding training window decided earlier
            # (before reward computation) so reward decodes and training are
            # consistent.
            if _iter_window is not None:
                _w_start, _w_end = _iter_window
                _iter_train_step_set = set(range(_w_start, _w_end + 1))
                _iter_n_train_steps = len(_iter_train_step_set)
            else:
                _iter_train_step_set = train_step_set
                _iter_n_train_steps = n_train_steps

            for inner_update in range(num_inner_updates):
                start = inner_update * chunk_size_per_update
                end = effective_G if (inner_update == num_inner_updates - 1) else start + chunk_size_per_update
                g_chunk = perm_indices[start:end]

                metrics = train_on_prompt(
                    sr=sr, data_info=data_info,
                    advantages=advantages, advantages_per_step=advantages_per_step,
                    K=K, g_indices=g_chunk,
                    is_last_prompt_in_accum=(is_last_in_accum and inner_update == num_inner_updates - 1),
                    diff_cfg_model=diff_cfg_model, model=model,
                    accelerator=accelerator, device=device,
                    args=args, train_step_set=_iter_train_step_set,
                    n_train_steps=_iter_n_train_steps, train_batch_size=train_batch_size,
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
                             "changed_count_total", "total_dim_count",
                             "dim_clip_count", "dim_clip_total"):
                    accum_metrics[key] += metrics[key]
        else:
            if accelerator.is_main_process:
                accelerator.print(f"  [filter_zero_std] ALL {cur_P} prompts filtered — skipping training")

        # Timing
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

        # Gather rewards across all ranks so wandb logs reflect the true
        # global batch (cheap: tensor is just (cur_P*G,) floats).
        # Each rank sees the same prompt-group layout (cur_P prompts × G images),
        # gather concatenates along dim 0 -> shape (world_size * cur_P * G,).
        # We keep the per-prompt-group structure intact: groups from rank r
        # occupy indices [r*cur_P*G : (r+1)*cur_P*G].
        rewards_all = accelerator.gather(rewards.detach()).cpu()
        accum_rewards_list.append(rewards_all)

        # Reward logging
        if accelerator.is_main_process:
            per_prompt_stds = []
            # Number of prompt groups in the most recent gathered batch.
            n_groups_last = accum_rewards_list[-1].shape[0] // G
            for pi in range(n_groups_last):
                r_cpu = accum_rewards_list[-1][pi * G : (pi + 1) * G]
                per_prompt_stds.append(r_cpu.std())
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
                accum_reward_log["reward/global_reward_std"] = _global_reward_std.item() if _global_reward_std is not None else 0.0
            if args.per_step_reward and rewards_per_step is not None:
                K_steps = len(rewards_per_step)
                step_std_means = []
                for k_idx in range(K_steps):
                    std_k = rewards_per_step[k_idx].std().item()
                    step_std_means.append(std_k)
                    accum_reward_log[f"reward/step{k_idx}_std_mean"] = std_k
                accum_reward_log["reward/per_step_std_mean_avg"] = sum(step_std_means) / len(step_std_means)
                # Mean of |R_{k+1} - R_k| (raw forward-difference magnitude per step)
                # — diagnoses how informative the per-step signal is.
                for k_idx in range(K_steps):
                    accum_reward_log[f"reward/x1diff_step{k_idx}_abs_mean"] = \
                        rewards_per_step[k_idx].abs().mean().item()

        # Free this step's SR data
        del sr, data_info, rewards, advantages, advantages_per_step, rewards_per_step
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

            # Reset accumulation
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
                        accelerator.print(f"  LoRA adapter (EMA) saved to {ema_lora_dir}")
                    if args.train_gen_modules:
                        _raw = accelerator.unwrap_model(model)
                        gen_state_ema = {n: p.data for n, p in _raw.named_parameters()
                                         if n.startswith(("gen_aligner.", "gen_head."))}
                        torch.save(gen_state_ema, os.path.join(checkpoint_dir, "gen_modules_state_ema.pt"))
                        accelerator.print(f"  gen_modules (EMA) saved")
                    ema.copy_temp_to(trainable_params)

    pbar.close()

    # ---- End of epoch: flush partial accumulation ----
    if prompts_since_step > 0:
        accelerator.print(f"  Flushing partial accumulation ({prompts_since_step}/{grad_accum_steps} steps)")
        if hasattr(model, 'module') and not getattr(accelerator.state, 'deepspeed_plugin', None):
            for p in model.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG)
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
            tc = accum_metrics["total_count"]
            log_dict = {
                "loss": accum_metrics["loss"],
                "kl_loss": accum_metrics["kl_loss"],
                "total_loss": accum_metrics["loss"] + accum_metrics["kl_loss"],
                "grad/grad_norm": grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm),
                "partial_flush": 1,
            }
            if tc > 0:
                log_dict["policy/clip_fraction"] = accum_metrics["clip_count"] / tc
                log_dict["policy/clip_high"] = accum_metrics["clip_high"] / tc
                log_dict["policy/clip_low"] = accum_metrics["clip_low"] / tc
                log_dict["policy/approx_kl"] = accum_metrics["approx_kl"] / tc
                log_dict["policy/ratio_mean"] = accum_metrics["ratio_sum"] / tc
            log_dict.update(accum_reward_log)
            wandb.log(log_dict, step=global_step)

        prompts_since_step = 0
        accum_metrics = _defaultdict(float)
        accum_reward_log = {}
        accum_rewards_list = []

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
    if accelerator.is_main_process:
        _meta = {"global_step": global_step, "epoch": epoch + 1}
        with open(os.path.join(epoch_dir, "training_meta.json"), "w") as f:
            json.dump(_meta, f)
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
