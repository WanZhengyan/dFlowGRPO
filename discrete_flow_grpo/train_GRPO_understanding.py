"""
Discrete Flow GRPO training for FUDOKI — multimodal UNDERSTANDING direction.

Mirrors ``train_GRPO.py`` but optimizes the model to pick the correct MCQ
option on ScienceQA. Reward = 1 if the sampled answer letter matches the
ground-truth choice, else 0.

Sampling direction: text tokens (conditioned on image + question).
Policy parameterization: same FUDOKI model, with LoRA on the language model.
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
from torch.utils.data import DataLoader

# Patch torch.load for older checkpoints
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from fudoki.janus.models import VLChatProcessor

from config import VOCABULARY_SIZE_TXT

# Reuse generation-side helpers where they're direction-agnostic.
from train_setup import (
    setup_model, make_disable_adapter_ctx, setup_optimizer, setup_ema,
    validate_ema_args, setup_old_policy, resume_from_checkpoint,
)
from training_grpo import parse_train_steps, merge_sampling_results

from train_args_understanding import parse_arguments
from data_understanding import ScienceQADataset
from model_utils_understanding import build_data_info_understanding
from sampling_understanding import sample_text_with_log_prob, sample_text_only
from cfg_model_understanding import (
    UnderstandingCFGModel, DifferentiableUnderstandingCFGModel,
)
from reward_understanding import build_mcq_reward_fn
from training_grpo_understanding import train_on_batch_u, collect_items

# ===================================================================
# Initialization
# ===================================================================
args = parse_arguments()

accelerator = Accelerator()
device = accelerator.device
torch.manual_seed(args.seed)

# Understanding doesn't need CFG — training path is single-forward.
TRAIN_CFG_SCALE = 0.0

# ---- Model ----
model, _gen_ref_state = setup_model(args, device, accelerator)
disable_adapter_and_gen_modules = make_disable_adapter_ctx(_gen_ref_state)

# ---- Tokenizer & paths (need both text and image paths like FUDOKI eval) ----
vl_chat_processor = VLChatProcessor.from_pretrained(args.checkpoint_path)
path_txt = MixtureDiscreteSoftmaxProbPath(
    mode="text", embedding_path=args.text_embedding_path)
path_txt.embedding = path_txt.embedding.to(device)
path_img = MixtureDiscreteSoftmaxProbPath(
    mode="image", embedding_path=args.image_embedding_path)
path_img.embedding = path_img.embedding.to(device)

# ---- Reward ----
# Optional API judge for ambiguous answers (VLMEvalKit-style fallback).
_api_judge = None
if getattr(args, "use_api_judge", False):
    from api_judge import APIJudge
    _api_key = os.environ.get(args.api_judge_key_env, None)
    _api_judge = APIJudge(
        model=args.api_judge_model,
        base_url=args.api_judge_base_url,
        api_key=_api_key,
        max_retries=args.api_judge_max_retries,
        max_workers=args.api_judge_max_workers,
        temperature=args.api_judge_temperature,
        timeout=args.api_judge_timeout,
        verbose=args.api_judge_verbose,
    )
    accelerator.print(
        f"API judge enabled: model={args.api_judge_model} "
        f"base_url={args.api_judge_base_url or '(default)'} "
        f"only_eval={args.api_judge_only_eval}"
    )

# Train-time reward fn: uses the judge unless --api_judge_only_eval.
reward_fn = build_mcq_reward_fn(
    api_judge=None if (args.use_api_judge and args.api_judge_only_eval)
    else _api_judge
)
# Eval-time reward fn always uses the judge when enabled.
reward_fn_eval = build_mcq_reward_fn(api_judge=_api_judge) if _api_judge is not None else reward_fn
accelerator.print("Reward: ScienceQA MCQ (1 if predicted letter == answer else 0)")# ---- Datasets ----
train_ds = ScienceQADataset(args.dataset_dir, split="train", require_image=True,
                            max_prompt_chars=args.max_prompt_chars)
val_ds = ScienceQADataset(args.dataset_dir, split="val", require_image=True,
                          max_prompt_chars=args.max_prompt_chars)
accelerator.print(f"Dataset: ScienceQA  train={len(train_ds)}  val={len(val_ds)}")

dataloader = DataLoader(train_ds, batch_size=1, shuffle=True,
                        collate_fn=ScienceQADataset.collate_fn)

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

# ---- CFG wrappers for understanding ----
raw_model = accelerator.unwrap_model(model)
sample_cfg_model = UnderstandingCFGModel(raw_model)       # for no-grad sampling
diff_cfg_model = DifferentiableUnderstandingCFGModel(raw_model)  # for training

# ---- Output dir ----
save_path = args.output_dir
os.makedirs(save_path, exist_ok=True)
run_tag = args.wandb_name or "grpo_understanding"

if accelerator.is_main_process:
    wandb.init(project=args.wandb_project, name=run_tag,
               settings=wandb.Settings(init_timeout=300), config=vars(args))

# ---- Derived constants ----
G = args.group_size
P = args.prompts_per_sample_batch
sample_batch_size = args.sample_batch_size if args.sample_batch_size is not None else G
n_mc = args.n_mc
step_size = 1.0 / args.discrete_fm_steps
epsilon_low = args.epsilon_low if args.epsilon_low is not None else args.epsilon
epsilon_high = args.epsilon_high if args.epsilon_high is not None else args.epsilon
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

global_step = _resumed_global_step
_total_items_consumed = _resumed_global_step * grad_accum_steps * P


# ===================================================================
# Helper: evaluation
# ===================================================================
@torch.no_grad()
def evaluate_val(n_samples_per_gpu, tag=None):
    """Run validation: lean sampling (no GRPO intermediates), compute accuracy.

    If ``tag`` is provided, rank-0 will dump its per-sample predictions
    (question/image-id/gold/pred/score) to ``<save_path>/eval_preds/<tag>.jsonl``.
    """
    n_total = min(n_samples_per_gpu * accelerator.num_processes, len(val_ds))
    n_per_gpu = n_total // accelerator.num_processes
    start = accelerator.process_index * n_per_gpu
    local_items = [val_ds[i] for i in range(start, start + n_per_gpu)]

    model.eval()
    correct = 0
    rank0_records = [] if accelerator.is_main_process and tag is not None else None
    for it in local_items:
        x_init, di = build_data_info_understanding(
            [it], vl_chat_processor, path_txt,
            VOCABULARY_SIZE_TXT, args.txt_max_length, device)
        final_tokens, text_mask = sample_text_only(
            model=sample_cfg_model, path_txt=path_txt,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT,
            x_init=x_init, data_info=di,
            step_size=step_size, cfg_scale=0.0, n_mc=1, device=device,
        )
        text_tok = final_tokens[text_mask == 1].reshape(1, -1).cpu()
        del final_tokens, x_init, di
        ans = vl_chat_processor.tokenizer.batch_decode(
            text_tok, skip_special_tokens=True)[0]
        scores, _details = reward_fn_eval([ans], [it])
        is_correct = int(scores[0].item() > 0.5)
        correct += is_correct
        if rank0_records is not None:
            rec = {
                "score": float(scores[0].item()),
                "correct": bool(is_correct),
                "gold": it.get("answer", None),
                "pred_raw": ans,
            }
            # Optional extras if present on the dataset item
            for key in ("question", "choices", "image_id", "id", "subject", "topic"):
                if key in it and not isinstance(it[key], (torch.Tensor,)):
                    try:
                        json.dumps(it[key])  # ensure JSON-serializable
                        rec[key] = it[key]
                    except (TypeError, ValueError):
                        pass
            if _details is not None:
                try:
                    d0 = _details[0] if isinstance(_details, (list, tuple)) else _details
                    if isinstance(d0, dict):
                        for k, v in d0.items():
                            if k not in rec:
                                try:
                                    json.dumps(v)
                                    rec[k] = v
                                except (TypeError, ValueError):
                                    pass
                except Exception:
                    pass
            rank0_records.append(rec)

    # All-reduce accuracy
    local = torch.tensor([correct, len(local_items)], dtype=torch.float64, device=device)
    gathered = accelerator.gather(local.unsqueeze(0)).sum(dim=0)
    acc = (gathered[0] / gathered[1]).item() if gathered[1] > 0 else 0.0

    # Dump rank-0 predictions
    if rank0_records is not None:
        eval_dir = os.path.join(save_path, "eval_preds")
        os.makedirs(eval_dir, exist_ok=True)
        out_path = os.path.join(eval_dir, f"{tag}.jsonl")
        with open(out_path, "w") as f:
            header = {
                "_meta": True, "tag": tag,
                "rank": 0, "n_rank0": len(rank0_records),
                "n_total_all_gpus": int(gathered[1].item()),
                "accuracy_all_gpus": acc,
                "rank0_accuracy": (correct / len(local_items)) if local_items else 0.0,
            }
            f.write(json.dumps(header) + "\n")
            for r in rank0_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        accelerator.print(f"  [Eval] rank0 predictions saved to {out_path}")

    model.train()
    torch.cuda.empty_cache()
    return acc, int(gathered[1].item())


# ===================================================================
# Optional step-0 baseline eval
# ===================================================================
if _resumed_global_step == 0:
    accelerator.print("Running step-0 baseline val evaluation...")
    torch.cuda.empty_cache()
    acc0, n0 = evaluate_val(args.eval_num_samples_per_gpu, tag="baseline_step0")
    if accelerator.is_main_process:
        wandb.log({"eval/accuracy": acc0, "eval/n_samples": n0}, step=0)
        accelerator.print(f"  [Baseline] val_accuracy={acc0:.4f} (n={n0})")


# ===================================================================
# Training loop
# ===================================================================
for epoch in range(num_epochs):
    _items_per_epoch = len(dataloader)
    if _total_items_consumed > 0:
        _start_epoch = _total_items_consumed // _items_per_epoch
        _skip_this_epoch = _total_items_consumed % _items_per_epoch
    else:
        _start_epoch, _skip_this_epoch = 0, 0
    if epoch < _start_epoch:
        continue
    _skip_steps = _skip_this_epoch if epoch == _start_epoch else 0

    accelerator.print(f"Epoch {epoch+1}/{num_epochs}"
                      + (f" (skipping {_skip_steps} items)" if _skip_steps else ""))
    dl_iter = iter(dataloader)
    pbar = tqdm(total=_items_per_epoch, desc=f"Epoch {epoch+1}",
                disable=not accelerator.is_main_process)

    prompts_since_step = 0
    accum_metrics = _defaultdict(float)
    accum_rewards_list = []
    accum_rewards_global_list = []  # gathered across GPUs, for wandb logging
    dl_consumed = 0
    # Counters for reward diagnostics (reset per optimizer step).
    _n_filtered_prompts = 0
    _n_prompts_seen_since_step = 0
    _n_judge_used_since_step = 0
    _n_samples_scored_since_step = 0

    while True:
        items = collect_items(dl_iter, P)
        if items is None:
            break
        cur_P = len(items)
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
            _old_policy_params is not None and not _using_ema_for_sampling)
        if _using_old_policy_for_sampling:
            _old_policy_temp = [p.data.clone() for p in trainable_params]
            for p, old_p in zip(trainable_params, _old_policy_params):
                p.data.copy_(old_p)

        # Build batch: each item repeated G times.
        flat_items = []
        for it in items:
            flat_items.extend([it] * G)

        sr_list = []
        remaining = total_G
        offset = 0
        while remaining > 0:
            bs = min(sample_batch_size, remaining)
            sub_items = flat_items[offset:offset + bs]
            x_init_r, di_r = build_data_info_understanding(
                sub_items, vl_chat_processor, path_txt,
                VOCABULARY_SIZE_TXT, args.txt_max_length, device)
            sr_r = sample_text_with_log_prob(
                model=sample_cfg_model, path_txt=path_txt,
                vocabulary_size_txt=VOCABULARY_SIZE_TXT,
                x_init=x_init_r, data_info=di_r,
                step_size=step_size, cfg_scale=0.0, n_mc=n_mc,
                device=device, offload_to_cpu=False,
            )
            sr_list.append(sr_r)
            del x_init_r, di_r
            remaining -= bs
            offset += bs

        # Merge sub-batches. text_mask plays the role of image_mask in the
        # generation-side merge helper; temporarily alias it for reuse.
        for s in sr_list:
            s["image_mask"] = s["text_mask"]
        sr = merge_sampling_results(sr_list)
        sr["text_mask"] = sr.pop("image_mask")
        del sr_list

        # Build full data_info
        _, data_info = build_data_info_understanding(
            flat_items, vl_chat_processor, path_txt,
            VOCABULARY_SIZE_TXT, args.txt_max_length, device)

        # Restore live θ after sampling
        if _using_ema_for_sampling:
            ema.copy_temp_to(trainable_params)
        if _using_old_policy_for_sampling and _old_policy_temp is not None:
            for p, live_p in zip(trainable_params, _old_policy_temp):
                p.data.copy_(live_p)
            _old_policy_temp = None

        torch.cuda.synchronize()
        _t_sample_end = _time.time()

        # ============ Decode answers & compute rewards ============
        text_mask = sr["text_mask"]
        text_tokens = sr["final_tokens"][text_mask == 1].reshape(total_G, -1)
        answers = vl_chat_processor.tokenizer.batch_decode(
            text_tokens, skip_special_tokens=True)
        scores, _details = reward_fn(answers, flat_items)
        rewards = scores.to(device).float()
        if isinstance(_details, dict):
            _n_judge_used_since_step += int(_details.get("n_judge_used", 0))
            _n_samples_scored_since_step += len(answers)
        del text_tokens, sr["final_tokens"]
        torch.cuda.empty_cache()
        _t_reward_end = _time.time()

        K = len(sr["all_x1_samples"])

        # ============ Phase 2: Advantages ============
        _global_std = rewards.std() if args.global_std else None
        advantages = torch.zeros_like(rewards)
        for pi in range(cur_P):
            r_group = rewards[pi * G: (pi + 1) * G]
            if args.global_std:
                denom = (_global_std + 1e-8)
                adv_g = (r_group - r_group.mean()) / denom
            else:
                std = r_group.std()
                adv_g = torch.zeros_like(r_group) if std < 1e-8 else \
                        (r_group - r_group.mean()) / (std + 1e-8)
            if args.adv_clip_max > 0:
                adv_g = torch.clamp(adv_g, -args.adv_clip_max, args.adv_clip_max)
            advantages[pi * G: (pi + 1) * G] = adv_g

        # Filter zero-std groups
        valid_g_indices = list(range(total_G))
        _n_filtered_this_batch = 0
        if args.filter_zero_std:
            valid_g_indices = []
            for pi in range(cur_P):
                s, e = pi * G, (pi + 1) * G
                if advantages[s:e].abs().sum().item() >= 1e-8:
                    valid_g_indices.extend(range(s, e))
                else:
                    _n_filtered_this_batch += 1
        _n_filtered_prompts += _n_filtered_this_batch
        _n_prompts_seen_since_step += cur_P

        _all_filtered = len(valid_g_indices) == 0

        # ============ Phase 3: Train ============
        model.train()
        if not _all_filtered:
            prompts_since_step += 1
            is_last = prompts_since_step >= grad_accum_steps
            eff_G = len(valid_g_indices)
            chunk = max(1, eff_G // num_inner_updates)
            perm = [valid_g_indices[i] for i in torch.randperm(eff_G).tolist()]

            # Adapter `text_mask` on ``data_info`` so downstream routines named
            # ``image_token_mask`` are not confused; we use ``text_token_mask`` already present.

            for inner in range(num_inner_updates):
                s0 = inner * chunk
                s1 = eff_G if inner == num_inner_updates - 1 else s0 + chunk
                g_chunk = perm[s0:s1]
                metrics = train_on_batch_u(
                    sr=sr, data_info=data_info, advantages=advantages,
                    K=K, g_indices=g_chunk,
                    is_last_prompt_in_accum=(is_last and inner == num_inner_updates - 1),
                    diff_cfg_model=diff_cfg_model, model=model,
                    accelerator=accelerator, device=device,
                    args=args, train_step_set=train_step_set,
                    n_train_steps=n_train_steps, train_batch_size=train_batch_size,
                    grad_accum_steps=grad_accum_steps, num_inner_updates=num_inner_updates,
                    epsilon_low=epsilon_low, epsilon_high=epsilon_high,
                    kl_beta=kl_beta, TRAIN_CFG_SCALE=TRAIN_CFG_SCALE,
                    disable_adapter_and_gen_modules=disable_adapter_and_gen_modules,
                    raw_model=raw_model,
                )
                for key, val in metrics.items():
                    accum_metrics[key] += val
        else:
            if accelerator.is_main_process:
                accelerator.print(f"  [filter_zero_std] all {cur_P} groups filtered — skipping")

        torch.cuda.synchronize()
        _t_train_end = _time.time()
        if accelerator.is_main_process and dl_consumed <= _skip_steps + 10 * P:
            accelerator.print(
                f"  [Timing P={cur_P}] total={_t_train_end - _t0:.1f}s  "
                f"sample={_t_sample_end - _t_sample_start:.1f}s  "
                f"reward={_t_reward_end - _t_sample_end:.1f}s  "
                f"train={_t_train_end - _t_reward_end:.1f}s")

        accum_rewards_list.append(rewards.detach().cpu())

        # Cross-GPU reward gather for logging (training-time accuracy / reward).
        # NOTE: advantages above are intentionally per-rank (each rank has its
        # own prompts/groups). The gather here is purely for wandb logging so
        # that `reward_mean` reflects the true global training accuracy.
        _rewards_global = accelerator.gather(rewards.detach()).cpu()
        accum_rewards_global_list.append(_rewards_global)

        # Build reward-log for this prompt-batch (to merge at optimizer step)
        if accelerator.is_main_process:
            per_prompt_stds = []
            for rew_cpu in accum_rewards_list:
                n_p = rew_cpu.shape[0] // G
                for pi in range(n_p):
                    per_prompt_stds.append(rew_cpu[pi * G:(pi + 1) * G].std())
            all_stds_t = torch.stack(per_prompt_stds) if per_prompt_stds else torch.zeros(1)
            zero_std_count = (all_stds_t < 1e-8).sum().item()
            all_rew_cat = torch.cat(accum_rewards_list)
            # Global (cross-GPU) reward / accuracy for logging.
            all_rew_global_cat = (torch.cat(accum_rewards_global_list)
                                  if accum_rewards_global_list else all_rew_cat)
            _zero_std_ratio = (zero_std_count / max(1, _n_prompts_seen_since_step))
            accum_reward_log = {
                # Top-level names match image-generation train_GRPO.py for
                # direct cross-run comparison in wandb. ``reward_mean`` /
                # ``reward_std`` are aggregated across all GPUs so the curve
                # tracks the true global training accuracy.
                "reward_mean": all_rew_global_cat.mean().item(),
                "reward_std": all_rew_global_cat.std().item() if all_rew_global_cat.numel() > 1 else 0.0,
                "reward/rank0_mean": all_rew_cat.mean().item(),
                "reward/rank0_std": all_rew_cat.std().item() if all_rew_cat.numel() > 1 else 0.0,
                "reward/per_prompt_std_mean": all_stds_t.mean().item(),
                "reward/per_prompt_std_min": all_stds_t.min().item(),
                "reward/per_prompt_std_max": all_stds_t.max().item(),
                "reward/zero_std_prompt_count": zero_std_count,
                "reward/zero_std_ratio": _zero_std_ratio,
                "reward/filtered_prompt_count": _n_filtered_prompts,
                "reward/judge_used_count": _n_judge_used_since_step,
                "reward/judge_used_ratio": (
                    _n_judge_used_since_step / max(1, _n_samples_scored_since_step)
                ),
                "advantage/mean": advantages.mean().item(),
                "advantage/std": advantages.std().item() if advantages.numel() > 1 else 0.0,
                "advantage/max": advantages.max().item(),
                "advantage/min": advantages.min().item(),
            }
            if args.global_std and _global_std is not None:
                accum_reward_log["reward/global_reward_std"] = _global_std.item()
        else:
            accum_reward_log = {}

        pbar.set_postfix(reward=f"{rewards.mean().item():.3f}",
                         accum=f"{prompts_since_step}/{grad_accum_steps}",
                         step=global_step)

        # Free step
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
                tc = accum_metrics["total_count"]
                log_dict = {
                    "loss": accum_metrics["loss"],
                    "kl_loss": accum_metrics["kl_loss"],
                    "total_loss": accum_metrics["loss"] + accum_metrics["kl_loss"],
                    "grad/grad_norm": grad_norm.item() if hasattr(grad_norm, "item") else float(grad_norm),
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
            accum_rewards_list = []
            accum_rewards_global_list = []
            accum_reward_log = {}
            optimizer.zero_grad(set_to_none=True)
            # Reset reward-diagnostic counters after each optimizer step so
            # `reward/filtered_prompt_count` and `reward/zero_std_ratio` are
            # per-step (not cumulative over the epoch).
            _n_filtered_prompts = 0
            _n_prompts_seen_since_step = 0
            _n_judge_used_since_step = 0
            _n_samples_scored_since_step = 0

            # ---- Periodic Eval ----
            if global_step % args.eval_every == 0:
                acc, n_ev = evaluate_val(args.eval_num_samples_per_gpu,
                                         tag=f"step{global_step:06d}")
                if accelerator.is_main_process:
                    wandb.log({"eval/accuracy": acc, "eval/n_samples": n_ev}, step=global_step)
                    accelerator.print(f"  [Eval step {global_step}] accuracy={acc:.4f} (n={n_ev})")
                torch.cuda.empty_cache()

            # ---- EMA eval ----
            if ema is not None and args.eval_ema_every > 0 and global_step % args.eval_ema_every == 0:
                try:
                    ema.copy_ema_to(trainable_params, store_temp=True)
                    acc_ema, n_ev = evaluate_val(args.eval_num_samples_per_gpu,
                                                 tag=f"step{global_step:06d}_ema")
                    if accelerator.is_main_process:
                        wandb.log({"eval_ema/accuracy": acc_ema, "eval_ema/n_samples": n_ev},
                                  step=global_step)
                        accelerator.print(f"  [EMA Eval {global_step}] accuracy={acc_ema:.4f}")
                finally:
                    ema.copy_temp_to(trainable_params)
                torch.cuda.empty_cache()

            # ---- Save ----
            if global_step % args.save_every == 0:
                ckpt = os.path.join(save_path,
                                    f"checkpoint_{run_tag}_epoch{epoch+1}_step{global_step}")
                os.makedirs(ckpt, exist_ok=True)
                if args.use_lora and accelerator.is_main_process:
                    accelerator.unwrap_model(model).language_model.save_pretrained(
                        os.path.join(ckpt, "lora_adapter"))
                accelerator.save_state(ckpt)
                if accelerator.is_main_process:
                    with open(os.path.join(ckpt, "training_meta.json"), "w") as f:
                        json.dump({"global_step": global_step, "epoch": epoch + 1}, f)
                if ema is not None and accelerator.is_main_process:
                    torch.save(ema.state_dict(), os.path.join(ckpt, "ema_state.pt"))

    pbar.close()

    # ---- End-of-epoch checkpoint ----
    ed = os.path.join(save_path, f"checkpoint_{run_tag}_epoch{epoch+1}")
    os.makedirs(ed, exist_ok=True)
    if args.use_lora and accelerator.is_main_process:
        accelerator.unwrap_model(model).language_model.save_pretrained(
            os.path.join(ed, "lora_adapter"))
    accelerator.save_state(ed)
    if accelerator.is_main_process:
        with open(os.path.join(ed, "training_meta.json"), "w") as f:
            json.dump({"global_step": global_step, "epoch": epoch + 1}, f)
    if ema is not None and accelerator.is_main_process:
        torch.save(ema.state_dict(), os.path.join(ed, "ema_state.pt"))
    accelerator.print(f"Epoch {epoch+1} saved to {ed}")
