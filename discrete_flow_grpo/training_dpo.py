"""
Core training functions for on-policy DPO on discrete flow matching.

Loss (per (winner +, loser -) pair, summed over the K-1 trainable Euler steps):

    Δ_pair = (1 / (K-1)) Σ_k [ log π*(x⁺_{t_k}|x⁺_{t_{k-1}}) / π_old(x⁺_{t_k}|x⁺_{t_{k-1}})
                              - log π*(x⁻_{t_k}|x⁻_{t_{k-1}}) / π_old(x⁻_{t_k}|x⁻_{t_{k-1}}) ]

    L_pair = - log σ( β' · Δ_pair )

The per-step log-ratio  log π*(x_{t_k}|x_{t_{k-1}}) / π_old(x_{t_k}|x_{t_{k-1}})
is computed exactly the same way as in GRPO via importance reweighting:

    ratio_per_dim   = (Σ_j p_θ(X_{1,j}^d | x_{t_{k-1}}) · factor_j^d) / p_old_times_factor^d
    log_ratio_k     = mean_d log( ratio_per_dim )

i.e. the same MC estimator used in training_grpo.py (sample-level geometric mean
of per-dim ratios over D image tokens). We pick "winner" / "loser" pairs by
reward inside each prompt group and optimize the Bradley-Terry style objective
above. NO advantage normalization, NO PPO clip — DPO naturally bounds gradient
magnitude through the sigmoid.
"""

import torch
import torch.nn.functional as F
from contextlib import nullcontext

from config import VOCABULARY_SIZE_IMG


# ---------------------------------------------------------------------------
# Reused helpers (same as training_grpo.py)
# ---------------------------------------------------------------------------
def parse_train_steps(spec, K, max_step):
    if spec is None or spec.lower() == "all":
        return list(range(max_step + 1))
    steps = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            steps.update(range(int(lo), int(hi) + 1))
        else:
            steps.add(int(part))
    for s in steps:
        if s < 0 or s > max_step:
            raise ValueError(f"--train_steps: step {s} out of range [0, {max_step}] for K={K}")
    return sorted(steps)


def merge_sampling_results(sr_list):
    K = len(sr_list[0]["all_x1_samples"])
    merged = {
        "final_tokens": torch.cat([s["final_tokens"] for s in sr_list], dim=0),
        "image_mask": torch.cat([s["image_mask"] for s in sr_list], dim=0),
        "is_last_step": sr_list[0]["is_last_step"],
        "n_mc": sr_list[0]["n_mc"],
    }
    merged["trajectory"] = [
        torch.cat([s["trajectory"][k] for s in sr_list], dim=0)
        for k in range(len(sr_list[0]["trajectory"]))
    ]
    merged["all_x1_samples"] = [
        [torch.cat([s["all_x1_samples"][k][j] for s in sr_list], dim=0)
         for j in range(len(sr_list[0]["all_x1_samples"][k]))]
        for k in range(K)
    ]
    merged["all_factors"] = [
        [torch.cat([s["all_factors"][k][j] for s in sr_list], dim=0)
         for j in range(len(sr_list[0]["all_factors"][k]))]
        for k in range(K)
    ]
    merged["all_p_old_times_factor"] = [
        torch.cat([s["all_p_old_times_factor"][k] for s in sr_list], dim=0)
        for k in range(K)
    ]
    merged["changed_masks"] = [
        torch.cat([s["changed_masks"][k] for s in sr_list], dim=0)
        for k in range(K)
    ]
    return merged


def collect_prompts(dataloader_iter, P, use_geneval_metadata=False):
    prompts, metadatas = [], []
    for _ in range(P):
        try:
            batch = next(dataloader_iter)
            if use_geneval_metadata:
                prompts.append(batch[0][0])
                metadatas.append(batch[1][0])
            else:
                prompts.append(batch[0])
        except StopIteration:
            break
    if not prompts:
        return None, None
    return prompts, metadatas if use_geneval_metadata else None


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------
def build_pairs(rewards, cur_P, G, num_pairs=0, filter_zero_std=False, eps=1e-8):
    """
    For each prompt group of G samples, sort by reward descending and pair
    top-i with bottom-i:
        pairs_pi = [(top[0], bot[0]), (top[1], bot[1]), ..., (top[m-1], bot[m-1])]
    where m = num_pairs (or G//2 if num_pairs<=0).

    Returns:
        pos_idx, neg_idx : LongTensors of global sample indices (length = total pairs)
        n_filtered_prompts : int
    """
    device = rewards.device
    m_target = num_pairs if num_pairs > 0 else (G // 2)
    m_target = max(1, min(m_target, G // 2))

    pos_list, neg_list = [], []
    n_filtered = 0
    for pi in range(cur_P):
        r_group = rewards[pi * G : (pi + 1) * G]
        if filter_zero_std and r_group.std().item() < eps:
            n_filtered += 1
            continue
        sorted_idx = torch.argsort(r_group, descending=True)  # local indices in group
        top = sorted_idx[:m_target] + pi * G
        bot = sorted_idx[-m_target:].flip(0) + pi * G  # worst first to align with best
        # require strict reward gap (skip ties)
        for t_g, b_g in zip(top.tolist(), bot.tolist()):
            if rewards[t_g].item() - rewards[b_g].item() > eps:
                pos_list.append(t_g)
                neg_list.append(b_g)
    if not pos_list:
        return (torch.empty(0, dtype=torch.long, device=device),
                torch.empty(0, dtype=torch.long, device=device),
                n_filtered)
    return (torch.tensor(pos_list, dtype=torch.long, device=device),
            torch.tensor(neg_list, dtype=torch.long, device=device),
            n_filtered)


# ---------------------------------------------------------------------------
# Per-step log-ratio (same estimator as GRPO)
# ---------------------------------------------------------------------------
def _compute_log_ratio_for_indices(
    sr, data_info, k, sample_idxs, diff_cfg_model, TRAIN_CFG_SCALE,
    need_ref, disable_adapter_and_gen_modules, raw_model, device,
):
    """
    Compute log( π_θ(x_{t_k}|x_{t_{k-1}}) / π_old(x_{t_k}|x_{t_{k-1}}) ) for each
    sample in sample_idxs at denoising step k.

    Mirrors training_grpo.train_on_prompt's per-step ratio computation, returning
    sample-level log-ratio (mean over D image tokens), differentiable w.r.t. θ.

    Returns:
        log_ratio_k : (B,) — differentiable
        log_ratio_ref_k : (B,) or None — for KL vs π_ref (if need_ref)
    """
    B = sample_idxs.shape[0]
    x_t_batch = sr["trajectory"][k][sample_idxs]
    di_batch = {key: (v[sample_idxs] if isinstance(v, torch.Tensor) else v)
                for key, v in data_info.items()}

    _, p_new, _ = diff_cfg_model(x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
    p_new_img = p_new[di_batch["image_token_mask"] == 1].reshape(B, -1, VOCABULARY_SIZE_IMG)
    D = p_new_img.shape[1]

    n_mc_k = len(sr["all_x1_samples"][k])

    if sr["is_last_step"][k]:
        x1_last = sr["all_x1_samples"][k][0][sample_idxs]
        fac_last = sr["all_factors"][k][0][sample_idxs]
        p_new_gathered = torch.gather(p_new_img, -1, x1_last.unsqueeze(-1)).squeeze(-1)
        numer = p_new_gathered * fac_last  # ratio_per_dim numerator (= ratio itself, since denom=1)
        ratio_per_dim = numer.clamp(min=1e-30)
        if need_ref:
            with torch.no_grad(), disable_adapter_and_gen_modules(raw_model):
                _, p_ref, _ = diff_cfg_model(x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
            p_ref_img = p_ref[di_batch["image_token_mask"] == 1].reshape(B, -1, VOCABULARY_SIZE_IMG)
            p_ref_gathered = torch.gather(p_ref_img, -1, x1_last.unsqueeze(-1)).squeeze(-1)
            numer_ref = p_ref_gathered * fac_last
        else:
            numer_ref = None
    else:
        numer = torch.zeros(B, D, device=device)
        if need_ref:
            with torch.no_grad(), disable_adapter_and_gen_modules(raw_model):
                _, p_ref, _ = diff_cfg_model(x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
            p_ref_img = p_ref[di_batch["image_token_mask"] == 1].reshape(B, -1, VOCABULARY_SIZE_IMG)
            numer_ref = torch.zeros(B, D, device=device)
        else:
            numer_ref = None

        for j in range(n_mc_k):
            x1_j = sr["all_x1_samples"][k][j][sample_idxs]
            fac_j = sr["all_factors"][k][j][sample_idxs]
            p_new_j = torch.gather(p_new_img, -1, x1_j.unsqueeze(-1)).squeeze(-1)
            numer = numer + p_new_j * fac_j
            if need_ref:
                p_ref_j = torch.gather(p_ref_img, -1, x1_j.unsqueeze(-1)).squeeze(-1)
                numer_ref = numer_ref + p_ref_j * fac_j
        numer = numer / n_mc_k
        if need_ref:
            numer_ref = numer_ref / n_mc_k

        denom = sr["all_p_old_times_factor"][k][sample_idxs]
        ratio_per_dim = numer.clamp(min=1e-30) / denom.clamp(min=1e-30)

    log_ratio_per_dim_raw = torch.log(ratio_per_dim.clamp(min=1e-30))
    # Mild per-token log-ratio clamp to prevent single-token spikes (when π_old
    # gives very low prob to a token that π_θ now upweights, ratio can blow up
    # to 1e3+ and dominate the mean). DPO has no PPO clip, so we cap here.
    log_ratio_per_dim = log_ratio_per_dim_raw.clamp(min=-5.0, max=5.0)
    # Diagnostic: fraction of per-dim log-ratios that hit the clamp boundary.
    with torch.no_grad():
        clamp_hit = ((log_ratio_per_dim_raw <= -5.0) | (log_ratio_per_dim_raw >= 5.0)).float()
        clamp_frac_k = clamp_hit.mean()  # scalar
    log_ratio_k = log_ratio_per_dim.mean(dim=-1)  # (B,) — sample-level geometric mean

    log_ratio_ref_k = None
    if need_ref:
        ratio_ref_per_dim = numer / numer_ref.clamp(min=1e-30)
        log_ratio_ref_per_dim = torch.log(ratio_ref_per_dim.clamp(min=1e-30))
        log_ratio_ref_per_dim = log_ratio_ref_per_dim.clamp(min=-5.0, max=5.0)
        log_ratio_ref_k = log_ratio_ref_per_dim.mean(dim=-1)

    return log_ratio_k, log_ratio_ref_k, clamp_frac_k


# ---------------------------------------------------------------------------
# DPO training step
# ---------------------------------------------------------------------------
def train_on_prompt(
    sr, data_info, pos_idx, neg_idx, K,
    is_last_prompt_in_accum,
    diff_cfg_model, model, accelerator, device,
    args, train_step_set, n_train_steps, train_batch_size,
    grad_accum_steps, num_inner_updates,
    beta_dpo, kl_beta, TRAIN_CFG_SCALE,
    disable_adapter_and_gen_modules=None, raw_model=None,
):
    """
    Per-step DPO loss (mirrors flow_grpo/scripts/train_sd3_dpo.py): each
    denoising step k contributes its own Bradley-Terry term, and we backward
    immediately after each step's forward, so peak memory matches GRPO.

        For each pair p = (i+, i-) and each trained step k ∈ S:
            δ_{p,k} = log_ratio_k[i+] - log_ratio_k[i-]
            L_{p,k} = - log σ( β' · δ_{p,k} )

        Total loss = Σ_p Σ_{k∈S} L_{p,k}, normalized by:
            normalizer = N_pairs · |S| · grad_accum_steps · num_inner_updates

    This is the discrete-flow analogue of `for j in train_timesteps: ...
    accelerator.backward(loss)` in train_sd3_dpo.py:864-944.

    Mini-batching: we process up to `train_batch_size` pairs per forward pass
    (=> 2 · train_batch_size samples in the model forward).

    Returns: dict of scalar metrics.
    """
    n_pairs = pos_idx.shape[0]
    if n_pairs == 0:
        return {k: 0.0 for k in (
            "loss", "kl_loss", "dpo_acc_count", "dpo_total", "delta_sum",
            "delta_sq_sum", "abs_delta_max", "logits_sum", "abs_logits_max",
            "sat_count", "win_prob_sum",
            "log_ratio_pos_sum", "log_ratio_pos_sq_sum",
            "log_ratio_neg_sum", "log_ratio_neg_sq_sum",
            "clamp_frac_sum", "clamp_frac_count", "ref_kl_sum",
        )}

    loss_total = 0.0
    kl_total = 0.0
    delta_sum = 0.0          # accumulates Σ_k δ_{p,k} (for "trajectory-level" Δ logging)
    delta_sq_sum = 0.0       # for std
    abs_delta_max = 0.0
    logits_sum = 0.0         # Σ β'·δ
    abs_logits_max = 0.0
    sat_count = 0            # |β'·δ| > 5 (σ saturation)
    win_prob_sum = 0.0       # Σ σ(β'·δ)
    log_ratio_pos_sum = 0.0
    log_ratio_pos_sq_sum = 0.0
    log_ratio_neg_sum = 0.0
    log_ratio_neg_sq_sum = 0.0
    clamp_frac_sum = 0.0     # Σ_k frac(per-dim log-ratio hit ±5)
    clamp_frac_count = 0
    dpo_acc_count = 0        # acc on per-step δ > 0 (token of comparison: each (pair, step) pair)
    dpo_total = 0
    ref_kl_sum = 0.0

    sync_context = (lambda: nullcontext()) if is_last_prompt_in_accum else (
        (lambda: accelerator.no_sync(model)) if hasattr(accelerator, 'no_sync') else
        (model.no_sync if hasattr(model, 'no_sync') else (lambda: nullcontext()))
    )

    need_ref = (kl_beta > 0 and args.use_lora and not args.kl_old_policy)
    n_steps_eff = max(n_train_steps, 1)

    # Per-step-and-per-pair normalizer: divides every backward'd scalar
    # consistently so that gradient magnitudes are independent of n_pairs / |S|.
    normalizer = n_pairs * n_steps_eff * grad_accum_steps * num_inner_updates

    with sync_context():
        # Outer loop over denoising steps -> inner loop over pair-batches.
        # Each (step, batch) does one independent forward + backward, so the
        # autograd graph is released between iterations (peak mem == GRPO).
        for k in range(K):
            if k not in train_step_set:
                continue

            for b_start in range(0, n_pairs, train_batch_size):
                b_end = min(b_start + train_batch_size, n_pairs)
                B_pair = b_end - b_start
                pos_b = pos_idx[b_start:b_end]
                neg_b = neg_idx[b_start:b_end]
                # Stack winners + losers into one mini-batch of 2 B_pair samples
                both_idx = torch.cat([pos_b, neg_b], dim=0)

                log_ratio_k, log_ratio_ref_k, clamp_frac_k = _compute_log_ratio_for_indices(
                    sr=sr, data_info=data_info, k=k, sample_idxs=both_idx,
                    diff_cfg_model=diff_cfg_model, TRAIN_CFG_SCALE=TRAIN_CFG_SCALE,
                    need_ref=need_ref,
                    disable_adapter_and_gen_modules=disable_adapter_and_gen_modules,
                    raw_model=raw_model, device=device,
                )
                lr_pos = log_ratio_k[:B_pair]
                lr_neg = log_ratio_k[B_pair:]
                delta_k = lr_pos - lr_neg                      # (B_pair,)

                # Per-step BT loss; β' applies directly (no /|S| division).
                logits_k = beta_dpo * delta_k
                pair_loss_k = F.softplus(-logits_k)            # = -log σ(logits_k)
                step_loss = pair_loss_k.sum() / normalizer

                if kl_beta > 0:
                    if need_ref and log_ratio_ref_k is not None:
                        ratio_ref = torch.exp(log_ratio_ref_k)
                        ref_kl_step = (ratio_ref - 1) - log_ratio_ref_k  # (2 B_pair,)
                        kl_per_pair_k = 0.5 * (ref_kl_step[:B_pair] + ref_kl_step[B_pair:])
                        ref_kl_sum += kl_per_pair_k.detach().sum().item()
                    else:
                        # vs π_old surrogate: penalize |log_ratio| magnitude.
                        kl_per_pair_k = 0.5 * (lr_pos.detach().abs() + lr_neg.detach().abs())
                    kl_loss_k = kl_beta * kl_per_pair_k.sum() / normalizer
                    step_loss = step_loss + kl_loss_k
                    kl_total += float(kl_loss_k.detach().item()) if torch.is_tensor(kl_loss_k) else float(kl_loss_k)

                accelerator.backward(step_loss)
                loss_total += float(step_loss.detach().item())

                with torch.no_grad():
                    delta_sum += delta_k.sum().item()
                    delta_sq_sum += (delta_k ** 2).sum().item()
                    abs_delta_max = max(abs_delta_max, delta_k.abs().max().item())
                    logits_sum += logits_k.sum().item()
                    abs_logits_max = max(abs_logits_max, logits_k.abs().max().item())
                    sat_count += (logits_k.abs() > 5.0).sum().item()
                    win_prob_sum += torch.sigmoid(logits_k).sum().item()
                    dpo_acc_count += (delta_k > 0).sum().item()
                    dpo_total += B_pair
                    log_ratio_pos_sum += lr_pos.sum().item()
                    log_ratio_pos_sq_sum += (lr_pos ** 2).sum().item()
                    log_ratio_neg_sum += lr_neg.sum().item()
                    log_ratio_neg_sq_sum += (lr_neg ** 2).sum().item()
                    clamp_frac_sum += clamp_frac_k.item()
                    clamp_frac_count += 1

    # Normalize "*_sum" stats so they are per-pair (averaged over the |S| steps),
    # matching the previous return-value semantics expected by the trainer.
    log_ratio_pos_sum = log_ratio_pos_sum / n_steps_eff
    log_ratio_pos_sq_sum = log_ratio_pos_sq_sum / n_steps_eff
    log_ratio_neg_sum = log_ratio_neg_sum / n_steps_eff
    log_ratio_neg_sq_sum = log_ratio_neg_sq_sum / n_steps_eff
    delta_sum = delta_sum / n_steps_eff
    delta_sq_sum = delta_sq_sum / n_steps_eff
    logits_sum = logits_sum / n_steps_eff
    win_prob_sum = win_prob_sum / n_steps_eff
    if need_ref:
        ref_kl_sum = ref_kl_sum / n_steps_eff

    return {
        "loss": loss_total,
        "kl_loss": kl_total,
        "dpo_acc_count": dpo_acc_count,
        "dpo_total": dpo_total,
        "delta_sum": delta_sum,
        "delta_sq_sum": delta_sq_sum,
        "abs_delta_max": abs_delta_max,
        "logits_sum": logits_sum,
        "abs_logits_max": abs_logits_max,
        "sat_count": sat_count,
        "win_prob_sum": win_prob_sum,
        "log_ratio_pos_sum": log_ratio_pos_sum,
        "log_ratio_pos_sq_sum": log_ratio_pos_sq_sum,
        "log_ratio_neg_sum": log_ratio_neg_sum,
        "log_ratio_neg_sq_sum": log_ratio_neg_sq_sum,
        "clamp_frac_sum": clamp_frac_sum,
        "clamp_frac_count": clamp_frac_count,
        "ref_kl_sum": ref_kl_sum,
    }
