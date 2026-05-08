"""
Core training functions for GRPO.

Contains:
  - _train_on_prompt()        — PPO surrogate loss computation (forward/backward)
  - _merge_sampling_results() — merge multi-round sampling results
  - _collect_prompts()        — collect P prompts from dataloader iterator
  - parse_train_steps()       — parse --train_steps spec into step indices
"""

import torch
from contextlib import nullcontext

from config import VOCABULARY_SIZE_IMG


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_train_steps(spec, K, max_step):
    """Parse step spec like '4-6', '0,1,4-6', 'all', or None -> sorted list of ints.
    max_step: inclusive upper bound for valid step indices."""
    if spec is None or spec.lower() == "all":
        return list(range(max_step + 1))
    steps = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            steps.update(range(lo, hi + 1))
        else:
            steps.add(int(part))
    for s in steps:
        if s < 0 or s > max_step:
            raise ValueError(f"--train_steps: step {s} out of range [0, {max_step}] for K={K}")
    return sorted(steps)


def merge_sampling_results(sr_list):
    """
    Merge multiple sampling results (each with batch dim = sub_batch) into one.
    Tensors stay on whichever device they were produced on.
    """
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
    """Collect P prompts from the dataloader iterator.

    Returns:
        (prompt_list, metadata_list) tuple.
        prompt_list: list of str, metadata_list: list of dict (or None).
        Returns (None, None) if exhausted.
    """
    prompts = []
    metadatas = []
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
# Core training step
# ---------------------------------------------------------------------------
def train_on_prompt(
    sr, data_info, advantages, advantages_per_step, K,
    g_indices, is_last_prompt_in_accum,
    # Model / accelerator
    diff_cfg_model, model, accelerator, device,
    # Hyper-params
    args, train_step_set, n_train_steps, train_batch_size,
    grad_accum_steps, num_inner_updates,
    epsilon_low, epsilon_high, kl_beta, TRAIN_CFG_SCALE,
    # Optional: KL reference
    disable_adapter_and_gen_modules=None, raw_model=None,
):
    """
    Run training forward/backward for one prompt's data over the given g_indices.
    Accumulates gradients (does NOT call optimizer.step()).

    Returns: dict of scalar metrics
    """
    chunk_size = len(g_indices)
    g_idxs_t = torch.tensor(g_indices, dtype=torch.long, device=device)

    loss_total = 0.0
    kl_total = 0.0
    clip_count = 0
    clip_high = 0
    clip_low = 0
    total_count = 0
    approx_kl = 0.0
    ratio_sum = 0.0
    log_ratio_mean_sum = 0.0
    log_ratio_sum_sum = 0.0
    token_clip_count = 0
    token_total_count = 0
    log_ratio_mean_pos_sum = 0.0
    log_ratio_mean_neg_sum = 0.0
    count_pos = 0
    count_neg = 0
    changed_count_total = 0
    total_dim_count = 0
    dim_clip_count = 0
    dim_clip_total = 0

    sync_context = (lambda: nullcontext()) if is_last_prompt_in_accum else (
        (lambda: accelerator.no_sync(model)) if hasattr(accelerator, 'no_sync') else
        (model.no_sync if hasattr(model, 'no_sync') else (lambda: nullcontext()))
    )

    with sync_context():
        for k in range(K):
            if k not in train_step_set:
                continue
            discount_k = args.gamma ** k

            for b_start in range(0, chunk_size, train_batch_size):
                b_end = min(b_start + train_batch_size, chunk_size)
                B_batch = b_end - b_start
                batch_g_idxs = g_idxs_t[b_start:b_end]

                x_t_batch = sr["trajectory"][k][batch_g_idxs]
                di_batch = {key: (v[batch_g_idxs] if isinstance(v, torch.Tensor) else v)
                            for key, v in data_info.items()}

                _, p_new, _ = diff_cfg_model(x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
                p_new_img = p_new[di_batch["image_token_mask"] == 1].reshape(
                    B_batch, -1, VOCABULARY_SIZE_IMG)

                n_mc_k = len(sr["all_x1_samples"][k])
                D = p_new_img.shape[1]

                need_ref = (kl_beta > 0 and args.use_lora and not args.kl_old_policy)

                # --- Fast path for the last denoising step (K-1) ---
                if sr["is_last_step"][k]:
                    x1_last = sr["all_x1_samples"][k][0][batch_g_idxs]
                    fac_last = sr["all_factors"][k][0][batch_g_idxs]
                    p_new_gathered = torch.gather(
                        p_new_img, -1, x1_last.unsqueeze(-1)
                    ).squeeze(-1)
                    ratio_per_dim = (p_new_gathered * fac_last).clamp(min=1e-30)
                    numer = p_new_gathered * fac_last

                    if need_ref:
                        with torch.no_grad(), disable_adapter_and_gen_modules(raw_model):
                            _, p_ref, _ = diff_cfg_model(
                                x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
                        p_ref_img = p_ref[di_batch["image_token_mask"] == 1].reshape(
                            B_batch, -1, VOCABULARY_SIZE_IMG)
                        p_ref_gathered = torch.gather(
                            p_ref_img, -1, x1_last.unsqueeze(-1)
                        ).squeeze(-1)
                        numer_ref = p_ref_gathered * fac_last
                        del p_ref, p_ref_img, p_ref_gathered

                    del x1_last, fac_last, p_new_gathered
                else:
                    # --- Generic MC path for intermediate steps ---
                    numer = torch.zeros(B_batch, D, device=device)

                    if need_ref:
                        with torch.no_grad(), disable_adapter_and_gen_modules(raw_model):
                            _, p_ref, _ = diff_cfg_model(
                                x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
                        p_ref_img = p_ref[di_batch["image_token_mask"] == 1].reshape(
                            B_batch, -1, VOCABULARY_SIZE_IMG)
                        numer_ref = torch.zeros(B_batch, D, device=device)

                    for j in range(n_mc_k):
                        x1_j = sr["all_x1_samples"][k][j][batch_g_idxs]
                        fac_j = sr["all_factors"][k][j][batch_g_idxs]
                        p_new_j = torch.gather(
                            p_new_img, -1, x1_j.unsqueeze(-1)
                        ).squeeze(-1)
                        numer += p_new_j * fac_j
                        if need_ref:
                            p_ref_j = torch.gather(
                                p_ref_img, -1, x1_j.unsqueeze(-1)
                            ).squeeze(-1)
                            numer_ref += p_ref_j * fac_j
                            del p_ref_j
                        del x1_j, fac_j, p_new_j
                    numer = numer / n_mc_k
                    if need_ref:
                        numer_ref = numer_ref / n_mc_k
                        del p_ref, p_ref_img

                    denom = sr["all_p_old_times_factor"][k][batch_g_idxs]
                    ratio_per_dim = numer.clamp(min=1e-30) / denom.clamp(min=1e-30)

                # log ratio per token: (B, D)
                log_ratio_per_dim = torch.log(ratio_per_dim.clamp(min=1e-30))

                # --changed_only masking
                if args.changed_only:
                    ch_mask = sr["changed_masks"][k][batch_g_idxs]
                    n_changed_per_sample = ch_mask.sum(dim=-1).clamp(min=1)
                    log_ratio_k = (log_ratio_per_dim * ch_mask).sum(dim=-1) / n_changed_per_sample
                    with torch.no_grad():
                        changed_count_total += ch_mask.sum().item()
                        total_dim_count += B_batch * D
                else:
                    ch_mask = None
                    log_ratio_k = log_ratio_per_dim.mean(dim=-1)
                    with torch.no_grad():
                        total_dim_count += B_batch * D
                        if "changed_masks" in sr:
                            _ch = sr["changed_masks"][k][batch_g_idxs]
                            changed_count_total += _ch.sum().item()
                            del _ch

                ratio_k = torch.exp(log_ratio_k)

                with torch.no_grad():
                    log_ratio_mean_sum += log_ratio_k.sum().item()
                    log_ratio_sum_sum += log_ratio_per_dim.sum(dim=-1).sum().item()
                    if args.dim_level_clip:
                        _high_dim = (ratio_per_dim > 1 + epsilon_high)
                        _low_dim = (ratio_per_dim < 1 - epsilon_low)
                        if args.changed_only and ch_mask is not None:
                            clip_high += (_high_dim & ch_mask).sum().item()
                            clip_low += (_low_dim & ch_mask).sum().item()
                            clip_count += ((_high_dim | _low_dim) & ch_mask).sum().item()
                            total_count += ch_mask.sum().item()
                        else:
                            clip_high += _high_dim.sum().item()
                            clip_low += _low_dim.sum().item()
                            clip_count += (_high_dim | _low_dim).sum().item()
                            total_count += B_batch * D
                        del _high_dim, _low_dim
                    else:
                        clipped_high_b = (ratio_k > 1 + epsilon_high).sum().item()
                        clipped_low_b = (ratio_k < 1 - epsilon_low).sum().item()
                        clip_count += clipped_high_b + clipped_low_b
                        clip_high += clipped_high_b
                        clip_low += clipped_low_b
                        total_count += B_batch
                    approx_kl += ((ratio_k - 1) - log_ratio_k).sum().item()
                    ratio_sum += ratio_k.sum().item()

                if args.per_step_reward \
                        and advantages_per_step is not None \
                        and k < len(advantages_per_step):
                    A_batch = advantages_per_step[k][batch_g_idxs] * discount_k
                else:
                    A_batch = advantages[batch_g_idxs] * discount_k

                with torch.no_grad():
                    pos_mask = A_batch > 0
                    neg_mask = A_batch < 0
                    n_pos = pos_mask.sum().item()
                    n_neg = neg_mask.sum().item()
                    if n_pos > 0:
                        log_ratio_mean_pos_sum += log_ratio_k[pos_mask].sum().item()
                        count_pos += n_pos
                    if n_neg > 0:
                        log_ratio_mean_neg_sum += log_ratio_k[neg_mask].sum().item()
                        count_neg += n_neg

                normalizer = chunk_size * n_train_steps * grad_accum_steps * num_inner_updates

                if args.token_level_loss:
                    A_token = A_batch.unsqueeze(-1)
                    surr1 = ratio_per_dim * A_token
                    surr2 = torch.clamp(ratio_per_dim, 1 - epsilon_low, 1 + epsilon_high) * A_token
                    token_normalizer = normalizer * D
                    if ch_mask is not None:
                        batch_loss = -(torch.min(surr1, surr2) * ch_mask).sum() / token_normalizer
                    else:
                        batch_loss = -torch.min(surr1, surr2).sum() / token_normalizer
                    with torch.no_grad():
                        if ch_mask is not None:
                            token_clip_count += (((ratio_per_dim > 1 + epsilon_high) |
                                                  (ratio_per_dim < 1 - epsilon_low)) & ch_mask).sum().item()
                            token_total_count += ch_mask.sum().item()
                        else:
                            token_clip_count += ((ratio_per_dim > 1 + epsilon_high) |
                                                 (ratio_per_dim < 1 - epsilon_low)).sum().item()
                            token_total_count += B_batch * D
                else:
                    surr1 = ratio_k * A_batch
                    if args.dim_level_clip:
                        clipped_ratio_per_dim = torch.clamp(ratio_per_dim, 1 - epsilon_low, 1 + epsilon_high)
                        with torch.no_grad():
                            _outside = (ratio_per_dim > 1 + epsilon_high) | (ratio_per_dim < 1 - epsilon_low)
                            if args.changed_only and ch_mask is not None:
                                dim_clip_count += (_outside & ch_mask).sum().item()
                                dim_clip_total += ch_mask.sum().item()
                            else:
                                dim_clip_count += _outside.sum().item()
                                dim_clip_total += B_batch * D
                            del _outside
                        log_clipped_per_dim = torch.log(clipped_ratio_per_dim.clamp(min=1e-30))
                        if args.changed_only and ch_mask is not None:
                            n_ch = ch_mask.sum(dim=-1).clamp(min=1)
                            log_clipped_k = (log_clipped_per_dim * ch_mask).sum(dim=-1) / n_ch
                        else:
                            log_clipped_k = log_clipped_per_dim.mean(dim=-1)
                        clipped_ratio_k = torch.exp(log_clipped_k)
                        surr2 = clipped_ratio_k * A_batch
                        del clipped_ratio_per_dim, log_clipped_per_dim, log_clipped_k, clipped_ratio_k
                    else:
                        surr2 = torch.clamp(ratio_k, 1 - epsilon_low, 1 + epsilon_high) * A_batch
                    batch_loss = -torch.min(surr1, surr2).sum() / normalizer

                if kl_beta > 0:
                    if need_ref:
                        ratio_ref_per_dim = numer / numer_ref.clamp(min=1e-30)
                        if ch_mask is not None:
                            n_ch = ch_mask.sum(dim=-1).clamp(min=1)
                            log_ratio_ref = (torch.log(ratio_ref_per_dim.clamp(min=1e-30)) * ch_mask).sum(dim=-1) / n_ch
                        else:
                            log_ratio_ref = torch.log(ratio_ref_per_dim.clamp(min=1e-30)).mean(dim=-1)
                        ratio_ref = torch.exp(log_ratio_ref)
                        kl_per_sample = (ratio_ref - 1) - log_ratio_ref
                        kl_loss = kl_beta * kl_per_sample.sum() / normalizer
                        batch_loss = batch_loss + kl_loss
                        kl_total += kl_loss.item()
                        del ratio_ref_per_dim, log_ratio_ref, ratio_ref, kl_per_sample, kl_loss, numer_ref
                    else:
                        kl_per_sample = (ratio_k - 1) - log_ratio_k
                        kl_loss = kl_beta * kl_per_sample.sum() / normalizer
                        batch_loss = batch_loss + kl_loss
                        kl_total += kl_loss.item()
                        del kl_per_sample, kl_loss

                accelerator.backward(batch_loss)
                loss_total += batch_loss.item()

                del x_t_batch, p_new, p_new_img, numer
                if not sr["is_last_step"][k]:
                    del denom
                del ratio_per_dim, log_ratio_per_dim, log_ratio_k, ratio_k
                del surr1, surr2, batch_loss, di_batch, A_batch, ch_mask

    return {
        "loss": loss_total,
        "kl_loss": kl_total,
        "clip_count": clip_count,
        "clip_high": clip_high,
        "clip_low": clip_low,
        "total_count": total_count,
        "approx_kl": approx_kl,
        "ratio_sum": ratio_sum,
        "log_ratio_mean_sum": log_ratio_mean_sum,
        "log_ratio_sum_sum": log_ratio_sum_sum,
        "token_clip_count": token_clip_count,
        "token_total_count": token_total_count,
        "log_ratio_mean_pos_sum": log_ratio_mean_pos_sum,
        "log_ratio_mean_neg_sum": log_ratio_mean_neg_sum,
        "count_pos": count_pos,
        "count_neg": count_neg,
        "changed_count_total": changed_count_total,
        "total_dim_count": total_dim_count,
        "dim_clip_count": dim_clip_count,
        "dim_clip_total": dim_clip_total,
    }
