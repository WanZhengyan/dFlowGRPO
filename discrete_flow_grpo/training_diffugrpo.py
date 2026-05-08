"""
DiffuGRPO loss: per-sample geometric-mean ratio over D image-token dims,
evaluated by a SINGLE forward pass of pi_theta at x_0 (initial noise).

Objective:
    J(theta) = (1/G) Σ_i [ min( r^i(theta) A^i,
                                 clip(r^i(theta), 1-eps, 1+eps) A^i )
                            - beta * D_KL(pi_theta || pi_ref) ]

with
    r^i(theta) = ( prod_d  p_theta(x_1^{(i),d} | x_0^{(i)}, c)
                          / p_{theta_old}(x_1^{(i),d} | x_0^{(i)}, c) )^{1/D}

The denominator log p_{theta_old}(x_1^d | x_0) is precomputed during sampling
and stored in sr["log_p_old_x1_given_x0"].

For KL we use the same geometric-mean ratio against pi_ref (or pi_old when
--kl_old_policy is set or when no LoRA reference is available) and the
unbiased estimator  ( ratio_ref - 1 ) - log_ratio_ref.
"""

import torch
from contextlib import nullcontext

from config import VOCABULARY_SIZE_IMG


def train_on_prompt_diffugrpo(
    sr, data_info, advantages,
    g_indices, is_last_prompt_in_accum,
    # Model / accelerator
    diff_cfg_model, model, accelerator, device,
    # Hyper-params
    args, train_batch_size,
    grad_accum_steps, num_inner_updates,
    epsilon_low, epsilon_high, kl_beta, TRAIN_CFG_SCALE,
    # Optional: KL reference
    disable_adapter_and_gen_modules=None, raw_model=None,
):
    """
    Run training forward/backward for one prompt's data over the given g_indices.
    Accumulates gradients (does NOT call optimizer.step()).

    Returns: dict of scalar metrics.
    """
    chunk_size = len(g_indices)
    g_idxs_t = torch.tensor(g_indices, dtype=torch.long, device=device)

    loss_total = 0.0
    kl_total = 0.0
    clip_count = 0
    clip_high_count = 0
    clip_low_count = 0
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

    need_ref = (kl_beta > 0 and args.use_lora and not args.kl_old_policy)

    with sync_context():
        for b_start in range(0, chunk_size, train_batch_size):
            b_end = min(b_start + train_batch_size, chunk_size)
            B_batch = b_end - b_start
            batch_g_idxs = g_idxs_t[b_start:b_end]

            # ---- Inputs (no grad needed for these) ----
            x_0_batch = sr["x_init"][batch_g_idxs].to(device)               # (B, L)
            x_1_img_batch = sr["x_1_img"][batch_g_idxs].to(device)           # (B, D)
            log_p_old_batch = sr["log_p_old_x1_given_x0"][batch_g_idxs].to(device)  # (B, D)
            ch_mask = sr["changed_mask"][batch_g_idxs].to(device)            # (B, D) bool

            di_batch = {key: (v[batch_g_idxs] if isinstance(v, torch.Tensor) else v)
                        for key, v in data_info.items()}

            # ---- Single forward of pi_theta at x_0 ----
            _, p_new, _ = diff_cfg_model(
                x=x_0_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
            p_new_img = p_new[di_batch["image_token_mask"] == 1].reshape(
                B_batch, -1, VOCABULARY_SIZE_IMG)
            D = p_new_img.shape[1]

            # log p_theta(x_1^d | x_0, c) gathered at x_1
            p_new_x1 = torch.gather(
                p_new_img, -1, x_1_img_batch.unsqueeze(-1)).squeeze(-1)      # (B, D)
            log_p_new = torch.log(p_new_x1.clamp(min=1e-30))                 # (B, D)

            # per-dim log-ratio: log p_theta - log p_old   (B, D)
            log_ratio_per_dim = log_p_new - log_p_old_batch
            ratio_per_dim = torch.exp(log_ratio_per_dim)

            # ---- Optional reference forward (for KL vs pi_ref) ----
            if need_ref:
                with torch.no_grad(), disable_adapter_and_gen_modules(raw_model):
                    _, p_ref, _ = diff_cfg_model(
                        x=x_0_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
                p_ref_img = p_ref[di_batch["image_token_mask"] == 1].reshape(
                    B_batch, -1, VOCABULARY_SIZE_IMG)
                p_ref_x1 = torch.gather(
                    p_ref_img, -1, x_1_img_batch.unsqueeze(-1)).squeeze(-1)
                log_p_ref = torch.log(p_ref_x1.clamp(min=1e-30))             # (B, D)
                log_ratio_ref_per_dim = log_p_new - log_p_ref                # (B, D)
                del p_ref, p_ref_img, p_ref_x1
            else:
                log_ratio_ref_per_dim = None

            # ---- Aggregate to per-sample geometric-mean log-ratio ----
            if args.changed_only:
                n_ch = ch_mask.sum(dim=-1).clamp(min=1)
                log_ratio = (log_ratio_per_dim * ch_mask).sum(dim=-1) / n_ch  # (B,)
                with torch.no_grad():
                    changed_count_total += ch_mask.sum().item()
                    total_dim_count += B_batch * D
            else:
                log_ratio = log_ratio_per_dim.mean(dim=-1)                    # (B,)
                with torch.no_grad():
                    total_dim_count += B_batch * D
                    changed_count_total += ch_mask.sum().item()

            ratio = torch.exp(log_ratio)                                      # (B,)

            # ---- Diagnostics ----
            with torch.no_grad():
                log_ratio_mean_sum += log_ratio.sum().item()
                log_ratio_sum_sum += log_ratio_per_dim.sum(dim=-1).sum().item()

                if args.dim_level_clip:
                    _high_dim = (ratio_per_dim > 1 + epsilon_high)
                    _low_dim = (ratio_per_dim < 1 - epsilon_low)
                    if args.changed_only:
                        clip_high_count += (_high_dim & ch_mask).sum().item()
                        clip_low_count += (_low_dim & ch_mask).sum().item()
                        clip_count += ((_high_dim | _low_dim) & ch_mask).sum().item()
                        total_count += ch_mask.sum().item()
                    else:
                        clip_high_count += _high_dim.sum().item()
                        clip_low_count += _low_dim.sum().item()
                        clip_count += (_high_dim | _low_dim).sum().item()
                        total_count += B_batch * D
                    del _high_dim, _low_dim
                else:
                    _high = (ratio > 1 + epsilon_high).sum().item()
                    _low = (ratio < 1 - epsilon_low).sum().item()
                    clip_high_count += _high
                    clip_low_count += _low
                    clip_count += _high + _low
                    total_count += B_batch

                approx_kl += ((ratio - 1) - log_ratio).sum().item()
                ratio_sum += ratio.sum().item()

                # Token-level clip count diagnostic
                if args.changed_only:
                    token_clip_count += (((ratio_per_dim > 1 + epsilon_high) |
                                          (ratio_per_dim < 1 - epsilon_low)) & ch_mask).sum().item()
                    token_total_count += ch_mask.sum().item()
                else:
                    token_clip_count += ((ratio_per_dim > 1 + epsilon_high) |
                                         (ratio_per_dim < 1 - epsilon_low)).sum().item()
                    token_total_count += B_batch * D

            A_batch = advantages[batch_g_idxs]                                # (B,)

            with torch.no_grad():
                pos_mask = A_batch > 0
                neg_mask = A_batch < 0
                n_pos = pos_mask.sum().item()
                n_neg = neg_mask.sum().item()
                if n_pos > 0:
                    log_ratio_mean_pos_sum += log_ratio[pos_mask].sum().item()
                    count_pos += n_pos
                if n_neg > 0:
                    log_ratio_mean_neg_sum += log_ratio[neg_mask].sum().item()
                    count_neg += n_neg

            # ---- PPO surrogate (sample-level; geometric-mean ratio) ----
            normalizer = chunk_size * grad_accum_steps * num_inner_updates

            if args.token_level_loss:
                # Token-level loss: clip + min done per dim, then averaged
                A_token = A_batch.unsqueeze(-1)
                surr1 = ratio_per_dim * A_token
                surr2 = torch.clamp(ratio_per_dim, 1 - epsilon_low, 1 + epsilon_high) * A_token
                token_normalizer = normalizer * D
                if args.changed_only:
                    batch_loss = -(torch.min(surr1, surr2) * ch_mask).sum() / token_normalizer
                else:
                    batch_loss = -torch.min(surr1, surr2).sum() / token_normalizer
            else:
                surr1 = ratio * A_batch
                if args.dim_level_clip:
                    # Clip per-dim ratio first, then aggregate via geometric mean
                    clipped_ratio_per_dim = torch.clamp(
                        ratio_per_dim, 1 - epsilon_low, 1 + epsilon_high)
                    log_clipped_per_dim = torch.log(clipped_ratio_per_dim.clamp(min=1e-30))
                    if args.changed_only:
                        n_ch = ch_mask.sum(dim=-1).clamp(min=1)
                        log_clipped = (log_clipped_per_dim * ch_mask).sum(dim=-1) / n_ch
                    else:
                        log_clipped = log_clipped_per_dim.mean(dim=-1)
                    clipped_ratio = torch.exp(log_clipped)
                    surr2 = clipped_ratio * A_batch
                    with torch.no_grad():
                        _outside = (ratio_per_dim > 1 + epsilon_high) | (ratio_per_dim < 1 - epsilon_low)
                        if args.changed_only:
                            dim_clip_count += (_outside & ch_mask).sum().item()
                            dim_clip_total += ch_mask.sum().item()
                        else:
                            dim_clip_count += _outside.sum().item()
                            dim_clip_total += B_batch * D
                        del _outside
                    del clipped_ratio_per_dim, log_clipped_per_dim, log_clipped, clipped_ratio
                else:
                    surr2 = torch.clamp(ratio, 1 - epsilon_low, 1 + epsilon_high) * A_batch
                batch_loss = -torch.min(surr1, surr2).sum() / normalizer

            # ---- KL term (geometric-mean ratio in log-space) ----
            if kl_beta > 0:
                if need_ref and log_ratio_ref_per_dim is not None:
                    if args.changed_only:
                        n_ch = ch_mask.sum(dim=-1).clamp(min=1)
                        log_ratio_ref = (log_ratio_ref_per_dim * ch_mask).sum(dim=-1) / n_ch
                    else:
                        log_ratio_ref = log_ratio_ref_per_dim.mean(dim=-1)
                    ratio_ref = torch.exp(log_ratio_ref)
                    kl_per_sample = (ratio_ref - 1) - log_ratio_ref
                else:
                    kl_per_sample = (ratio - 1) - log_ratio
                kl_loss = kl_beta * kl_per_sample.sum() / normalizer
                batch_loss = batch_loss + kl_loss
                kl_total += kl_loss.item()
                del kl_per_sample, kl_loss

            accelerator.backward(batch_loss)
            loss_total += batch_loss.item()

            del x_0_batch, x_1_img_batch, log_p_old_batch, ch_mask
            del p_new, p_new_img, p_new_x1, log_p_new
            del log_ratio_per_dim, ratio_per_dim, log_ratio, ratio
            del surr1, surr2, batch_loss, A_batch, di_batch
            if log_ratio_ref_per_dim is not None:
                del log_ratio_ref_per_dim

    return {
        "loss": loss_total,
        "kl_loss": kl_total,
        "clip_count": clip_count,
        "clip_high": clip_high_count,
        "clip_low": clip_low_count,
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
