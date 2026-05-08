"""
Core GRPO training step for multimodal understanding.

Mirrors ``training_grpo.train_on_prompt`` but operates on TEXT tokens:
the policy distribution is taken from ``p_new_txt`` / ``p_old_txt`` (returned
by ``CFGScaledModel(..., g_or_u='understanding')`` or the differentiable
counterpart wrapping ``g_or_u='understanding'``).
"""

import torch
from contextlib import nullcontext

from config import VOCABULARY_SIZE_TXT


def collect_items(dataloader_iter, P):
    """Collect P items (dicts) from the dataloader iterator.
    Returns (items_list, ) or (None,) if exhausted."""
    items = []
    for _ in range(P):
        try:
            batch = next(dataloader_iter)
            # collate_fn returns a list of dicts, batch_size=1 here
            items.append(batch[0])
        except StopIteration:
            break
    return items if items else None


def train_on_batch_u(
    sr, data_info, advantages, K, g_indices, is_last_prompt_in_accum,
    diff_cfg_model, model, accelerator, device,
    args, train_step_set, n_train_steps, train_batch_size,
    grad_accum_steps, num_inner_updates,
    epsilon_low, epsilon_high, kl_beta, TRAIN_CFG_SCALE,
    disable_adapter_and_gen_modules=None, raw_model=None,
):
    """Run PPO-style forward/backward over selected samples (text tokens).

    ``sr`` is the dict returned by ``sample_text_with_log_prob``.
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
        (lambda: accelerator.no_sync(model)) if hasattr(accelerator, "no_sync") else
        (model.no_sync if hasattr(model, "no_sync") else (lambda: nullcontext()))
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

                # Understanding CFGScaledModel returns (p_txt_softmax, None, data_info).
                p_new_txt, _, _ = diff_cfg_model(
                    x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
                p_new_t = p_new_txt[di_batch["text_token_mask"] == 1].reshape(
                    B_batch, -1, VOCABULARY_SIZE_TXT)

                n_mc_k = len(sr["all_x1_samples"][k])
                D = p_new_t.shape[1]

                need_ref = (kl_beta > 0 and args.use_lora and not args.kl_old_policy)

                if sr["is_last_step"][k]:
                    x1_last = sr["all_x1_samples"][k][0][batch_g_idxs]
                    fac_last = sr["all_factors"][k][0][batch_g_idxs]
                    p_new_g = torch.gather(p_new_t, -1, x1_last.unsqueeze(-1)).squeeze(-1)
                    numer = p_new_g * fac_last
                    ratio_per_dim = numer.clamp(min=1e-30)

                    if need_ref:
                        with torch.no_grad(), disable_adapter_and_gen_modules(raw_model):
                            p_ref_txt, _, _ = diff_cfg_model(
                                x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
                        p_ref_t = p_ref_txt[di_batch["text_token_mask"] == 1].reshape(
                            B_batch, -1, VOCABULARY_SIZE_TXT)
                        p_ref_g = torch.gather(p_ref_t, -1, x1_last.unsqueeze(-1)).squeeze(-1)
                        numer_ref = p_ref_g * fac_last
                else:
                    numer = torch.zeros(B_batch, D, device=device)
                    if need_ref:
                        with torch.no_grad(), disable_adapter_and_gen_modules(raw_model):
                            p_ref_txt, _, _ = diff_cfg_model(
                                x=x_t_batch, cfg_scale=TRAIN_CFG_SCALE, datainfo=di_batch)
                        p_ref_t = p_ref_txt[di_batch["text_token_mask"] == 1].reshape(
                            B_batch, -1, VOCABULARY_SIZE_TXT)
                        numer_ref = torch.zeros(B_batch, D, device=device)

                    for j in range(n_mc_k):
                        x1_j = sr["all_x1_samples"][k][j][batch_g_idxs]
                        fac_j = sr["all_factors"][k][j][batch_g_idxs]
                        p_new_j = torch.gather(p_new_t, -1, x1_j.unsqueeze(-1)).squeeze(-1)
                        numer += p_new_j * fac_j
                        if need_ref:
                            p_ref_j = torch.gather(p_ref_t, -1, x1_j.unsqueeze(-1)).squeeze(-1)
                            numer_ref += p_ref_j * fac_j
                    numer = numer / n_mc_k
                    if need_ref:
                        numer_ref = numer_ref / n_mc_k
                    denom = sr["all_p_old_times_factor"][k][batch_g_idxs]
                    ratio_per_dim = numer.clamp(min=1e-30) / denom.clamp(min=1e-30)

                log_ratio_per_dim = torch.log(ratio_per_dim.clamp(min=1e-30))

                if args.changed_only:
                    ch_mask = sr["changed_masks"][k][batch_g_idxs]
                    n_ch = ch_mask.sum(dim=-1).clamp(min=1)
                    log_ratio_k = (log_ratio_per_dim * ch_mask).sum(dim=-1) / n_ch
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
                    clipped_high_b = (ratio_k > 1 + epsilon_high).sum().item()
                    clipped_low_b = (ratio_k < 1 - epsilon_low).sum().item()
                    clip_high += clipped_high_b
                    clip_low += clipped_low_b
                    clip_count += clipped_high_b + clipped_low_b
                    approx_kl += ((ratio_k - 1) - log_ratio_k).sum().item()
                    ratio_sum += ratio_k.sum().item()
                    total_count += B_batch

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
                    A_tok = A_batch.unsqueeze(-1)
                    surr1 = ratio_per_dim * A_tok
                    surr2 = torch.clamp(ratio_per_dim, 1 - epsilon_low, 1 + epsilon_high) * A_tok
                    tok_norm = normalizer * D
                    if ch_mask is not None:
                        batch_loss = -(torch.min(surr1, surr2) * ch_mask).sum() / tok_norm
                    else:
                        batch_loss = -torch.min(surr1, surr2).sum() / tok_norm
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
                    surr2 = torch.clamp(ratio_k, 1 - epsilon_low, 1 + epsilon_high) * A_batch
                    batch_loss = -torch.min(surr1, surr2).sum() / normalizer
                    with torch.no_grad():
                        _outside = (ratio_per_dim > 1 + epsilon_high) | (ratio_per_dim < 1 - epsilon_low)
                        if ch_mask is not None:
                            dim_clip_count += (_outside & ch_mask).sum().item()
                            dim_clip_total += ch_mask.sum().item()
                        else:
                            dim_clip_count += _outside.sum().item()
                            dim_clip_total += B_batch * D
                        del _outside

                if kl_beta > 0:
                    if need_ref:
                        ratio_ref_pd = numer / numer_ref.clamp(min=1e-30)
                        if ch_mask is not None:
                            n_ch = ch_mask.sum(dim=-1).clamp(min=1)
                            log_ratio_ref = (torch.log(ratio_ref_pd.clamp(min=1e-30))
                                             * ch_mask).sum(dim=-1) / n_ch
                        else:
                            log_ratio_ref = torch.log(ratio_ref_pd.clamp(min=1e-30)).mean(dim=-1)
                        ratio_ref = torch.exp(log_ratio_ref)
                        kl_ps = (ratio_ref - 1) - log_ratio_ref
                    else:
                        kl_ps = (ratio_k - 1) - log_ratio_k
                    kl_loss = kl_beta * kl_ps.sum() / normalizer
                    batch_loss = batch_loss + kl_loss
                    kl_total += kl_loss.item()

                accelerator.backward(batch_loss)
                loss_total += batch_loss.item()

    return {
        "loss": loss_total, "kl_loss": kl_total,
        "clip_count": clip_count, "clip_high": clip_high, "clip_low": clip_low,
        "total_count": total_count,
        "approx_kl": approx_kl, "ratio_sum": ratio_sum,
        "log_ratio_mean_sum": log_ratio_mean_sum,
        "log_ratio_sum_sum": log_ratio_sum_sum,
        "token_clip_count": token_clip_count,
        "token_total_count": token_total_count,
        "log_ratio_mean_pos_sum": log_ratio_mean_pos_sum,
        "log_ratio_mean_neg_sum": log_ratio_mean_neg_sum,
        "count_pos": count_pos, "count_neg": count_neg,
        "changed_count_total": changed_count_total,
        "total_dim_count": total_dim_count,
        "dim_clip_count": dim_clip_count,
        "dim_clip_total": dim_clip_total,
    }
