"""
Sampling for DiffuGRPO.

DiffuGRPO objective uses ONLY two endpoints of the denoising trajectory:
    x_0 (initial noise) and x_1 (final tokens).

The per-sample importance ratio is the geometric mean (over D image-token
dimensions) of single-step probabilities evaluated at x_0:

    r^i(theta) = prod_d ( p_theta(x_1^{(i),d} | x_0^{(i)}, c)
                          / p_{theta_old}(x_1^{(i),d} | x_0^{(i)}, c) )^{1/D}

So during sampling (with the old-policy snapshot loaded into the model) we only
need to save:
    x_init            : (B, L) full input with x_0 at image positions
    final_tokens      : (B, L) full input with x_1 at image positions
    image_mask        : (B, L)
    log_p_old_x1_given_x0 : (B, D)   gathered log p_{theta_old}(x_1^d | x_0, c)
    changed_mask      : (B, D) bool  True where x_0^d != x_1^d (for diagnostics
                                     and optional changed_only masking)

Intermediate states / MC samples / per-step factors are NOT stored — this is
much cheaper in memory than the GRPO sampler.
"""

import torch
from math import ceil

from sampling import safe_categorical, _compute_rate


@torch.no_grad()
def sample_with_log_prob_diffugrpo(
    model,
    path_img,
    vocabulary_size_img,
    x_init,
    data_info,
    step_size,
    cfg_scale,
    n_mc=1,
    dtype_categorical=torch.float32,
    device="cuda",
    offload_to_cpu=False,
):
    """
    Standard Euler sampling on image tokens (NO trajectory storage), plus a
    single extra forward pass at x_0 to gather log p_{theta_old}(x_1^d | x_0).

    Returns dict with keys:
        x_init                : (B, L)         saved input at t=0
        final_tokens          : (B, L)         x at t=1
        image_mask            : (B, L)
        x_1_img               : (B, D)         x_1 restricted to image positions
        log_p_old_x1_given_x0 : (B, D)         log p_{theta_old}(x_1^d | x_0, c)
        changed_mask          : (B, D) bool    x_0^d != x_1^d
        n_mc                  : int            (kept for compatibility)
    """
    n_steps = ceil(1.0 / step_size)
    ts = torch.tensor(
        [step_size * i for i in range(n_steps)] + [1.0], device=device
    )

    image_mask = data_info["image_token_mask"]
    B = x_init.shape[0]

    x_0_full = x_init.clone()
    x_0_img = x_0_full[image_mask == 1].reshape(B, -1)  # (B, D)
    D = x_0_img.shape[1]

    # GRPO-style per-step changed-frac diagnostic: count token jumps over
    # intermediate steps only (k = 0 .. n_steps-2), excluding the final step
    # (matches GRPO default, which trains on steps 0..K-2 unless
    # --include_last_step is set).
    sample_changed_count = 0
    sample_total_count = 0

    # ---- run plain Euler sampling, keeping only x_t (no trajectory) ----
    x_t = x_0_full.clone()
    buf = x_t.clone()
    for i in range(n_steps):
        t = ts[i : i + 1]
        h = ts[i + 1 : i + 2] - t

        _, p_1t_img, _ = model(x=x_t, cfg_scale=cfg_scale, datainfo=data_info)

        x_t_img = x_t[image_mask == 1].reshape(B, -1)
        posterior = p_1t_img[image_mask == 1].reshape(B, -1, vocabulary_size_img)
        p_flat = posterior.reshape(B * D, vocabulary_size_img)

        if i == n_steps - 1:
            # Last step: sample x_1 directly from posterior
            x_1 = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
            buf[image_mask == 1] = x_1.flatten()
            x_t = buf.clone()
        else:
            # Euler step with MC averaging of u
            u_accum = torch.zeros(B, D, vocabulary_size_img, device=device)
            for _ in range(n_mc):
                x1_j = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
                u_j = _compute_rate(
                    path_img, x_t_img, x1_j, t, vocabulary_size_img, device
                )
                u_accum += u_j

            u_avg = u_accum / n_mc
            ps_avg = torch.exp(-h * u_avg.sum(-1))

            next_img = x_t_img.clone()
            jump = torch.rand(B, D, device=device) > ps_avg
            if jump.any():
                next_img[jump] = safe_categorical(u_avg[jump].to(dtype_categorical))

            # GRPO-style: changed = (x_t != x_{t+1}) on (B, D)
            sample_changed_count += int((x_t_img != next_img).sum().item())
            sample_total_count += B * D

            buf[image_mask == 1] = next_img.flatten()
            x_t = buf.clone()

        del posterior, p_1t_img, p_flat

    final_tokens = x_t                                      # (B, L)
    x_1_img = final_tokens[image_mask == 1].reshape(B, -1)  # (B, D)

    # ---- Extra forward at x_0 to compute log p_{theta_old}(x_1^d | x_0, c) ----
    _, p_at_x0, _ = model(x=x_0_full, cfg_scale=cfg_scale, datainfo=data_info)
    p_at_x0_img = p_at_x0[image_mask == 1].reshape(B, -1, vocabulary_size_img)
    p_old_x1 = torch.gather(
        p_at_x0_img, -1, x_1_img.unsqueeze(-1)
    ).squeeze(-1)                                           # (B, D)
    log_p_old_x1 = torch.log(p_old_x1.clamp(min=1e-30))     # (B, D)
    del p_at_x0, p_at_x0_img, p_old_x1

    changed_mask = (x_0_img != x_1_img)                     # (B, D) bool

    if offload_to_cpu:
        out_x_init = x_0_full.cpu()
        out_final_tokens = final_tokens.cpu()
        out_x_1_img = x_1_img.cpu()
        out_log_p_old = log_p_old_x1.cpu()
        out_changed = changed_mask.cpu()
    else:
        out_x_init = x_0_full
        out_final_tokens = final_tokens
        out_x_1_img = x_1_img
        out_log_p_old = log_p_old_x1
        out_changed = changed_mask

    return {
        "x_init": out_x_init,                          # (B, L) keep on GPU by default
        "final_tokens": out_final_tokens,              # (B, L)
        "image_mask": image_mask,                      # (B, L)
        "x_1_img": out_x_1_img,                        # (B, D)
        "log_p_old_x1_given_x0": out_log_p_old,        # (B, D)
        "changed_mask": out_changed,                   # (B, D) bool
        # GRPO-style sampling-step changed-frac diagnostic (intermediate
        # steps only, single-GPU local counts):
        "sample_changed_count": sample_changed_count,
        "sample_total_count": sample_total_count,
        "n_mc": n_mc,
    }


def merge_sampling_results_diffugrpo(sr_list):
    """Merge multi-round DiffuGRPO sampling results (concat along batch dim)."""
    return {
        "x_init":              torch.cat([s["x_init"]              for s in sr_list], dim=0),
        "final_tokens":        torch.cat([s["final_tokens"]        for s in sr_list], dim=0),
        "image_mask":          torch.cat([s["image_mask"]          for s in sr_list], dim=0),
        "x_1_img":             torch.cat([s["x_1_img"]             for s in sr_list], dim=0),
        "log_p_old_x1_given_x0": torch.cat([s["log_p_old_x1_given_x0"] for s in sr_list], dim=0),
        "changed_mask":        torch.cat([s["changed_mask"]        for s in sr_list], dim=0),
        "sample_changed_count": sum(s["sample_changed_count"] for s in sr_list),
        "sample_total_count":   sum(s["sample_total_count"]   for s in sr_list),
        "n_mc":                sr_list[0]["n_mc"],
    }
