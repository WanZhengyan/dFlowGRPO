"""
Sampling for FUDOKI discrete flow matching with saved intermediates for GRPO.

Saves the minimal quantities needed to reconstruct differentiable transition
probabilities during training via importance-sampling reweighting:

    hat{p}_theta(x_t^d | x_s) = (1/n_mc) sum_j  p_theta(X_{1,j}^d | x_s) * factor_j^d

where factor_j^d (stored in all_factors) absorbs everything except p_theta:

    factor_j^d = (1 / p_{theta_old}(X_{1,j}^d | x_s)) * {
        1[x_s^d != x_t^d] * Q(x_s^d, x_t^d | X_1) / lambda_s^d * (1 - exp(-h * lambda_s^d))
      + 1[x_s^d == x_t^d] * exp(-h * lambda_s^d)
    }

Saved per step k, per MC sample j:
    all_x1_samples[k][j]: (B, D) — MC sample token ids (to gather p_theta during training)
    all_factors[k][j]:    (B, D) — the factor above (theta-independent, no grad needed)

Also saved per step k (scalar, no MC index):
    all_p_old_times_factor[k]: (B, D) — MC-averaged p_old * factor = (1/n_mc) sum_j braces_j
        This is the denominator of the per-dim ratio, i.e. hat{p}_{theta_old}(x_t^d | x_s).

Plus trajectory[k]: (B, L) — full token sequence at each step boundary.
"""

import torch
import torch.nn.functional as F
from math import ceil
from tqdm import tqdm
from contextlib import nullcontext

from flow_matching.utils import categorical


def safe_categorical(probs):
    """Sanitize probability tensor before calling torch.multinomial.
    Replaces nan/inf/negative values with 0 and ensures each row sums > 0.
    """
    probs = probs.clone()
    probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    probs = probs.clamp(min=0.0)
    # If any row is all-zero, set uniform distribution to avoid multinomial crash
    row_sum = probs.sum(dim=-1, keepdim=True)
    zero_rows = (row_sum == 0).expand_as(probs)
    probs[zero_rows] = 1.0 / probs.shape[-1]
    return categorical(probs)


def _compute_rate(path_img, x_t_img, x_1_sample, t, vocabulary_size_img, device):
    """
    Compute rate matrix row u^d = Q_s(x_s^d, . | X_1^d) with diagonal zeroed.

    Returns:
        u: (B, D, V) — off-diagonal rates
    """
    batch_size, D = x_t_img.shape

    emb_x_1 = path_img.embedding(x_1_sample)                         # (B, D, E)
    prob = path_img.get_prob_distribution(emb_x_1, t)                 # (B, D, V)

    emb_x_t = path_img.embedding(x_t_img)                            # (B, D, E)
    emb_x_t_f = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
    emb_x_1_f = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)

    dist_xt_x1 = (
        (emb_x_t_f ** 2).sum(1, keepdim=True)
        + (emb_x_1_f ** 2).sum(1, keepdim=True)
        - 2 * torch.einsum("bd,bd->b", emb_x_t_f, emb_x_1_f).unsqueeze(1)
    ) ** 2                                                            # (B*D, 1)

    dist_x1_all = path_img.metric(emb_x_1)                           # (B*D, V)
    distance = F.relu(dist_xt_x1 - dist_x1_all)                      # (B*D, V)

    if t.item() == 0:
        d_beta_t = torch.tensor(0.0, device=device)
    else:
        d_beta_t = (
            path_img.c * path_img.a
            * ((t / (1 - t)) ** (path_img.a - 1))
            / ((1 - t) ** 2)
        )

    u = (prob.reshape(-1, vocabulary_size_img) * d_beta_t * distance)
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    u = u.reshape(batch_size, D, vocabulary_size_img)

    # zero diagonal
    mask = F.one_hot(x_t_img, num_classes=vocabulary_size_img).bool()
    u = u.masked_fill(mask, 0.0)

    return u


@torch.no_grad()
def sample_with_log_prob(
    model,
    path_img,
    vocabulary_size_img,
    x_init,
    data_info,
    step_size,
    cfg_scale,
    n_mc=1,
    dtype_categorical=torch.float32,
    verbose=False,
    device="cuda",
    offload_to_cpu=True,
):
    """
    Euler sampling on image tokens, saving intermediates for GRPO training.
    L: sequence length, D: number of image tokens, V: vocabulary size for image tokens.
    Returns dict with keys:
        final_tokens:           (B, L)
        trajectory:             list of K+1 tensors (B, L)
        all_x1_samples:         [k][j] -> (B, D)
        all_factors:            [k][j] -> (B, D)
        all_p_old_times_factor: [k] -> (B, D)   denominator for ratio
        is_last_step:           [k] -> bool
        image_mask:             (B, L)
        n_mc:                   int
    """
    n_steps = ceil(1.0 / step_size)
    ts = torch.tensor(
        [step_size * i for i in range(n_steps)] + [1.0], device=device
    )

    x_t = x_init.clone()
    B = x_t.shape[0]
    image_mask = data_info["image_token_mask"]   # (B, L)

    trajectory = [x_t.clone()]
    all_x1_samples = []          # [step][j] -> (B, D)
    all_factors = []              # [step][j] -> (B, D)
    all_p_old_times_factor = []  # [step] -> (B, D)
    changed_masks = []           # [step] -> (B, D) bool — True where token jumped
    is_last_step = []

    ctx = tqdm(total=1.0, desc="Sampling") if verbose else nullcontext()
    with ctx:
        buf = x_t.clone()
        for i in range(n_steps):
            t = ts[i : i + 1]
            h = ts[i + 1 : i + 2] - t

            # --- model forward (p_{theta_old}) ---
            _, p_1t_img, _ = model(x=x_t, cfg_scale=cfg_scale, datainfo=data_info)

            x_t_img = x_t[image_mask == 1].reshape(B, -1)               # (B, D)
            posterior = p_1t_img[image_mask == 1].reshape(
                B, -1, vocabulary_size_img
            )                                                            # (B, D, V)
            D = x_t_img.shape[1]
            p_flat = posterior.reshape(B * D, vocabulary_size_img)

            if i == n_steps - 1:
                # ===== last step: sample x_1 directly from posterior =====
                is_last_step.append(True)
                x_1 = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)

                p_old = torch.gather(
                    posterior, -1, x_1.unsqueeze(-1)
                ).squeeze(-1)                                            # (B, D)

                # factor = 1 / p_old  =>  p_theta * factor = p_theta / p_old
                factor = 1.0 / p_old.clamp(min=1e-30)                   # (B, D)

                all_x1_samples.append([x_1])
                all_factors.append([factor])
                # p_old * factor = 1.0
                all_p_old_times_factor.append(torch.ones(B, D, device=device))
                # Last step: all tokens are sampled fresh from posterior → all changed
                changed_masks.append(torch.ones(B, D, dtype=torch.bool, device=device))

                buf[image_mask == 1] = x_1.flatten()
                x_t = buf.clone()
            else:
                # ===== Euler step =====
                is_last_step.append(False)

                step_x1 = []
                step_factors = []
                u_accum = torch.zeros(B, D, vocabulary_size_img, device=device)
                u_list, lam_list, ps_list, p_old_list = [], [], [], []

                # -- Phase 1: draw n_mc posterior samples, accumulate u --
                for _ in range(n_mc):
                    x1_j = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
                    step_x1.append(x1_j)

                    p_old_j = torch.gather(
                        posterior, -1, x1_j.unsqueeze(-1)
                    ).squeeze(-1)                                        # (B, D)
                    p_old_list.append(p_old_j)

                    u_j = _compute_rate(
                        path_img, x_t_img, x1_j, t, vocabulary_size_img, device
                    )                                                    # (B, D, V)
                    lam_j = u_j.sum(-1)                                  # (B, D)
                    ps_j = torch.exp(-h * lam_j)                         # (B, D)

                    u_accum += u_j
                    u_list.append(u_j)
                    lam_list.append(lam_j)
                    ps_list.append(ps_j)

                # -- Phase 2: average u -> sample next state --
                u_avg = u_accum / n_mc
                ps_avg = torch.exp(-h * u_avg.sum(-1))

                next_img = x_t_img.clone()
                jump = torch.rand(B, D, device=device) > ps_avg
                if jump.any():
                    next_img[jump] = safe_categorical(
                        u_avg[jump].to(dtype_categorical)
                    )

                # -- Phase 3: compute factor per MC sample --
                changed = (x_t_img != next_img)                          # (B, D)
                next_oh = F.one_hot(next_img, vocabulary_size_img).float()

                denom_accum = torch.zeros(B, D, device=device)

                for j in range(n_mc):
                    q_to_next = (u_list[j] * next_oh).sum(-1)            # (B, D)

                    # jump term: Q/lambda * (1 - exp(-h*lambda))
                    jump_term = torch.where(
                        lam_list[j] > 1e-20,
                        (q_to_next / lam_list[j].clamp(min=1e-30)) * (1.0 - ps_list[j]),
                        torch.zeros_like(q_to_next),
                    )
                    braces = torch.where(changed, jump_term, ps_list[j]) # formula: 1[x_s^d != x_t^d] * Q(x_s^d, x_t^d | X_1) / lambda_s^d * (1 - exp(-h * lambda_s^d)) + 1[x_s^d == x_t^d] * exp(-h * lambda_s^d)

                    factor_j = braces / p_old_list[j].clamp(min=1e-30)  # (B, D)
                    step_factors.append(factor_j)
                    denom_accum += braces

                all_x1_samples.append(step_x1)
                all_factors.append(step_factors)
                all_p_old_times_factor.append(denom_accum / n_mc)        # (B, D)
                changed_masks.append(changed)                            # (B, D) bool

                buf[image_mask == 1] = next_img.flatten()
                x_t = buf.clone()

            trajectory.append(x_t.clone())
            if verbose and hasattr(ctx, "n"):
                ctx.n = (t + h).item()
                ctx.refresh()

    # Optionally move stored intermediates to CPU to free GPU memory.
    # They will be moved back to GPU on-demand during training (one (g,k) at a time).
    if offload_to_cpu:
        trajectory_out = [t.cpu() for t in trajectory]
        all_x1_out = [[x.cpu() for x in step] for step in all_x1_samples]
        all_factors_out = [[f.cpu() for f in step] for step in all_factors]
        all_p_old_out = [p.cpu() for p in all_p_old_times_factor]
        changed_masks_out = [m.cpu() for m in changed_masks]
    else:
        trajectory_out = trajectory
        all_x1_out = all_x1_samples
        all_factors_out = all_factors
        all_p_old_out = all_p_old_times_factor
        changed_masks_out = changed_masks

    return {
        "final_tokens": x_t,                             # keep on GPU for reward decode
        "trajectory": trajectory_out,                    # [0..K], each (B, L)
        "all_x1_samples": all_x1_out,                   # [k][j] -> (B, D)
        "all_factors": all_factors_out,                  # [k][j] -> (B, D)
        "all_p_old_times_factor": all_p_old_out,         # [k] -> (B, D)
        "changed_masks": changed_masks_out,              # [k] -> (B, D) bool
        "is_last_step": is_last_step,                    # [k] -> bool
        "image_mask": image_mask,                        # (B, L) keep on GPU
        "n_mc": n_mc,
    }


@torch.no_grad()
def sample_only(
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
):
    """
    Lightweight Euler sampling for evaluation — only returns final tokens.
    No GRPO intermediates are saved, much less memory usage.
    """
    n_steps = ceil(1.0 / step_size)
    ts = torch.tensor(
        [step_size * i for i in range(n_steps)] + [1.0], device=device
    )

    x_t = x_init.clone()
    B = x_t.shape[0]
    image_mask = data_info["image_token_mask"]

    buf = x_t.clone()
    for i in range(n_steps):
        t = ts[i : i + 1]
        h = ts[i + 1 : i + 2] - t

        _, p_1t_img, _ = model(x=x_t, cfg_scale=cfg_scale, datainfo=data_info)

        x_t_img = x_t[image_mask == 1].reshape(B, -1)
        posterior = p_1t_img[image_mask == 1].reshape(B, -1, vocabulary_size_img)
        D = x_t_img.shape[1]
        p_flat = posterior.reshape(B * D, vocabulary_size_img)

        if i == n_steps - 1:
            # Last step: sample x_1 directly from posterior
            x_1 = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
            buf[image_mask == 1] = x_1.flatten()
            x_t = buf.clone()
        else:
            # Euler step with MC averaging
            u_accum = torch.zeros(B, D, vocabulary_size_img, device=device)
            for _ in range(n_mc):
                x1_j = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
                u_j = _compute_rate(path_img, x_t_img, x1_j, t, vocabulary_size_img, device)
                u_accum += u_j

            u_avg = u_accum / n_mc
            ps_avg = torch.exp(-h * u_avg.sum(-1))

            next_img = x_t_img.clone()
            jump = torch.rand(B, D, device=device) > ps_avg
            if jump.any():
                next_img[jump] = safe_categorical(u_avg[jump].to(dtype_categorical))

            buf[image_mask == 1] = next_img.flatten()
            x_t = buf.clone()

    return x_t, image_mask

