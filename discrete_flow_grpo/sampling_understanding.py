"""
Euler sampling with saved intermediates for multimodal-understanding GRPO.

This mirrors ``sampling.sample_with_log_prob`` but samples TEXT tokens
instead of image tokens: the model is wrapped as
``CFGScaledModel(..., g_or_u='understanding')``, which returns
``(p_1t_txt_softmax, None, datainfo)``.
"""

import torch
import torch.nn.functional as F
from math import ceil
from contextlib import nullcontext
from tqdm import tqdm

from sampling import safe_categorical


@torch.no_grad()
def sample_text_only(
    model, path_txt, vocabulary_size_txt,
    x_init, data_info, step_size, cfg_scale=0.0,
    n_mc=1, dtype_categorical=torch.float32, device="cuda",
):
    """Lean Euler sampling for evaluation — only returns final tokens.

    Mirrors ``sampling.sample_only`` (image generation): no trajectory /
    factors / per-step softmax kept, so evaluation memory stays flat.
    """
    n_steps = ceil(1.0 / step_size)
    ts = torch.tensor(
        [step_size * i for i in range(n_steps)] + [1.0], device=device)

    x_t = x_init.clone()
    B = x_t.shape[0]
    text_mask = data_info["text_token_mask"]

    buf = x_t.clone()
    for i in range(n_steps):
        t = ts[i:i + 1]
        h = ts[i + 1:i + 2] - t

        p_1t_txt, _, _ = model(x=x_t, cfg_scale=cfg_scale, datainfo=data_info)

        x_t_txt = x_t[text_mask == 1].reshape(B, -1)
        posterior = p_1t_txt[text_mask == 1].reshape(B, -1, vocabulary_size_txt)
        D = x_t_txt.shape[1]
        p_flat = posterior.reshape(B * D, vocabulary_size_txt)

        if i == n_steps - 1:
            x_1 = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
            buf[text_mask == 1] = x_1.flatten()
            x_t = buf.clone()
        else:
            u_accum = torch.zeros(B, D, vocabulary_size_txt, device=device)
            for _ in range(n_mc):
                x1_j = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
                u_j = _compute_rate_txt(path_txt, x_t_txt, x1_j, t,
                                        vocabulary_size_txt, device)
                u_accum += u_j
            u_avg = u_accum / n_mc
            ps_avg = torch.exp(-h * u_avg.sum(-1))

            next_txt = x_t_txt.clone()
            jump = torch.rand(B, D, device=device) > ps_avg
            if jump.any():
                next_txt[jump] = safe_categorical(u_avg[jump].to(dtype_categorical))

            buf[text_mask == 1] = next_txt.flatten()
            x_t = buf.clone()

        del p_1t_txt, posterior, p_flat

    return x_t, text_mask


def _compute_rate_txt(path_txt, x_t_txt, x_1_sample, t, vocabulary_size_txt, device):
    """Same idea as ``sampling._compute_rate`` but for text tokens."""
    B, D = x_t_txt.shape

    emb_x_1 = path_txt.embedding(x_1_sample)                         # (B, D, E)
    prob = path_txt.get_prob_distribution(emb_x_1, t)                 # (B, D, V)

    emb_x_t = path_txt.embedding(x_t_txt)                            # (B, D, E)
    emb_x_t_f = F.normalize(emb_x_t.view(-1, emb_x_t.shape[-1]), p=2, dim=-1)
    emb_x_1_f = F.normalize(emb_x_1.view(-1, emb_x_1.shape[-1]), p=2, dim=-1)

    dist_xt_x1 = (
        (emb_x_t_f ** 2).sum(1, keepdim=True)
        + (emb_x_1_f ** 2).sum(1, keepdim=True)
        - 2 * torch.einsum("bd,bd->b", emb_x_t_f, emb_x_1_f).unsqueeze(1)
    ) ** 2

    dist_x1_all = path_txt.metric(emb_x_1)
    distance = F.relu(dist_xt_x1 - dist_x1_all)

    if t.item() == 0:
        d_beta_t = torch.tensor(0.0, device=device)
    else:
        d_beta_t = (
            path_txt.c * path_txt.a
            * ((t / (1 - t)) ** (path_txt.a - 1))
            / ((1 - t) ** 2)
        )

    u = (prob.reshape(-1, vocabulary_size_txt) * d_beta_t * distance)
    u = torch.nan_to_num(u, nan=0.0, posinf=0.0, neginf=0.0)
    u = u.reshape(B, D, vocabulary_size_txt)

    mask = F.one_hot(x_t_txt, num_classes=vocabulary_size_txt).bool()
    u = u.masked_fill(mask, 0.0)
    return u


@torch.no_grad()
def sample_text_with_log_prob(
    model, path_txt, vocabulary_size_txt,
    x_init, data_info, step_size, cfg_scale=0.0,
    n_mc=1, dtype_categorical=torch.float32,
    verbose=False, device="cuda", offload_to_cpu=True,
):
    """Euler sampling on TEXT tokens with saved intermediates for GRPO."""
    n_steps = ceil(1.0 / step_size)
    ts = torch.tensor(
        [step_size * i for i in range(n_steps)] + [1.0], device=device)

    x_t = x_init.clone()
    B = x_t.shape[0]
    text_mask = data_info["text_token_mask"]

    trajectory = [x_t.clone()]
    all_x1_samples, all_factors = [], []
    all_p_old_times_factor = []
    changed_masks = []
    is_last_step = []

    ctx = tqdm(total=1.0, desc="Sampling-U") if verbose else nullcontext()
    with ctx:
        buf = x_t.clone()
        for i in range(n_steps):
            t = ts[i:i + 1]
            h = ts[i + 1:i + 2] - t

            # model(..., g_or_u='understanding') -> (p_1t_txt_softmax, None, datainfo)
            p_1t_txt, _, _ = model(x=x_t, cfg_scale=cfg_scale, datainfo=data_info)

            x_t_txt = x_t[text_mask == 1].reshape(B, -1)
            posterior = p_1t_txt[text_mask == 1].reshape(B, -1, vocabulary_size_txt)
            D = x_t_txt.shape[1]
            p_flat = posterior.reshape(B * D, vocabulary_size_txt)

            if i == n_steps - 1:
                is_last_step.append(True)
                x_1 = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
                p_old = torch.gather(posterior, -1, x_1.unsqueeze(-1)).squeeze(-1)
                factor = 1.0 / p_old.clamp(min=1e-30)

                all_x1_samples.append([x_1])
                all_factors.append([factor])
                all_p_old_times_factor.append(torch.ones(B, D, device=device))
                changed_masks.append(torch.ones(B, D, dtype=torch.bool, device=device))

                buf[text_mask == 1] = x_1.flatten()
                x_t = buf.clone()
            else:
                is_last_step.append(False)
                step_x1, step_factors = [], []
                u_accum = torch.zeros(B, D, vocabulary_size_txt, device=device)
                u_list, lam_list, ps_list, p_old_list = [], [], [], []

                for _ in range(n_mc):
                    x1_j = safe_categorical(p_flat.to(dtype_categorical)).reshape(B, D)
                    step_x1.append(x1_j)
                    p_old_list.append(
                        torch.gather(posterior, -1, x1_j.unsqueeze(-1)).squeeze(-1))
                    u_j = _compute_rate_txt(path_txt, x_t_txt, x1_j, t,
                                            vocabulary_size_txt, device)
                    lam_j = u_j.sum(-1)
                    ps_j = torch.exp(-h * lam_j)
                    u_accum += u_j
                    u_list.append(u_j); lam_list.append(lam_j); ps_list.append(ps_j)

                u_avg = u_accum / n_mc
                ps_avg = torch.exp(-h * u_avg.sum(-1))
                next_txt = x_t_txt.clone()
                jump = torch.rand(B, D, device=device) > ps_avg
                if jump.any():
                    next_txt[jump] = safe_categorical(u_avg[jump].to(dtype_categorical))

                changed = (x_t_txt != next_txt)
                next_oh = F.one_hot(next_txt, vocabulary_size_txt).float()
                denom_accum = torch.zeros(B, D, device=device)
                for j in range(n_mc):
                    q_to_next = (u_list[j] * next_oh).sum(-1)
                    jump_term = torch.where(
                        lam_list[j] > 1e-20,
                        (q_to_next / lam_list[j].clamp(min=1e-30)) * (1.0 - ps_list[j]),
                        torch.zeros_like(q_to_next),
                    )
                    braces = torch.where(changed, jump_term, ps_list[j])
                    step_factors.append(braces / p_old_list[j].clamp(min=1e-30))
                    denom_accum += braces

                all_x1_samples.append(step_x1)
                all_factors.append(step_factors)
                all_p_old_times_factor.append(denom_accum / n_mc)
                changed_masks.append(changed)

                buf[text_mask == 1] = next_txt.flatten()
                x_t = buf.clone()

            trajectory.append(x_t.clone())
            if verbose and hasattr(ctx, "n"):
                ctx.n = (t + h).item(); ctx.refresh()

    if offload_to_cpu:
        traj_out = [t.cpu() for t in trajectory]
        x1_out = [[x.cpu() for x in s] for s in all_x1_samples]
        fac_out = [[f.cpu() for f in s] for s in all_factors]
        p_old_out = [p.cpu() for p in all_p_old_times_factor]
        ch_out = [m.cpu() for m in changed_masks]
    else:
        traj_out = trajectory; x1_out = all_x1_samples; fac_out = all_factors
        p_old_out = all_p_old_times_factor; ch_out = changed_masks

    return {
        "final_tokens": x_t,
        "trajectory": traj_out,
        "all_x1_samples": x1_out,
        "all_factors": fac_out,
        "all_p_old_times_factor": p_old_out,
        "changed_masks": ch_out,
        "is_last_step": is_last_step,
        "text_mask": text_mask,
        "n_mc": n_mc,
    }
