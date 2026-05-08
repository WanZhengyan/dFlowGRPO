"""
Argument parsing for on-policy DPO training.

Mirrors train_args.py (so all GRPO knobs / paths / sampling / EMA / LoRA flags
stay available) and adds DPO-specific options:
    --beta_dpo         : β' coefficient inside log σ(...)
    --dpo_num_pairs    : number of (winner, loser) pairs per prompt group.
                         0 => G // 2 pairs (top vs bottom by reward).
"""

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Discrete Flow on-policy DPO Training")

    # ---- Paths ----
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--text_embedding_path", type=str, required=True)
    parser.add_argument("--image_embedding_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True)

    # ---- General ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discrete_fm_steps", type=int, default=8)
    parser.add_argument("--txt_max_length", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=5)

    # ---- Optimizer ----
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1)

    # ---- Group / sampling ----
    parser.add_argument("--group_size", type=int, default=24,
                        help="Group size G (images per prompt).")
    parser.add_argument("--sample_batch_size", type=int, default=24)
    parser.add_argument("--n_mc", type=int, default=24)

    # ---- CFG ----
    parser.add_argument("--no_cfg", action="store_true")
    parser.add_argument("--train_cfg_scale", type=float, default=None)

    # ---- DPO core ----
    parser.add_argument("--beta_dpo", type=float, default=100,
                        help="DPO temperature β' inside -log σ(β' · Δ). Larger -> sharper preference.")
    parser.add_argument("--dpo_num_pairs", type=int, default=1,
                        help="Number of (winner, loser) pairs per prompt. "
                             "0 => G//2 pairs from sorted-by-reward top vs bottom.")

    # ---- Pair filtering ----
    parser.add_argument("--filter_zero_std", action="store_true",
                        help="Skip prompt groups whose reward std is zero (no informative pairs).")

    # ---- KL penalty (optional, vs π_ref via disable_adapter; same convention as GRPO) ----
    parser.add_argument("--kl_beta", type=float, default=0,
                        help="KL penalty coefficient for KL(π_θ||π_ref). 0 to disable.")
    parser.add_argument("--kl_old_policy", action="store_true",
                        help="Compute KL(π_θ||π_old) using importance ratio (no extra ref forward).")

    # ---- Training steps ----
    parser.add_argument("--train_steps", type=str, default=None)
    parser.add_argument("--include_last_step", action="store_true")

    # ---- Batching / accumulation ----
    parser.add_argument("--num_inner_updates", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=12,
                        help="Number of PAIRS per training forward pass (each pair => 2 samples).")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--prompts_per_sample_batch", type=int, default=2)

    # ---- EMA ----
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.9)
    parser.add_argument("--ema_update_interval", type=int, default=10)
    parser.add_argument("--sample_with_ema", action="store_true")
    parser.add_argument("--old_policy_update_interval", type=int, default=48)

    # ---- LoRA ----
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=48)
    parser.add_argument("--lora_alpha", type=int, default=96)

    # ---- Generation modules ----
    parser.add_argument("--train_gen_modules", action="store_true")

    # ---- Reward (only pickscore + geneval supported in this script) ----
    parser.add_argument("--reward_type", type=str, default=None,
                        choices=["aesthetic", "clip", "aesthetic_clip", "pickscore", "jpeg"])
    parser.add_argument("--reward_dict", type=str, default=None,
                        help='JSON dict, e.g. \'{"pickscore": 1.0}\' or \'{"geneval": 1.0}\'.')
    parser.add_argument("--prompt_dataset", type=str, default=None,
                        choices=["pickscore", "geneval"])
    parser.add_argument("--aesthetic_weight", type=float, default=0.5)
    parser.add_argument("--clip_weight", type=float, default=0.5)
    parser.add_argument("--reward_on_gpu", action="store_true")

    # ---- Eval / checkpoint ----
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_samples_per_prompt", type=int, default=1)
    parser.add_argument("--eval_num_prompts_per_gpu", type=int, default=10)
    parser.add_argument("--eval_ema_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=100)

    # ---- Logging ----
    parser.add_argument("--wandb_name", type=str, default=None)

    # ---- Resume ----
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    # ---- Compatibility shims (consumed by setup helpers; unused in DPO loss) ----
    # These are kept so that train_setup helpers (validate_ema_args, etc.) don't fail.
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--epsilon_low", type=float, default=None)
    parser.add_argument("--epsilon_high", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--adv_clip_max", type=float, default=0)
    parser.add_argument("--global_std", action="store_true")
    parser.add_argument("--top_bottom_k", type=int, default=0)
    parser.add_argument("--changed_only", action="store_true")
    parser.add_argument("--changed_frac_reward_bonus", type=float, default=0)
    parser.add_argument("--changed_frac_threshold", type=float, default=0)
    parser.add_argument("--token_level_loss", action="store_true")
    parser.add_argument("--dim_level_clip", action="store_true")
    parser.add_argument("--per_step_reward", action="store_true")

    return parser.parse_args()
