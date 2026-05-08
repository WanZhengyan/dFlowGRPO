"""
Argument parsing for GRPO training.

All training hyper-parameters and paths are exposed as CLI arguments.
No hardcoded paths — pass everything via CLI or shell scripts.
"""

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Discrete Flow GRPO Training")

    # ---- Paths (all required — no hardcoded defaults) ----
    parser.add_argument("--checkpoint_path", type=str, required=True,
                        help="Path to the base FUDOKI model checkpoint directory.")
    parser.add_argument("--text_embedding_path", type=str, required=True,
                        help="Path to the text embedding .pt file.")
    parser.add_argument("--image_embedding_path", type=str, required=True,
                        help="Path to the image embedding .pt file.")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save checkpoints and eval images.")
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Root directory for the prompt dataset (e.g. .../dataset/geneval "
                             "or .../dataset/pickscore). Must contain train.txt (pickscore) or "
                             "train_metadata.jsonl (geneval) depending on --prompt_dataset.")

    # ---- General ----
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--discrete_fm_steps", type=int, default=8,
                        help="Inference steps for discrete flow matching.")
    parser.add_argument("--txt_max_length", type=int, default=500,
                        help="Text length maximum.")
    parser.add_argument("--num_epochs", type=int, default=5,
                        help="Number of training epochs.")

    # ---- Optimizer ----
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="AdamW weight decay. Flow-GRPO default: 1e-4.")
    parser.add_argument("--adam_beta1", type=float, default=0.9,
                        help="AdamW beta1.")
    parser.add_argument("--adam_beta2", type=float, default=0.999,
                        help="AdamW beta2.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8,
                        help="AdamW epsilon.")
    parser.add_argument("--max_grad_norm", type=float, default=1,
                        help="Max gradient norm for clipping. 0 to disable.")

    # ---- GRPO hyper-params ----
    parser.add_argument("--group_size", type=int, default=24,
                        help="Group size G (images per prompt).")
    parser.add_argument("--sample_batch_size", type=int, default=24,
                        help="Max images per sampling call. If group_size > sample_batch_size, "
                             "sampling is split into ceil(G/sample_batch_size) rounds and merged.")
    parser.add_argument("--n_mc", type=int, default=24,
                        help="MC posterior samples per Euler step.")
    parser.add_argument("--epsilon", type=float, default=1e-3,
                        help="PPO clip range (symmetric default).")
    parser.add_argument("--epsilon_low", type=float, default=1e-3,
                        help="PPO clip lower bound: ratio clipped to [1 - epsilon_low, ...]. "
                             "If not set, falls back to --epsilon for symmetric clipping.")
    parser.add_argument("--epsilon_high", type=float, default=1.5e-3,
                        help="PPO clip upper bound: ratio clipped to [..., 1 + epsilon_high]. "
                             "If not set, falls back to --epsilon for symmetric clipping.")
    parser.add_argument("--gamma", type=float, default=1,
                        help="Discount factor for Euler steps: step k advantage *= gamma^k.")

    # ---- CFG ----
    parser.add_argument("--no_cfg", action="store_true",
                        help="Disable CFG during training (sets train_cfg_scale=0). "
                             "Halves model forwards during both sampling and training. "
                             "Eval still uses full CFG for quality.")
    parser.add_argument("--train_cfg_scale", type=float, default=None,
                        help="CFG scale for training (sampling + training forward). "
                             "If not set, uses the global CFG_SCALE (5.0). "
                             "--no_cfg overrides this to 0.")

    # ---- Advantage normalization ----
    parser.add_argument("--adv_clip_max", type=float, default=0,
                        help="Clip advantages to [-adv_clip_max, adv_clip_max] after z-score "
                             "normalization. 0 to disable.")
    parser.add_argument("--filter_zero_std", action="store_true",
                        help="Skip training on prompt groups whose reward std is zero.")
    parser.add_argument("--global_std", action="store_true",
                        help="Use global std (across all prompts in the batch) for advantage "
                             "normalization instead of per-prompt std.")

    # ---- Sample selection ----
    parser.add_argument("--top_bottom_k", type=int, default=0,
                        help="Per-prompt group, keep only the top-K + bottom-K samples. "
                             "0 to disable.")

    # ---- Loss variants ----
    parser.add_argument("--changed_only", action="store_true",
                        help="Only use tokens that actually changed (jumped) at each denoising "
                             "step for computing the policy ratio and loss.")
    parser.add_argument("--changed_frac_reward_bonus", type=float, default=0,
                        help="Reward bonus for images with changed_frac below threshold. "
                             "0 to disable.")
    parser.add_argument("--changed_frac_threshold", type=float, default=0,
                        help="Threshold for --changed_frac_reward_bonus.")
    parser.add_argument("--token_level_loss", action="store_true",
                        help="Compute the PPO surrogate loss at the per-token level.")
    parser.add_argument("--dim_level_clip", action="store_true",
                        help="Clip the importance ratio at per-token level before aggregating "
                             "into the sample-level geometric mean.")

    # ---- KL penalty ----
    parser.add_argument("--kl_beta", type=float, default=0,
                        help="KL penalty coefficient for KL(π_θ||π_ref). 0 to disable.")
    parser.add_argument("--kl_old_policy", action="store_true",
                        help="When kl_beta > 0, compute KL(π_θ||π_old) using the existing "
                             "importance ratio instead of running an extra reference forward.")

    # ---- Training steps ----
    parser.add_argument("--train_steps", type=str, default=None,
                        help="Subset of denoising steps to train on. "
                             "Comma-separated indices or ranges: '4,5,6' or '4-6'. "
                             "'all' or None = use all available steps.")
    parser.add_argument("--include_last_step", action="store_true",
                        help="Include the final denoising step (step K-1) in training.")

    # ---- Flow-GRPO-Fast: random contiguous training-step window ----
    parser.add_argument("--fast_window_size", type=int, default=0,
                        help="Flow-GRPO-Fast: if > 0, every sampling iteration we randomly "
                             "pick a contiguous window of this many denoising steps to "
                             "train on (e.g. window=3 → {0,1,2} or {1,2,3} or {2,3,4} ...). "
                             "The window's right boundary (start + window - 1) is constrained "
                             "to be <= --fast_window_max_end (default: K // 2). "
                             "Overrides --train_steps. Set 0 to disable.")
    parser.add_argument("--fast_window_max_end", type=int, default=-1,
                        help="Upper bound (inclusive) on the right edge of the random "
                             "fast-window. -1 (default) = K // 2, where K = discrete_fm_steps. "
                             "Only used when --fast_window_size > 0.")

    # ---- Batching / accumulation ----
    parser.add_argument("--num_inner_updates", type=int, default=1,
                        help="Number of training passes over each prompt's G samples.")
    parser.add_argument("--train_batch_size", type=int, default=24,
                        help="Number of samples to batch together per forward pass during training.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce VRAM usage.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of sampling calls to accumulate gradients over per GPU.")
    parser.add_argument("--prompts_per_sample_batch", type=int, default=2,
                        help="Number of prompts (P) to batch together per sampling call per GPU.")

    # ---- EMA ----
    parser.add_argument("--ema", action="store_true",
                        help="Enable EMA for sampling/inference.")
    parser.add_argument("--ema_decay", type=float, default=0.9,
                        help="EMA decay rate.")
    parser.add_argument("--ema_update_interval", type=int, default=10,
                        help="EMA update step interval.")
    parser.add_argument("--sample_with_ema", action="store_true",
                        help="Use EMA parameters as old policy for sampling.")
    parser.add_argument("--old_policy_update_interval", type=int, default=48,
                        help="Refresh old-policy snapshot every N optimizer steps. "
                             "0 = disabled. Mutually exclusive with --sample_with_ema.")

    # ---- LoRA ----
    parser.add_argument("--use_lora", action="store_true",
                        help="Enable LoRA for language_model.")
    parser.add_argument("--lora_r", type=int, default=48,
                        help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=96,
                        help="LoRA alpha.")

    # ---- Generation modules ----
    parser.add_argument("--train_gen_modules", action="store_true",
                        help="Also train gen_aligner, gen_head (image I/O layers).")

    # ---- Reward ----
    parser.add_argument("--reward_type", type=str, default=None,
                        choices=["aesthetic", "clip", "aesthetic_clip", "pickscore", "jpeg"],
                        help="(Legacy) Single reward function. Use --reward_dict instead.")
    parser.add_argument("--reward_dict", type=str, default=None,
                        help='JSON dict of {reward_name: weight}. '
                             'E.g. \'{"aesthetic": 0.5, "clip": 0.5}\' or \'{"geneval": 1.0}\'.')
    parser.add_argument("--prompt_dataset", type=str, default=None,
                        choices=["pickscore", "geneval"],
                        help="Which prompt dataset to use. Auto-detected from --reward_dict if not set.")
    parser.add_argument("--aesthetic_weight", type=float, default=0.5,
                        help="Weight for aesthetic score in combo reward (legacy).")
    parser.add_argument("--clip_weight", type=float, default=0.5,
                        help="Weight for clip score in combo reward (legacy).")
    parser.add_argument("--reward_on_gpu", action="store_true",
                        help="Run reward scorers on GPU instead of CPU.")

    # ---- Eval ----
    parser.add_argument("--eval_every", type=int, default=10,
                        help="Evaluate every N optimizer steps.")
    parser.add_argument("--eval_samples_per_prompt", type=int, default=1,
                        help="Number of images per prompt during eval.")
    parser.add_argument("--eval_num_prompts_per_gpu", type=int, default=10,
                        help="Number of test prompts per GPU during eval.")
    parser.add_argument("--eval_ema_every", type=int, default=20,
                        help="Evaluate EMA parameters every N optimizer steps. "
                             "Only effective when --ema is enabled. 0 to disable.")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save checkpoint every N optimizer steps.")

    # ---- Per-step reward (forward-difference of x_1 estimates) ----
    parser.add_argument("--per_step_reward", action="store_true",
                        help="Per-step rewards via forward-difference of x_1 estimates: "
                             "A_k_raw = R(decode(x_hat_1^{k+1})) - R(decode(x_hat_1^{k})), "
                             "where x_hat_1^{k} is one randomly-picked MC sample from "
                             "the posterior at denoising step k. The last step's "
                             "advantage is 0 (telescoping baseline). Group-relative "
                             "standardization is then applied per step.")

    # ---- Logging ----
    parser.add_argument("--wandb_name", type=str, default=None,
                        help="Wandb run name. If not set, auto-generated from flags.")

    # ---- Resume ----
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to a checkpoint directory to resume training from.")

    return parser.parse_args()
