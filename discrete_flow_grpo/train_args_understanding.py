"""
CLI arguments for multimodal-understanding GRPO training.

Kept deliberately close to ``train_args.py`` (generation). Image-specific flags
(CFG, gen_modules) are dropped or made no-ops.
"""
import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Discrete Flow GRPO — Understanding")

    # ---- Paths ----
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--text_embedding_path", type=str, required=True)
    parser.add_argument("--image_embedding_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, required=True,
                        help="Path to .../dataset_understanding/ScienceQA")

    # ---- General ----
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--discrete_fm_steps", type=int, default=8)
    parser.add_argument("--txt_max_length", type=int, default=500)
    parser.add_argument("--max_prompt_chars", type=int, default=1800,
                        help="Drop ScienceQA items whose formatted prompt (Q + "
                             "options + hint) exceeds this many characters. "
                             "Prevents 'Split token not found' crashes from "
                             "over-long prompts that get truncated past the "
                             "'Assistant:' marker. ~1800 chars ≈ 450 tokens.")
    parser.add_argument("--num_epochs", type=int, default=20)

    # ---- Optimizer ----
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # ---- GRPO hyper-params ----
    parser.add_argument("--group_size", type=int, default=24)
    parser.add_argument("--sample_batch_size", type=int, default=24)
    parser.add_argument("--n_mc", type=int, default=1,
                        help="MC posterior samples per Euler step. "
                             "Memory scales linearly with n_mc × V_txt (102400). "
                             "Keep low for understanding; image-gen uses 24 but V_img=16384.")
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--epsilon_low", type=float, default=1e-3)
    parser.add_argument("--epsilon_high", type=float, default=1.5e-3)
    parser.add_argument("--gamma", type=float, default=1.0)

    # ---- Adv normalization ----
    parser.add_argument("--adv_clip_max", type=float, default=0.0)
    parser.add_argument("--filter_zero_std", action="store_true")
    parser.add_argument("--global_std", action="store_true")

    # ---- Loss variants ----
    parser.add_argument("--changed_only", action="store_true")
    parser.add_argument("--token_level_loss", action="store_true")

    # ---- KL penalty ----
    parser.add_argument("--kl_beta", type=float, default=0.0)
    parser.add_argument("--kl_old_policy", action="store_true")

    # ---- Training steps ----
    parser.add_argument("--train_steps", type=str, default=None)
    parser.add_argument("--include_last_step", action="store_true")

    # ---- Batching / accumulation ----
    parser.add_argument("--num_inner_updates", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--prompts_per_sample_batch", type=int, default=1) # For understanding, we cannot tune this.

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

    # Kept for API compatibility with train_setup helpers; not used.
    parser.add_argument("--train_gen_modules", action="store_true")

    # ---- Eval / save ----
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--eval_num_samples_per_gpu", type=int, default=20)
    parser.add_argument("--eval_ema_every", type=int, default=20)
    parser.add_argument("--save_every", type=int, default=100)

    # ---- Logging ----
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="discrete_flow_grpo_u")

    # ---- API judge (optional LLM-assisted answer extraction) ----
    parser.add_argument("--use_api_judge", action="store_true",
                        help="When local extraction (can_infer + regex) fails, "
                             "fall back to an OpenAI-compatible LLM to map "
                             "the free-form answer to a choice letter. "
                             "Mirrors VLMEvalKit's extract_answer_from_item.")
    parser.add_argument("--api_judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--api_judge_base_url", type=str, default=None,
                        help="Override endpoint, e.g. https://api.deepseek.com/v1")
    parser.add_argument("--api_judge_key_env", type=str, default="OPENAI_API_KEY",
                        help="Name of env var holding the API key.")
    parser.add_argument("--api_judge_max_retries", type=int, default=3)
    parser.add_argument("--api_judge_max_workers", type=int, default=8)
    parser.add_argument("--api_judge_temperature", type=float, default=0.0)
    parser.add_argument("--api_judge_timeout", type=float, default=30.0)
    parser.add_argument("--api_judge_verbose", action="store_true")
    parser.add_argument("--api_judge_only_eval", action="store_true",
                        help="Use the API judge only during periodic eval (not "
                             "during RL reward computation). Useful to avoid "
                             "spending API budget during training while still "
                             "getting accurate eval numbers.")

    # ---- Resume ----
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    return parser.parse_args()
