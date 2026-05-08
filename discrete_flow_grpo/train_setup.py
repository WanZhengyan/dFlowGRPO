"""
Training setup / initialization for GRPO training.

Handles:
  - Model loading, freezing, gradient checkpointing
  - LoRA setup
  - Optimizer creation
  - EMA initialization
  - Old-policy snapshot
  - Checkpoint resume
  - CFG wrapper creation
  - Dataset + dataloader creation
  - Eval dataset splitting
"""

import json
import os
import re
import torch
from contextlib import contextmanager
from torch.utils.data import DataLoader

from fudoki.janus.models import VLChatProcessor
from fudoki.model import instantiate_model
from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from peft import LoraConfig, get_peft_model
from flow_grpo.ema import EMAModuleWrapper
from cfg_model import DifferentiableCFGScaledModel

from config import CFG_SCALE
from data import TextPromptDataset, GenevalPromptDataset
from reward_utils import resolve_training_reward


def resolve_train_cfg_scale(args):
    """Resolve the effective CFG scale for training."""
    if args.no_cfg:
        return 0.0
    elif args.train_cfg_scale is not None:
        return args.train_cfg_scale
    else:
        return CFG_SCALE  # 5.0


def setup_model(args, device, accelerator):
    """
    Load and configure the FUDOKI model:
      - Freeze non-trainable modules
      - Enable gradient checkpointing (optional)
      - Apply LoRA (optional)
      - Save reference weights for KL (if needed)

    Returns: (model, gen_ref_state)
    """
    model = instantiate_model(args.checkpoint_path).to(device).to(torch.float32)

    # Freeze modules that should NOT be trained:
    #   gen_vision_model (VQ-VAE decoder), vision_model & aligner (understanding only)
    #   gen_embed: always frozen (0.13M, tiny embedding table)
    #   gen_aligner, gen_head: optionally trainable via --train_gen_modules (~42M params)
    _always_frozen = ("gen_vision_model.", "vision_model.", "aligner.", "gen_embed.")
    _gen_trainable = ("gen_aligner.", "gen_head.")
    for name, param in model.named_parameters():
        if name.startswith(_always_frozen):
            param.requires_grad = False
        elif name.startswith(_gen_trainable):
            param.requires_grad = args.train_gen_modules

    # Enable gradient checkpointing BEFORE LoRA
    if args.gradient_checkpointing:
        model.language_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        accelerator.print("Gradient checkpointing enabled for language_model")

    # LoRA: apply to language_model only
    if args.use_lora:
        if args.gradient_checkpointing:
            model.language_model.enable_input_require_grads()
        lora_target_modules = [
            "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
            "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
        ]
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian", target_modules=lora_target_modules,
        )
        model.language_model = get_peft_model(model.language_model, lora_config)
        for name, param in model.language_model.named_parameters():
            if 'lora' in name:
                param.data = param.data.to(torch.float32)
        lora_params = sum(p.numel() for p in model.language_model.parameters() if p.requires_grad)
        total_lm_params = sum(p.numel() for p in model.language_model.parameters())
        accelerator.print(f"LoRA enabled: r={args.lora_r}, alpha={args.lora_alpha}")
        accelerator.print(f"  LM trainable: {lora_params/1e6:.1f}M / {total_lm_params/1e6:.1f}M "
                          f"({100*lora_params/total_lm_params:.1f}%)")
        if args.train_gen_modules:
            gen_params = sum(p.numel() for n, p in model.named_parameters()
                             if n.startswith(("gen_aligner.", "gen_head.")) and p.requires_grad)
            accelerator.print(f"  gen_aligner/gen_head: TRAINABLE ({gen_params/1e6:.1f}M params), gen_embed: frozen")
        else:
            accelerator.print(f"  gen_embed/gen_aligner/gen_head: frozen")

    # Save reference (init) weights for gen_modules if trainable + KL is used
    gen_ref_state = None
    if args.train_gen_modules and args.kl_beta > 0 and args.use_lora:
        gen_ref_state = {}
        for n, p in model.named_parameters():
            if n.startswith(("gen_aligner.", "gen_head.")):
                gen_ref_state[n] = p.data.clone().cpu()
        accelerator.print(f"  Saved gen_module reference weights for KL ({len(gen_ref_state)} tensors)")

    model.train()
    return model, gen_ref_state


def make_disable_adapter_ctx(gen_ref_state):
    """
    Build a context manager factory that disables LoRA adapters AND restores
    gen_module weights to their reference (init) values for a clean π_ref forward.

    Returns: disable_adapter_and_gen_modules(raw_model) context manager
    """
    @contextmanager
    def disable_adapter_and_gen_modules(raw_model):
        if gen_ref_state is not None:
            _gen_temp = {}
            for n, p in raw_model.named_parameters():
                if n in gen_ref_state:
                    _gen_temp[n] = p.data.clone()
                    p.data.copy_(gen_ref_state[n].to(p.device))
        with raw_model.language_model.disable_adapter():
            yield
        if gen_ref_state is not None:
            for n, p in raw_model.named_parameters():
                if n in _gen_temp:
                    p.data.copy_(_gen_temp[n])
            del _gen_temp

    return disable_adapter_and_gen_modules


def setup_optimizer(model, args, accelerator):
    """Create AdamW optimizer for trainable parameters."""
    lr = args.learning_rate
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=args.adam_epsilon,
    )
    accelerator.print(f"Optimizer: AdamW(lr={lr}, betas=({args.adam_beta1}, {args.adam_beta2}), "
                      f"weight_decay={args.weight_decay}, eps={args.adam_epsilon})")
    return optimizer


def setup_ema(args, trainable_params, device, accelerator):
    """Initialize EMA wrapper if enabled."""
    if not args.ema:
        return None
    _ema_device = str(device)
    ema = EMAModuleWrapper(
        trainable_params, decay=args.ema_decay,
        update_step_interval=args.ema_update_interval, device=_ema_device,
    )
    accelerator.print(f"EMA enabled: decay={args.ema_decay}, "
                      f"update_interval={args.ema_update_interval} (stored on {_ema_device})")
    return ema


def validate_ema_args(args, ema, accelerator):
    """Validate EMA-related argument combinations."""
    if args.sample_with_ema and ema is None:
        raise ValueError("--sample_with_ema requires --ema to be enabled")
    if args.sample_with_ema and args.old_policy_update_interval > 0:
        raise ValueError("--sample_with_ema and --old_policy_update_interval are mutually exclusive.")
    if args.sample_with_ema:
        accelerator.print("Old-policy sampling will use EMA parameters (π_ema)")


def setup_old_policy(args, trainable_params, accelerator):
    """Initialize old-policy snapshot if enabled."""
    if args.old_policy_update_interval <= 0:
        return None
    old_policy_params = [p.data.clone() for p in trainable_params]
    accelerator.print(f"Old-policy snapshot initialized ({len(old_policy_params)} tensors, "
                      f"refresh every {args.old_policy_update_interval} optimizer steps)")
    return old_policy_params


def resume_from_checkpoint(args, accelerator, model, ema):
    """Resume training state from a checkpoint directory.

    Returns: resumed_global_step (int).
    The epoch and within-epoch position are derived purely from global_step
    and the dataloader length in the training loop.
    """
    if args.resume_from_checkpoint is None:
        return 0

    ckpt_dir = args.resume_from_checkpoint
    accelerator.print(f"Resuming from checkpoint: {ckpt_dir}")

    # ------------------------------------------------------------------
    # Workaround for a FUDOKI quirk: `VectorQuantizer2` does
    #     self.register_buffer("codebook_used", nn.Parameter(torch.zeros(65536)))
    # which leaves a `nn.Parameter` sitting inside `_buffers`. Depending on the
    # torch version, `module.load_state_dict(...)` may silently promote it to a
    # real Parameter (see Module.__setattr__ / _load_from_state_dict). That
    # turns it into a "frozen parameter" from DeepSpeed's point of view when it
    # collects `FROZEN_PARAM_FRAGMENTS` on resume, but the checkpoint we try to
    # resume from was saved WITHOUT that key, so we get
    #     KeyError: 'gen_vision_model.quantize.codebook_used'
    # in deepspeed/runtime/engine.py::load_module_state_dict.
    #
    # Normalize every such "buffer-disguised-as-Parameter" into a plain tensor
    # buffer BEFORE calling accelerator.load_state().
    _raw = accelerator.unwrap_model(model)
    _normalized = 0
    for _mod in _raw.modules():
        for _bname, _bval in list(_mod._buffers.items()):
            if isinstance(_bval, torch.nn.Parameter):
                _plain = _bval.data.detach().clone()
                # Drop the Parameter wrapper; keep it as a normal persistent buffer.
                del _mod._buffers[_bname]
                _mod.register_buffer(_bname, _plain, persistent=True)
                _normalized += 1
        # Also handle the case where a previous load_state_dict already
        # promoted it into _parameters: demote it back to a buffer so the
        # DeepSpeed frozen-param iteration won't enumerate it.
        for _pname in list(_mod._parameters.keys()):
            _pval = _mod._parameters[_pname]
            if _pval is None:
                continue
            # Only demote things that were originally buffers-with-Parameter.
            # Heuristic: name == 'codebook_used'.
            if _pname == "codebook_used":
                _plain = _pval.data.detach().clone()
                del _mod._parameters[_pname]
                _mod.register_buffer(_pname, _plain, persistent=True)
                _normalized += 1
    if _normalized:
        accelerator.print(
            f"  Normalized {_normalized} 'buffer-as-Parameter' tensor(s) "
            f"to plain buffers before resume (FUDOKI codebook_used workaround)."
        )

    accelerator.load_state(ckpt_dir)
    accelerator.print("  Restored model + optimizer + RNG states via accelerator.load_state")

    # Restore gen_modules weights if they were saved
    gen_modules_path = os.path.join(ckpt_dir, "gen_modules_state.pt")
    if os.path.exists(gen_modules_path) and args.train_gen_modules:
        gen_state = torch.load(gen_modules_path, map_location="cpu")
        _raw = accelerator.unwrap_model(model)
        for n, p in _raw.named_parameters():
            if n in gen_state:
                p.data.copy_(gen_state[n].to(p.device))
        accelerator.print(f"  Restored gen_modules ({len(gen_state)} tensors)")
        del gen_state

    # Restore EMA state if available
    ema_path = os.path.join(ckpt_dir, "ema_state.pt")
    if ema is not None and os.path.exists(ema_path):
        ema.load_state_dict(torch.load(ema_path, map_location="cpu"))
        accelerator.print("  Restored EMA state")

    # Determine global_step: try training_meta.json first, then parse from dir name
    meta_path = os.path.join(ckpt_dir, "training_meta.json")
    if os.path.exists(meta_path):
        import json
        with open(meta_path, "r") as f:
            meta = json.load(f)
        resumed_step = meta.get("global_step", 0)
        accelerator.print(f"  Loaded training_meta.json: global_step={resumed_step}")
        return resumed_step

    _step_match = re.search(r"step(\d+)", os.path.basename(ckpt_dir))
    if _step_match:
        resumed_step = int(_step_match.group(1))
        accelerator.print(f"  Resumed global_step = {resumed_step}")
    else:
        resumed_step = 0
        accelerator.print("  WARNING: could not parse global_step from checkpoint dir name, starting from 0")

    return resumed_step


def setup_cfg_wrapper(model, accelerator, device):
    """Build CFG wrapper after accelerator.prepare, ensure VQ decoder is on GPU."""
    raw_model = accelerator.unwrap_model(model)
    if torch.cuda.is_available():
        try:
            if hasattr(raw_model, 'gen_vision_model'):
                raw_model.gen_vision_model.to(device).to(torch.float32)
                raw_model.gen_vision_model.eval()
                accelerator.print(f"gen_vision_model moved to {device} (float32) for GPU decoding")
        except Exception as e:
            accelerator.print(f"Warning: failed to move gen_vision_model to {device}: {e}")
    diff_cfg_model = DifferentiableCFGScaledModel(model=raw_model, g_or_u='generation')
    return diff_cfg_model, raw_model


def setup_datasets(args, accelerator):
    """
    Create train/eval datasets and dataloader.

    Returns: (dataloader, eval_dataset, use_geneval_metadata, prompt_dataset_name)
    """
    P = args.prompts_per_sample_batch
    prompt_dataset_name = args.prompt_dataset
    if prompt_dataset_name is None:
        if args.reward_dict:
            _rd = json.loads(args.reward_dict)
            prompt_dataset_name = "geneval" if "geneval" in _rd else "pickscore"
        else:
            prompt_dataset_name = "pickscore"

    use_geneval_metadata = (prompt_dataset_name == "geneval")
    dataset_path = args.dataset_dir

    if use_geneval_metadata:
        dataset = GenevalPromptDataset(dataset_path, split="train")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                collate_fn=GenevalPromptDataset.collate_fn)
        eval_dataset = GenevalPromptDataset(dataset_path, split="test")
        accelerator.print(f"Prompt dataset: geneval ({len(dataset)} train, {len(eval_dataset)} test)")
    else:
        dataset = TextPromptDataset(dataset_path, split="train")
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True,
                                collate_fn=TextPromptDataset.collate_fn)
        eval_dataset = TextPromptDataset(dataset_path, split="test")
        accelerator.print(f"Prompt dataset: pickscore ({len(dataset)} train)")

    return dataloader, eval_dataset, use_geneval_metadata, prompt_dataset_name


def setup_eval_split(eval_dataset, use_geneval_metadata, accelerator, args):
    """
    Split eval dataset across GPUs.

    Returns: (eval_prompts_list, eval_metadatas_list, n_prompts_per_gpu)
    """
    n_prompts_per_gpu = args.eval_num_prompts_per_gpu
    total_eval_prompts = n_prompts_per_gpu * accelerator.num_processes
    total_eval_prompts = min(total_eval_prompts, len(eval_dataset))
    total_eval_prompts = (total_eval_prompts // accelerator.num_processes) * accelerator.num_processes
    n_prompts_per_gpu = total_eval_prompts // accelerator.num_processes

    all_eval_prompts = eval_dataset.prompts[:total_eval_prompts]
    eval_prompts_list = all_eval_prompts[
        accelerator.process_index * n_prompts_per_gpu
        : (accelerator.process_index + 1) * n_prompts_per_gpu
    ]

    eval_metadatas_list = None
    if use_geneval_metadata:
        all_eval_metadatas = eval_dataset.metadatas[:total_eval_prompts]
        eval_metadatas_list = all_eval_metadatas[
            accelerator.process_index * n_prompts_per_gpu
            : (accelerator.process_index + 1) * n_prompts_per_gpu
        ]

    accelerator.print(f"Eval: {total_eval_prompts} total test prompts, {n_prompts_per_gpu} per GPU, "
                      f"{args.eval_samples_per_prompt} images/prompt, every {args.eval_every} steps")
    return eval_prompts_list, eval_metadatas_list, n_prompts_per_gpu
