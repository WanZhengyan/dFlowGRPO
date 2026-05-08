#!/usr/bin/env python3
# filepath: /your_path/DiscreteFlowRL/discrete_flow_grpo/eval_scienceqa.py
"""
Minimal ScienceQA test-split evaluator for FUDOKI understanding.

Uses the local ``data_understanding.ScienceQADataset`` (test split with images)
and the same reward pipeline as training
(``reward_understanding.build_mcq_reward_fn`` + optional ``api_judge.APIJudge``).

Decoding is **top-1 (greedy)**: we wrap the model with
``fudoki.eval_loop.CFGScaledModel(g_or_u='understanding')`` which applies
``top_k_logits(logits, top_k=1)`` before softmax, so each denoising step
takes the argmax of the text-token posterior.

Example
-------
  python eval_scienceqa.py \\
      --checkpoint_path /path/to/FUDOKI/checkpoints \\
      --api_key sk-xxxx

  # with a trained LoRA (EMA auto-picked next to it)
  python eval_scienceqa.py \\
      --checkpoint_path /path/to/FUDOKI/checkpoints \\
      --lora_path   /path/to/output_grpo_u_scienceqa/checkpoint_step_XXX/lora_adapter \\
      --api_key sk-xxxx
"""

import argparse
import json
import os
import sys
import time

import torch

# Base FUDOKI checkpoints live outside the GRPO dir -- make sure we can import
# the repo modules (config / model_utils / sampling_understanding / ...).
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Patch torch.load for older checkpoints (PyTorch 2.6+ defaults weights_only=True).
_orig_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)
torch.load = _patched_torch_load  # type: ignore

from flow_matching.path import MixtureDiscreteSoftmaxProbPath
from fudoki.eval_loop import CFGScaledModel
from fudoki.janus.models import VLChatProcessor

from config import VOCABULARY_SIZE_TXT
from data_understanding import ScienceQADataset
from model_utils import load_model
from model_utils_understanding import build_data_info_understanding
from sampling_understanding import sample_text_only
from reward_understanding import build_mcq_reward_fn
from api_judge import APIJudge

try:
    from accelerate import Accelerator
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_path", type=str, required=True,
                   help="Path to the base FUDOKI checkpoint directory.")
    p.add_argument("--dataset_dir", type=str,
                   default=os.path.join(_HERE, "dataset_understanding", "ScienceQA"),
                   help="Path to ScienceQA dataset root.")
    p.add_argument("--split", type=str, default="test", choices=["test", "val", "train"])

    # Optional RL-trained weights (same convention as run_eval.sh).
    p.add_argument("--lora_path", type=str, default=None)
    p.add_argument("--ema_path", type=str, default=None)
    p.add_argument("--gen_modules_path", type=str, default=None)

    # Sampling.
    p.add_argument("--steps", type=int, default=16,
                   help="Discrete-flow-matching denoising steps (top-1 per step).")
    p.add_argument("--txt_max_length", type=int, default=500)
    p.add_argument("--max_prompt_chars", type=int, default=1800)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_samples", type=int, default=0,
                   help="Evaluate only the first N items (0 = all).")
    p.add_argument("--device", type=str, default="cuda")

    # OpenAI-compatible API judge (LLM answer-letter extractor).
    p.add_argument("--api_key", type=str, default=None,
                   help="OpenAI API key. If unset, falls back to $OPENAI_API_KEY.")
    p.add_argument("--api_base_url", type=str, default=None)
    p.add_argument("--api_model", type=str, default="gpt-4o-mini")
    p.add_argument("--api_max_workers", type=int, default=8)
    p.add_argument("--api_max_retries", type=int, default=3)
    p.add_argument("--no_api_judge", action="store_true",
                   help="Disable the LLM judge fallback (pure local extraction).")

    # Output.
    p.add_argument("--output_dir", type=str, default=None,
                   help="Dir to dump per-sample predictions (jsonl) and summary.")
    return p.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ---- Accelerator (single-rank when launched with plain python) ----
    if _HAS_ACCELERATE:
        accelerator = Accelerator()
        is_main = accelerator.is_main_process
        world_size = accelerator.num_processes
        rank = accelerator.process_index
        device = accelerator.device
    else:
        accelerator = None
        is_main = True
        world_size = 1
        rank = 0
        device = torch.device(args.device)

    def mprint(*a, **kw):
        if is_main:
            print(*a, **kw)

    # ---- Load model (base + optional LoRA/EMA/gen_modules) ----
    # Auto-detect EMA state next to the LoRA adapter if not explicitly given.
    if args.lora_path and not args.ema_path:
        cand = os.path.join(os.path.dirname(os.path.normpath(args.lora_path)),
                            "ema_state.pt")
        if os.path.isfile(cand):
            args.ema_path = cand
            mprint(f"[eval] auto-detected EMA weights: {cand}")

    mprint(f"[eval] world_size={world_size}  device={device}")
    mprint(f"[eval] loading model ckpt={args.checkpoint_path} "
           f"lora={args.lora_path} ema={args.ema_path} gen={args.gen_modules_path}")
    model = load_model(
        checkpoint_path=args.checkpoint_path,
        lora_path=args.lora_path,
        ema_path=args.ema_path,
        gen_modules_path=args.gen_modules_path,
        device=device,
    )
    model.eval()
    model.training = False

    vl_chat_processor = VLChatProcessor.from_pretrained(args.checkpoint_path)
    path_txt = MixtureDiscreteSoftmaxProbPath(
        mode="text",
        embedding_path=os.path.join(args.checkpoint_path, "text_embedding.pt"),
    )
    path_txt.embedding = path_txt.embedding.to(device)

    # Top-1 wrapper (understanding direction). See fudoki.eval_loop.CFGScaledModel.
    cfg_model = CFGScaledModel(model=model, g_or_u="understanding")

    # ---- Dataset ----
    ds = ScienceQADataset(
        args.dataset_dir, split=args.split, require_image=True,
        max_prompt_chars=args.max_prompt_chars,
    )
    n_total = len(ds)
    n_eval = n_total if args.num_samples <= 0 else min(args.num_samples, n_total)

    # Shard across ranks: each rank owns indices [rank::world_size] within [0, n_eval).
    local_indices = list(range(rank, n_eval, world_size))
    mprint(f"[eval] ScienceQA split={args.split}  n_total={n_total}  "
           f"n_eval={n_eval}  per_rank~{len(local_indices)}")

    # ---- API judge (only on rank 0 to avoid hammering the API) ----
    api_judge = None
    if is_main and not args.no_api_judge:
        api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("[eval] WARNING: no API key provided — LLM judge disabled.")
        else:
            api_judge = APIJudge(
                model=args.api_model,
                base_url=args.api_base_url,
                api_key=api_key,
                max_retries=args.api_max_retries,
                max_workers=args.api_max_workers,
                verbose=True,
            )
            print(f"[eval] API judge enabled: model={args.api_model}")
    # NOTE: local reward_fn below is used for the local letter extraction (no
    # API call); the authoritative judge pass happens on rank 0 after gather.
    reward_fn_local = build_mcq_reward_fn(api_judge=None)
    reward_fn_judge = build_mcq_reward_fn(api_judge=api_judge)

    # ---- Output dir ----
    out_dir = args.output_dir
    if out_dir is None:
        tag = "base" if not args.lora_path else os.path.basename(
            os.path.dirname(os.path.normpath(args.lora_path)))
        out_dir = os.path.join(_HERE, "eval_scienceqa_results",
                               f"{tag}_steps{args.steps}_{args.split}")
    if is_main:
        os.makedirs(out_dir, exist_ok=True)
        preds_path = os.path.join(out_dir, "predictions.jsonl")
        summary_path = os.path.join(out_dir, "summary.json")
        print(f"[eval] writing predictions -> {preds_path}")

    step_size = 1.0 / args.steps
    t0 = time.time()

    # Each rank generates text for its shard, no API call yet.
    local_records = []  # list of dicts (qid, gold, raw, it)
    for local_i, i in enumerate(local_indices):
        it = ds[i]

        x_init, data_info = build_data_info_understanding(
            [it], vl_chat_processor, path_txt,
            VOCABULARY_SIZE_TXT, args.txt_max_length, device,
        )
        final_tokens, text_mask = sample_text_only(
            model=cfg_model, path_txt=path_txt,
            vocabulary_size_txt=VOCABULARY_SIZE_TXT,
            x_init=x_init, data_info=data_info,
            step_size=step_size, cfg_scale=0.0, n_mc=1, device=device,
        )
        text_tok = final_tokens[text_mask == 1].reshape(1, -1).cpu()
        ans = vl_chat_processor.tokenizer.batch_decode(
            text_tok, skip_special_tokens=True)[0]
        for _eos in ("<｜end▁of▁sentence｜>", "<|end_of_sentence|>"):
            k = ans.find(_eos)
            if k != -1:
                ans = ans[:k]
                break

        local_records.append({
            "dataset_index": i,
            "qid": it["qid"],
            "gold": it["answer_letter"],
            "raw": ans,
            # Keep the raw dataset item so rank 0 can re-run reward_fn for judging.
            "_it": it,
        })

        if is_main and ((local_i + 1) % 20 == 0 or local_i == len(local_indices) - 1):
            elapsed = time.time() - t0
            rate = (local_i + 1) / max(elapsed, 1e-6)
            eta = (len(local_indices) - local_i - 1) / max(rate, 1e-6)
            print(f"[rank0 {local_i+1}/{len(local_indices)}]  "
                  f"rate={rate:.2f} it/s  eta={eta/60:.1f} min")

        del final_tokens, x_init, data_info, text_tok

    # ---- Gather records from all ranks to rank 0 for judging + writing ----
    if accelerator is not None and world_size > 1:
        accelerator.wait_for_everyone()
        # gather_object handles arbitrary python objects.
        from accelerate.utils import gather_object
        all_records_nested = gather_object(local_records)
        # gather_object returns the concatenation already.
        all_records = list(all_records_nested)
    else:
        all_records = local_records

    if not is_main:
        # Non-main ranks are done.
        if accelerator is not None:
            accelerator.wait_for_everyone()
        return

    # Deduplicate + sort by dataset_index (gather_object should already
    # concatenate in rank order, but be safe).
    seen = set()
    ordered = []
    for rec in sorted(all_records, key=lambda r: r["dataset_index"]):
        if rec["dataset_index"] in seen:
            continue
        seen.add(rec["dataset_index"])
        ordered.append(rec)

    print(f"[eval] gathered {len(ordered)} predictions, running judge/extraction on rank 0")

    n_correct = 0
    with open(preds_path, "w") as fp:
        for rec in ordered:
            it = rec["_it"]
            ans = rec["raw"]
            scores, details = reward_fn_judge([ans], [it])
            correct = int(scores[0].item() > 0.5)
            n_correct += correct
            fp.write(json.dumps({
                "qid": rec["qid"],
                "gold": rec["gold"],
                "pred_letter": details["pred_letters"][0],
                "judge_used": bool(details["judge_used"][0]),
                "correct": bool(correct),
                "raw": ans,
            }, ensure_ascii=False) + "\n")

    n_eval_actual = len(ordered)
    acc = n_correct / max(n_eval_actual, 1)
    summary = {
        "checkpoint_path": args.checkpoint_path,
        "lora_path": args.lora_path,
        "ema_path": args.ema_path,
        "gen_modules_path": args.gen_modules_path,
        "split": args.split,
        "steps": args.steps,
        "n_eval": n_eval_actual,
        "n_correct": n_correct,
        "accuracy": acc,
        "world_size": world_size,
        "decoding": "top-1 (greedy)",
        "api_judge": None if api_judge is None else {
            "model": args.api_model,
            "base_url": args.api_base_url,
            "n_calls": getattr(api_judge, "n_calls", 0),
            "n_success": getattr(api_judge, "n_success", 0),
            "n_fail": getattr(api_judge, "n_fail", 0),
        },
    }
    with open(summary_path, "w") as fp:
        json.dump(summary, fp, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"ScienceQA {args.split}  accuracy = {acc:.4f}  ({n_correct}/{n_eval_actual})")
    print(f"Summary: {summary_path}")
    print("=" * 60)

    if accelerator is not None:
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
