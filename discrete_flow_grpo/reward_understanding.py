"""
Reward function for multimodal-understanding GRPO on ScienceQA.

Given a decoded answer string and the ground-truth choice letter, returns 1.0
if the extracted option letter matches the correct option, else 0.0.

Extraction pipeline (mirrors VLMEvalKit ``extract_answer_from_item``):

  1. Local heuristics — VLMEvalKit ``can_infer`` = ``can_infer_option`` OR
     ``can_infer_text``:
        * option: isolated letter token (A/B/C/...) in the answer.
        * text:   full choice text appearing in the answer.
     Covers e.g. "Answer: B", "I pick (B) because ...", and
     "the answer is teddy bear" when a choice text is 'teddy bear'.

  2. Regex fall-backs (local): "Answer: X", leading "X.", "(X)", etc.

  3. Optional **API judge** (OpenAI-compatible LLM) — when local extraction
     fails we batch-call an external model with the VLMEvalKit prompt
     template and ask it to map the prediction to A/B/C/D/Z. See
     ``api_judge.py``.

If all three fail, reward is 0.
"""

import os
import sys
import copy as cp
import re

import torch


# ---------------------------------------------------------------------------
# Lazy import of VLMEvalKit matchers.
# ---------------------------------------------------------------------------
def _add_vlmevalkit_to_path():
    here = os.path.dirname(os.path.abspath(__file__))
    cand = os.path.join(here, "dataset_understanding", "VLMEvalKit")
    if os.path.isdir(cand) and cand not in sys.path:
        sys.path.insert(0, cand)


_can_infer = None
_can_infer_option = None


def _get_can_infer():
    """Return VLMEvalKit's ``can_infer`` (option + text)."""
    global _can_infer
    if _can_infer is not None:
        return _can_infer
    try:
        _add_vlmevalkit_to_path()
        from vlmeval.utils.matching_util import can_infer  # type: ignore
        _can_infer = can_infer
    except Exception:
        # Fallback: minimal reimplementation of option + text matching.
        def _can_infer_option_local(answer, choices):
            if not isinstance(answer, str):
                return False
            mod = cp.copy(answer)
            for c in ".()[],:;!*#{}":
                mod = mod.replace(c, " ")
            splits = [x.strip() for x in mod.split()]
            hits = [c for c in choices if c in splits]
            return hits[0] if len(hits) == 1 else False

        def _can_infer_text_local(answer, choices):
            if not isinstance(answer, str) or not isinstance(choices, dict):
                return False
            low = answer.lower()
            cands = [k for k, v in choices.items() if str(v).lower() in low]
            return cands[0] if len(cands) == 1 else False

        def can_infer(answer, choices):
            r = _can_infer_option_local(str(answer), choices)
            return r if r else _can_infer_text_local(str(answer), choices)

        _can_infer = can_infer
    return _can_infer


def _get_can_infer_option():
    """Option-only matcher (kept for back-compat)."""
    global _can_infer_option
    if _can_infer_option is not None:
        return _can_infer_option
    try:
        _add_vlmevalkit_to_path()
        from vlmeval.utils.matching_util import can_infer_option  # type: ignore
        _can_infer_option = can_infer_option
    except Exception:
        def can_infer_option(answer, choices):
            if not isinstance(answer, str):
                return False
            mod = cp.copy(answer)
            for c in ".()[],:;!*#{}":
                mod = mod.replace(c, " ")
            splits = [x.strip() for x in mod.split()]
            hits = [c for c in choices if c in splits]
            return hits[0] if len(hits) == 1 else False
        _can_infer_option = can_infer_option
    return _can_infer_option


# ---------------------------------------------------------------------------
# Local letter extraction.
# ---------------------------------------------------------------------------
def extract_option_letter(answer_str, choices_letters, choices_texts=None):
    """Return a single letter in ``choices_letters`` or ``None``.

    Uses VLMEvalKit ``can_infer`` (option + text) then regex fall-backs.
    ``choices_texts`` (list[str]) enables the text-matching branch.
    """
    if not isinstance(answer_str, str):
        return None

    choices_dict = {ch: (choices_texts[i] if choices_texts else ch)
                    for i, ch in enumerate(choices_letters)}
    res = _get_can_infer()(answer_str, choices_dict)
    if res and res in choices_letters:
        return res

    # Regex patterns (cover cases can_infer misses: "Answer: B", "**B**", etc.)
    patterns = [
        r"(?:^|\W)(?:answer|Answer|ANSWER)\s*[:：]?\s*\*{0,2}\s*([A-Z])\b",
        r"(?:^|\W)(?:option|Option)\s*([A-Z])\b",
        r"\(([A-Z])\)",
        r"\b([A-Z])\s*[.):]",
        r"^\s*([A-Z])\b",
    ]
    for p in patterns:
        m = re.search(p, answer_str)
        if m and m.group(1) in choices_letters:
            return m.group(1)

    for tok in re.findall(r"\b([A-Z])\b", answer_str):
        if tok in choices_letters:
            return tok
    return None


# ---------------------------------------------------------------------------
# Reward function.
# ---------------------------------------------------------------------------
def build_mcq_reward_fn(api_judge=None):
    """Return a reward function with signature

        reward_fn(answers: list[str], items: list[dict]) -> (tensor[B], details)

    ``items`` must have ``choices_letters``, ``choices``, ``answer_letter``
    and (optional, for the API judge) ``prompt`` / ``question``.

    ``api_judge``: optional ``APIJudge`` instance. When provided, samples where
    local extraction fails are batched together and sent to the LLM judge to
    extract a letter.
    """
    @torch.no_grad()
    def reward_fn(answers, items):
        B = len(answers)
        scores = torch.zeros(B, dtype=torch.float32)
        pred_letters = [None] * B
        judge_used = [False] * B

        # ---- Pass 1: local extraction ----
        need_judge = []
        for i, (ans, it) in enumerate(zip(answers, items)):
            letters = it["choices_letters"]
            texts = it.get("choices", None)
            pred = extract_option_letter(ans, letters, texts)
            if pred is None and api_judge is not None:
                need_judge.append(i)
            else:
                pred_letters[i] = pred

        # ---- Pass 2: API judge (batched across failed samples) ----
        if need_judge and api_judge is not None:
            batch = []
            for i in need_judge:
                it = items[i]
                q = it.get("question") or it.get("prompt") or ""
                batch.append((q, it["choices_letters"], it["choices"], answers[i]))
            judged = api_judge.judge_many(batch)
            for i, pred in zip(need_judge, judged):
                pred_letters[i] = pred
                judge_used[i] = True

        # ---- Score ----
        for i, it in enumerate(items):
            if pred_letters[i] == it["answer_letter"]:
                scores[i] = 1.0

        details = {
            "pred_letters": [p if p is not None else "?" for p in pred_letters],
            "judge_used": judge_used,
            "n_judge_used": sum(judge_used),
        }
        return scores, details

    return reward_fn
