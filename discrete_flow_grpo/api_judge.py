"""
OpenAI-compatible API judge for MCQ answer extraction.

Mirrors VLMEvalKit's ``extract_answer_from_item`` fallback: when local
heuristics (``can_infer_option`` / ``can_infer_text``) cannot extract a letter
from a free-form answer, we ask an external LLM to map the answer to one of
the listed options (A/B/C/... or Z for "none of the above").

Design:
  * Synchronous OpenAI-style ``chat.completions`` API (works with OpenAI,
    DeepSeek, Qwen-DashScope OpenAI-compatible endpoints, local vLLM server,
    Ollama-OpenAI, etc.).
  * Batch interface: ``judge_many(requests)`` uses a ThreadPoolExecutor so
    the reward function does not block reward computation per-sample.
  * Up to N retries per request; on total failure returns ``None`` (caller
    then scores 0).
  * Prompt template copied verbatim from VLMEvalKit ``build_prompt`` (EN)
    and ``build_prompt_cn`` (ZH) — selected by detecting Chinese characters
    in the question.
"""

from __future__ import annotations

import os
import re
import time
import string
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Prompt templates (copied from VLMEvalKit/vlmeval/dataset/utils/multiple_choice.py)
# ---------------------------------------------------------------------------
_PROMPT_EN = (
    "You are an AI assistant who will help me to match an answer with several options of a single-choice question. "
    "You are provided with a question, several options, and an answer, "
    "and you need to find which option is most similar to the answer. "
    "If the meaning of all options are significantly different from the answer, output Z. "
    "Your should output a single uppercase character in A, B, C, D "
    "(if they are valid options), and Z. \n"
    "Example 1: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: a cute teddy bear\nYour output: A\n"
    "Example 2: \n"
    "Question: What is the main object in image?\nOptions: A. teddy bear B. rabbit C. cat D. dog\n"
    "Answer: Spider\nYour output: Z\n"
    "Example 3: \n"
    "Question: {}?\nOptions: {}\nAnswer: {}\nYour output: "
)

_PROMPT_CN = (
    "你是一个帮助我匹配答案与单选题中多个选项的 AI 助手。"
    "你会被提供：一个问题，多个选项，一个答案。你的任务是找到与答案意义最相近的选项。"
    "如果所有选项的意义都与答案显著不同，则输出 Z。"
    "你应该输出一个单个的大写字母，例如 A, B, C, D（如果它们是有效选项），或 Z。"
    "例 1:"
    "问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 一只可爱的泰迪熊\n输出: A\n"
    "例 2: \n"
    "问题: 图中最主要的物体是什么?\n选项: A. 泰迪熊 B. 兔子 C. 猫 D. 狗\n答案: 蜘蛛\n输出: Z\n"
    "例 3: \n"
    "问题: {}?\n选项: {}\n答案: {}\n输出: "
)


def _cn_string(s: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", s or ""))


def build_judge_prompt(question: str, choices_letters: Sequence[str],
                       choices: Sequence[str], prediction: str) -> str:
    opt_str = " ".join(f"{ch}. {txt}" for ch, txt in zip(choices_letters, choices))
    tmpl = _PROMPT_CN if _cn_string(question) else _PROMPT_EN
    return tmpl.format(question, opt_str, prediction)


# ---------------------------------------------------------------------------
# OpenAI-compatible client (lazy import).
# ---------------------------------------------------------------------------

class APIJudge:
    """Calls an OpenAI-compatible chat endpoint to extract option letters.

    Parameters
    ----------
    model : str
        Model name, e.g. ``gpt-4o-mini`` or ``deepseek-chat``.
    base_url : str | None
        Override endpoint (e.g. ``https://api.deepseek.com/v1``). If ``None``
        uses the OpenAI default.
    api_key : str
        API key (mandatory).
    max_retries : int
        Retries per request (default 3, same as VLMEvalKit).
    max_workers : int
        Thread pool size for batch calls.
    temperature : float
        Sampling temperature. 0.0 recommended for deterministic extraction.
    timeout : float
        Per-request timeout (seconds).
    verbose : bool
        Whether to print a message when a request totally fails.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        max_retries: int = 3,
        max_workers: int = 8,
        temperature: float = 0.0,
        timeout: float = 30.0,
        verbose: bool = False,
    ):
        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "API judge requires the 'openai' package. Install with: "
                "pip install openai"
            ) from e

        if api_key is None:
            api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "API judge requires an API key. Pass --api_judge_key_env to "
                "point to an env var, or set OPENAI_API_KEY."
            )

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model
        self.max_retries = int(max_retries)
        self.max_workers = int(max_workers)
        self.temperature = float(temperature)
        self.timeout = float(timeout)
        self.verbose = bool(verbose)

        # Running counters useful for wandb.
        self.n_calls = 0
        self.n_success = 0
        self.n_fail = 0

    # ---- single call ----
    def _one(self, prompt: str, choices_letters: Sequence[str]) -> Optional[str]:
        from reward_understanding import _get_can_infer  # flat import
        can_infer = _get_can_infer()
        choices_dict = {ch: ch for ch in choices_letters}

        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}],
                    timeout=self.timeout,
                )
                txt = (resp.choices[0].message.content or "").strip()
            except Exception as e:
                if self.verbose:
                    print(f"[api_judge] call failed (attempt {attempt+1}): {e}")
                time.sleep(min(2 ** attempt, 8))
                continue
            ret = can_infer(txt, choices_dict)
            if ret and (ret in choices_letters or ret == "Z"):
                return None if ret == "Z" else ret
            # If the model literally said just a letter we may still catch it:
            m = re.search(r"\b([A-Z])\b", txt)
            if m and m.group(1) in choices_letters:
                return m.group(1)
        return None

    # ---- batch call ----
    def judge_many(
        self,
        items: Sequence[Tuple[str, Sequence[str], Sequence[str], str]],
    ) -> List[Optional[str]]:
        """items: list of (question, choices_letters, choices, prediction).

        Returns a list of extracted letters (or ``None`` if extraction failed).
        """
        if not items:
            return []
        prompts = [
            build_judge_prompt(q, cl, c, p) for (q, cl, c, p) in items
        ]
        choice_letters_list = [it[1] for it in items]
        results: List[Optional[str]] = [None] * len(items)

        def _task(i):
            r = self._one(prompts[i], choice_letters_list[i])
            return i, r

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for i, r in pool.map(_task, range(len(items))):
                results[i] = r

        self.n_calls += len(items)
        self.n_success += sum(1 for r in results if r is not None)
        self.n_fail += sum(1 for r in results if r is None)
        return results
