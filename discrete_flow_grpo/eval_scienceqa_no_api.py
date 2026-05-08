"""
Re-score ScienceQA predictions WITHOUT calling any external API judge.

For every ``predictions.jsonl`` under
``eval_scienceqa_scienceqa_kl0.01_results/<subdir>/``:
  * Re-extract the answer letter from the raw decoded text using only local
    heuristics (VLMEvalKit-style ``can_infer_option`` + a few regex
    fall-backs).  If nothing matches -> score 0.
  * Compare against the gold letter and dump a fresh
    ``summary_no_api.json`` (and ``predictions_no_api.jsonl``) next to the
    originals, so the original API-judged numbers are not overwritten.

Usage:
    python eval_scienceqa_no_api.py
    python eval_scienceqa_no_api.py --root <path> --letters ABCDE
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parent / "eval_scienceqa_scienceqa_kl0.01_results"
DEFAULT_LETTERS = "ABCDE"  # ScienceQA has at most 5 options


def can_infer_option_local(answer: str, choices_letters: str) -> str | None:
    """Isolated-letter token match (mirrors VLMEvalKit can_infer_option)."""
    if not isinstance(answer, str):
        return None
    mod = answer
    for c in ".()[],:;!*#{}\"'":
        mod = mod.replace(c, " ")
    splits = [t.strip() for t in mod.split()]
    hits = [t for t in splits if t in choices_letters]
    return hits[0] if len(hits) == 1 else None


def extract_letter(answer: str, choices_letters: str) -> str | None:
    """Local-only extractor — no API."""
    if not isinstance(answer, str) or not answer:
        return None

    # 1) isolated-token match (most robust for "B", " B ", "(B)")
    r = can_infer_option_local(answer, choices_letters)
    if r is not None:
        return r

    # 2) keyword-anchored regexes
    patterns = [
        r"(?:^|\W)(?:answer|Answer|ANSWER)\s*[:：]?\s*\*{0,2}\s*([A-Z])\b",
        r"(?:^|\W)(?:option|Option)\s*([A-Z])\b",
        r"\(([A-Z])\)",
        r"\b([A-Z])\s*[.):]",
        r"^\s*([A-Z])\b",
    ]
    for p in patterns:
        m = re.search(p, answer)
        if m and m.group(1) in choices_letters:
            return m.group(1)

    # 3) any standalone uppercase letter that is in the choice set,
    #    only if there is exactly one such occurrence.
    toks = [t for t in re.findall(r"\b([A-Z])\b", answer) if t in choices_letters]
    if len(set(toks)) == 1:
        return toks[0]
    return None


def rescore_subdir(sub: Path, choices_letters: str) -> dict | None:
    pred_path = sub / "predictions.jsonl"
    if not pred_path.is_file():
        return None

    n_eval = 0
    n_correct = 0
    n_extract_fail = 0
    rows = []
    with open(pred_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            gold = obj.get("gold")
            raw = obj.get("raw") or ""
            pred = extract_letter(raw, choices_letters)
            correct = bool(pred is not None and pred == gold)
            n_eval += 1
            if pred is None:
                n_extract_fail += 1
            if correct:
                n_correct += 1
            rows.append({
                "qid": obj.get("qid"),
                "gold": gold,
                "pred_letter": pred,        # None if extraction failed
                "correct": correct,
                "raw": raw,
            })

    summary = {
        "subdir": sub.name,
        "n_eval": n_eval,
        "n_correct": n_correct,
        "n_extract_fail": n_extract_fail,
        "accuracy": (n_correct / n_eval) if n_eval else 0.0,
        "letters_universe": choices_letters,
        "method": "local_only_no_api",
    }

    # Dump artefacts next to the originals.
    out_pred = sub / "predictions_no_api.jsonl"
    with open(out_pred, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_sum = sub / "summary_no_api.json"
    with open(out_sum, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--letters", type=str, default=DEFAULT_LETTERS,
                    help="Universe of valid option letters (default: ABCDE).")
    args = ap.parse_args()

    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"[err] root dir not found: {root}")

    summaries = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        s = rescore_subdir(sub, args.letters)
        if s is None:
            continue
        summaries.append(s)
        print(f"{sub.name:35s} acc={s['accuracy']*100:6.2f}%  "
              f"({s['n_correct']}/{s['n_eval']}, "
              f"extract_fail={s['n_extract_fail']})")

    # Aggregated table.
    out = root / "summary_no_api_all.json"
    with open(out, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nSaved aggregate -> {out}")


if __name__ == "__main__":
    main()
