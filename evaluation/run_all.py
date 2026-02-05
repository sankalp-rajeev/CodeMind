"""
Run evaluations one at a time or all together.

CodeMind evaluating itself: benchmarks use actual CodeMind code and files.
Resume-grade metrics: strict definitions, binary relevance, per-response hallucination.

Usage:
  # Run one eval at a time (recommended for low-memory machines):
  python run_all.py --intent
  python run_all.py --retrieval
  python run_all.py --self-correction
  python run_all.py --hallucination

  # Run multiple:
  python run_all.py --intent --retrieval

  # Run all (original behavior):
  python run_all.py --all
  python run_all.py

Output:
  evaluation_report.json
  evaluation_report.md
  (Report merges with existing; run evals separately and results accumulate)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import REPORT_PATH, REPORT_MD_PATH, INDEX_DIR


EMPTY_RESULT = {"baseline": 0, "improved": 0, "delta": 0}


def load_existing_report() -> dict:
    """Load existing report if present, else return empty template."""
    if REPORT_PATH.exists():
        try:
            with open(REPORT_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {
        "timestamp": datetime.now().isoformat(),
        "intent_classification": None,
        "retrieval_recall_at_5": None,
        "self_correction_pass_rate": None,
        "hallucination_invalid_rate": None,
    }


def run_intent(report: dict) -> None:
    print("\n" + "=" * 60)
    print("1. Intent Classification")
    print("=" * 60)
    try:
        from eval_intent import main as eval_intent
        report["intent_classification"] = eval_intent()
    except Exception as e:
        print(f"Intent eval failed: {e}")
        report["intent_classification"] = EMPTY_RESULT.copy()


def run_retrieval(report: dict) -> None:
    print("\n" + "=" * 60)
    print("2. Hybrid Retrieval (Recall@5)")
    print("=" * 60)
    try:
        from eval_retrieval import main as eval_retrieval
        report["retrieval_recall_at_5"] = eval_retrieval()
    except Exception as e:
        print(f"Retrieval eval failed: {e}")
        report["retrieval_recall_at_5"] = EMPTY_RESULT.copy()


def run_self_correction(report: dict) -> None:
    print("\n" + "=" * 60)
    print("3. Self-Correction (Test Pass Rate)")
    print("=" * 60)
    try:
        from eval_self_correction import main as eval_self_correction
        report["self_correction_pass_rate"] = eval_self_correction()
    except Exception as e:
        print(f"Self-correction eval failed: {e}")
        report["self_correction_pass_rate"] = EMPTY_RESULT.copy()


def run_hallucination(report: dict) -> None:
    print("\n" + "=" * 60)
    print("4. Hallucination")
    print("=" * 60)
    try:
        from eval_hallucination import main as eval_hallucination
        report["hallucination_invalid_rate"] = eval_hallucination()
    except Exception as e:
        print(f"Hallucination eval failed: {e}")
        report["hallucination_invalid_rate"] = EMPTY_RESULT.copy()


def _fmt(r, key):
    """Format value for report; handle None."""
    v = r.get(key) if r else None
    if v is None:
        return "N/A"
    return f"{v:.2%}"


def _fmt_delta(r, key="delta"):
    v = r.get(key) if r else None
    if v is None:
        return "N/A"
    return f"{v:+.2%}"


def write_report(report: dict):
    """Write report; handle missing sections (N/A)."""
    ic = report.get("intent_classification") or {}
    rr = report.get("retrieval_recall_at_5") or {}
    sc = report.get("self_correction_pass_rate") or {}
    hl = report.get("hallucination_invalid_rate") or {}

    # JSON
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Markdown
    md = f"""# CodeMind Evaluation Report

**Generated:** {report.get('timestamp', 'N/A')}

**Scope:** CodeMind evaluating itself (index: {INDEX_DIR})

---

## 1. Intent Classification

| Metric | Baseline | Improved | Delta |
|--------|----------|----------|-------|
| Accuracy | {_fmt(ic, 'baseline')} | {_fmt(ic, 'improved')} | {_fmt_delta(ic)} |

**Resume line:** Improved intent routing accuracy from {_fmt(ic, 'baseline')} → {_fmt(ic, 'improved')} via QLoRA fine-tuning of Qwen2.5 for multi-agent orchestration.

---

## 2. Hybrid Retrieval (Recall@5)

Binary relevance: does any relevant file appear in top 5?

| Metric | Baseline (vector-only) | Improved (RRF) | Delta |
|--------|------------------------|----------------|-------|
| Recall@5 | {_fmt(rr, 'baseline')} | {_fmt(rr, 'improved')} | {_fmt_delta(rr)} |

**Resume line:** Hybrid RRF retrieval improved Recall@5 by {f"{rr.get('delta', 0)*100:.0f}" if rr.get('delta') is not None else "N/A"}% over vector-only baseline (file-level binary relevance).

---

## 3. Self-Correction (Test Pass Rate)

Pass@1 = single attempt, Pass@3 = up to 3 iterations. CodeMind code snippets.

| Metric | Pass@1 | Pass@3 | Delta |
|--------|--------|--------|-------|
| Pass Rate | {_fmt(sc, 'baseline')} | {_fmt(sc, 'improved')} | {_fmt_delta(sc)} |

**Resume line:** Iterative self-correction improved test pass rate from {_fmt(sc, 'baseline')} → {_fmt(sc, 'improved')} (Pass@1 → Pass@3).

---

## 4. Hallucination (Strict Definition)

% of responses containing at least one invalid reference (file/function/import not in repo).

| Metric | Baseline | Improved (anti-hallucination) | Delta |
|--------|----------|-------------------------------|-------|
| Hallucination Rate | {_fmt(hl, 'baseline')} | {_fmt(hl, 'improved')} | {_fmt_delta(hl)} |

**Resume line:** Anti-hallucination verification reduced hallucination (invalid file/function/import refs) from {_fmt(hl, 'baseline')} → {_fmt(hl, 'improved')}.

---

## Summary

| Metric | Baseline | Improved | Delta |
|--------|----------|----------|-------|
| Intent Accuracy | {_fmt(ic, 'baseline')} | {_fmt(ic, 'improved')} | {_fmt_delta(ic)} |
| Recall@5 | {_fmt(rr, 'baseline')} | {_fmt(rr, 'improved')} | {_fmt_delta(rr)} |
| Test Pass Rate | {_fmt(sc, 'baseline')} | {_fmt(sc, 'improved')} | {_fmt_delta(sc)} |
| Hallucination | {_fmt(hl, 'baseline')} | {_fmt(hl, 'improved')} | {_fmt_delta(hl)} |
"""
    with open(REPORT_MD_PATH, "w", encoding="utf-8") as f:
        f.write(md)


def main():
    parser = argparse.ArgumentParser(
        description="Run CodeMind evaluations (one at a time or all).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_all.py --intent              # Run intent only
  python run_all.py --retrieval           # Run retrieval only
  python run_all.py --self-correction     # Run self-correction only
  python run_all.py --hallucination       # Run hallucination only
  python run_all.py --intent --retrieval  # Run two evals
  python run_all.py --all                 # Run all four
        """,
    )
    parser.add_argument("--intent", action="store_true", help="Run intent classification")
    parser.add_argument("--retrieval", action="store_true", help="Run hybrid retrieval")
    parser.add_argument("--self-correction", action="store_true", help="Run self-correction (TestingCrew)")
    parser.add_argument("--hallucination", action="store_true", help="Run hallucination eval")
    parser.add_argument("--all", action="store_true", help="Run all four evals (original behavior)")
    args = parser.parse_args()

    # If no flags, default to --all
    run_any = args.intent or args.retrieval or args.self_correction or args.hallucination
    if not run_any or args.all:
        args.intent = args.retrieval = args.self_correction = args.hallucination = True

    report = load_existing_report()
    report["timestamp"] = datetime.now().isoformat()

    if args.intent:
        run_intent(report)
    if args.retrieval:
        run_retrieval(report)
    if args.self_correction:
        run_self_correction(report)
    if args.hallucination:
        run_hallucination(report)

    write_report(report)
    print("\n" + "=" * 60)
    print(f"Report saved to {REPORT_PATH} and {REPORT_MD_PATH}")
    return report


if __name__ == "__main__":
    main()
