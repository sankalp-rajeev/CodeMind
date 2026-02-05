"""
Intent Classification Evaluation

Compares baseline (qwen2.5:7b) vs improved (orchestrator-ft) on 50 queries.
Metric: Accuracy (correct intent / total).
"""

import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BENCHMARKS_DIR, BASELINE_ORCHESTRATOR, IMPROVED_ORCHESTRATOR


def load_benchmark():
    with open(BENCHMARKS_DIR / "intent_classification.json") as f:
        return json.load(f)


def run_intent_eval(model: str, verbose: bool = False) -> tuple:
    """Run intent classification on benchmark. Returns (accuracy, per_intent_stats)."""
    from src.agents.orchestrator import OrchestratorAgent, QueryIntent

    bench = load_benchmark()
    orch = OrchestratorAgent(model=model)
    correct = 0
    total = len(bench["tasks"])
    per_intent = {}  # expected -> {correct, total, wrong_preds: [(query, pred), ...]}

    for task in bench["tasks"]:
        query = task["query"]
        expected = task["intent"]
        per_intent.setdefault(expected, {"correct": 0, "total": 0, "wrong": []})
        per_intent[expected]["total"] += 1
        try:
            intent = orch.classify_intent(query)
            pred = intent.name
            if pred == expected:
                correct += 1
                per_intent[expected]["correct"] += 1
            else:
                per_intent[expected]["wrong"].append((query[:50], pred))
        except Exception as e:
            per_intent[expected]["wrong"].append((query[:50], f"ERR:{e}"))

    acc = correct / total if total > 0 else 0.0
    return acc, per_intent


def print_per_intent(per_intent: dict, label: str):
    """Print per-intent accuracy (lowest first)."""
    rows = []
    for intent, d in per_intent.items():
        total = d["total"]
        correct = d["correct"]
        acc = correct / total if total else 0
        rows.append((intent, correct, total, acc))
    rows.sort(key=lambda r: r[3])  # lowest accuracy first
    print(f"\n  Per-intent ({label}) — lowest first:")
    for intent, c, t, acc in rows:
        wrong = per_intent[intent]["wrong"]
        samples = "; ".join(f"{q[:35]}→{p}" for q, p in wrong[:2]) if wrong else ""
        print(f"    {intent}: {c}/{t} = {acc:.0%}  {samples}")


def main():
    print("Intent Classification Evaluation")
    print("=" * 50)

    # Baseline
    print(f"\nBaseline ({BASELINE_ORCHESTRATOR})...")
    try:
        baseline_acc, baseline_per = run_intent_eval(BASELINE_ORCHESTRATOR)
    except Exception as e:
        print(f"  Baseline failed: {e}")
        baseline_acc, baseline_per = 0.0, {}
    print(f"  Accuracy: {baseline_acc:.2%}")
    print_per_intent(baseline_per, BASELINE_ORCHESTRATOR)

    # Improved (orchestrator-ft; fallback to baseline if not available)
    print(f"\nImproved ({IMPROVED_ORCHESTRATOR})...")
    try:
        improved_acc, improved_per = run_intent_eval(IMPROVED_ORCHESTRATOR)
    except Exception as e:
        print(f"  Improved failed (fallback to baseline): {e}")
        improved_acc, improved_per = baseline_acc, baseline_per
    print(f"  Accuracy: {improved_acc:.2%}")
    print_per_intent(improved_per, IMPROVED_ORCHESTRATOR)

    delta = improved_acc - baseline_acc
    print(f"\nDelta: {delta:+.2%}")

    return {
        "baseline": round(baseline_acc, 4),
        "improved": round(improved_acc, 4),
        "delta": round(delta, 4),
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
