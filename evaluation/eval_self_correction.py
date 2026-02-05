"""
Self-Correction Evaluation (TestingCrew)

CodeMind evaluating itself: actual CodeMind code snippets.
Compares Pass@1 (single attempt) vs Pass@3 (up to 3 iterations).
Metric: Test pass rate — does pytest pass on generated tests?
"""

import json
import sys
import tempfile
import subprocess
import re
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BENCHMARKS_DIR, PROJECT_ROOT


def load_benchmark():
    with open(BENCHMARKS_DIR / "self_correction.json") as f:
        return json.load(f)


def extract_python_code(text: str) -> str:
    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, text, re.DOTALL)
        valid = [m.strip() for m in matches if m.strip() and ("def " in m or "import " in m)]
        if valid:
            return max(valid, key=len)
    if "def test_" in text or "import pytest" in text:
        return text.strip()
    return ""


def run_pytest(test_code: str) -> bool:
    """Run pytest on test code. Returns True if all pass."""
    if not test_code or len(test_code.strip()) < 50:
        return False
    with tempfile.NamedTemporaryFile(mode="w", suffix="_test.py", delete=False) as f:
        f.write(test_code)
        path = f.name
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-v", "--tb=short", "-q"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.returncode == 0
    except Exception:
        return False
    finally:
        Path(path).unlink(missing_ok=True)


def run_self_correction_eval(max_iterations: int) -> float:
    """Run TestingCrew with given max_iterations. Returns pass rate."""
    from src.crews.testing_crew import TestingCrew

    bench = load_benchmark()
    crew = TestingCrew(max_iterations=max_iterations)
    passed = 0
    total = len(bench["tasks"])

    for i, task in enumerate(bench["tasks"]):
        code = task["code"]
        file_path = task.get("file", "src/utils/__init__.py")
        # Resolve to absolute path if relative
        if not Path(file_path).is_absolute():
            file_path = str(PROJECT_ROOT / file_path)
        try:
            for progress in crew.run(code, file_path):
                if progress.get("type") == "crew_complete":
                    tests = progress.get("final_tests", "")
                    parsed = extract_python_code(tests)
                    if run_pytest(parsed):
                        passed += 1
                    break
        except Exception as e:
            print(f"  [WARN] Task {i+1} ({task.get('id','?')}): {e}")

    return passed / total if total > 0 else 0.0


def main():
    print("Self-Correction Evaluation (TestingCrew)")
    print("=" * 50)

    # Pass@1: single attempt (baseline)
    print("\nPass@1 (max_iterations=1)...")
    try:
        baseline_rate = run_self_correction_eval(max_iterations=1)
    except Exception as e:
        print(f"  Pass@1 failed: {e}")
        baseline_rate = 0.0
    print(f"  Pass rate: {baseline_rate:.2%}")

    # Pass@3: up to 3 iterations (improved)
    print("\nPass@3 (max_iterations=3)...")
    try:
        improved_rate = run_self_correction_eval(max_iterations=3)
    except Exception as e:
        print(f"  Pass@3 failed: {e}")
        improved_rate = baseline_rate
    print(f"  Pass rate: {improved_rate:.2%}")

    delta = improved_rate - baseline_rate
    print(f"\nDelta: {delta:+.2%}")

    return {
        "pass_at_1": round(baseline_rate, 4),
        "pass_at_3": round(improved_rate, 4),
        "baseline": round(baseline_rate, 4),
        "improved": round(improved_rate, 4),
        "delta": round(delta, 4),
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
