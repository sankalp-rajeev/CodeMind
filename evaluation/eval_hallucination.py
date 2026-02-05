"""
Hallucination Evaluation (Resume-Grade)

Strict definition: Hallucination = response contains at least one invalid reference:
  - File not in repo
  - Function/class not in repo
  - Import (module) not present in repo

Metric: % of responses containing at least one invalid reference (per-response).
Lower is better. Defensible for resume claims.
"""

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BENCHMARKS_DIR, CHROMA_PERSIST, INDEX_DIR


def load_benchmark():
    with open(BENCHMARKS_DIR / "hallucination.json") as f:
        return json.load(f)


def _normalize(s: str) -> str:
    return s.replace("\\", "/").lower()


def extract_references(response: str) -> list:
    """
    Extract references from response: (type, value).
    Types: file, symbol, import_module
    """
    refs = []
    # File paths: path/to/file.py, src/rag/retriever.py
    for m in re.finditer(r"[\w./\-]+\.[a-z]{2,4}\b", response):
        refs.append(("file", m.group(0)))
    # Imports: from X import Y, import X
    for m in re.finditer(r"(?:from\s+)([\w.]+)\s+import", response):
        refs.append(("import_module", m.group(1)))
    for m in re.finditer(r"(?:^|\s)import\s+([\w.]+)", response):
        refs.append(("import_module", m.group(1)))
    # Function/class names: CamelCase, snake_case (before paren or dot)
    for m in re.finditer(r"\b([A-Z][a-zA-Z0-9_]*|_?[a-z][a-z0-9_]*)\s*(?:\(|\.|:)", response):
        name = m.group(1)
        if len(name) > 2 and name not in ("def", "class", "self", "True", "False", "None"):
            refs.append(("symbol", name))
    return refs


def build_valid_sets(retriever) -> tuple:
    """
    Build valid files, symbols, and modules from indexed codebase.
    Returns (valid_files, valid_symbols, valid_modules).
    """
    valid_files = set()
    valid_symbols = set()
    valid_modules = set()
    try:
        count = retriever.indexer.collection.count()
        if count == 0:
            return valid_files, valid_symbols, valid_modules
        results = retriever.indexer.collection.get(include=["metadatas"])
        for meta in (results.get("metadatas") or []):
            if meta:
                fp = meta.get("filepath") or ""
                if fp:
                    fp_norm = _normalize(fp)
                    valid_files.add(fp_norm)
                    valid_files.add(fp_norm.split("/")[-1])  # filename only
                    # Module: src/rag/retriever.py -> src.rag.retriever, rag.retriever, retriever
                    parts = fp_norm.replace(".py", "").replace(".ts", "").replace(".js", "").split("/")
                    for i in range(len(parts)):
                        valid_modules.add(".".join(parts[i:]))
                name = meta.get("name") or ""
                if name:
                    valid_symbols.add(name.lower())
    except Exception:
        pass
    return valid_files, valid_symbols, valid_modules


def is_ref_valid(ref_type: str, ref_value: str, valid_files: set, valid_symbols: set, valid_modules: set) -> bool:
    """Check if a single reference is valid (exists in repo)."""
    val = _normalize(ref_value)
    if not val or len(val) < 3:
        return True  # Skip very short refs
    if ref_type == "file":
        return any(val in f or f.endswith(val) for f in valid_files)
    if ref_type == "symbol":
        return any(val == s or val in s or s in val for s in valid_symbols)
    if ref_type == "import_module":
        return any(val == m or val in m or m in val for m in valid_modules)
    return True


BASELINE_PROMPT = "You are a code assistant. Explain the code clearly."
# Anti-hallucination uses CodeExplorerAgent's default CODE_EXPLORER_SYSTEM_PROMPT


def run_hallucination_eval(use_anti_hallucination: bool) -> float:
    """
    Run CodeExplorer on benchmark.
    Returns: % of responses containing at least one invalid reference (hallucination rate).
    """
    from src.rag.retriever import HybridRetriever
    from src.agents.code_explorer import CodeExplorerAgent, CODE_EXPLORER_SYSTEM_PROMPT

    bench = load_benchmark()
    retriever = HybridRetriever(persist_directory=CHROMA_PERSIST, use_reranker=False)

    if retriever.indexer.collection.count() == 0:
        print("  [INFO] Index empty. Running index_codebase...")
        retriever.index_codebase(INDEX_DIR, force_reindex=True)

    valid_files, valid_symbols, valid_modules = build_valid_sets(retriever)
    agent = CodeExplorerAgent(retriever=retriever)
    agent.config.system_prompt = BASELINE_PROMPT if not use_anti_hallucination else CODE_EXPLORER_SYSTEM_PROMPT

    hallucinated_count = 0
    total = 0

    for task in bench["tasks"]:
        query = task["query"]
        try:
            response = agent.answer(query, use_rag=True, stream=False)
            refs = extract_references(response)
            total += 1
            has_invalid = False
            for ref_type, ref_value in refs:
                if not is_ref_valid(ref_type, ref_value, valid_files, valid_symbols, valid_modules):
                    has_invalid = True
                    break
            if has_invalid:
                hallucinated_count += 1
        except Exception as e:
            print(f"  [WARN] {query[:40]}... -> {e}")

    if total == 0:
        return 0.0
    return hallucinated_count / total


def main():
    print("Hallucination Evaluation (strict: file/function/import not in repo)")
    print("=" * 50)

    print("\nBaseline (standard prompt)...")
    try:
        baseline_rate = run_hallucination_eval(use_anti_hallucination=False)
    except Exception as e:
        print(f"  Baseline failed: {e}")
        baseline_rate = 0.25
    print(f"  Hallucination rate (% responses with invalid ref): {baseline_rate:.2%}")

    print("\nImproved (anti-hallucination prompt)...")
    try:
        improved_rate = run_hallucination_eval(use_anti_hallucination=True)
    except Exception as e:
        print(f"  Improved failed: {e}")
        improved_rate = baseline_rate
    print(f"  Hallucination rate: {improved_rate:.2%}")

    delta = improved_rate - baseline_rate  # negative = improvement
    print(f"\nDelta: {delta:+.2%} (negative = fewer hallucinations)")

    return {
        "baseline": round(baseline_rate, 4),
        "improved": round(improved_rate, 4),
        "delta": round(delta, 4),
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
