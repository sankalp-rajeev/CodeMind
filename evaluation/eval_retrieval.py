"""
Hybrid Retrieval Evaluation

Compares vector-only (baseline) vs hybrid RRF (improved) on 50 queries.
Metric: Recall@5 — binary relevance. Does any relevant file appear in top 5 results?

Resume-grade: Predefined relevant files per query (actual CodeMind files).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import BENCHMARKS_DIR, CHROMA_PERSIST, INDEX_DIR


def load_benchmark():
    with open(BENCHMARKS_DIR / "retrieval.json") as f:
        return json.load(f)


def _normalize_path(p: str) -> str:
    """Normalize path for comparison (forward slash, lowercase)."""
    return p.replace("\\", "/").lower()


def recall_at_5(results: list, relevant_files: list) -> bool:
    """
    Binary relevance: True if any result's filepath matches any relevant file.
    Handles path normalization (forward/backslash, case).
    """
    if not relevant_files:
        return False
    result_paths = set()
    for r in results[:5]:
        fp = r.get("metadata", {}).get("filepath") or r.get("filepath") or ""
        if fp:
            result_paths.add(_normalize_path(fp))
    for rel in relevant_files:
        rel_norm = _normalize_path(rel)
        for rp in result_paths:
            if rel_norm in rp or rp.endswith(rel_norm):
                return True
    return False


def run_retrieval_eval(use_hybrid: bool, retriever=None) -> float:
    """Run retrieval on benchmark. use_hybrid=False = vector-only, True = hybrid RRF."""
    from src.rag.retriever import HybridRetriever

    bench = load_benchmark()
    if retriever is None:
        retriever = HybridRetriever(persist_directory=CHROMA_PERSIST, use_reranker=False)
        # Reindex from INDEX_DIR (CodeMind src/) so benchmark relevant_files match
        print(f"  [INFO] Indexing {INDEX_DIR}...")
        retriever.index_codebase(INDEX_DIR, force_reindex=True)

    hits = 0
    total = len(bench["tasks"])

    for i, task in enumerate(bench["tasks"]):
        query = task["query"]
        relevant = task.get("relevant_files") or task.get("expected_in_top5") or []
        if not relevant:
            continue
        try:
            if use_hybrid:
                results = retriever.search(query, n_results=5)
            else:
                results = retriever.vector_search(query, n_results=5)
            if recall_at_5(results, relevant):
                hits += 1
            elif i == 0:
                # Debug first miss
                fps = [r.get("metadata", {}).get("filepath", "") for r in results[:5]]
                print(f"  [DEBUG] First query miss: relevant={relevant}, got filepaths={fps[:3]}")
        except Exception as e:
            print(f"  [WARN] {query[:40]}... -> {e}")

    return hits / total if total > 0 else 0.0


def main():
    print("Hybrid Retrieval Evaluation (file-level binary relevance)")
    print("=" * 50)

    # Index once (CodeMind src/), reuse for both baseline and improved
    from src.rag.retriever import HybridRetriever
    retriever = HybridRetriever(persist_directory=CHROMA_PERSIST, use_reranker=False)
    print(f"\n  [INFO] Indexing {INDEX_DIR}...")
    retriever.index_codebase(INDEX_DIR, force_reindex=True)

    print("\nBaseline (vector-only)...")
    try:
        baseline_recall = run_retrieval_eval(use_hybrid=False, retriever=retriever)
    except Exception as e:
        print(f"  Baseline failed: {e}")
        baseline_recall = 0.0
    print(f"  Recall@5: {baseline_recall:.2%}")

    print("\nImproved (vector + BM25 + RRF)...")
    try:
        improved_recall = run_retrieval_eval(use_hybrid=True, retriever=retriever)
    except Exception as e:
        print(f"  Improved failed: {e}")
        improved_recall = baseline_recall
    print(f"  Recall@5: {improved_recall:.2%}")

    delta = improved_recall - baseline_recall
    print(f"\nDelta: {delta:+.2%}")

    return {
        "baseline": round(baseline_recall, 4),
        "improved": round(improved_recall, 4),
        "delta": round(delta, 4),
    }


if __name__ == "__main__":
    result = main()
    print(f"\nResult: {result}")
