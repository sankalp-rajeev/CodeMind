"""
Evaluation configuration.
Frozen paths and model names for reproducible runs.
"""

from pathlib import Path

# Paths (relative to project root)
PROJECT_ROOT = Path(__file__).parent.parent
BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"
REPORT_PATH = Path(__file__).parent / "evaluation_report.json"
REPORT_MD_PATH = Path(__file__).parent / "evaluation_report.md"

# Models
BASELINE_ORCHESTRATOR = "qwen2.5:7b"
IMPROVED_ORCHESTRATOR = "orchestrator-ft"  # fine-tuned; fallback to qwen2.5:7b if not present

# RAG
CHROMA_PERSIST = str(PROJECT_ROOT / "data" / "chroma_db")
# Index dir: CodeMind on itself. Prefer src/ (CodeMind's own code).
_INDEX_CANDIDATES = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "data" / "test-repo",
    PROJECT_ROOT,
]
INDEX_DIR = str(next((p for p in _INDEX_CANDIDATES if p.exists()), _INDEX_CANDIDATES[0]))

# Intent labels (must match QueryIntent enum)
INTENT_LABELS = ["EXPLORE", "REFACTOR", "TEST", "SECURITY", "DOCUMENT", "GENERAL"]
