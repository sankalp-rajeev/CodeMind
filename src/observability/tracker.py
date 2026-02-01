"""
MLflow Tracker for CodeMind AI

Tracks:
- Query metrics (latency, tokens, agent used)
- RAG metrics (retrieval time, chunks, scores)
- Crew metrics (execution time, agents, tasks)
- User feedback (thumbs up/down)
"""

import time
from typing import Dict, Any, Optional, List
from datetime import datetime
from dataclasses import dataclass, field
from contextlib import contextmanager

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: mlflow not installed. Run: pip install mlflow")


@dataclass
class QueryMetrics:
    """Metrics for a single query."""
    query: str
    intent: str
    agent_used: str
    latency_ms: float
    tokens_generated: int
    success: bool
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass  
class RAGMetrics:
    """Metrics for RAG retrieval."""
    query: str
    retrieval_time_ms: float
    chunks_returned: int
    avg_relevance_score: float
    top_chunk_score: float


@dataclass
class CrewMetrics:
    """Metrics for crew execution."""
    crew_name: str
    agents_involved: List[str]
    total_time_s: float
    tasks_completed: int
    iterations: int = 1
    success: bool = True


class MLflowTracker:
    """
    Tracks CodeMind metrics using MLflow.
    
    Features:
    - Query tracking
    - RAG performance tracking
    - Crew execution tracking
    - User feedback
    """
    
    def __init__(
        self,
        experiment_name: str = "codemind-ai",
        tracking_uri: str = "./data/mlruns"
    ):
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self.enabled = MLFLOW_AVAILABLE
        
        if self.enabled:
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(experiment_name)
            self.client = MlflowClient()
            print(f"MLflow tracking enabled: {tracking_uri}")
        else:
            print("MLflow disabled - install with: pip install mlflow")
    
    @contextmanager
    def track_query(self, query: str, intent: str, agent: str):
        """Context manager to track a query."""
        start_time = time.time()
        tokens = 0
        success = True
        
        try:
            yield lambda t: setattr(self, '_tokens', t)
        except Exception as e:
            success = False
            raise
        finally:
            latency = (time.time() - start_time) * 1000  # ms
            tokens = getattr(self, '_tokens', 0)
            
            if self.enabled:
                with mlflow.start_run(run_name=f"query_{int(time.time())}"):
                    mlflow.log_params({
                        "query": query[:100],
                        "intent": intent,
                        "agent": agent
                    })
                    mlflow.log_metrics({
                        "latency_ms": latency,
                        "tokens_generated": tokens,
                        "success": 1 if success else 0
                    })
    
    def log_rag_metrics(self, metrics: RAGMetrics):
        """Log RAG retrieval metrics."""
        if not self.enabled:
            return
            
        with mlflow.start_run(run_name=f"rag_{int(time.time())}"):
            mlflow.log_params({
                "query": metrics.query[:100]
            })
            mlflow.log_metrics({
                "retrieval_time_ms": metrics.retrieval_time_ms,
                "chunks_returned": metrics.chunks_returned,
                "avg_relevance_score": metrics.avg_relevance_score,
                "top_chunk_score": metrics.top_chunk_score
            })
    
    def log_crew_metrics(self, metrics: CrewMetrics):
        """Log crew execution metrics."""
        if not self.enabled:
            return
            
        with mlflow.start_run(run_name=f"crew_{metrics.crew_name}_{int(time.time())}"):
            mlflow.log_params({
                "crew_name": metrics.crew_name,
                "agents": ",".join(metrics.agents_involved)
            })
            mlflow.log_metrics({
                "total_time_s": metrics.total_time_s,
                "tasks_completed": metrics.tasks_completed,
                "iterations": metrics.iterations,
                "success": 1 if metrics.success else 0,
                "agent_count": len(metrics.agents_involved)
            })
    
    def log_feedback(self, query: str, feedback: str, rating: int):
        """Log user feedback (thumbs up/down)."""
        if not self.enabled:
            return
            
        with mlflow.start_run(run_name=f"feedback_{int(time.time())}"):
            mlflow.log_params({
                "query": query[:100],
                "feedback_type": feedback
            })
            mlflow.log_metrics({
                "rating": rating  # 1 = thumbs up, -1 = thumbs down
            })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all tracked metrics."""
        if not self.enabled:
            return {"enabled": False}
        
        experiment = self.client.get_experiment_by_name(self.experiment_name)
        if not experiment:
            return {"enabled": True, "runs": 0}
        
        runs = self.client.search_runs(
            experiment_ids=[experiment.experiment_id],
            max_results=100
        )
        
        return {
            "enabled": True,
            "runs": len(runs),
            "experiment_name": self.experiment_name,
            "tracking_uri": self.tracking_uri
        }


# Global tracker instance
_tracker: Optional[MLflowTracker] = None


def get_tracker() -> MLflowTracker:
    """Get or create the global tracker."""
    global _tracker
    if _tracker is None:
        _tracker = MLflowTracker()
    return _tracker


def test_tracker():
    """Test the MLflow tracker."""
    print("Testing MLflow Tracker...")
    print("=" * 50)
    
    tracker = get_tracker()
    
    # Test query tracking
    with tracker.track_query("What is the login function?", "EXPLORE", "code_explorer") as log_tokens:
        time.sleep(0.1)  # Simulate work
        log_tokens(150)
    
    print("✅ Query tracking OK")
    
    # Test RAG metrics
    tracker.log_rag_metrics(RAGMetrics(
        query="test query",
        retrieval_time_ms=45.2,
        chunks_returned=5,
        avg_relevance_score=0.78,
        top_chunk_score=0.92
    ))
    print("✅ RAG metrics OK")
    
    # Test crew metrics
    tracker.log_crew_metrics(CrewMetrics(
        crew_name="RefactoringCrew",
        agents_involved=["Explorer", "Security", "Optimizer"],
        total_time_s=45.3,
        tasks_completed=5,
        success=True
    ))
    print("✅ Crew metrics OK")
    
    # Get summary
    summary = tracker.get_summary()
    print(f"\n📊 Summary: {summary}")
    
    print("\n✅ MLflow Tracker working!")
    print(f"   View dashboard: mlflow ui --backend-store-uri {tracker.tracking_uri}")


if __name__ == "__main__":
    test_tracker()
