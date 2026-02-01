"""
Hybrid Retriever for CodeMind AI

Combines vector search (ChromaDB) with keyword search (BM25)
using Reciprocal Rank Fusion (RRF) for optimal retrieval.

Also includes cross-encoder reranking for improved precision.
"""

from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict

try:
    from sentence_transformers import CrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    CROSS_ENCODER_AVAILABLE = False

from .indexer import CodebaseIndexer
from .bm25_index import BM25Index
from .chunker import SemanticCodeChunker


class HybridRetriever:
    """
    Hybrid retrieval system combining:
    1. Vector search (semantic similarity)
    2. BM25 search (keyword matching)
    3. Cross-encoder reranking (optional)
    
    Uses Reciprocal Rank Fusion to combine results.
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        use_reranker: bool = False,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            persist_directory: ChromaDB persistence directory
            use_reranker: Whether to use cross-encoder reranking
            reranker_model: Model for reranking
        """
        # Initialize vector search (ChromaDB)
        self.indexer = CodebaseIndexer(
            persist_directory=persist_directory,
            use_fallback_embeddings=True
        )
        
        # Initialize BM25 index
        self.bm25 = BM25Index()
        self.bm25_built = False
        
        # Initialize reranker if requested
        self.use_reranker = use_reranker and CROSS_ENCODER_AVAILABLE
        self.reranker = None
        
        if self.use_reranker:
            print(f"Loading reranker: {reranker_model}...")
            self.reranker = CrossEncoder(reranker_model)
    
    def index_codebase(
        self,
        directory: str,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Index a codebase for hybrid search.
        
        Args:
            directory: Path to the codebase
            force_reindex: If True, rebuild the entire index
            
        Returns:
            Indexing statistics
        """
        # Index using ChromaDB (vector search)
        stats = self.indexer.index_codebase(directory, force_reindex=force_reindex)
        
        # Build BM25 index from the same chunks
        chunker = SemanticCodeChunker()
        chunks = chunker.chunk_directory(directory)
        
        if chunks:
            self.bm25.build_index(chunks)
            self.bm25_built = True
            stats['bm25_indexed'] = len(chunks)
        else:
            self.bm25_built = False
            stats['bm25_indexed'] = 0
        
        return stats
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        vector_weight: float = 0.6,
        bm25_weight: float = 0.4,
        use_rrf: bool = True
    ) -> List[Dict]:
        """
        Hybrid search combining vector and BM25 results.
        
        Args:
            query: Search query
            n_results: Number of results to return
            vector_weight: Weight for vector search (0-1)
            bm25_weight: Weight for BM25 search (0-1)
            use_rrf: If True, use RRF; otherwise use weighted scores
            
        Returns:
            List of search results with combined scores
        """
        # Get more candidates than needed for fusion
        n_candidates = n_results * 3
        
        # Vector search
        vector_results = self.indexer.search(query, n_results=n_candidates)
        
        # BM25 search
        if self.bm25_built:
            bm25_results = self.bm25.search_with_chunks(query, n_results=n_candidates)
        else:
            bm25_results = []
        
        # Combine results
        if use_rrf:
            combined = self._reciprocal_rank_fusion(
                vector_results, 
                bm25_results,
                k=60  # RRF constant
            )
        else:
            combined = self._weighted_combination(
                vector_results,
                bm25_results,
                vector_weight,
                bm25_weight
            )
        
        # Rerank if enabled
        if self.use_reranker and self.reranker is not None:
            combined = self._rerank(query, combined, n_results * 2)
        
        return combined[:n_results]
    
    def vector_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Pure vector search."""
        return self.indexer.search(query, n_results=n_results)
    
    def keyword_search(self, query: str, n_results: int = 5) -> List[Dict]:
        """Pure BM25 keyword search."""
        if not self.bm25_built:
            return []
        return self.bm25.search_with_chunks(query, n_results=n_results)
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict]:
        """
        Combine results using Reciprocal Rank Fusion.
        
        RRF score = sum(1 / (k + rank_i)) for each ranking
        
        Args:
            vector_results: Results from vector search
            bm25_results: Results from BM25 search
            k: RRF constant (default 60)
            
        Returns:
            Combined and sorted results
        """
        rrf_scores = defaultdict(float)
        result_data = {}
        
        # Add vector search ranks
        for rank, result in enumerate(vector_results):
            doc_id = result.get('id') or result.get('metadata', {}).get('name', str(rank))
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            result_data[doc_id] = result
        
        # Add BM25 ranks
        for rank, result in enumerate(bm25_results):
            doc_id = result.get('id') or result.get('name', str(rank))
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            if doc_id not in result_data:
                result_data[doc_id] = result
        
        # Sort by RRF score
        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        
        # Build results
        combined = []
        for doc_id in sorted_ids:
            result = result_data[doc_id].copy()
            result['rrf_score'] = rrf_scores[doc_id]
            combined.append(result)
        
        return combined
    
    def _weighted_combination(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        vector_weight: float,
        bm25_weight: float
    ) -> List[Dict]:
        """
        Combine results using weighted score combination.
        
        Normalizes scores and combines with weights.
        """
        combined_scores = defaultdict(float)
        result_data = {}
        
        # Normalize and add vector scores
        if vector_results:
            max_score = max(r.get('score', 0) for r in vector_results) or 1
            for result in vector_results:
                doc_id = result.get('id') or result.get('metadata', {}).get('name')
                norm_score = result.get('score', 0) / max_score
                combined_scores[doc_id] += vector_weight * norm_score
                result_data[doc_id] = result
        
        # Normalize and add BM25 scores
        if bm25_results:
            max_score = max(r.get('score', 0) for r in bm25_results) or 1
            for result in bm25_results:
                doc_id = result.get('id') or result.get('name')
                norm_score = result.get('score', 0) / max_score
                combined_scores[doc_id] += bm25_weight * norm_score
                if doc_id not in result_data:
                    result_data[doc_id] = result
        
        # Sort by combined score
        sorted_ids = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)
        
        combined = []
        for doc_id in sorted_ids:
            result = result_data[doc_id].copy()
            result['combined_score'] = combined_scores[doc_id]
            combined.append(result)
        
        return combined
    
    def _rerank(
        self,
        query: str,
        results: List[Dict],
        n_results: int
    ) -> List[Dict]:
        """
        Rerank results using cross-encoder.
        
        Args:
            query: Original query
            results: Results to rerank
            n_results: Number of results after reranking
            
        Returns:
            Reranked results
        """
        if not results or not self.reranker:
            return results
        
        # Prepare pairs for cross-encoder
        pairs = []
        for result in results[:n_results]:
            content = result.get('content', '')[:500]  # Limit content length
            pairs.append([query, content])
        
        # Get reranker scores
        scores = self.reranker.predict(pairs)
        
        # Add scores and sort
        for i, result in enumerate(results[:n_results]):
            result['rerank_score'] = float(scores[i])
        
        reranked = sorted(
            results[:n_results], 
            key=lambda x: x.get('rerank_score', 0), 
            reverse=True
        )
        
        return reranked
    
    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics."""
        return {
            "vector_stats": self.indexer.get_stats(),
            "bm25_built": self.bm25_built,
            "reranker_enabled": self.use_reranker
        }


def test_hybrid_retriever():
    """Test the hybrid retriever."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("=" * 60)
    print("Testing Hybrid Retriever")
    print("=" * 60)
    
    # Initialize retriever
    retriever = HybridRetriever(
        persist_directory="./data/chroma_db_test",
        use_reranker=False  # Skip reranker for speed
    )
    
    # Index codebase
    stats = retriever.index_codebase("./data/test-repo", force_reindex=True)
    print(f"\nIndexed: {stats}")
    
    # Test queries
    test_queries = [
        "dataset loading preprocessing",
        "train model",
        "evaluate metrics accuracy"
    ]
    
    for query in test_queries:
        print(f"\n{'='*40}")
        print(f"Query: '{query}'")
        print(f"{'='*40}")
        
        # Compare methods
        print("\n📊 Vector only:")
        for r in retriever.vector_search(query, 2):
            print(f"   • {r.get('metadata', {}).get('name', 'N/A')}")
        
        print("\n📊 BM25 only:")
        for r in retriever.keyword_search(query, 2):
            print(f"   • {r.get('name', 'N/A')}")
        
        print("\n📊 Hybrid (RRF):")
        for r in retriever.search(query, 2):
            name = r.get('metadata', {}).get('name') or r.get('name', 'N/A')
            print(f"   • {name} (RRF: {r.get('rrf_score', 0):.4f})")
    
    print("\n" + "=" * 60)
    print("✅ Hybrid Retriever working!")
    print("=" * 60)


if __name__ == "__main__":
    test_hybrid_retriever()
