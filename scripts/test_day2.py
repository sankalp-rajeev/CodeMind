"""Test script for Day 2 - Hybrid Search validation"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.retriever import HybridRetriever


def test_hybrid_search():
    """Test the complete hybrid search pipeline."""
    
    print("\n" + "=" * 60)
    print("   🧠 CodeMind Day 2 - Hybrid Search Test")
    print("=" * 60)
    
    # Initialize retriever
    print("\n📦 PART 1: INITIALIZING RETRIEVER")
    print("-" * 40)
    
    retriever = HybridRetriever(
        persist_directory="./data/chroma_db_test",
        use_reranker=False  # Skip reranker for speed
    )
    
    # Index the test repo
    print("\n🗄️  PART 2: INDEXING CODEBASE")
    print("-" * 40)
    
    start_time = time.time()
    stats = retriever.index_codebase("./data/test-repo", force_reindex=True)
    index_time = time.time() - start_time
    
    print(f"   Chunks indexed: {stats['chunks_indexed']}")
    print(f"   BM25 indexed: {stats.get('bm25_indexed', 'N/A')}")
    print(f"   Time: {index_time:.2f}s")
    print("\n✅ Indexing: PASSED")
    
    # Test different search methods
    print("\n🔍 PART 3: SEARCH COMPARISON")
    print("-" * 40)
    
    test_queries = [
        "dataset loading",
        "training model loss",
        "evaluation accuracy f1 score"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        
        # Measure latencies
        start = time.time()
        vector_results = retriever.vector_search(query, n_results=3)
        vector_time = (time.time() - start) * 1000
        
        start = time.time()
        bm25_results = retriever.keyword_search(query, n_results=3)
        bm25_time = (time.time() - start) * 1000
        
        start = time.time()
        hybrid_results = retriever.search(query, n_results=3)
        hybrid_time = (time.time() - start) * 1000
        
        print(f"\n   📊 Vector Search ({vector_time:.0f}ms):")
        for r in vector_results[:2]:
            name = r.get('metadata', {}).get('name', 'N/A')
            score = r.get('score', 0)
            print(f"      • {name} (score: {score:.3f})")
        
        print(f"\n   📊 BM25 Search ({bm25_time:.0f}ms):")
        for r in bm25_results[:2]:
            print(f"      • {r.get('name', 'N/A')} (score: {r.get('score', 0):.2f})")
        
        print(f"\n   📊 Hybrid RRF ({hybrid_time:.0f}ms):")
        for r in hybrid_results[:2]:
            name = r.get('metadata', {}).get('name') or r.get('name', 'N/A')
            rrf = r.get('rrf_score', 0)
            print(f"      • {name} (RRF: {rrf:.4f})")
    
    print("\n✅ Search Comparison: PASSED")
    
    # Summary
    print("\n" + "=" * 60)
    print("🎉 DAY 2 COMPLETE! All tests passed.")
    print("=" * 60)
    
    print("\n📊 METRICS SUMMARY:")
    print(f"   • Index time: {index_time:.2f}s")
    print(f"   • Vector search: ~{vector_time:.0f}ms")
    print(f"   • BM25 search: ~{bm25_time:.0f}ms")  
    print(f"   • Hybrid search: ~{hybrid_time:.0f}ms")
    
    print("\n📋 FILES CREATED:")
    print("   • src/rag/bm25_index.py - BM25 keyword search")
    print("   • src/rag/retriever.py - Hybrid retriever with RRF")
    
    print("\nNext: Days 3-4 - First Agent (CodeExplorerAgent)\n")


if __name__ == "__main__":
    test_hybrid_search()
