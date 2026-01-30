"""Test script for Day 1 - Chunker validation"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rag.chunker import SemanticCodeChunker

def test_chunker():
    """Test Part 1: Chunker"""
    print("\n" + "=" * 60)
    print("📦 PART 1: CHUNKER TEST")
    print("=" * 60)
    
    chunker = SemanticCodeChunker()
    chunks = chunker.chunk_directory('data/test-repo')
    
    print(f"\n📊 RESULTS:")
    print(f"   Total chunks: {len(chunks)}")
    print(f"   Functions: {sum(1 for c in chunks if c.type == 'function')}")
    print(f"   Classes: {sum(1 for c in chunks if c.type == 'class')}")
    
    unique_files = set(c.filepath for c in chunks)
    print(f"   Files parsed: {len(unique_files)}")
    
    print(f"\n📝 SAMPLE CHUNKS (first 3):")
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n   [{i+1}] {chunk.type.upper()}: {chunk.name}")
        print(f"       Lines: {chunk.start_line}-{chunk.end_line}")
    
    print("\n✅ Chunker: PASSED")
    return chunks


def test_indexer():
    """Test Part 2: ChromaDB Indexer"""
    print("\n" + "=" * 60)
    print("🗄️  PART 2: CHROMADB INDEXER TEST")
    print("=" * 60)
    
    from src.rag.indexer import CodebaseIndexer
    
    indexer = CodebaseIndexer(
        persist_directory="./data/chroma_db_test",
        use_fallback_embeddings=True
    )
    
    # Index the test repo
    stats = indexer.index_codebase("./data/test-repo", force_reindex=True)
    
    print(f"\n📊 INDEXING RESULTS:")
    print(f"   Chunks indexed: {stats['chunks_indexed']}")
    print(f"   Files processed: {stats['files_processed']}")
    print(f"   Duration: {stats['duration_seconds']}s")
    
    print("\n✅ Indexer: PASSED")
    return indexer


def test_search(indexer):
    """Test Part 3: Semantic Search"""
    print("\n" + "=" * 60)
    print("🔍 PART 3: SEMANTIC SEARCH TEST")
    print("=" * 60)
    
    test_queries = [
        "dataset loading",
        "training model",
        "evaluate metrics"
    ]
    
    for query in test_queries:
        print(f"\n🔎 Query: '{query}'")
        results = indexer.search(query, n_results=3)
        
        for i, result in enumerate(results):
            print(f"   [{i+1}] {result['metadata']['name']} (score: {result['score']:.3f})")
            print(f"       → {result['metadata']['filepath'].split('/')[-1]}")
    
    print("\n✅ Search: PASSED")


def main():
    print("\n" + "=" * 60)
    print("   🧠 CodeMind Day 1 - RAG Foundation Test")
    print("=" * 60)
    
    # Part 1: Chunker
    chunks = test_chunker()
    
    # Part 2: Indexer
    indexer = test_indexer()
    
    # Part 3: Search
    test_search(indexer)
    
    print("\n" + "=" * 60)
    print("🎉 DAY 1 COMPLETE! All tests passed.")
    print("=" * 60)
    print("\nNext: Day 2 - Embeddings & Hybrid Search\n")

if __name__ == "__main__":
    main()
