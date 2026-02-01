"""
BM25 Index for CodeMind AI

Provides keyword-based search using BM25 algorithm.
Used in combination with vector search for hybrid retrieval.
"""

import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False
    print("Warning: rank_bm25 not installed. BM25 search unavailable.")


class BM25Index:
    """
    BM25 keyword search index for code chunks.
    
    BM25 excels at:
    - Exact keyword matching
    - Finding specific function/class names
    - API and variable name search
    """
    
    def __init__(self):
        if not BM25_AVAILABLE:
            raise RuntimeError("rank_bm25 is not installed")
        
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List = []
        self.tokenized_corpus: List[List[str]] = []
    
    def build_index(self, chunks: List) -> None:
        """
        Build BM25 index from code chunks.
        
        Args:
            chunks: List of CodeChunk objects or dicts with 'content'
        """
        if not chunks:
            print("Warning: No chunks to index for BM25")
            self.bm25 = None
            self.chunks = []
            self.tokenized_corpus = []
            return
        
        self.chunks = chunks
        self.tokenized_corpus = []
        
        for chunk in chunks:
            # Get content from chunk
            if hasattr(chunk, 'content'):
                text = chunk.content
            elif isinstance(chunk, dict):
                text = chunk.get('content', '')
            else:
                text = str(chunk)
            
            # Tokenize for BM25
            tokens = self._tokenize(text)
            self.tokenized_corpus.append(tokens)
        
        # Build BM25 index
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        
        print(f"BM25 index built with {len(chunks)} documents")
    
    def search(
        self, 
        query: str, 
        n_results: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Search for relevant chunks using BM25.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of (chunk_index, score) tuples
        """
        if self.bm25 is None:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Tokenize query
        query_tokens = self._tokenize(query)
        
        # Get BM25 scores
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        
        return indexed_scores[:n_results]
    
    def search_with_chunks(
        self,
        query: str,
        n_results: int = 10
    ) -> List[Dict]:
        """
        Search and return chunks with metadata.
        
        Args:
            query: Search query
            n_results: Number of results
            
        Returns:
            List of dicts with chunk data and scores
        """
        results = self.search(query, n_results)
        
        formatted = []
        for idx, score in results:
            if score > 0:  # Only include matches
                chunk = self.chunks[idx]
                
                # Handle both CodeChunk objects and dicts
                if hasattr(chunk, 'id'):
                    formatted.append({
                        "id": chunk.id,
                        "content": chunk.content,
                        "name": chunk.name,
                        "filepath": chunk.filepath,
                        "type": chunk.type,
                        "score": float(score)
                    })
                else:
                    formatted.append({
                        "idx": idx,
                        "content": chunk.get('content', ''),
                        "score": float(score)
                    })
        
        return formatted
    
    def _tokenize(self, text: str) -> List[str]:
        """
        Tokenize text for BM25 indexing.
        
        Splits on whitespace and punctuation, lowercases,
        and handles code-specific patterns.
        """
        # Convert camelCase and snake_case to separate tokens
        text = self._split_identifiers(text)
        
        # Split on whitespace and punctuation
        tokens = re.findall(r'\b\w+\b', text.lower())
        
        # Filter very short tokens
        tokens = [t for t in tokens if len(t) > 1]
        
        return tokens
    
    def _split_identifiers(self, text: str) -> str:
        """
        Split camelCase and snake_case identifiers.
        
        Examples:
            getUserData -> get User Data
            get_user_data -> get user data
        """
        # Split camelCase
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Split snake_case
        text = text.replace('_', ' ')
        
        return text


def test_bm25():
    """Test BM25 index."""
    from src.rag.chunker import SemanticCodeChunker
    
    print("Testing BM25 Index...")
    print("=" * 50)
    
    # Get chunks
    chunker = SemanticCodeChunker()
    chunks = chunker.chunk_directory('data/test-repo')
    
    # Build index
    bm25 = BM25Index()
    bm25.build_index(chunks)
    
    # Test queries
    test_queries = [
        "dataset",
        "train model",
        "evaluate metrics f1"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = bm25.search_with_chunks(query, n_results=3)
        
        for i, result in enumerate(results):
            print(f"  [{i+1}] {result['name']} (score: {result['score']:.2f})")
    
    print("\n✅ BM25 Index working!")


if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    test_bm25()
