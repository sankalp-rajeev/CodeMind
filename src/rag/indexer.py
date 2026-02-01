"""
Indexer Module for CodeMind AI

Handles indexing code chunks into ChromaDB vector database.
Supports:
- Adding chunks with embeddings
- Persistent storage
- Metadata filtering
- Python, JavaScript, and TypeScript files
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("Warning: chromadb not installed")

from .chunker import CodeChunk, SemanticCodeChunker
from .embeddings import CodeEmbedder

# Import JS chunker
try:
    from .js_chunker import JSChunker, JSCodeChunk
    JS_CHUNKER_AVAILABLE = True
except ImportError:
    JS_CHUNKER_AVAILABLE = False


class CodebaseIndexer:
    """
    Indexes code chunks into ChromaDB for semantic search.
    
    Features:
    - Persistent storage
    - Metadata preservation
    - Batch indexing
    - Collection management
    """
    
    def __init__(
        self,
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "code_chunks",
        use_fallback_embeddings: bool = True
    ):
        """
        Initialize the indexer.
        
        Args:
            persist_directory: Directory for persistent storage
            collection_name: Name of the ChromaDB collection
            use_fallback_embeddings: Use smaller, faster embedding model
        """
        if not CHROMADB_AVAILABLE:
            raise RuntimeError("chromadb is not installed")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory)
        )
        
        # Initialize embedder
        self.embedder = CodeEmbedder(use_fallback=use_fallback_embeddings)
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Track indexed files
        self.indexed_files: set = set()
        
        print(f"Indexer initialized. Collection '{collection_name}' has {self.collection.count()} items.")
    
    def index_codebase(
        self,
        directory: str,
        batch_size: int = 32,
        force_reindex: bool = False
    ) -> Dict[str, Any]:
        """
        Index an entire codebase directory.
        
        Args:
            directory: Path to the codebase
            batch_size: Batch size for embedding
            force_reindex: If True, reindex all files
            
        Returns:
            Statistics about the indexing operation
        """
        start_time = datetime.now()
        
        # Clear collection if force reindex
        if force_reindex:
            self.clear_collection()
        
        # Chunk Python files
        chunker = SemanticCodeChunker()
        python_chunks = chunker.chunk_directory(directory)
        
        # Also chunk JS/TS files if available
        all_chunks = list(python_chunks)
        js_chunks = []
        
        if JS_CHUNKER_AVAILABLE:
            js_chunker = JSChunker()
            js_code_chunks = js_chunker.chunk_directory(directory)
            
            # Convert JSCodeChunk to CodeChunk for compatibility
            for jc in js_code_chunks:
                chunk = CodeChunk(
                    id=jc.id,
                    content=jc.content,
                    type=jc.type,
                    name=jc.name,
                    filepath=jc.filepath,
                    language=jc.language,
                    start_line=jc.start_line,
                    end_line=jc.end_line,
                    metadata=jc.metadata
                )
                all_chunks.append(chunk)
            
            print(f"Found {len(js_code_chunks)} JS/TS chunks")
        
        chunks = all_chunks
        
        if not chunks:
            return {
                "status": "no_files",
                "chunks_indexed": 0,
                "files_processed": 0,
                "duration_seconds": 0
            }
        
        py_count = len(python_chunks)
        js_count = len(chunks) - py_count
        print(f"Found {len(chunks)} total chunks ({py_count} Python, {js_count} JS/TS) from {len(set(c.filepath for c in chunks))} files")
        
        # Generate embeddings
        print("Generating embeddings...")
        embeddings = self.embedder.embed_chunks(chunks, batch_size=batch_size)
        
        # Prepare data for ChromaDB (sanitize content for Windows cp1252 compatibility)
        ids = [chunk.id for chunk in chunks]
        documents = [self._sanitize_for_chromadb(chunk.content) for chunk in chunks]
        metadatas = [self._chunk_to_metadata(chunk) for chunk in chunks]
        
        # Add to collection in batches
        print("Adding to ChromaDB...")
        for i in range(0, len(chunks), batch_size):
            end_idx = min(i + batch_size, len(chunks))
            
            self.collection.add(
                ids=ids[i:end_idx],
                documents=documents[i:end_idx],
                embeddings=embeddings[i:end_idx].tolist(),
                metadatas=metadatas[i:end_idx]
            )
        
        duration = (datetime.now() - start_time).total_seconds()
        
        stats = {
            "status": "success",
            "chunks_indexed": len(chunks),
            "files_processed": len(set(c.filepath for c in chunks)),
            "functions": sum(1 for c in chunks if c.type == 'function'),
            "classes": sum(1 for c in chunks if c.type == 'class'),
            "duration_seconds": round(duration, 2),
            "collection_size": self.collection.count()
        }
        
        print(f"[OK] Indexed {stats['chunks_indexed']} chunks in {stats['duration_seconds']}s")
        
        return stats
    
    def search(
        self,
        query: str,
        n_results: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict]:
        """
        Search for relevant code chunks.
        
        Args:
            query: Natural language query
            n_results: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of matching chunks with scores
        """
        # Get query embedding
        query_embedding = self.embedder.embed(query)
        
        # Search in ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=filter_metadata,
            include=["documents", "metadatas", "distances"]
        )
        
        # Format results
        formatted_results = []
        
        if results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                result = {
                    "id": chunk_id,
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i],
                    "score": 1 - results['distances'][0][i]  # Convert distance to similarity
                }
                formatted_results.append(result)
        
        return formatted_results
    
    def clear_collection(self):
        """Clear all items from the collection."""
        # Delete and recreate collection
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Collection cleared. Now has {self.collection.count()} items.")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed codebase."""
        return {
            "total_chunks": self.collection.count(),
            "persist_directory": str(self.persist_directory),
            "collection_name": self.collection.name
        }
    
    def _sanitize_for_chromadb(self, text: str) -> str:
        """Replace emojis/Unicode that cause cp1252 encoding errors on Windows."""
        if not text:
            return text
        # Replace common emojis that break Windows cp1252
        replacements = {
            '\u2705': '[OK]',   # checkmark
            '\u274c': '[X]',    # cross mark
            '\u26a0\ufe0f': '[!]',  # warning
            '\u26a0': '[!]',
        }
        result = text
        for emoji, replacement in replacements.items():
            result = result.replace(emoji, replacement)
        return result

    def _chunk_to_metadata(self, chunk: CodeChunk) -> Dict[str, Any]:
        """Convert a CodeChunk to ChromaDB-compatible metadata."""
        # ChromaDB only supports string, int, float, bool for metadata
        metadata = {
            "type": chunk.type,
            "name": chunk.name,
            "filepath": chunk.filepath,
            "language": chunk.language,
            "start_line": chunk.start_line,
            "end_line": chunk.end_line,
        }
        
        # Add optional metadata
        if chunk.metadata.get('complexity'):
            metadata['complexity'] = chunk.metadata['complexity']
        
        if chunk.metadata.get('is_async'):
            metadata['is_async'] = chunk.metadata['is_async']
        
        # Store params as comma-separated string
        if chunk.metadata.get('params'):
            metadata['params'] = ','.join(chunk.metadata['params'])
        
        # Store first 200 chars of docstring (sanitize for Windows encoding)
        if chunk.metadata.get('docstring'):
            metadata['docstring'] = self._sanitize_for_chromadb(chunk.metadata['docstring'][:200])
        
        return metadata


# Test function
def test_indexer():
    """Test the indexer on a sample directory."""
    indexer = CodebaseIndexer(
        persist_directory="./data/chroma_db_test",
        use_fallback_embeddings=True
    )
    
    # Index test repository
    stats = indexer.index_codebase("./data/test-repo", force_reindex=True)
    print(f"\nIndexing stats: {stats}")
    
    # Test search
    print("\n" + "=" * 50)
    print("Testing search...")
    
    test_queries = [
        "dataset loading and preprocessing",
        "training the model",
        "evaluation metrics"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = indexer.search(query, n_results=3)
        
        for i, result in enumerate(results):
            print(f"  [{i+1}] {result['metadata']['name']} ({result['metadata']['type']})")
            print(f"      File: {result['metadata']['filepath']}")
            print(f"      Score: {result['score']:.3f}")


if __name__ == "__main__":
    test_indexer()
