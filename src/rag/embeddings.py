"""
Embeddings Module for CodeMind AI

Handles embedding generation for code chunks using sentence-transformers.
Uses CodeBERT for code embeddings (optimized for code understanding).
"""

import numpy as np
from typing import List, Union, Optional
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not installed. Embeddings will not work.")


class CodeEmbedder:
    """
    Generates embeddings for code chunks using CodeBERT.
    
    Supports:
    - Single text embedding
    - Batch embedding with GPU acceleration
    - Code-specific embeddings (CodeBERT)
    """
    
    # Default model for code embeddings
    DEFAULT_CODE_MODEL = 'microsoft/codebert-base'
    # Fallback model (smaller, faster)
    FALLBACK_MODEL = 'all-MiniLM-L6-v2'
    
    def __init__(
        self, 
        model_name: Optional[str] = None,
        device: str = 'cuda',
        use_fallback: bool = False
    ):
        """
        Initialize the embedder.
        
        Args:
            model_name: Name of the sentence-transformer model
            device: 'cuda' for GPU, 'cpu' for CPU
            use_fallback: If True, use smaller fallback model
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise RuntimeError("sentence-transformers is not installed")
        
        if model_name is None:
            model_name = self.FALLBACK_MODEL if use_fallback else self.DEFAULT_CODE_MODEL
        
        self.model_name = model_name
        self.device = device
        
        print(f"Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        
        # Move to GPU if available
        try:
            self.model = self.model.to(device)
            print(f"Model loaded on {device}")
        except Exception:
            self.model = self.model.to('cpu')
            self.device = 'cpu'
            print("Falling back to CPU")
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
    
    def embed(self, text: str) -> np.ndarray:
        """
        Embed a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as numpy array
        """
        return self.model.encode(text, convert_to_numpy=True)
    
    def embed_batch(
        self, 
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Embed multiple texts in batches.
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            show_progress: Whether to show progress bar
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            device=self.device
        )
    
    def embed_chunks(
        self,
        chunks: List,
        use_enhanced: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Embed code chunks with optional enhancement.
        
        Args:
            chunks: List of CodeChunk objects
            use_enhanced: If True, include metadata in embedding text
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings
        """
        if use_enhanced:
            texts = [self._enhance_chunk_text(chunk) for chunk in chunks]
        else:
            texts = [chunk.content for chunk in chunks]
        
        return self.embed_batch(texts, batch_size=batch_size)
    
    def _enhance_chunk_text(self, chunk) -> str:
        """
        Enhance chunk content with metadata for better retrieval.
        
        This technique (contextual retrieval) improves retrieval accuracy
        by including context about what the code does.
        """
        parts = []
        
        # Add file context
        parts.append(f"File: {chunk.filepath}")
        
        # Add type and name
        parts.append(f"{chunk.type.title()}: {chunk.name}")
        
        # Add docstring if available
        docstring = chunk.metadata.get('docstring')
        if docstring:
            parts.append(f"Description: {docstring[:200]}")
        
        # Add parameters for functions
        params = chunk.metadata.get('params')
        if params:
            parts.append(f"Parameters: {', '.join(params)}")
        
        # Add the actual code
        parts.append(f"\nCode:\n{chunk.content}")
        
        return '\n'.join(parts)


# Simple test
def test_embedder():
    """Test embedding functionality."""
    embedder = CodeEmbedder(use_fallback=True)  # Use fallback for faster testing
    
    # Test single embedding
    code = "def hello_world():\n    print('Hello, World!')"
    embedding = embedder.embed(code)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sample: {embedding[:5]}...")
    
    # Test batch embedding
    codes = [
        "def add(a, b): return a + b",
        "class User: pass",
        "import os\nos.listdir('.')"
    ]
    embeddings = embedder.embed_batch(codes, show_progress=False)
    print(f"Batch embeddings shape: {embeddings.shape}")


if __name__ == "__main__":
    test_embedder()
