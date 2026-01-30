"""
Code Chunking Module for CodeMind AI

This module provides semantic chunking of source code files by parsing
them into logical units (functions, classes, methods) using AST analysis.
"""

import ast
import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime


@dataclass
class CodeChunk:
    """
    Represents a semantic code chunk (function, class, or method).
    
    Attributes:
        id: Unique identifier for this chunk
        content: The raw source code content
        type: Type of chunk (function, class, method)
        name: Name of the function/class
        filepath: Path to the source file
        language: Programming language
        start_line: Starting line number (1-indexed)
        end_line: Ending line number (1-indexed)
        metadata: Additional metadata about the chunk
    """
    id: str
    content: str
    type: str  # 'function', 'class', 'method'
    name: str
    filepath: str
    language: str
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, CodeChunk):
            return self.id == other.id
        return False


class SemanticCodeChunker:
    """
    Chunks code by semantic units (functions, classes) using AST parsing.
    
    Supports:
    - Python (.py)
    - JavaScript/TypeScript (.js, .ts, .jsx, .tsx) - via tree-sitter (TODO)
    """
    
    SUPPORTED_EXTENSIONS = {'.py'}  # Will add JS/TS later
    
    def __init__(self):
        self.file_summaries: Dict[str, str] = {}
    
    def chunk_file(self, filepath: str) -> List[CodeChunk]:
        """
        Chunk a single file into semantic units.
        
        Args:
            filepath: Path to the source file
            
        Returns:
            List of CodeChunk objects
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
        
        if path.suffix not in self.SUPPORTED_EXTENSIONS:
            return []
        
        content = path.read_text(encoding='utf-8', errors='ignore')
        
        if path.suffix == '.py':
            return self._chunk_python(content, str(path))
        
        return []
    
    def chunk_directory(
        self, 
        directory: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[CodeChunk]:
        """
        Chunk all supported files in a directory.
        
        Args:
            directory: Path to the directory
            exclude_patterns: Patterns to exclude (e.g., 'venv', 'node_modules')
            
        Returns:
            List of all CodeChunk objects from the directory
        """
        exclude_patterns = exclude_patterns or [
            'venv', 'env', '.venv', '__pycache__', 
            'node_modules', '.git', 'dist', 'build'
        ]
        
        chunks = []
        path = Path(directory)
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for file_path in path.rglob(f'*{ext}'):
                # Skip excluded patterns
                if any(pattern in str(file_path) for pattern in exclude_patterns):
                    continue
                
                try:
                    file_chunks = self.chunk_file(str(file_path))
                    chunks.extend(file_chunks)
                except Exception as e:
                    print(f"Warning: Failed to parse {file_path}: {e}")
        
        return chunks
    
    def _chunk_python(self, code: str, filepath: str) -> List[CodeChunk]:
        """
        Chunk Python code using AST.
        
        Args:
            code: Python source code
            filepath: Path to the file (for metadata)
            
        Returns:
            List of CodeChunk objects
        """
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            print(f"Syntax error in {filepath}: {e}")
            return []
        
        chunks = []
        lines = code.split('\n')
        
        for node in ast.walk(tree):
            chunk = None
            
            # Handle function definitions
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                # Skip if this is a method (has parent class)
                chunk = self._create_function_chunk(node, code, filepath, lines)
            
            # Handle class definitions
            elif isinstance(node, ast.ClassDef):
                chunk = self._create_class_chunk(node, code, filepath, lines)
            
            if chunk:
                chunks.append(chunk)
        
        return chunks
    
    def _create_function_chunk(
        self, 
        node: ast.FunctionDef, 
        code: str, 
        filepath: str,
        lines: List[str]
    ) -> CodeChunk:
        """Create a CodeChunk from a function AST node."""
        
        # Get source segment
        try:
            content = ast.get_source_segment(code, node)
            if content is None:
                # Fallback: extract by line numbers
                content = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        except Exception:
            content = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        
        # Extract metadata
        params = [arg.arg for arg in node.args.args]
        
        # Get return type if annotated
        returns = None
        if node.returns:
            try:
                returns = ast.unparse(node.returns)
            except Exception:
                pass
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass
        
        # Extract function calls made in this function
        calls = self._extract_function_calls(node)
        
        # Calculate simple complexity (number of branches)
        complexity = self._calculate_complexity(node)
        
        # Generate unique ID
        chunk_id = self._generate_id(filepath, node.name, node.lineno)
        
        return CodeChunk(
            id=chunk_id,
            content=content,
            type='function',
            name=node.name,
            filepath=filepath,
            language='python',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            metadata={
                'params': params,
                'returns': returns,
                'docstring': docstring,
                'decorators': decorators,
                'calls': calls,
                'complexity': complexity,
                'is_async': isinstance(node, ast.AsyncFunctionDef),
            }
        )
    
    def _create_class_chunk(
        self, 
        node: ast.ClassDef, 
        code: str, 
        filepath: str,
        lines: List[str]
    ) -> CodeChunk:
        """Create a CodeChunk from a class AST node."""
        
        # Get source segment
        try:
            content = ast.get_source_segment(code, node)
            if content is None:
                content = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        except Exception:
            content = '\n'.join(lines[node.lineno - 1:node.end_lineno])
        
        # Get base classes
        bases = []
        for base in node.bases:
            try:
                bases.append(ast.unparse(base))
            except Exception:
                pass
        
        # Get docstring
        docstring = ast.get_docstring(node)
        
        # Get method names
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(item.name)
        
        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            try:
                decorators.append(ast.unparse(dec))
            except Exception:
                pass
        
        # Calculate complexity
        complexity = self._calculate_complexity(node)
        
        # Generate unique ID
        chunk_id = self._generate_id(filepath, node.name, node.lineno)
        
        return CodeChunk(
            id=chunk_id,
            content=content,
            type='class',
            name=node.name,
            filepath=filepath,
            language='python',
            start_line=node.lineno,
            end_line=node.end_lineno or node.lineno,
            metadata={
                'bases': bases,
                'docstring': docstring,
                'methods': methods,
                'decorators': decorators,
                'complexity': complexity,
            }
        )
    
    def _extract_function_calls(self, node: ast.AST) -> List[str]:
        """Extract all function calls from an AST node."""
        calls = []
        
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
        
        return list(set(calls))  # Unique calls
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Calculate cyclomatic complexity of a node.
        
        Simple approximation: count branches (if, for, while, try, etc.)
        """
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.IfExp)):
                complexity += 1
            elif isinstance(child, (ast.For, ast.AsyncFor, ast.While)):
                complexity += 1
            elif isinstance(child, (ast.Try, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, (ast.And, ast.Or)):
                complexity += 1
            elif isinstance(child, ast.comprehension):
                complexity += 1
        
        return complexity
    
    def _generate_id(self, filepath: str, name: str, line: int) -> str:
        """Generate a unique ID for a chunk."""
        unique_string = f"{filepath}:{name}:{line}"
        return hashlib.sha256(unique_string.encode()).hexdigest()[:16]


# Test function
def test_chunker():
    """Test the chunker on a sample file."""
    chunker = SemanticCodeChunker()
    
    # Test on this file itself
    chunks = chunker.chunk_file(__file__)
    
    print(f"Found {len(chunks)} chunks in {__file__}")
    print("-" * 50)
    
    for chunk in chunks:
        print(f"  [{chunk.type}] {chunk.name}")
        print(f"    Lines: {chunk.start_line}-{chunk.end_line}")
        print(f"    Complexity: {chunk.metadata.get('complexity', 'N/A')}")
        if chunk.type == 'function':
            print(f"    Params: {chunk.metadata.get('params', [])}")
            print(f"    Calls: {chunk.metadata.get('calls', [])[:5]}...")
        print()


if __name__ == "__main__":
    test_chunker()
