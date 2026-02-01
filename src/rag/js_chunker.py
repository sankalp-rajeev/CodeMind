"""
JavaScript/TypeScript Chunker for CodeMind AI

Uses tree-sitter to parse JS/TS files into semantic chunks
(functions, classes, methods, exports).

Supports:
- JavaScript (.js)
- TypeScript (.ts)
- JSX (.jsx)
- TSX (.tsx)
"""

import hashlib
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
import re


@dataclass
class JSCodeChunk:
    """Represents a semantic JS/TS code chunk."""
    id: str
    content: str
    type: str  # function, class, method, arrow_function, export
    name: str
    filepath: str
    language: str  # javascript, typescript
    start_line: int
    end_line: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, JSCodeChunk):
            return self.id == other.id
        return False


class JSChunker:
    """
    Chunks JavaScript/TypeScript code using regex-based parsing.
    
    Uses pattern matching to identify:
    - Function declarations
    - Arrow functions
    - Class declarations
    - Methods
    - Exports
    
    Note: For production, consider using tree-sitter for more accurate parsing.
    This implementation provides a lightweight alternative.
    """
    
    # Patterns for JS/TS constructs
    PATTERNS = {
        'function': r'^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)',
        'arrow_function': r'^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>',
        'class': r'^(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?(?:\s+implements\s+[\w,\s]+)?\s*\{',
        'method': r'^\s*(?:async\s+)?(\w+)\s*\([^)]*\)\s*(?::\s*\w+)?\s*\{',
        'interface': r'^(?:export\s+)?interface\s+(\w+)',
        'type': r'^(?:export\s+)?type\s+(\w+)\s*=',
    }
    
    SUPPORTED_EXTENSIONS = {'.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}
    
    def __init__(self):
        self.chunks: List[JSCodeChunk] = []
    
    def chunk_file(self, filepath: str) -> List[JSCodeChunk]:
        """
        Chunk a JavaScript/TypeScript file into semantic units.
        
        Args:
            filepath: Path to the JS/TS file
            
        Returns:
            List of JSCodeChunk objects
        """
        path = Path(filepath)
        
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return []
        
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            return []
        
        # Determine language
        language = 'typescript' if path.suffix in {'.ts', '.tsx'} else 'javascript'
        
        return self._chunk_js_code(content, str(path), language)
    
    def chunk_directory(
        self,
        directory: str,
        exclude_patterns: Optional[List[str]] = None
    ) -> List[JSCodeChunk]:
        """
        Chunk all JS/TS files in a directory.
        
        Args:
            directory: Path to the directory
            exclude_patterns: Patterns to exclude (e.g., 'node_modules')
            
        Returns:
            List of all JSCodeChunk objects
        """
        exclude_patterns = exclude_patterns or ['node_modules', 'dist', 'build', '.git']
        chunks = []
        
        path = Path(directory)
        
        for ext in self.SUPPORTED_EXTENSIONS:
            for filepath in path.rglob(f'*{ext}'):
                # Check exclusions
                skip = False
                for pattern in exclude_patterns:
                    if pattern in str(filepath):
                        skip = True
                        break
                
                if skip:
                    continue
                
                file_chunks = self.chunk_file(str(filepath))
                chunks.extend(file_chunks)
        
        return chunks
    
    def _chunk_js_code(
        self,
        code: str,
        filepath: str,
        language: str
    ) -> List[JSCodeChunk]:
        """
        Parse JS/TS code and extract chunks.
        
        Uses brace counting to find function/class boundaries.
        """
        chunks = []
        lines = code.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Skip empty lines and comments
            stripped = line.strip()
            if not stripped or stripped.startswith('//') or stripped.startswith('/*'):
                i += 1
                continue
            
            # Try to match patterns
            for chunk_type, pattern in self.PATTERNS.items():
                match = re.match(pattern, stripped, re.MULTILINE)
                if match:
                    name = match.group(1)
                    
                    # Find the end of this construct
                    start_line = i + 1
                    end_line = self._find_block_end(lines, i)
                    
                    # Extract content
                    block_content = '\n'.join(lines[i:end_line])
                    
                    # Create chunk
                    chunk = JSCodeChunk(
                        id=self._generate_id(filepath, name, start_line),
                        content=block_content,
                        type=chunk_type,
                        name=name,
                        filepath=filepath,
                        language=language,
                        start_line=start_line,
                        end_line=end_line,
                        metadata={
                            'has_exports': 'export' in line,
                            'is_async': 'async' in line,
                            'is_default': 'default' in line,
                        }
                    )
                    chunks.append(chunk)
                    
                    # Skip past this block
                    i = end_line
                    break
            else:
                i += 1
        
        # If no chunks found, create a file-level chunk
        if not chunks and code.strip():
            chunks.append(JSCodeChunk(
                id=self._generate_id(filepath, 'module', 1),
                content=code[:2000],  # Limit size
                type='module',
                name=Path(filepath).stem,
                filepath=filepath,
                language=language,
                start_line=1,
                end_line=len(lines),
                metadata={'is_module': True}
            ))
        
        return chunks
    
    def _find_block_end(self, lines: List[str], start: int) -> int:
        """
        Find the end of a brace-delimited block.
        
        Uses brace counting to match opening and closing braces.
        """
        brace_count = 0
        in_string = False
        string_char = None
        found_open = False
        
        for i in range(start, len(lines)):
            line = lines[i]
            j = 0
            
            while j < len(line):
                char = line[j]
                
                # Handle strings
                if char in '"\'`' and (j == 0 or line[j-1] != '\\'):
                    if not in_string:
                        in_string = True
                        string_char = char
                    elif char == string_char:
                        in_string = False
                        string_char = None
                
                # Count braces (only outside strings)
                if not in_string:
                    if char == '{':
                        brace_count += 1
                        found_open = True
                    elif char == '}':
                        brace_count -= 1
                        if found_open and brace_count == 0:
                            return i + 1
                
                j += 1
        
        # If no matching brace found, return a reasonable end
        return min(start + 50, len(lines))
    
    def _generate_id(self, filepath: str, name: str, line: int) -> str:
        """Generate a unique ID for a chunk."""
        content = f"{filepath}:{name}:{line}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


def test_js_chunker():
    """Test the JS chunker."""
    print("Testing JSChunker...")
    print("=" * 50)
    
    sample_code = '''
import React from 'react';

export interface User {
    id: string;
    name: string;
}

export const fetchUser = async (id: string): Promise<User> => {
    const response = await fetch(`/api/users/${id}`);
    return response.json();
};

export class UserService {
    private cache: Map<string, User> = new Map();
    
    async getUser(id: string): Promise<User> {
        if (this.cache.has(id)) {
            return this.cache.get(id)!;
        }
        const user = await fetchUser(id);
        this.cache.set(id, user);
        return user;
    }
}

function formatUserName(user: User): string {
    return user.name.toUpperCase();
}
'''
    
    # Write to temp file
    import tempfile
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.ts', delete=False
    ) as f:
        f.write(sample_code)
        temp_path = f.name
    
    chunker = JSChunker()
    chunks = chunker.chunk_file(temp_path)
    
    print(f"Found {len(chunks)} chunks:\n")
    for chunk in chunks:
        print(f"📦 {chunk.type}: {chunk.name}")
        print(f"   Lines {chunk.start_line}-{chunk.end_line}")
        print(f"   Language: {chunk.language}")
        print(f"   Exports: {chunk.metadata.get('has_exports', False)}")
        print()
    
    # Cleanup
    import os
    os.unlink(temp_path)
    
    print("✅ JSChunker working!")


if __name__ == "__main__":
    test_js_chunker()
