"""
Custom Tools for CrewAI Agents

Provides tools that agents can use to interact with the codebase:
- RAG search
- File reading
- Code analysis
"""

from typing import Type, Optional
from pydantic import BaseModel, Field

try:
    from crewai.tools import BaseTool
    CREWAI_AVAILABLE = True
except ImportError:
    CREWAI_AVAILABLE = False
    # Fallback base class
    class BaseTool:
        pass


class SearchCodeInput(BaseModel):
    """Input for searching the codebase."""
    query: str = Field(description="Natural language query to search for relevant code")
    n_results: int = Field(default=5, description="Number of results to return")


class SearchCodeTool(BaseTool):
    """
    Tool for searching the codebase using hybrid RAG.
    
    Uses vector + BM25 search to find relevant code chunks.
    """
    name: str = "search_code"
    description: str = "Search the codebase for code related to a query. Returns relevant code snippets with file paths."
    args_schema: Type[BaseModel] = SearchCodeInput
    _retriever: Optional[object] = None
    
    def __init__(self, retriever=None, **kwargs):
        super().__init__(**kwargs)
        object.__setattr__(self, '_retriever', retriever)
    
    def _run(self, query: str, n_results: int = 5) -> str:
        """Execute the search."""
        if self._retriever is None:
            return "Error: No retriever configured"
        
        results = self._retriever.search(query, n_results=n_results)
        
        if not results:
            return "No relevant code found."
        
        output = []
        for i, r in enumerate(results):
            name = r.get('metadata', {}).get('name') or r.get('name', 'Unknown')
            filepath = r.get('metadata', {}).get('filepath') or r.get('filepath', 'Unknown')
            chunk_type = r.get('metadata', {}).get('type') or r.get('type', 'code')
            content = r.get('content', '')[:500]
            
            output.append(f"[{i+1}] {chunk_type}: {name}\nFile: {filepath}\n```\n{content}\n```\n")
        
        return "\n".join(output)


class ReadFileInput(BaseModel):
    """Input for reading a file."""
    filepath: str = Field(description="Path to the file to read")
    start_line: int = Field(default=1, description="Starting line number")
    end_line: int = Field(default=100, description="Ending line number")


class ReadFileTool(BaseTool):
    """
    Tool for reading file contents.
    
    Returns the contents of a file, optionally limited to specific lines.
    """
    name: str = "read_file"
    description: str = "Read the contents of a file. Specify start and end lines to read a portion."
    args_schema: Type[BaseModel] = ReadFileInput
    
    def _run(self, filepath: str, start_line: int = 1, end_line: int = 100) -> str:
        """Read the file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Adjust line numbers (1-indexed)
            start_idx = max(0, start_line - 1)
            end_idx = min(len(lines), end_line)
            
            selected_lines = lines[start_idx:end_idx]
            
            if not selected_lines:
                return f"No content in lines {start_line}-{end_line}"
            
            # Add line numbers
            numbered = []
            for i, line in enumerate(selected_lines, start=start_line):
                numbered.append(f"{i:4d} | {line.rstrip()}")
            
            return f"File: {filepath}\n" + "\n".join(numbered)
            
        except FileNotFoundError:
            return f"Error: File not found: {filepath}"
        except Exception as e:
            return f"Error reading file: {e}"


class AnalyzeCodeInput(BaseModel):
    """Input for code analysis."""
    code: str = Field(description="The code to analyze")
    analysis_type: str = Field(
        default="general",
        description="Type of analysis: 'general', 'security', 'performance', 'style'"
    )


class AnalyzeCodeTool(BaseTool):
    """
    Tool for analyzing code snippets.
    
    Provides structured analysis of code for various concerns.
    """
    name: str = "analyze_code"
    description: str = "Analyze a code snippet for issues. Specify analysis type: general, security, performance, or style."
    args_schema: Type[BaseModel] = AnalyzeCodeInput
    
    def _run(self, code: str, analysis_type: str = "general") -> str:
        """Analyze the code."""
        # Simple static analysis
        issues = []
        
        if analysis_type in ["general", "security"]:
            # Security checks
            if "eval(" in code:
                issues.append("SECURITY: Use of eval() detected - potential code injection risk")
            if "pickle.load" in code:
                issues.append("SECURITY: pickle.load() can execute arbitrary code")
            if "subprocess" in code and "shell=True" in code:
                issues.append("SECURITY: shell=True in subprocess can enable shell injection")
            if "password" in code.lower() and "=" in code:
                issues.append("SECURITY: Possible hardcoded password")
        
        if analysis_type in ["general", "performance"]:
            # Performance checks
            if "for" in code and "append" in code:
                issues.append("PERFORMANCE: Consider list comprehension instead of loop with append")
            if ".read()" in code and "for line in" not in code:
                issues.append("PERFORMANCE: Reading entire file into memory - consider iterating lines")
        
        if analysis_type in ["general", "style"]:
            # Style checks
            lines = code.split('\n')
            long_lines = [i+1 for i, l in enumerate(lines) if len(l) > 100]
            if long_lines:
                issues.append(f"STYLE: Lines exceeding 100 chars: {long_lines[:5]}")
            if "TODO" in code or "FIXME" in code:
                issues.append("STYLE: Contains TODO/FIXME comments")
        
        if not issues:
            return f"No issues found in {analysis_type} analysis."
        
        return f"Analysis ({analysis_type}):\n" + "\n".join(f"- {i}" for i in issues)


def get_tools(retriever=None):
    """Get all tools for CrewAI agents."""
    return [
        SearchCodeTool(retriever=retriever),
        ReadFileTool(),
        AnalyzeCodeTool()
    ]
