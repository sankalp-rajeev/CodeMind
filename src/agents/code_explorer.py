"""
Code Explorer Agent for CodeMind AI

Specializes in code understanding and navigation.
Uses RAG to retrieve relevant code context and answer questions.
"""

from typing import List, Dict, Any, Optional, Generator
from .base import BaseAgent, AgentConfig

# System prompt for code exploration
CODE_EXPLORER_SYSTEM_PROMPT = """You are CodeExplorer, an expert software engineer specialized in understanding and explaining code.

Your capabilities:
- Explain what code does and how it works
- Trace function calls and dependencies
- Identify design patterns and architecture
- Answer questions about codebase structure

Guidelines:
- Be precise and technical when explaining code
- Reference specific functions, classes, and files
- If you're unsure, say so rather than guessing
- Keep explanations concise but thorough
- Use code snippets when helpful

When given CODE CONTEXT, use it to answer the user's question accurately.
"""

# Template for RAG-augmented prompts
RAG_PROMPT_TEMPLATE = """Use the following code context to answer the question.

=== CODE CONTEXT ===
{context}
==================

QUESTION: {question}

ANSWER:"""


class CodeExplorerAgent(BaseAgent):
    """
    Agent for exploring and understanding code.
    
    Features:
    - RAG-augmented responses
    - Code explanation
    - Dependency tracing
    - Design pattern identification
    """
    
    def __init__(
        self,
        retriever=None,
        model: str = "deepseek-coder:6.7b",
        n_context_chunks: int = 8
    ):
        """
        Initialize the Code Explorer agent.
        
        Args:
            retriever: HybridRetriever instance for RAG
            model: Ollama model to use
            n_context_chunks: Number of code chunks to retrieve
        """
        config = AgentConfig(
            model=model,
            temperature=0.3,  # Lower temp for factual responses
            max_tokens=2048,
            system_prompt=CODE_EXPLORER_SYSTEM_PROMPT
        )
        super().__init__(config)
        
        self.retriever = retriever
        self.n_context_chunks = n_context_chunks
    
    def answer(
        self,
        question: str,
        use_rag: bool = True,
        stream: bool = False,
        conversation_history: Optional[str] = None
    ) -> str | Generator[str, None, None]:
        """
        Answer a question about the codebase.
        
        Args:
            question: User's question
            use_rag: Whether to use RAG for context
            stream: Whether to stream the response
            conversation_history: Optional previous conversation context
            
        Returns:
            Answer string or generator for streaming
        """
        if conversation_history:
            question = f"{conversation_history}\n\nCurrent question: {question}"
        
        if use_rag and self.retriever:
            # Retrieve relevant code context
            context = self._get_context(question)
            prompt = RAG_PROMPT_TEMPLATE.format(
                context=context,
                question=question
            )
        else:
            prompt = question
        
        return self.generate(prompt, stream=stream)
    
    def explain_function(self, function_name: str) -> str:
        """
        Explain what a specific function does.
        
        Args:
            function_name: Name of the function to explain
            
        Returns:
            Explanation of the function
        """
        question = f"Explain what the function '{function_name}' does, including its parameters, return value, and purpose."
        return self.answer(question)
    
    def trace_dependencies(self, function_name: str) -> str:
        """
        Trace the dependencies of a function.
        
        Args:
            function_name: Name of the function
            
        Returns:
            Description of dependencies
        """
        question = f"What functions and classes does '{function_name}' depend on? Trace its call graph."
        return self.answer(question)
    
    def find_similar_code(self, description: str) -> List[Dict]:
        """
        Find code similar to a description.
        
        Args:
            description: Natural language description
            
        Returns:
            List of matching code chunks
        """
        if not self.retriever:
            return []
        
        return self.retriever.search(description, n_results=self.n_context_chunks)
    
    def _get_context(self, question: str) -> str:
        """
        Retrieve relevant code context for a question.
        
        Args:
            question: The question to get context for
            
        Returns:
            Formatted context string
        """
        if not self.retriever:
            return "No code context available."
        
        # Search for relevant chunks
        results = self.retriever.search(
            question, 
            n_results=self.n_context_chunks
        )
        
        if not results:
            return "No relevant code found."
        
        # Format context
        context_parts = []
        for i, result in enumerate(results):
            # Get metadata
            metadata = result.get('metadata', {})
            name = metadata.get('name') or result.get('name', 'Unknown')
            filepath = metadata.get('filepath') or result.get('filepath', 'Unknown')
            chunk_type = metadata.get('type') or result.get('type', 'code')
            
            # Get content
            content = result.get('content', '')[:800]  # Limit length
            
            context_parts.append(f"""
[{i+1}] {chunk_type.upper()}: {name}
File: {filepath}
```
{content}
```
""")
        
        return "\n".join(context_parts)


def test_code_explorer():
    """Test the Code Explorer agent."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.rag.retriever import HybridRetriever
    
    print("Testing Code Explorer Agent...")
    print("=" * 50)
    
    # Initialize retriever
    retriever = HybridRetriever(
        persist_directory="./data/chroma_db_test",
        use_reranker=False
    )
    
    # Make sure we have indexed data
    if retriever.indexer.collection.count() == 0:
        print("Indexing test repository...")
        retriever.index_codebase("./data/test-repo", force_reindex=True)
    
    # Initialize agent
    agent = CodeExplorerAgent(
        retriever=retriever,
        model="deepseek-coder:6.7b"
    )
    
    # Test a question
    question = "How does the dataset loading work in this project?"
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    response = agent.answer(question)
    print(f"\nAnswer:\n{response[:500]}...")
    
    print("\n✅ Code Explorer Agent working!")


if __name__ == "__main__":
    test_code_explorer()
