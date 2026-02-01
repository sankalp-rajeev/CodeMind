"""
RefactoringCrew for CodeMind AI

A multi-agent crew specializing in code refactoring.
Agents work sequentially to analyze and improve code.

READ-ONLY: Agents access and analyze code but do NOT modify files.
They output proposed changes in their response; the user reviews and
decides whether to apply them.

NOTE: Ollama models have limited tool/function calling support in CrewAI.
We pre-load file content and RAG results into task descriptions so agents
receive the code directly without needing to invoke tools.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Generator, Tuple
from crewai import Agent, Task, Crew, Process
from crewai import LLM

from .tools import get_tools


def create_ollama_llm(model: str = "deepseek-coder:6.7b") -> LLM:
    """Create an Ollama LLM for CrewAI."""
    return LLM(
        model=f"ollama/{model}",
        base_url="http://localhost:11434"
    )


class RefactoringCrew:
    """
    Multi-agent crew for comprehensive code refactoring.
    
    Agents:
    1. Explorer - Understands the code structure
    2. Security - Identifies security vulnerabilities
    3. Algorithm - Finds performance improvements
    4. Tester - Suggests test cases
    5. Documenter - Writes documentation
    """
    
    def __init__(
        self,
        retriever=None,
        model: str = "deepseek-coder:6.7b"
    ):
        """
        Initialize the RefactoringCrew.
        
        Args:
            retriever: HybridRetriever for RAG search
            model: Ollama model to use
        """
        self.retriever = retriever
        self.llm = create_ollama_llm(model)
        self.tools = get_tools(retriever)
        
        # Create agents
        self.explorer = self._create_explorer()
        self.security = self._create_security()
        self.algorithm = self._create_algorithm()
        self.tester = self._create_tester()
        self.documenter = self._create_documenter()
        
        self.agents = [
            self.explorer,
            self.security,
            self.algorithm,
            self.tester,
            self.documenter
        ]
    
    def _create_explorer(self) -> Agent:
        """Create the Code Explorer agent."""
        return Agent(
            role="Code Explorer",
            goal="Understand the code structure, dependencies, and purpose",
            backstory="""You are an expert software architect who excels at 
            understanding complex codebases. You trace function calls, identify 
            design patterns, and explain code clearly.
            
            IMPORTANT: You have READ-ONLY access to files. You may read and analyze 
            code but must NOT modify or write any files. Output your analysis and 
            proposed changes in your response - the user will review and decide.
            
            ANALYSIS PRINCIPLES:
            - Only describe what you can actually see in the provided code
            - If code is truncated, acknowledge what you cannot see
            - Be specific: cite function names, line numbers, and actual code snippets
            - If you're uncertain about something, say "I'm not certain, but..."
            """,
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
    
    def _create_security(self) -> Agent:
        """Create the Security Analyst agent."""
        return Agent(
            role="Security Analyst",
            goal="Identify ONLY verifiable security vulnerabilities with evidence",
            backstory="""You are a security expert specializing in code security.
            
            CRITICAL ANTI-HALLUCINATION RULES:
            1. ONLY report vulnerabilities you can PROVE exist in the provided code
            2. For EACH finding, you MUST cite:
               - The exact line number or function name
               - The actual vulnerable code snippet
               - Why specifically this code is vulnerable
            3. If you cannot point to specific code, DO NOT report the issue
            
            VERIFICATION CHECKLIST (check before reporting):
            - SQL Injection: Is there ANY SQL/database code? If no SQL imports or queries, do NOT mention SQL injection
            - XSS: Is there ANY HTML output or web rendering? If no HTML/web code, do NOT mention XSS
            - CSRF: Is there ANY web form handling? If not a web app, do NOT mention CSRF
            - Auth issues: Is there ANY authentication code? If none, do NOT mention auth vulnerabilities
            
            ABSTENTION: If the code has no security issues, say "No security vulnerabilities found in the provided code."
            It is BETTER to report nothing than to report false positives.
            
            FOCUS ON WHAT'S ACTUALLY IN THE CODE:
            - Hardcoded secrets/API keys (look for actual strings)
            - Unsafe file operations (look for actual file I/O)
            - Input validation gaps (look for actual user input handling)
            - Dangerous eval/exec usage (look for actual eval calls)
            """,
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
    
    def _create_algorithm(self) -> Agent:
        """Create the Algorithm Optimizer agent."""
        return Agent(
            role="Algorithm Optimizer",
            goal="Find SPECIFIC, PROVABLE performance improvements",
            backstory="""You are a performance engineer who optimizes code for 
            speed and efficiency.
            
            EVIDENCE-BASED ANALYSIS:
            1. For EACH performance issue, cite the exact code location
            2. Explain the current complexity (e.g., "This loop at line X is O(n^2) because...")
            3. Show the specific improvement with actual code
            
            VERIFICATION BEFORE REPORTING:
            - Can you point to the exact line causing the issue?
            - Can you explain WHY it's inefficient with specifics?
            - Is your proposed fix actually better? Show the complexity improvement
            
            DO NOT:
            - Suggest generic optimizations without evidence
            - Claim "memory leaks" without pointing to specific unreleased resources
            - Suggest "caching" without identifying specific repeated computations
            
            ABSTENTION: If the code is already reasonably efficient, say so.
            "The code appears efficient for its purpose. No major optimizations needed."
            """,
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
    
    def _create_tester(self) -> Agent:
        """Create the Test Engineer agent."""
        return Agent(
            role="Test Engineer",
            goal="Suggest test cases based on ACTUAL functions in the code",
            backstory="""You are a QA engineer who designs test cases.
            
            GROUNDED TEST DESIGN:
            1. Only write tests for functions that ACTUALLY EXIST in the provided code
            2. Use the ACTUAL function signatures (parameters, return types)
            3. Reference ACTUAL edge cases based on the code logic you can see
            
            FOR EACH TEST:
            - Name the specific function being tested
            - Use realistic inputs based on what the function actually accepts
            - Expected outputs should be based on actual code logic
            
            DO NOT:
            - Invent functions that don't exist
            - Assume database/API calls unless you see them
            - Write tests for imaginary features
            
            FORMAT: Provide pytest-compatible test code with clear docstrings
            """,
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
    
    def _create_documenter(self) -> Agent:
        """Create the Documentation Writer agent."""
        return Agent(
            role="Documentation Writer",
            goal="Write accurate documentation based on actual code",
            backstory="""You are a technical writer who creates clear, 
            comprehensive documentation.
            
            ACCURACY PRINCIPLES:
            1. Document what the code ACTUALLY does, not what you assume
            2. Use the ACTUAL parameter names and types from the code
            3. Describe ACTUAL return values based on the code
            4. If behavior is unclear, say "Based on the code, this appears to..."
            
            DOCUMENTATION FORMAT:
            - Docstrings: Follow Google/NumPy style with Args, Returns, Raises
            - Include actual examples using real function signatures
            - Note any assumptions or limitations you observe
            
            DO NOT:
            - Invent parameters that don't exist
            - Describe features not present in the code
            - Add documentation for imaginary error handling
            """,
            tools=self.tools,
            llm=self.llm,
            verbose=True
        )
    
    def run(
        self,
        target: str,
        focus: Optional[str] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run the refactoring crew via generator (standard interface).
        
        Yields progress updates.
        """
        tasks = self._create_tasks(target, focus)
        agents = self.agents if not focus else self._get_focused_agents(focus)
        
        yield {
            "type": "crew_start",
            "crew_name": "RefactoringCrew",
            "target": target,
            "focus": focus,
            "agents": [a.role for a in agents]
        }
        
        # Create crew
        crew = Crew(
            tracing=os.environ.get("CREWAI_TRACING_ENABLED", "false").lower() == "true",
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Run the crew (Blocking in current CrewAI version)
        result = crew.kickoff()
        
        yield {
            "type": "crew_complete",
            "success": True,
            "result": str(result),
            "target": target,
            "focus": focus,
            "tasks_completed": len(tasks)
        }

    def refactor(
        self,
        target: str,
        focus: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run the refactoring crew on a target (Synchronous wrapper).
        Returns individual agent outputs for detailed analysis.
        """
        tasks = self._create_tasks(target, focus)
        agents = self.agents if not focus else self._get_focused_agents(focus)
        
        # Create crew
        crew = Crew(
            tracing=os.environ.get("CREWAI_TRACING_ENABLED", "false").lower() == "true",
            agents=agents,
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Run the crew
        result = crew.kickoff()
        
        # Extract individual task outputs
        agent_outputs = []
        agent_names = ["Code Explorer", "Security Analyst", "Algorithm Optimizer", "Test Engineer", "Documentation Writer"]
        
        if hasattr(result, 'tasks_output') and result.tasks_output:
            for i, task_output in enumerate(result.tasks_output):
                agent_name = agent_names[i] if i < len(agent_names) else f"Agent {i+1}"
                if hasattr(task_output, 'agent') and task_output.agent:
                    agent_name = task_output.agent
                output_text = str(task_output.raw) if hasattr(task_output, 'raw') else str(task_output)
                agent_outputs.append({
                    "agent": agent_name,
                    "output": output_text
                })
        else:
            # Fallback: single result
            agent_outputs.append({
                "agent": "RefactoringCrew",
                "output": str(result)
            })
        
        return {
            "target": target,
            "focus": focus,
            "result": str(result),
            "tasks_completed": len(tasks),
            "agent_outputs": agent_outputs
        }
    
    def _create_tasks(
        self,
        target: str,
        focus: Optional[str]
    ) -> List[Task]:
        """Create tasks for the crew."""
        tasks = []
        
        # Pre-load code content (Ollama has limited tool calling - we inject code directly)
        code_content, source_desc = self._load_target_content(target)
        
        if code_content:
            # Code pre-loaded - inject directly into task (no tools needed)
            lang = "python" if ".py" in source_desc else ("typescript" if ".ts" in source_desc else "javascript" if ".js" in source_desc else "")
            code_block = f"```{lang}\n{code_content}\n```" if lang else f"```\n{code_content}\n```"
            explore_desc = f"""Analyze the following code. The code has been loaded for you - it is provided below.

SOURCE: {source_desc}

CODE TO ANALYZE:
{code_block}

Provide a clear summary of:
            1. What the code does
            2. Key functions and classes involved
            3. Dependencies and imports
            4. Overall structure"""
        else:
            # No pre-loaded content - agent must use tools (may fail with Ollama)
            explore_desc = f"""Analyze the code related to: {target}

Use the search_code tool to find relevant code. Then provide:
1. What the code does
2. Key functions and classes involved
3. Dependencies and imports
4. Overall structure

Include any code you find so downstream agents can refactor it."""
        
        explore_task = Task(
            description=explore_desc,
            agent=self.explorer,
            expected_output="Analysis: what the code does, key functions/classes, structure"
        )
        tasks.append(explore_task)
        
        # Inject code into downstream tasks so they get it directly (Explorer may output placeholders)
        code_injection = ""
        if code_content:
            lang = "python" if ".py" in source_desc else ("typescript" if ".ts" in source_desc else "javascript" if ".js" in source_desc else "")
            code_injection = f"""

=== FULL SOURCE CODE (from {source_desc}) - REFACTOR THIS ===
```{lang}
{code_content}
```
==="""
        
        if not focus or focus == "security":
            security_task = Task(
                description=f"""Based on the code analysis above, identify security issues.{code_injection}
                
                Look for:
                - Input validation issues
                - Authentication/authorization problems
                - Data exposure risks
                - Injection vulnerabilities
                
                You MUST provide specific fixes with actual code. For each fix:
                1. Show the BEFORE code (vulnerable)
                2. Show the AFTER code (fixed) in a ```python ... ``` block
                Do NOT give only explanations - include concrete code changes.
                Do NOT modify any files - only output your proposed changes. The user will review and apply if desired.""",
                agent=self.security,
                expected_output="Security issues with BEFORE/AFTER code fixes in code blocks",
                context=[explore_task]
            )
            tasks.append(security_task)
        
        if not focus or focus == "performance":
            algo_intro = "REFACTOR this code for performance. The full source is below:" if code_injection else "REFACTOR the code from the Explorer's analysis above for performance."
            algorithm_task = Task(
                description=f"""{algo_intro}{code_injection}
                
                Look for:
                - Algorithm complexity (O(n^2) -> O(n))
                - Unnecessary loops or duplicate work
                - Memory inefficiencies
                - Redundant chunking (index_codebase chunks twice - fix this)
                - Caching opportunities
                
                OUTPUT: Output the COMPLETE refactored Python file in a code block.
                - Write the FULL modified code in ```python ... ```
                - Do NOT modify any actual files - only output your proposed code in this response
                - The user will review and decide whether to apply your changes
                - Preserve all functionality""",
                agent=self.algorithm,
                expected_output="Complete refactored Python file in a code block",
                context=[explore_task]
            )
            tasks.append(algorithm_task)
        
        if not focus or focus == "tests":
            test_task = Task(
                description=f"""Based on the code analysis above, suggest test cases.{code_injection}
                
                Include:
                - Unit test cases for each function
                - Edge cases and boundary conditions
                - Error handling scenarios
                
                Provide test code (pytest format) and expected outcomes.""",
                agent=self.tester,
                expected_output="Test code in pytest format with descriptions",
                context=[explore_task]
            )
            tasks.append(test_task)
        
        if not focus or focus == "docs":
            # Only use explorer context to avoid massive input that hangs Ollama
            doc_context = [explore_task]
            
            doc_task = Task(
                description=f"""Based on the code analysis and any refactoring above, write documentation.{code_injection}
                
                Provide:
                - Docstrings for functions/classes
                - README section for this functionality
                - Inline comments for complex logic
                - Usage examples
                
                Format in standard Python style.""",
                agent=self.documenter,
                expected_output="Complete documentation for the analyzed code",
                context=doc_context
            )
            tasks.append(doc_task)
        
        return tasks
    
    def _load_target_content(self, target: str) -> Tuple[Optional[str], str]:
        """
        Pre-load code content for the target (bypasses tool calling).
        Ollama models have limited tool support in CrewAI - we inject code directly.
        
        Supports:
        - Single files: "src/rag/retriever.py"
        - Directories: "src/rag/" (loads all code files in directory)
        
        Returns (content, source_description).
        content is None if we couldn't load anything.
        """
        import re
        target = target.strip()
        
        project_root = Path(__file__).parent.parent.parent
        bases = [
            Path.cwd(),
            project_root,
            project_root / "data" / "Carla-Autonomous-Vehicle",
            project_root / "data" / "Carla-Autonomous-Vehicle" / "carla_simulation code",
        ]
        
        CODE_EXTENSIONS = ('.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.go', '.rs')
        MAX_TOTAL_CHARS = 15000  # Limit total content to avoid overwhelming Ollama
        
        # Check if target is a directory
        for base in bases:
            candidate = (base / target).resolve()
            if candidate.exists() and candidate.is_dir():
                # Load all code files from directory
                files_content = []
                total_chars = 0
                
                for file_path in sorted(candidate.rglob('*')):
                    if file_path.is_file() and file_path.suffix in CODE_EXTENSIONS:
                        try:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            rel_path = file_path.relative_to(candidate)
                            
                            # Truncate individual files if needed
                            if len(content) > 4000:
                                content = content[:4000] + "\n\n# ... truncated ..."
                            
                            if total_chars + len(content) > MAX_TOTAL_CHARS:
                                files_content.append(f"\n# ... additional files truncated for size ...")
                                break
                            
                            lang = "python" if file_path.suffix == '.py' else file_path.suffix[1:]
                            files_content.append(f"### FILE: {rel_path}\n```{lang}\n{content}\n```")
                            total_chars += len(content)
                        except Exception:
                            continue
                
                if files_content:
                    return "\n\n".join(files_content), f"Directory: {target} ({len(files_content)} files)"
        
        # Extract file path from target (e.g. "src/rag/retriever.py" or "retriever.py")
        file_path = None
        match = re.search(r'[\w./\\-]+\.(py|js|ts|jsx|tsx)', target)
        if match:
            file_path = match.group(0)
        elif any(target.endswith(ext) for ext in CODE_EXTENSIONS):
            file_path = target
        
        # Try reading single file
        if file_path:
            for base in bases:
                candidate = (base / file_path).resolve()
                if candidate.exists() and candidate.is_file():
                    try:
                        content = candidate.read_text(encoding='utf-8', errors='ignore')
                        return content, str(candidate)
                    except Exception as e:
                        return None, f"Could not read: {e}"
        
        # Fallback: RAG search if retriever available
        if self.retriever:
            try:
                results = self.retriever.search(target, n_results=5)
                if results:
                    parts = []
                    for i, r in enumerate(results, 1):
                        name = r.get('metadata', {}).get('name') or r.get('name', 'Unknown')
                        filepath = r.get('metadata', {}).get('filepath') or r.get('filepath', 'Unknown')
                        content_slice = (r.get('content') or '')[:600]
                        parts.append(f"[{i}] {name} ({filepath}):\n```\n{content_slice}\n```")
                    return "\n\n".join(parts), f"RAG: {target[:50]}"
            except Exception:
                pass
        
        return None, target
    
    def _get_focused_agents(self, focus: str) -> List[Agent]:
        """Get agents for a focused analysis."""
        focus_map = {
            "security": [self.explorer, self.security],
            "performance": [self.explorer, self.algorithm],
            "tests": [self.explorer, self.tester],
            "docs": [self.explorer, self.documenter]
        }
        return focus_map.get(focus, self.agents)


def test_refactoring_crew():
    """Test the RefactoringCrew."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.rag.retriever import HybridRetriever
    
    print("Testing RefactoringCrew...")
    print("=" * 60)
    
    # Initialize retriever
    retriever = HybridRetriever(
        persist_directory="./data/chroma_db_test",
        use_reranker=False
    )
    
    # Create crew
    crew = RefactoringCrew(retriever=retriever)
    print(f"Created crew with {len(crew.agents)} agents")
    
    # Run a focused analysis
    print("\nRunning security-focused analysis on 'data loading'...")
    result = crew.refactor("data loading", focus="security")
    
    print(f"\nResult:\n{result['result'][:500]}...")
    print("\nRefactoringCrew test complete!")


if __name__ == "__main__":
    test_refactoring_crew()
