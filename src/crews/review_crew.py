"""
CodeReviewCrew for CodeMind AI

Implements a PARALLEL review workflow:
1. Security review
2. Performance review  
3. Test coverage review
4. Documentation review

All 4 reviews run in parallel, then a Synthesis agent combines them.

This is a key agentic pattern: parallel execution with synthesis.
"""

import os
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process

from .refactoring_crew import create_ollama_llm


@dataclass
class ReviewResult:
    """Result of a code review."""
    category: str
    severity: str  # critical, high, medium, low
    issues: List[str]
    suggestions: List[str]


class CodeReviewCrew:
    """
    Crew that performs parallel code reviews and synthesizes results.
    
    This implements the PARALLEL EXECUTION pattern:
    - 4 specialized reviewers run simultaneously
    - Results are collected and synthesized
    - Unified report with prioritized issues
    """
    
    def __init__(self, model: str = "deepseek-coder:6.7b"):
        self.llm = create_ollama_llm(model)
        self._create_agents()
    
    def _create_agents(self):
        """Create the review crew agents."""
        
        self.security_reviewer = Agent(
            role="Security Analyst",
            goal="Identify security vulnerabilities and risks",
            backstory="""You are a security expert who finds vulnerabilities
            like SQL injection, XSS, hardcoded secrets, and unsafe patterns.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.performance_reviewer = Agent(
            role="Performance Engineer",
            goal="Identify performance issues and optimizations",
            backstory="""You are a performance expert who identifies
            inefficient algorithms, memory leaks, N+1 queries, and bottlenecks.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.test_reviewer = Agent(
            role="Test Coverage Analyst",
            goal="Assess test coverage and quality",
            backstory="""You analyze test coverage, identify untested paths,
            and suggest additional test cases for edge cases.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.docs_reviewer = Agent(
            role="Documentation Reviewer",
            goal="Assess documentation quality and completeness",
            backstory="""You review docstrings, comments, and README files
            to ensure code is well-documented and understandable.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.synthesizer = Agent(
            role="Review Synthesizer",
            goal="Combine all reviews into unified report",
            backstory="""You synthesize multiple review perspectives into
            a prioritized action plan with severity ratings.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
    
    def run(
        self,
        code: str,
        file_path: str,
        progress_callback: Optional[callable] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run parallel code review workflow.
        
        Yields progress updates for streaming to frontend.
        """
        yield {
            "type": "crew_start",
            "crew_name": "CodeReviewCrew",
            "agents": ["Security", "Performance", "Tests", "Docs", "Synthesis"]
        }
        
        # Create parallel review tasks - be explicit that code is provided in the message
        code_block = f"""
=== CODE TO REVIEW (file: {file_path}) ===

{code}

=== END OF CODE ===
"""
        security_task = Task(
            description=f"""The Python code to analyze is provided below. You have full access to it in this message. Analyze it directly.

Review the code for security issues:

{code_block}

Look for:
- SQL injection risks
- XSS vulnerabilities
- Hardcoded secrets/credentials
- Unsafe file operations
- Input validation issues

Rate severity: critical, high, medium, low""",
            expected_output="Security review with issues and severity ratings",
            agent=self.security_reviewer,
            async_execution=True  # Enable parallel
        )
        
        performance_task = Task(
            description=f"""The Python code to analyze is provided below. You have full access to it in this message. Analyze it directly.

Review the code for performance issues:

{code_block}

Look for:
- Inefficient algorithms (O(n²) or worse)
- Memory leaks
- Unnecessary loops
- N+1 query patterns
- Blocking operations

Rate severity: critical, high, medium, low""",
            expected_output="Performance review with issues and severity ratings",
            agent=self.performance_reviewer,
            async_execution=True
        )
        
        test_task = Task(
            description=f"""The Python code to analyze is provided below. You have full access to it in this message. Analyze it directly.

Review test coverage for this code:

{code_block}

Assess:
- Are critical paths tested?
- Are edge cases covered?
- Is error handling tested?
- Are there missing test cases?

Suggest specific tests to add.""",
            expected_output="Test coverage review with suggestions",
            agent=self.test_reviewer,
            async_execution=True
        )
        
        docs_task = Task(
            description=f"""The Python code to analyze is provided below. You have full access to it in this message. Analyze it directly.

Review documentation for this code:

{code_block}

Check:
- Are functions documented?
- Are parameters described?
- Are return values explained?
- Are complex logic sections commented?

Suggest documentation improvements.""",
            expected_output="Documentation review with suggestions",
            agent=self.docs_reviewer,
            async_execution=True
        )
        
        yield {
            "type": "phase",
            "phase": "parallel_reviews",
            "status": "running",
            "agents": ["Security", "Performance", "Tests", "Docs"]
        }
        
        # Synthesis task depends on all parallel reviews
        synthesis_task = Task(
            description="""Synthesize the code reviews from your fellow agents into a unified report.
The Security, Performance, Test, and Documentation agents have each analyzed the code. Combine their findings.

You MUST include sections for ALL four review dimensions:
1. **Security** - Combine security findings (vulnerabilities, risks)
2. **Performance** - Combine performance findings (bottlenecks, inefficiencies)
3. **Test Coverage** - Combine test coverage findings (missing tests, edge cases)
4. **Documentation** - Combine documentation findings (missing docstrings, comments)

Then create a prioritized action plan:
- List all issues by severity (critical → low)
- Group related issues
- Suggest order of fixes
- Provide estimated effort for each

Format as a clear, actionable report with all four sections.""",
            expected_output="Unified review report with prioritized actions",
            agent=self.synthesizer,
            context=[security_task, performance_task, test_task, docs_task]
        )
        
        # Create crew with hierarchical process
        crew = Crew(
            tracing=os.environ.get("CREWAI_TRACING_ENABLED", "false").lower() == "true",
            agents=[
                self.security_reviewer,
                self.performance_reviewer,
                self.test_reviewer,
                self.docs_reviewer,
                self.synthesizer
            ],
            tasks=[security_task, performance_task, test_task, docs_task, synthesis_task],
            process=Process.sequential,  # Tasks with async_execution run in parallel
            verbose=True
        )
        
        yield {
            "type": "phase",
            "phase": "synthesis",
            "status": "running"
        }
        
        # Execute crew
        result = crew.kickoff()
        
        yield {
            "type": "crew_complete",
            "success": True,
            "reviews": {
                "security": str(security_task.output)[:500] if security_task.output else "",
                "performance": str(performance_task.output)[:500] if performance_task.output else "",
                "tests": str(test_task.output)[:500] if test_task.output else "",
                "docs": str(docs_task.output)[:500] if docs_task.output else "",
            },
            "synthesis": str(result)
        }


def test_review_crew():
    """Test the CodeReviewCrew."""
    print("Testing CodeReviewCrew...")
    print("=" * 50)
    
    sample_code = '''
def get_user(user_id):
    """Get user from database."""
    query = f"SELECT * FROM users WHERE id = {user_id}"  # SQL injection!
    result = db.execute(query)
    return result

def process_items(items):
    """Process all items."""
    results = []
    for item in items:
        for i in range(len(items)):  # O(n²) performance issue
            results.append(item + items[i])
    return results

API_KEY = "sk-secret123"  # Hardcoded secret!
'''
    
    crew = CodeReviewCrew()
    
    for progress in crew.run(sample_code, "sample.py"):
        print(f"[{progress['type']}]", end=" ")
        if progress['type'] == 'phase':
            print(f"{progress['phase']}: {progress['status']}")
        elif progress['type'] == 'crew_complete':
            print("\n✅ Review Complete!")
            print(f"\nSynthesis:\n{progress['synthesis'][:500]}...")
    
    print("\n✅ CodeReviewCrew working!")


if __name__ == "__main__":
    test_review_crew()
