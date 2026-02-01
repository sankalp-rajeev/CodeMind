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
        """Create the review crew agents with anti-hallucination prompting."""
        
        self.security_reviewer = Agent(
            role="Security Analyst",
            goal="Analyze code using structured output format",
            backstory="""You are a security analyst who follows a STRICT structured process.

            YOUR METHOD:
            1. First, LIST all functions/classes you see (creates ground truth)
            2. Then, LIST all imports you see
            3. Answer specific YES/NO questions with code citations
            4. Only report findings based on YES answers
            
            CRITICAL RULES:
            - You can ONLY reference code you listed in steps 1-2
            - If you didn't list it, you cannot claim it exists
            - NO is the safe default answer
            - "No security vulnerabilities identified" is a valid and good answer
            
            You value ACCURACY over finding issues. It is better to report nothing
            than to invent fake vulnerabilities.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.performance_reviewer = Agent(
            role="Performance Engineer",
            goal="Analyze code using structured output format",
            backstory="""You are a performance engineer who follows a STRICT structured process.

            YOUR METHOD:
            1. First, LIST all functions you see (creates ground truth)
            2. For each function, count and note loops
            3. Answer specific YES/NO questions
            4. Only report findings based on YES answers
            
            CRITICAL RULES:
            - You can ONLY reference functions you listed in step 1
            - If you didn't list it, you cannot claim it exists
            - NO is the safe default answer
            - "No significant performance issues" is a valid and good answer
            
            You value ACCURACY over finding issues.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.test_reviewer = Agent(
            role="Test Coverage Analyst",
            goal="Assess test coverage based on ACTUAL code structure",
            backstory="""You analyze what tests would be valuable.

            GROUNDING RULES:
            1. Only suggest tests for functions that EXIST in the code
            2. Use ACTUAL function names and signatures
            3. Base edge cases on ACTUAL code logic you can see
            
            FORMAT for each suggestion:
            - Function: [actual name from code]
            - Test case: [specific scenario]
            - Why: [based on actual code logic]
            
            DO NOT invent functions or features not present.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.docs_reviewer = Agent(
            role="Documentation Reviewer",
            goal="Assess documentation based on ACTUAL code",
            backstory="""You review documentation quality.

            ASSESSMENT RULES:
            1. Check if existing docstrings match actual parameters
            2. Note missing docs for functions that ACTUALLY exist
            3. Suggest improvements based on actual code complexity
            
            DO NOT:
            - Suggest documenting features that don't exist
            - Criticize missing docs for functions not in the code
            
            If well-documented: "Documentation is adequate for the code provided." """,
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.synthesizer = Agent(
            role="Review Synthesizer",
            goal="Create verified, evidence-based unified report",
            backstory="""You synthesize reviews into a prioritized report.

            SYNTHESIS VERIFICATION:
            1. Only include findings that cite specific code
            2. Remove any finding that lacks evidence
            3. If a reviewer reported "no issues," reflect that honestly
            
            CHAIN-OF-VERIFICATION:
            Before including each finding, verify:
            - Is there a specific code reference?
            - Does the issue match what's actually in the code?
            - Remove generic advice that doesn't apply
            
            OUTPUT FORMAT:
            - Prioritized issues with severity
            - Each issue must have code evidence
            - Acknowledge areas with no issues found""",
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
            description=f"""Analyze this code for security vulnerabilities using STRUCTURED OUTPUT.

{code_block}

YOU MUST FOLLOW THIS EXACT FORMAT:

## STEP 1: CODE INVENTORY
List every function/class name you see in the code above:
- [function_name_1]
- [function_name_2]
- ...

## STEP 2: IMPORT CHECK
List all import statements from the code:
- [import statement 1]
- [import statement 2]
- ...
Or write "No imports" if none.

## STEP 3: SECURITY CHECKLIST
Answer each question with YES or NO, then quote the specific code if YES:

Q1: Are there database libraries imported (sqlite3, pymysql, psycopg2, sqlalchemy)?
A1: [YES/NO]. Code: [quote the import line or "N/A"]

Q2: Is there any eval() or exec() call?
A2: [YES/NO]. Code: [quote the line or "N/A"]

Q3: Are there hardcoded strings that look like API keys/passwords (long random strings)?
A3: [YES/NO]. Code: [quote the line or "N/A"]

Q4: Is there file I/O with user-controlled paths (open, Path with variables)?
A4: [YES/NO]. Code: [quote the line or "N/A"]

## STEP 4: FINDINGS
Based ONLY on YES answers in Step 3:
- If all answers are NO: "No security vulnerabilities identified."
- If any YES: Describe the specific vulnerability with the quoted code.

DO NOT skip steps. DO NOT invent code. Only reference what you listed in Steps 1-2.""",
            expected_output="Structured security analysis following the 4-step format",
            agent=self.security_reviewer,
            async_execution=True  # Enable parallel
        )
        
        performance_task = Task(
            description=f"""Analyze this code for performance issues using STRUCTURED OUTPUT.

{code_block}

YOU MUST FOLLOW THIS EXACT FORMAT:

## STEP 1: CODE INVENTORY
List every function name you see:
- [function_name_1]
- [function_name_2]
- ...

## STEP 2: LOOP ANALYSIS
For each function, count loops:
- [function_name]: [0/1/2+] loops. Nested: [YES/NO]

## STEP 3: PERFORMANCE CHECKLIST
Q1: Are there nested loops (loop inside loop)?
A1: [YES/NO]. Location: [function name or "N/A"]

Q2: Are there database queries inside loops?
A2: [YES/NO]. Location: [function name or "N/A"]

Q3: Are there repeated identical computations?
A3: [YES/NO]. Location: [function name or "N/A"]

## STEP 4: FINDINGS
Based ONLY on YES answers in Step 3:
- If all NO: "No significant performance issues identified."
- If any YES: Describe with the specific function name and issue.

DO NOT invent issues. Only reference functions you listed in Step 1.""",
            expected_output="Structured performance analysis following the 4-step format",
            agent=self.performance_reviewer,
            async_execution=True
        )
        
        test_task = Task(
            description=f"""TEST COVERAGE ANALYSIS TASK - You are a Test Coverage Analyst.

{code_block}

INSTRUCTIONS:
1. List the ACTUAL functions/classes in this code
2. Suggest tests ONLY for functions that EXIST above
3. Use the ACTUAL function signatures (parameter names, types)

FIRST: List all functions you see in the code:
- Function name, parameters, return type

THEN: For each function, suggest test cases:
- Function: [actual name from code]
- Test: [specific scenario]
- Input: [realistic based on actual parameters]
- Expected: [based on actual code logic]

DO NOT:
- Invent functions not in the code
- Assume database/API calls unless you see them
- Write tests for imaginary features""",
            expected_output="List of actual functions found, then specific test suggestions for each",
            agent=self.test_reviewer,
            async_execution=True
        )
        
        docs_task = Task(
            description=f"""DOCUMENTATION REVIEW TASK - You are a Documentation Reviewer.

{code_block}

INSTRUCTIONS:
1. List ACTUAL functions/classes in the code
2. For each, note if it has a docstring (yes/no)
3. If yes, check if docstring matches actual parameters

FIRST: Inventory what exists:
- Function/Class name | Has docstring? | Parameters match?

THEN: Suggest improvements ONLY for functions that EXIST:
- Function: [actual name]
- Current state: [has/missing docstring]
- Suggestion: [specific improvement]

If well-documented:
  "Documentation is adequate. All functions have appropriate docstrings."

DO NOT suggest documenting functions that don't exist.""",
            expected_output="Inventory of actual functions with docstring status, then specific suggestions",
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
            description="""SYNTHESIS TASK - Combine reviews into a verified report.

VERIFICATION RULES (apply before including ANY finding):
1. Does the finding cite a specific line number or function name? If NO, exclude it.
2. Does the finding match code that actually exists? If uncertain, exclude it.
3. Is the finding generic advice without evidence? If YES, exclude it.

SECTIONS TO INCLUDE:

## Security
- Include findings WITH specific code references
- If security agent said "No vulnerabilities", report: "No security issues identified"

## Performance  
- Include findings WITH specific complexity analysis
- If performance agent said "No issues", report: "Code is efficient"

## Test Coverage
- Include ONLY tests for functions that exist in the code
- List actual functions found

## Documentation
- Include ONLY suggestions for real functions
- Note current docstring status

## Priority Actions
Only list items that have specific code evidence.

IMPORTANT: It is BETTER to have a short, accurate report than a long report with generic advice.
If agents found no issues, honestly report "No issues found in [area]".""",
            expected_output="Verified unified report with only evidence-based findings",
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
