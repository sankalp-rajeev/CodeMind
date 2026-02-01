"""
TestingCrew for CodeMind AI

Implements an ITERATIVE testing workflow:
1. Generate tests for target code
2. Execute tests (subprocess pytest)
3. Analyze failures
4. Refine tests based on failures
5. Repeat until all pass or max iterations

This is a key agentic pattern: iterative refinement with feedback loop.
"""

from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from crewai import Agent, Task, Crew, Process
from crewai.tools import BaseTool

from .refactoring_crew import create_ollama_llm
import subprocess
import tempfile
import os
import re


@dataclass
class TestResult:
    """Result of running tests."""
    passed: int
    failed: int
    errors: int
    output: str
    coverage: float
    failure_details: List[str]


def _extract_python_code(text: str) -> str:
    """Extract Python code from markdown code blocks. Used for deterministic handoff."""
    if not text or not text.strip():
        return ""
    text = str(text)
    # Match ```python ... ``` or ``` ... ``` - prefer python blocks
    for pattern in [r"```python\s*\n(.*?)```", r"```\s*\n(.*?)```"]:
        matches = re.findall(pattern, text, re.DOTALL)
        valid = [m.strip() for m in matches if m.strip() and ("def " in m or "import " in m)]
        if valid:
            return max(valid, key=len)
    # Fallback: entire text if it looks like test code
    if "def test_" in text or "import pytest" in text:
        return text.strip()
    return ""


def _repair_target_path(path: str) -> str:
    """Repair path when LLM strips backslashes (e.g. C:Umich...Carla...environment_final.py)."""
    if not path or "\\" in path or "/" in path:
        return path
    # If path contains Carla markers but no separators, try known Carla file paths
    norm = path.replace("\\", "").replace("/", "").lower()
    if "carla" in norm and "environment_final" in norm:
        for root in [os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))]:
            candidate = os.path.join(root, "data", "Carla-Autonomous-Vehicle", "carla_simulation code", "environment_final.py")
            if os.path.exists(candidate):
                return candidate
    if "carla" in norm and "inference_final" in norm:
        for root in [os.getcwd(), os.path.dirname(os.path.dirname(os.path.dirname(__file__)))]:
            candidate = os.path.join(root, "data", "Carla-Autonomous-Vehicle", "carla_simulation code", "inference_final.py")
            if os.path.exists(candidate):
                return candidate
    return path


class ExecuteTestsTool(BaseTool):
    """Tool to execute pytest on generated tests."""
    name: str = "execute_tests"
    description: str = "Execute pytest on test code. Pass the FULL test code from the previous task's output as test_code - never use placeholders."
    
    def _run(self, test_code: str, target_file: str = "") -> str:
        """Run pytest on the test code."""
        # Reject placeholder/invalid test code
        if not test_code or len(test_code.strip()) < 50:
            return "ERROR: test_code is empty or too short. Pass the FULL Python test code from the generator's output."
        if any(p in test_code.lower() for p in ["paste the above", "see above", "insert code here", "paste the above code"]):
            return "ERROR: Do not use placeholders. Pass the ACTUAL test code from the previous task's output."
        
        # Repair malformed path (LLM sometimes strips backslashes: C:\...\file.py -> C:...file.py)
        target_file = _repair_target_path(target_file)
        
        # Create temp file for tests
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='_test.py', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            # Use current Python (venv) for pytest; avoid plugin conflicts (flask, cacheprovider)
            import sys
            pytest_cmd = [sys.executable, '-m', 'pytest', test_file, '-v', '--tb=short',
                         '-p', 'no:cacheprovider', '-p', 'no:flask', '-p', 'no:flask_sqlalchemy']
            cwd = None
            if target_file and os.path.exists(target_file):
                cwd = os.path.dirname(target_file)
            elif target_file and os.path.exists(os.path.normpath(target_file.replace('/', os.sep))):
                cwd = os.path.dirname(os.path.normpath(target_file.replace('/', os.sep)))
            result = subprocess.run(
                pytest_cmd,
                capture_output=True,
                text=True,
                timeout=60,
                cwd=cwd
            )
            
            output = result.stdout + result.stderr
            # If pytest failed due to missing flask (common with system Python plugins), suggest venv
            if "ModuleNotFoundError: No module named 'flask'" in output:
                output = ("NOTE: pytest failed due to missing 'flask' (often from system Python plugins). "
                         "Run from project venv: python -m venv .venv && .venv\\Scripts\\activate\n\n") + output
            
            # Parse results
            passed = len(re.findall(r'PASSED', output))
            failed = len(re.findall(r'FAILED', output))
            errors = len(re.findall(r'ERROR', output))
            
            return f"""Test Results:
- Passed: {passed}
- Failed: {failed}
- Errors: {errors}

Output:
{output[:2000]}"""
        
        except subprocess.TimeoutExpired:
            return "ERROR: Tests timed out after 60 seconds"
        except Exception as e:
            return f"ERROR: {str(e)}"
        finally:
            # Cleanup
            try:
                os.unlink(test_file)
            except:
                pass


class AnalyzeFailuresTool(BaseTool):
    """Tool to analyze test failures and suggest fixes."""
    name: str = "analyze_failures"
    description: str = "Analyze test failures and suggest improvements"
    
    def _run(self, test_output: str, original_tests: str) -> str:
        """Analyze failures and suggest fixes."""
        # Extract failure details
        failures = re.findall(r'FAILED.*?(?=FAILED|PASSED|$)', test_output, re.DOTALL)
        
        analysis = []
        for i, failure in enumerate(failures[:5], 1):  # Limit to 5 failures
            analysis.append(f"Failure {i}:\n{failure[:500]}")
        
        if not failures:
            return "No failures to analyze. All tests passed!"
        
        return f"""Failure Analysis:
{chr(10).join(analysis)}

Suggestions:
1. Check assertions match expected behavior
2. Verify imports are correct
3. Check for missing fixtures
4. Ensure test isolation"""


class TestingCrew:
    """
    Crew that generates, runs, and iteratively refines tests.
    
    This implements the ITERATIVE REFINEMENT pattern:
    - Generate initial tests
    - Run them
    - Analyze failures
    - Refine based on feedback
    - Repeat until success or max iterations
    """
    
    def __init__(
        self,
        model: str = "deepseek-coder:6.7b",
        max_iterations: int = 3,
        target_coverage: float = 0.85
    ):
        self.llm = create_ollama_llm(model)
        self.max_iterations = max_iterations
        self.target_coverage = target_coverage
        self.tools = [ExecuteTestsTool(), AnalyzeFailuresTool()]
        
        # Create agents
        self._create_agents()
    
    def _create_agents(self):
        """Create the testing crew agents with anti-hallucination prompting."""
        
        self.test_generator = Agent(
            role="Test Engineer",
            goal="Generate pytest tests for ACTUAL functions in the code",
            backstory="""You are an expert test engineer who writes pytest tests.

            GROUNDING RULES (critical):
            1. ONLY write tests for functions that ACTUALLY EXIST in the provided code
            2. Use the EXACT function signatures (parameter names, types) from the code
            3. Import statements must match the actual module/file structure
            4. Test realistic inputs based on what the function actually accepts
            
            VERIFICATION BEFORE WRITING:
            - Does this function exist in the provided code?
            - Am I using the correct parameter names?
            - Are my expected outputs based on actual code logic?
            
            DO NOT:
            - Invent functions not in the code
            - Assume parameter names not shown
            - Create tests for imaginary features
            - Add mock database/API calls unless the code actually uses them
            
            FORMAT: pytest-compatible with clear docstrings""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
        
        self.test_executor = Agent(
            role="Test Executor",
            goal="Execute tests and collect results accurately",
            backstory="""You run tests using pytest and parse the results,
            identifying passed, failed, and error counts accurately.""",
            verbose=True,
            llm=self.llm,
            tools=[self.tools[0]],  # ExecuteTestsTool
            allow_delegation=False
        )
        
        self.failure_analyzer = Agent(
            role="Failure Analyst",
            goal="Analyze ACTUAL test failures from the output",
            backstory="""You analyze test failures from pytest output.

            ANALYSIS RULES:
            1. Only analyze failures shown in the actual test output
            2. Quote the specific error messages
            3. Identify whether it's an assertion, import, or runtime error
            4. Suggest fixes based on actual error content
            
            DO NOT invent or assume failures not in the output.""",
            verbose=True,
            llm=self.llm,
            tools=[self.tools[1]],  # AnalyzeFailuresTool
            allow_delegation=False
        )
        
        self.test_refiner = Agent(
            role="Test Refiner",
            goal="Fix tests based on ACTUAL failure analysis",
            backstory="""You refine tests to fix specific issues.

            REFINEMENT RULES:
            1. Only fix issues identified in the failure analysis
            2. Keep tests grounded in actual code functionality
            3. Don't add new tests for imaginary functions
            4. Preserve correct tests that passed
            
            VERIFICATION: Before submitting, check that your tests
            still only reference functions from the original code.""",
            verbose=True,
            llm=self.llm,
            allow_delegation=False
        )
    
    def run_iteration(
        self,
        target_code: str,
        target_file: str,
        previous_tests: str = "",
        previous_failures: str = "",
        iteration: int = 1
    ) -> Dict[str, Any]:
        """
        Run one iteration of the test-refine cycle.
        
        Production-style flow: Generate (agent) -> Parse (orchestrator) -> Execute (pure fn) -> Analyze (pure fn).
        No agent involvement in data handoff; avoids placeholder/copy failures.
        
        Returns:
            Dict with tests, results, and whether to continue
        """
        # Step 1: Generate/Refine tests (agent)
        if iteration == 1:
            generate_task = Task(
                description=f"""Generate pytest tests for this code:

{target_code}

Requirements:
- Use pytest framework
- Include edge cases
- Add docstrings
- Use appropriate fixtures
- Target 85% coverage
- Output the complete test code in a ```python code block""",
                expected_output="Complete pytest test file in a code block",
                agent=self.test_generator
            )
        elif previous_tests and ("def test_" in previous_tests or "import pytest" in previous_tests):
            # Refine: we had valid tests that failed
            generate_task = Task(
                description=f"""Refine these tests based on failures:

Previous Tests:
{previous_tests}

Failures:
{previous_failures}

Fix the failing tests while keeping passing ones.
Output the refined test code in a ```python code block""",
                expected_output="Refined pytest test file in a code block",
                agent=self.test_refiner
            )
        else:
            # Generate from scratch: previous attempt produced no valid test code
            generate_task = Task(
                description=f"""The previous attempt produced NO valid pytest code. Generate tests from scratch.

Target code:
{target_code}

Requirements:
- Use pytest framework
- Include edge cases
- Add docstrings
- Output the complete test code in a ```python code block
- You MUST produce executable Python with def test_ or import pytest""",
                expected_output="Complete pytest test file in a code block",
                agent=self.test_generator
            )
        
        crew = Crew(
            tracing=os.environ.get("CREWAI_TRACING_ENABLED", "false").lower() == "true",
            agents=[generate_task.agent],
            tasks=[generate_task],
            process=Process.sequential,
            verbose=True
        )
        crew.kickoff()
        
        generated_output = str(generate_task.output or "")
        tests = generated_output
        
        # Step 2: Parse (orchestrator) - deterministic extraction
        parsed_code = _extract_python_code(generated_output)
        
        # Step 3: Execute (pure function) - no agent, no placeholder risk
        if not parsed_code or len(parsed_code.strip()) < 50:
            execution_result = (
                "ERROR: No valid Python test code found in generator output. "
                "Expected a ```python code block with def test_ or import pytest."
            )
        else:
            execution_result = self.tools[0]._run(test_code=parsed_code, target_file=target_file)
        
        # Step 4: Analyze (pure function) - deterministic failure analysis
        analysis = self.tools[1]._run(test_output=execution_result, original_tests=generated_output)
        
        all_passed = "FAILED" not in execution_result and "ERROR" not in execution_result
        
        return {
            "iteration": iteration,
            "tests": tests,
            "execution_result": execution_result,
            "analysis": analysis,
            "all_passed": all_passed,
            "should_continue": not all_passed and iteration < self.max_iterations
        }
    
    def run(
        self,
        target_code: str,
        target_file: str,
        progress_callback: Optional[callable] = None
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Run the full iterative testing workflow.
        
        Yields progress updates for streaming to frontend.
        """
        yield {
            "type": "crew_start",
            "crew_name": "TestingCrew",
            "max_iterations": self.max_iterations
        }
        
        previous_tests = ""
        previous_failures = ""
        
        for iteration in range(1, self.max_iterations + 1):
            yield {
                "type": "iteration_start",
                "iteration": iteration,
                "max_iterations": self.max_iterations
            }
            
            # Run iteration
            result = self.run_iteration(
                target_code=target_code,
                target_file=target_file,
                previous_tests=previous_tests,
                previous_failures=previous_failures,
                iteration=iteration
            )
            
            yield {
                "type": "iteration_complete",
                "iteration": iteration,
                "all_passed": result["all_passed"],
                "tests": result["tests"][:500],
                "analysis": result["analysis"][:500]
            }
            
            if result["all_passed"]:
                yield {
                    "type": "crew_complete",
                    "success": True,
                    "iterations": iteration,
                    "final_tests": result["tests"]
                }
                return
            
            # Prepare for next iteration
            previous_tests = result["tests"]
            previous_failures = result["analysis"]
        
        # Max iterations exhausted
        yield {
            "type": "crew_complete",
            "success": False,
            "iterations": self.max_iterations,
            "final_tests": previous_tests,
            "message": "Max iterations reached, some tests still failing"
        }


def test_testing_crew():
    """Test the TestingCrew."""
    print("Testing TestingCrew...")
    print("=" * 50)
    
    # Sample code to test
    sample_code = '''
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b
'''
    
    crew = TestingCrew(max_iterations=2)
    
    for progress in crew.run(sample_code, "sample.py"):
        print(f"[{progress['type']}] ", end="")
        if progress['type'] == 'iteration_start':
            print(f"Iteration {progress['iteration']}/{progress['max_iterations']}")
        elif progress['type'] == 'iteration_complete':
            status = "✅ PASSED" if progress['all_passed'] else "❌ FAILED"
            print(f"Iteration {progress['iteration']} {status}")
        elif progress['type'] == 'crew_complete':
            status = "✅ SUCCESS" if progress['success'] else "⚠️ INCOMPLETE"
            print(f"\n{status} after {progress['iterations']} iterations")
    
    print("\n✅ TestingCrew working!")


if __name__ == "__main__":
    test_testing_crew()
