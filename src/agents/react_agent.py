"""
ReAct Agent for CodeMind AI

Implements the ReAct (Reasoning + Acting) loop pattern:
1. Think - Reason about what to do
2. Act - Execute an action (tool call)
3. Observe - Process the result
4. Repeat until done

This is a core pattern for true agentic AI.
"""

from typing import Dict, Any, List, Optional, Callable, Generator
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from .base import BaseAgent, AgentConfig


class ActionType(Enum):
    """Available actions for the ReAct agent."""
    SEARCH = "search"       # Search codebase
    READ = "read"           # Read a file
    ANALYZE = "analyze"     # Analyze code
    THINK = "think"         # Internal reasoning
    ANSWER = "answer"       # Final answer


@dataclass
class Thought:
    """A reasoning step."""
    content: str
    confidence: float = 0.8


@dataclass 
class Action:
    """An action to execute."""
    type: ActionType
    params: Dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""


@dataclass
class Observation:
    """Result of an action."""
    content: str
    success: bool = True
    error: Optional[str] = None


@dataclass
class ReActStep:
    """A single step in the ReAct loop."""
    step_num: int
    thought: Thought
    action: Action
    observation: Optional[Observation] = None


REACT_SYSTEM_PROMPT = """You are a reasoning agent that solves problems step-by-step.

For each step, you must output in this exact format:

THOUGHT: [Your reasoning about what to do next]
ACTION: [action_type]
PARAMS: [JSON parameters for the action]

Available actions:
- search: Search the codebase. Params: {"query": "search terms"}
- read: Read a file. Params: {"path": "file/path"}
- analyze: Analyze code. Params: {"code": "code snippet", "aspect": "security|performance|structure"}
- think: Internal reasoning. Params: {"topic": "what to think about"}
- answer: Provide final answer. Params: {"response": "your final answer"}

Rules:
1. Always start with a THOUGHT
2. Use search/read to gather information before answering
3. Use answer action when you have enough information
4. Maximum 5 steps before you must answer"""


class ReActAgent(BaseAgent):
    """
    Agent that uses ReAct loop for autonomous reasoning.
    
    The ReAct pattern enables:
    - Multi-step problem solving
    - Tool usage with reasoning
    - Self-directed exploration
    - Observable thought process
    """
    
    def __init__(
        self,
        tools: Optional[Dict[str, Callable]] = None,
        retriever=None,
        model: str = "qwen2.5:7b",
        max_steps: int = 5
    ):
        config = AgentConfig(
            model=model,
            temperature=0.3,
            max_tokens=1000,
            system_prompt=REACT_SYSTEM_PROMPT
        )
        super().__init__(config)
        
        self.tools = tools or {}
        self.retriever = retriever
        self.max_steps = max_steps
        self.steps: List[ReActStep] = []
        
        # Register default tools
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register built-in tools."""
        if "search" not in self.tools:
            self.tools["search"] = self._search_tool
        if "read" not in self.tools:
            self.tools["read"] = self._read_tool
        if "analyze" not in self.tools:
            self.tools["analyze"] = self._analyze_tool
        if "think" not in self.tools:
            self.tools["think"] = self._think_tool
    
    def _search_tool(self, query: str) -> str:
        """Search the codebase."""
        if self.retriever:
            results = self.retriever.search(query, n_results=3)
            if results:
                return "\n\n".join([
                    f"[{r['metadata'].get('source', 'unknown')}]\n{r['content'][:500]}"
                    for r in results
                ])
        return "No results found."
    
    def _read_tool(self, path: str) -> str:
        """Read a file."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
                if len(content) > 2000:
                    return content[:2000] + "\n... (truncated)"
                return content
        except Exception as e:
            return f"Error reading file: {e}"
    
    def _analyze_tool(self, code: str, aspect: str = "structure") -> str:
        """Analyze code for a specific aspect."""
        prompt = f"Analyze this code for {aspect}:\n\n{code}\n\nProvide a brief analysis."
        return self.generate(prompt)
    
    def _think_tool(self, topic: str) -> str:
        """Internal reasoning step."""
        prompt = f"Think deeply about: {topic}\n\nProvide your reasoning."
        return self.generate(prompt)
    
    def _parse_response(self, response: str) -> tuple[Thought, Action]:
        """Parse LLM response into Thought and Action."""
        thought_match = re.search(r'THOUGHT:\s*(.+?)(?=ACTION:|$)', response, re.DOTALL)
        action_match = re.search(r'ACTION:\s*(\w+)', response)
        params_match = re.search(r'PARAMS:\s*({.+?})', response, re.DOTALL)
        
        thought = Thought(
            content=thought_match.group(1).strip() if thought_match else "No explicit thought"
        )
        
        action_type = ActionType.THINK
        if action_match:
            try:
                action_type = ActionType(action_match.group(1).lower())
            except ValueError:
                action_type = ActionType.THINK
        
        params = {}
        if params_match:
            try:
                params = json.loads(params_match.group(1))
            except json.JSONDecodeError:
                params = {}
        
        action = Action(
            type=action_type,
            params=params,
            reasoning=thought.content
        )
        
        return thought, action
    
    def _execute_action(self, action: Action) -> Observation:
        """Execute an action and return observation."""
        try:
            if action.type == ActionType.ANSWER:
                return Observation(
                    content=action.params.get("response", "No response provided"),
                    success=True
                )
            
            action_name = action.type.value
            if action_name in self.tools:
                tool = self.tools[action_name]
                result = tool(**action.params)
                return Observation(content=str(result), success=True)
            
            return Observation(
                content=f"Unknown action: {action_name}",
                success=False,
                error=f"No tool registered for {action_name}"
            )
        except Exception as e:
            return Observation(
                content="",
                success=False,
                error=str(e)
            )
    
    def _build_context(self, query: str) -> str:
        """Build context from previous steps."""
        context = f"Original query: {query}\n\n"
        
        for step in self.steps:
            context += f"Step {step.step_num}:\n"
            context += f"  Thought: {step.thought.content}\n"
            context += f"  Action: {step.action.type.value} {step.action.params}\n"
            if step.observation:
                obs_preview = step.observation.content[:200]
                context += f"  Observation: {obs_preview}...\n"
            context += "\n"
        
        return context
    
    def reason(self, query: str, stream: bool = False) -> Generator[Dict[str, Any], None, None]:
        """
        Run the ReAct loop on a query.
        
        Yields step-by-step progress for streaming.
        
        Args:
            query: The user's query
            stream: Whether to stream intermediate steps
            
        Yields:
            Dict with step information
        """
        self.steps = []
        
        for step_num in range(1, self.max_steps + 1):
            # Build context from previous steps
            context = self._build_context(query)
            
            # Get next thought and action from LLM
            prompt = f"{context}\n\nWhat is your next step? Remember the format."
            response = self.generate(prompt)
            
            # Parse response
            thought, action = self._parse_response(response)
            
            # Create step
            step = ReActStep(
                step_num=step_num,
                thought=thought,
                action=action
            )
            
            # Yield thought before action
            if stream:
                yield {
                    "type": "thought",
                    "step": step_num,
                    "content": thought.content
                }
            
            # Execute action
            observation = self._execute_action(action)
            step.observation = observation
            self.steps.append(step)
            
            # Yield action result
            if stream:
                yield {
                    "type": "action",
                    "step": step_num,
                    "action": action.type.value,
                    "params": action.params,
                    "result": observation.content[:500] if observation.content else None,
                    "success": observation.success
                }
            
            # Check if we got a final answer
            if action.type == ActionType.ANSWER:
                yield {
                    "type": "answer",
                    "step": step_num,
                    "content": observation.content
                }
                return
            
            # Check for failure
            if not observation.success:
                if stream:
                    yield {
                        "type": "error",
                        "step": step_num,
                        "error": observation.error
                    }
        
        # Max steps reached - force an answer
        yield {
            "type": "answer",
            "step": self.max_steps,
            "content": self._force_answer(query)
        }
    
    def _force_answer(self, query: str) -> str:
        """Force a final answer after max steps."""
        context = self._build_context(query)
        prompt = f"""{context}

You've reached the maximum steps. Based on everything you've learned, 
provide your best answer to the original query: {query}"""
        return self.generate(prompt)
    
    def answer(self, query: str, stream: bool = True) -> Generator[str, None, None]:
        """
        Answer a query using ReAct loop, compatible with existing API.
        
        Args:
            query: User's query
            stream: Whether to stream tokens
            
        Yields:
            Tokens or full response
        """
        for step in self.reason(query, stream=True):
            if step["type"] == "thought":
                yield f"\n💭 **Thinking:** {step['content']}\n"
            elif step["type"] == "action":
                yield f"\n🔧 **Action:** {step['action']}({step.get('params', {})})\n"
                if step.get("result"):
                    yield f"📋 **Result:** {step['result'][:200]}...\n"
            elif step["type"] == "answer":
                yield f"\n✅ **Answer:**\n{step['content']}"
            elif step["type"] == "error":
                yield f"\n⚠️ **Error:** {step['error']}\n"


def test_react_agent():
    """Test the ReAct agent."""
    print("Testing ReAct Agent...")
    print("=" * 50)
    
    agent = ReActAgent()
    
    query = "What is the structure of the BaseAgent class?"
    
    print(f"Query: {query}\n")
    
    for output in agent.answer(query, stream=True):
        print(output, end="", flush=True)
    
    print("\n\n✅ ReAct Agent working!")


if __name__ == "__main__":
    test_react_agent()
