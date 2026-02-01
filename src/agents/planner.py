"""
Dynamic Planner Agent for CodeMind AI

Implements dynamic goal planning capabilities:
1. Decompose complex goals into sub-tasks
2. Prioritize and order tasks
3. Replan based on feedback

This enables true agentic behavior - agents that plan their own work.
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from enum import Enum
from .base import BaseAgent, AgentConfig


class TaskStatus(Enum):
    """Status of a planned task."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class SubTask:
    """A sub-task in a plan."""
    id: str
    description: str
    priority: int = 1  # 1 = highest
    dependencies: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    agent: Optional[str] = None  # Which agent should handle this


@dataclass
class Plan:
    """A complete execution plan."""
    goal: str
    tasks: List[SubTask] = field(default_factory=list)
    current_task_idx: int = 0
    is_complete: bool = False
    
    def get_next_task(self) -> Optional[SubTask]:
        """Get the next pending task."""
        for task in self.tasks:
            if task.status == TaskStatus.PENDING:
                # Check dependencies
                deps_met = all(
                    self.get_task(dep).status == TaskStatus.COMPLETED
                    for dep in task.dependencies
                    if self.get_task(dep)
                )
                if deps_met:
                    return task
        return None
    
    def get_task(self, task_id: str) -> Optional[SubTask]:
        """Get task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        return None
    
    def mark_complete(self, task_id: str, result: str):
        """Mark a task as complete."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.COMPLETED
            task.result = result
    
    def mark_failed(self, task_id: str, error: str):
        """Mark a task as failed."""
        task = self.get_task(task_id)
        if task:
            task.status = TaskStatus.FAILED
            task.result = error


PLANNER_SYSTEM_PROMPT = """You are a planning agent that breaks down complex goals into actionable sub-tasks.

When given a goal, output a JSON array of tasks in this format:
[
  {"id": "task_1", "description": "...", "priority": 1, "dependencies": [], "agent": "explorer|security|optimizer|tester|documenter"},
  {"id": "task_2", "description": "...", "priority": 2, "dependencies": ["task_1"], "agent": "..."}
]

Rules:
1. Break complex goals into 2-5 sub-tasks
2. Lower priority number = higher priority
3. List dependencies by task_id
4. Assign appropriate agent for each task
5. Tasks should be specific and actionable"""


class PlannerAgent(BaseAgent):
    """
    Agent that creates and manages execution plans.
    
    Enables dynamic goal planning - a key pattern for true agentic AI
    where agents decide what tasks to perform to achieve a goal.
    """
    
    def __init__(self, model: str = "qwen2.5:7b"):
        config = AgentConfig(
            model=model,
            temperature=0.2,
            max_tokens=1500,
            system_prompt=PLANNER_SYSTEM_PROMPT
        )
        super().__init__(config)
        self.current_plan: Optional[Plan] = None
    
    def create_plan(self, goal: str) -> Plan:
        """
        Create a plan for achieving a goal.
        
        Args:
            goal: High-level goal to achieve
            
        Returns:
            Plan with ordered sub-tasks
        """
        prompt = f"""Create a plan to achieve this goal:

{goal}

Output ONLY a JSON array of tasks, nothing else."""

        response = self.generate(prompt)
        
        # Parse tasks from JSON
        tasks = self._parse_tasks(response)
        
        self.current_plan = Plan(goal=goal, tasks=tasks)
        return self.current_plan
    
    def _parse_tasks(self, response: str) -> List[SubTask]:
        """Parse tasks from LLM response."""
        import json
        import re
        
        # Try to extract JSON array
        json_match = re.search(r'\[[\s\S]*\]', response)
        if json_match:
            try:
                tasks_json = json.loads(json_match.group())
                tasks = []
                for i, t in enumerate(tasks_json):
                    tasks.append(SubTask(
                        id=t.get("id", f"task_{i+1}"),
                        description=t.get("description", ""),
                        priority=t.get("priority", i + 1),
                        dependencies=t.get("dependencies", []),
                        agent=t.get("agent")
                    ))
                return tasks
            except json.JSONDecodeError:
                pass
        
        # Fallback: create single task
        return [SubTask(
            id="task_1",
            description=response[:200],
            priority=1
        )]
    
    def replan(self, feedback: str) -> Plan:
        """
        Adjust the current plan based on feedback.
        
        Args:
            feedback: Information about what went wrong or changed
            
        Returns:
            Updated plan
        """
        if not self.current_plan:
            raise ValueError("No current plan to adjust")
        
        # Build context from current plan
        completed = [t for t in self.current_plan.tasks if t.status == TaskStatus.COMPLETED]
        pending = [t for t in self.current_plan.tasks if t.status == TaskStatus.PENDING]
        failed = [t for t in self.current_plan.tasks if t.status == TaskStatus.FAILED]
        
        prompt = f"""Original goal: {self.current_plan.goal}

Completed tasks:
{self._format_tasks(completed)}

Failed tasks:
{self._format_tasks(failed)}

Pending tasks:
{self._format_tasks(pending)}

Feedback: {feedback}

Create a revised plan to achieve the goal, considering what's done and what failed.
Output ONLY a JSON array of remaining tasks."""

        response = self.generate(prompt)
        new_tasks = self._parse_tasks(response)
        
        # Keep completed tasks, add new ones
        all_tasks = completed + new_tasks
        self.current_plan.tasks = all_tasks
        
        return self.current_plan
    
    def _format_tasks(self, tasks: List[SubTask]) -> str:
        """Format tasks for prompt."""
        if not tasks:
            return "  (none)"
        return "\n".join([
            f"  - [{t.id}] {t.description[:50]}..."
            for t in tasks
        ])
    
    def get_next_action(self) -> Optional[Dict[str, Any]]:
        """
        Get the next action to take from the current plan.
        
        Returns:
            Dict with task info or None if plan complete
        """
        if not self.current_plan:
            return None
        
        next_task = self.current_plan.get_next_task()
        if not next_task:
            self.current_plan.is_complete = True
            return None
        
        next_task.status = TaskStatus.IN_PROGRESS
        
        return {
            "task_id": next_task.id,
            "description": next_task.description,
            "agent": next_task.agent,
            "priority": next_task.priority
        }
    
    def complete_task(self, task_id: str, result: str, success: bool = True):
        """Mark a task as complete or failed."""
        if not self.current_plan:
            return
        
        if success:
            self.current_plan.mark_complete(task_id, result)
        else:
            self.current_plan.mark_failed(task_id, result)


def test_planner():
    """Test the Planner agent."""
    print("Testing Planner Agent...")
    print("=" * 50)
    
    planner = PlannerAgent()
    
    goal = "Analyze the BaseAgent class for security issues and suggest improvements"
    
    print(f"Goal: {goal}\n")
    
    plan = planner.create_plan(goal)
    
    print("Generated Plan:")
    for task in plan.tasks:
        deps = f" (depends on: {task.dependencies})" if task.dependencies else ""
        print(f"  [{task.id}] P{task.priority}: {task.description[:60]}...{deps}")
    
    print("\n✅ Planner Agent working!")


if __name__ == "__main__":
    test_planner()
