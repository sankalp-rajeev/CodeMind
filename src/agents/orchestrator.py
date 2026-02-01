"""
Orchestrator Agent for CodeMind AI

Routes user queries to appropriate agents or crews.
Classifies intent and decides between single-agent and multi-agent responses.
"""

from typing import Dict, Any, Optional, List, Callable
from enum import Enum, auto
from .base import BaseAgent, AgentConfig


class QueryIntent(Enum):
    """Possible query intents."""
    EXPLORE = auto()      # Code understanding, Q&A
    REFACTOR = auto()     # Code improvement, optimization
    TEST = auto()         # Test generation
    SECURITY = auto()     # Security analysis
    DOCUMENT = auto()     # Documentation generation
    GENERAL = auto()      # General conversation


# System prompt for intent classification
ORCHESTRATOR_SYSTEM_PROMPT = """You are an intelligent query router for a codebase assistant.

Your job is to classify user queries into one of these categories:
- EXPLORE: Questions about code, understanding, navigation (e.g., "what does X do?", "where is Y defined?")
- REFACTOR: Requests to improve code (e.g., "optimize this", "clean up", "refactor")
- TEST: Requests for test generation (e.g., "write tests for", "add unit tests")
- SECURITY: Security-related queries (e.g., "find vulnerabilities", "security review")
- DOCUMENT: Documentation requests (e.g., "document this", "add docstrings")
- GENERAL: General conversation or unclear intent

Respond with ONLY the category name, nothing else."""


class OrchestratorAgent(BaseAgent):
    """
    Routes queries to appropriate agents or crews.
    
    Features:
    - Intent classification
    - Agent routing
    - Complexity assessment
    - Multi-agent coordination (for crews)
    - ReAct loop for complex reasoning
    """
    
    def __init__(
        self,
        agents: Optional[Dict[str, BaseAgent]] = None,
        model: str = "orchestrator-ft"
    ):
        """
        Initialize the Orchestrator.
        
        Args:
            agents: Dictionary of agent_name -> agent instance
            model: Ollama model for classification
        """
        config = AgentConfig(
            model=model,
            temperature=0.1,  # Very low for classification
            max_tokens=50,
            system_prompt=ORCHESTRATOR_SYSTEM_PROMPT
        )
        super().__init__(config)
        
        self.agents = agents or {}
        self.intent_handlers: Dict[QueryIntent, Callable] = {}
        self.react_agent = None  # Lazy load for complex queries
    
    def register_agent(self, name: str, agent: BaseAgent):
        """Register an agent for routing."""
        self.agents[name] = agent
    
    def register_handler(self, intent: QueryIntent, handler: Callable):
        """Register a handler for an intent."""
        self.intent_handlers[intent] = handler
    
    def classify_intent(self, query: str) -> QueryIntent:
        """
        Classify the intent of a user query.
        
        Args:
            query: User's query
            
        Returns:
            Classified intent
        """
        # Pre-check: vague "what's wrong" / debug queries -> GENERAL (user needs exploration first)
        q_lower = query.lower().strip()
        vague_patterns = [
            "what's wrong", "whats wrong", "what is wrong",
            "why isn't it working", "why isnt it working",
            "help me with", "something wrong", "not working",
            "debug this", "fix this"  # "fix this" alone is vague
        ]
        if any(p in q_lower for p in vague_patterns) and len(query.split()) < 12:
            # Short vague query -> GENERAL (explore first to understand)
            return QueryIntent.GENERAL
        
        prompt = f"Classify this query: {query}"
        response = self.generate(prompt).strip().upper()
        
        # Map response to intent (using keyword search for robustness)
        response_upper = response.upper()
        
        # Check for specific intents in the response
        if "EXPLORE" in response_upper:
            return QueryIntent.EXPLORE
        elif "REFACTOR" in response_upper:
            return QueryIntent.REFACTOR
        elif "TEST" in response_upper:
            return QueryIntent.TEST
        elif "SECURITY" in response_upper:
            return QueryIntent.SECURITY
        elif "DOCUMENT" in response_upper:
            return QueryIntent.DOCUMENT
        elif "GENERAL" in response_upper:
            return QueryIntent.GENERAL
            
        # Fallback to exact map if needed (though above covers it)
        intent_map = {
            "EXPLORE": QueryIntent.EXPLORE,
            "REFACTOR": QueryIntent.REFACTOR,
            "TEST": QueryIntent.TEST,
            "SECURITY": QueryIntent.SECURITY,
            "DOCUMENT": QueryIntent.DOCUMENT,
            "GENERAL": QueryIntent.GENERAL,
        }
        
        return intent_map.get(response, QueryIntent.GENERAL)
    
    def route_query(self, query: str) -> Dict[str, Any]:
        """
        Route a query to the appropriate agent.
        
        Args:
            query: User's query
            
        Returns:
            Response dict with agent, response, and metadata
        """
        # Classify intent
        intent = self.classify_intent(query)
        
        # Determine routing
        routing = self._get_routing(intent)
        
        result = {
            "intent": intent.name,
            "agent": routing["agent"],
            "use_crew": routing["use_crew"],
            "response": None
        }
        
        # Execute if handler or agent available
        if intent in self.intent_handlers:
            result["response"] = self.intent_handlers[intent](query)
        elif routing["agent"] in self.agents:
            agent = self.agents[routing["agent"]]
            if hasattr(agent, 'answer'):
                result["response"] = agent.answer(query)
            else:
                result["response"] = agent.generate(query)
        else:
            result["response"] = f"No handler available for {intent.name} queries."
        
        return result
    
    def _get_routing(self, intent: QueryIntent) -> Dict[str, Any]:
        """
        Get routing configuration for an intent.
        
        Args:
            intent: Query intent
            
        Returns:
            Routing configuration
        """
        routing_config = {
            QueryIntent.EXPLORE: {
                "agent": "code_explorer",
                "use_crew": False,
                "description": "Single agent for code understanding"
            },
            QueryIntent.REFACTOR: {
                "agent": "refactoring_crew",
                "use_crew": True,
                "description": "Multi-agent refactoring crew"
            },
            QueryIntent.TEST: {
                "agent": "testing_crew",
                "use_crew": True,
                "description": "Iterative test generation crew"
            },
            QueryIntent.SECURITY: {
                "agent": "review_crew",
                "use_crew": True,
                "description": "Code review and security analysis crew"
            },
            QueryIntent.DOCUMENT: {
                "agent": "refactoring_crew",  # Documenter is in RefactoringCrew
                "use_crew": True,
                "description": "Documentation generation"
            },
            QueryIntent.GENERAL: {
                "agent": "code_explorer",
                "use_crew": False,
                "description": "General assistance"
            },
        }
        
        return routing_config.get(intent, routing_config[QueryIntent.GENERAL])
    
    def assess_complexity(self, query: str) -> Dict[str, Any]:
        """
        Assess the complexity of a query.
        
        Used to decide between single agent and crew.
        
        Args:
            query: User's query
            
        Returns:
            Complexity assessment
        """
        # Simple heuristics for now
        complexity_indicators = [
            "refactor", "optimize", "improve", "redesign",
            "test", "coverage", "security", "vulnerability",
            "document", "explain all", "entire", "whole"
        ]
        
        query_lower = query.lower()
        matches = sum(1 for ind in complexity_indicators if ind in query_lower)
        
        if matches >= 3:
            level = "high"
            needs_crew = True
        elif matches >= 1:
            level = "medium"
            needs_crew = False
        else:
            level = "low"
            needs_crew = False
        
        return {
            "level": level,
            "needs_crew": needs_crew,
            "matched_indicators": matches
        }


def test_orchestrator():
    """Test the Orchestrator agent."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    print("Testing Orchestrator Agent...")
    print("=" * 50)
    
    # Initialize orchestrator
    orchestrator = OrchestratorAgent(model="orchestrator-ft")
    
    # Test intent classification
    test_queries = [
        "What does the dataset loader do?",
        "Refactor this function to be more efficient",
        "Write unit tests for the training module",
        "Are there any security vulnerabilities?",
        "Add docstrings to all functions",
        "Hello, how are you?"
    ]
    
    print("\nIntent Classification:")
    print("-" * 40)
    
    for query in test_queries:
        intent = orchestrator.classify_intent(query)
        complexity = orchestrator.assess_complexity(query)
        print(f"\nQuery: '{query[:40]}...'")
        print(f"  Intent: {intent.name}")
        print(f"  Complexity: {complexity['level']}")
    
    print("\n✅ Orchestrator Agent working!")


if __name__ == "__main__":
    test_orchestrator()
