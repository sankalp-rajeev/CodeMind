"""
Base Agent for CodeMind AI

Provides common functionality for all agents:
- Ollama LLM integration
- Streaming responses
- Configurable models
"""

import json
from typing import Optional, Generator, Dict, Any, List
from dataclasses import dataclass

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Warning: ollama package not installed")


@dataclass
class AgentConfig:
    """Configuration for an agent."""
    model: str = "deepseek-coder:6.7b"
    temperature: float = 0.7
    max_tokens: int = 2048
    system_prompt: str = ""


class BaseAgent:
    """
    Base class for all CodeMind agents.
    
    Features:
    - Ollama LLM integration
    - Streaming and non-streaming generation
    - Configurable models and parameters
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("ollama package not installed")
        
        self.config = config or AgentConfig()
        self.conversation_history: List[Dict[str, str]] = []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False
    ) -> str | Generator[str, None, None]:
        """
        Generate a response from the LLM.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            stream: If True, stream the response
            
        Returns:
            Response string or generator for streaming
        """
        messages = []
        
        # Add system prompt
        sys_prompt = system_prompt or self.config.system_prompt
        if sys_prompt:
            messages.append({"role": "system", "content": sys_prompt})
        
        # Add user message
        messages.append({"role": "user", "content": prompt})
        
        if stream:
            return self._stream_response(messages)
        else:
            return self._generate_response(messages)
    
    def _generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate a complete response."""
        try:
            response = ollama.chat(
                model=self.config.model,
                messages=messages,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            )
            return response['message']['content']
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _stream_response(
        self, 
        messages: List[Dict[str, str]]
    ) -> Generator[str, None, None]:
        """Stream response tokens."""
        try:
            stream = ollama.chat(
                model=self.config.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens,
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        except Exception as e:
            yield f"Error: {e}"
    
    def add_to_history(self, role: str, content: str):
        """Add a message to conversation history."""
        self.conversation_history.append({"role": role, "content": content})
    
    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
    
    def get_available_models(self) -> List[str]:
        """Get list of available Ollama models."""
        try:
            models = ollama.list()
            return [m['name'] for m in models.get('models', [])]
        except Exception:
            return []
    
    # ==================== Self-Correction ====================
    
    def execute_with_retry(
        self,
        task: str,
        validator: Optional[callable] = None,
        max_retries: int = 3
    ) -> Dict[str, Any]:
        """
        Execute a task with self-correction on failure.
        
        This is a core pattern for true agentic AI - the agent can
        detect failures and retry with a different approach.
        
        Args:
            task: The task/prompt to execute
            validator: Optional function to validate result
            max_retries: Maximum retry attempts
            
        Returns:
            Dict with result, attempts, and success status
        """
        attempts = []
        
        for attempt in range(1, max_retries + 1):
            # Execute the task
            result = self.generate(task)
            
            # Validate result
            is_valid = True
            error_msg = None
            
            if validator:
                try:
                    is_valid = validator(result)
                except Exception as e:
                    is_valid = False
                    error_msg = str(e)
            else:
                # Default validation: check for error indicators
                is_valid = not self._detect_failure(result)
                if not is_valid:
                    error_msg = "Response indicates failure or uncertainty"
            
            attempts.append({
                "attempt": attempt,
                "result": result,
                "valid": is_valid,
                "error": error_msg
            })
            
            if is_valid:
                return {
                    "success": True,
                    "result": result,
                    "attempts": attempts,
                    "total_attempts": attempt
                }
            
            # Self-correct: diagnose and adjust approach
            if attempt < max_retries:
                diagnosis = self._diagnose_failure(task, result, error_msg)
                task = self._adjust_approach(task, diagnosis)
        
        # All retries exhausted - return best attempt
        return {
            "success": False,
            "result": attempts[-1]["result"],
            "attempts": attempts,
            "total_attempts": max_retries,
            "fallback": True
        }
    
    def _detect_failure(self, result: str) -> bool:
        """Detect if a result indicates failure."""
        failure_indicators = [
            "i don't know",
            "i'm not sure",
            "error:",
            "cannot find",
            "unable to",
            "i apologize",
            "no information",
        ]
        result_lower = result.lower()
        return any(ind in result_lower for ind in failure_indicators)
    
    def _diagnose_failure(self, task: str, result: str, error: Optional[str]) -> str:
        """Diagnose why a task failed."""
        prompt = f"""The following task failed or produced an unsatisfactory result.

Task: {task}
Result: {result[:500]}
Error: {error or 'None'}

Diagnose what went wrong in 1-2 sentences."""
        
        return self.generate(prompt)
    
    def _adjust_approach(self, task: str, diagnosis: str) -> str:
        """Adjust the approach based on diagnosis."""
        prompt = f"""Original task: {task}

Diagnosis of failure: {diagnosis}

Rewrite the task with a different approach that addresses the diagnosed issue.
Be more specific or try a different angle."""
        
        return self.generate(prompt)


def test_base_agent():
    """Test the base agent."""
    print("Testing Base Agent...")
    print("=" * 50)
    
    # List available models
    agent = BaseAgent()
    models = agent.get_available_models()
    print(f"Available models: {models}")
    
    # Test generation
    response = agent.generate(
        "What is a Python decorator? Answer in 1-2 sentences.",
        system_prompt="You are a helpful coding assistant. Be concise."
    )
    print(f"\nResponse:\n{response}")
    
    print("\n✅ Base Agent working!")


if __name__ == "__main__":
    test_base_agent()
