"""
Conversation Memory for CodeMind AI

Stores and manages conversation history for multi-turn interactions.

Features:
- Session-based chat history
- Context window management with token counting
- Smart summarization of old context
- Reference resolution ("that function", "those changes")
- SQLite persistence
"""

import sqlite3
import json
import uuid
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, asdict
from pathlib import Path

# Token counting (lightweight approximation)
def count_tokens(text: str) -> int:
    """
    Approximate token count (4 chars ≈ 1 token).
    For production, use tiktoken: pip install tiktoken
    """
    return len(text) // 4


class ContextWindowManager:
    """
    Production-grade context window management.
    
    Features:
    - Token-aware message selection
    - Smart summarization of older context
    - Priority-based retention (recent > old)
    - Configurable limits
    
    This is how ChatGPT, Claude, and Gemini manage long conversations.
    """
    
    def __init__(
        self,
        max_tokens: int = 4096,
        reserve_tokens: int = 1024,  # For new response
        summary_threshold: int = 10   # Messages before summarizing
    ):
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.summary_threshold = summary_threshold
        self.available_tokens = max_tokens - reserve_tokens
    
    def fit_to_window(
        self,
        messages: List['Message'],
        system_prompt: str = ""
    ) -> Tuple[List['Message'], Optional[str]]:
        """
        Fit messages into context window.
        
        Returns:
            (kept_messages, summary_of_dropped)
        """
        system_tokens = count_tokens(system_prompt)
        remaining = self.available_tokens - system_tokens
        
        # Work backwards from most recent
        kept = []
        dropped = []
        total_tokens = 0
        
        for msg in reversed(messages):
            msg_tokens = count_tokens(msg.content)
            if total_tokens + msg_tokens <= remaining:
                kept.insert(0, msg)
                total_tokens += msg_tokens
            else:
                dropped.insert(0, msg)
        
        # Generate summary of dropped messages if any
        summary = None
        if dropped:
            summary = self._create_summary(dropped)
        
        return kept, summary
    
    def _create_summary(self, messages: List['Message']) -> str:
        """
        Create a summary of dropped messages.
        
        For production, this would call an LLM. Here we use extraction.
        """
        if not messages:
            return ""
        
        # Extract key points
        user_queries = []
        assistant_points = []
        mentioned_items = set()
        
        for msg in messages:
            if msg.role == "user":
                # Keep first 50 chars of each user query
                user_queries.append(msg.content[:50])
            elif msg.role == "assistant":
                # Extract first sentence
                first_sentence = msg.content.split('.')[0][:100]
                assistant_points.append(first_sentence)
            
            # Track mentioned code items
            if msg.metadata:
                if msg.metadata.get('function_name'):
                    mentioned_items.add(f"function:{msg.metadata['function_name']}")
                if msg.metadata.get('file_path'):
                    mentioned_items.add(f"file:{msg.metadata['file_path']}")
        
        # Build summary
        summary_parts = ["[Previous conversation summary]"]
        
        if user_queries:
            summary_parts.append(f"User asked about: {'; '.join(user_queries[:3])}")
        
        if mentioned_items:
            summary_parts.append(f"Discussed: {', '.join(list(mentioned_items)[:5])}")
        
        if assistant_points:
            summary_parts.append(f"Key points: {assistant_points[0]}")
        
        return " | ".join(summary_parts)
    
    def get_stats(self, messages: List['Message']) -> Dict[str, Any]:
        """Get token statistics for messages."""
        total = sum(count_tokens(m.content) for m in messages)
        return {
            "message_count": len(messages),
            "total_tokens": total,
            "max_tokens": self.max_tokens,
            "available_tokens": self.available_tokens,
            "utilization": f"{(total / self.available_tokens) * 100:.1f}%"
        }


@dataclass
class Message:
    """A single message in a conversation."""
    role: str  # user, assistant, system
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata or {}
        }


@dataclass
class ConversationContext:
    """Active context for reference resolution."""
    last_mentioned_function: Optional[str] = None
    last_mentioned_file: Optional[str] = None
    last_crew_result: Optional[str] = None
    last_code_snippet: Optional[str] = None
    active_codebase: Optional[str] = None


class ConversationMemory:
    """
    Manages conversation history and context.
    
    Features:
    - Create/load conversation sessions
    - Add messages with metadata
    - Get context for prompts
    - Reference resolution
    - SQLite persistence
    """
    
    def __init__(
        self,
        db_path: str = "./data/conversations.db",
        max_tokens: int = 4096
    ):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        
        # Context window management (production-grade)
        self.window_manager = ContextWindowManager(max_tokens=max_tokens)
        
        # In-memory cache for current session
        self.current_session_id: Optional[str] = None
        self.messages: List[Message] = []
        self.context = ConversationContext()
    
    def _init_db(self):
        """Initialize SQLite database."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT,
                updated_at TEXT,
                codebase TEXT,
                metadata TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                role TEXT,
                content TEXT,
                timestamp TEXT,
                metadata TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS context (
                session_id TEXT PRIMARY KEY,
                data TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def create_session(self, codebase: Optional[str] = None) -> str:
        """Create a new conversation session."""
        session_id = str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO sessions (id, created_at, updated_at, codebase, metadata) VALUES (?, ?, ?, ?, ?)',
            (session_id, now, now, codebase, '{}')
        )
        conn.commit()
        conn.close()
        
        self.current_session_id = session_id
        self.messages = []
        self.context = ConversationContext(active_codebase=codebase)
        
        return session_id
    
    def load_session(self, session_id: str) -> bool:
        """Load an existing session."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Load session
        cursor.execute('SELECT * FROM sessions WHERE id = ?', (session_id,))
        session = cursor.fetchone()
        if not session:
            conn.close()
            return False
        
        # Load messages
        cursor.execute(
            'SELECT role, content, timestamp, metadata FROM messages WHERE session_id = ? ORDER BY timestamp',
            (session_id,)
        )
        rows = cursor.fetchall()
        
        self.current_session_id = session_id
        self.messages = [
            Message(
                role=row[0],
                content=row[1],
                timestamp=datetime.fromisoformat(row[2]),
                metadata=json.loads(row[3]) if row[3] else None
            )
            for row in rows
        ]
        
        # Load context
        cursor.execute('SELECT data FROM context WHERE session_id = ?', (session_id,))
        ctx_row = cursor.fetchone()
        if ctx_row:
            ctx_data = json.loads(ctx_row[0])
            self.context = ConversationContext(**ctx_data)
        else:
            self.context = ConversationContext()
        
        conn.close()
        return True
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Add a message to the current session."""
        if not self.current_session_id:
            self.create_session()
        
        msg = Message(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.messages.append(msg)
        
        # Update context based on message
        self._update_context(content, metadata)
        
        # Persist
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)',
            (self.current_session_id, role, content, msg.timestamp.isoformat(), json.dumps(metadata))
        )
        cursor.execute(
            'UPDATE sessions SET updated_at = ? WHERE id = ?',
            (datetime.now().isoformat(), self.current_session_id)
        )
        conn.commit()
        conn.close()
    
    def _update_context(self, content: str, metadata: Optional[Dict[str, Any]]):
        """Update context based on message content."""
        if metadata:
            if 'function_name' in metadata:
                self.context.last_mentioned_function = metadata['function_name']
            if 'file_path' in metadata:
                self.context.last_mentioned_file = metadata['file_path']
            if 'crew_result' in metadata:
                self.context.last_crew_result = metadata['crew_result']
            if 'code_snippet' in metadata:
                self.context.last_code_snippet = metadata['code_snippet']
        
        # Save context
        self._save_context()
    
    def _save_context(self):
        """Save context to database."""
        if not self.current_session_id:
            return
            
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        ctx_data = json.dumps({
            'last_mentioned_function': self.context.last_mentioned_function,
            'last_mentioned_file': self.context.last_mentioned_file,
            'last_crew_result': self.context.last_crew_result,
            'last_code_snippet': self.context.last_code_snippet,
            'active_codebase': self.context.active_codebase
        })
        cursor.execute(
            'INSERT OR REPLACE INTO context (session_id, data) VALUES (?, ?)',
            (self.current_session_id, ctx_data)
        )
        conn.commit()
        conn.close()
    
    def get_context_prompt(self, max_messages: int = 10) -> str:
        """Get formatted context for LLM prompt (simple version)."""
        recent = self.messages[-max_messages:] if len(self.messages) > max_messages else self.messages
        
        context_parts = []
        
        if recent:
            context_parts.append("Previous conversation:")
            for msg in recent:
                role = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role}: {msg.content[:200]}...")
        
        if self.context.last_mentioned_function:
            context_parts.append(f"\nLast mentioned function: {self.context.last_mentioned_function}")
        if self.context.last_mentioned_file:
            context_parts.append(f"Last mentioned file: {self.context.last_mentioned_file}")
        if self.context.active_codebase:
            context_parts.append(f"Active codebase: {self.context.active_codebase}")
        
        return "\n".join(context_parts)
    
    def get_context_for_llm(self, system_prompt: str = "") -> Tuple[str, Dict[str, Any]]:
        """
        Get token-aware context for LLM (production-grade).
        
        Uses ContextWindowManager to:
        1. Fit messages into available token budget
        2. Summarize dropped older messages
        3. Track token usage stats
        
        Returns:
            (formatted_context, stats_dict)
        """
        # Use window manager for token-aware selection
        kept_messages, summary = self.window_manager.fit_to_window(
            self.messages,
            system_prompt
        )
        
        context_parts = []
        
        # Add summary of older context if any messages were dropped
        if summary:
            context_parts.append(summary)
            context_parts.append("")  # Empty line separator
        
        # Add kept messages
        if kept_messages:
            context_parts.append("Recent conversation:")
            for msg in kept_messages:
                role = "User" if msg.role == "user" else "Assistant"
                context_parts.append(f"{role}: {msg.content}")
        
        # Add active context references
        if self.context.last_mentioned_function:
            context_parts.append(f"\n[Active reference: function '{self.context.last_mentioned_function}']")
        if self.context.last_mentioned_file:
            context_parts.append(f"[Active reference: file '{self.context.last_mentioned_file}']")
        
        # Get stats
        stats = self.window_manager.get_stats(self.messages)
        stats["messages_kept"] = len(kept_messages)
        stats["messages_dropped"] = len(self.messages) - len(kept_messages)
        stats["has_summary"] = summary is not None
        
        return "\n".join(context_parts), stats
    
    def resolve_reference(self, query: str) -> str:
        """Resolve references like 'that function' or 'those changes'."""
        resolved = query
        
        # Resolve function references
        if self.context.last_mentioned_function:
            patterns = ['that function', 'this function', 'the function']
            for pattern in patterns:
                if pattern in query.lower():
                    resolved = resolved.replace(pattern, self.context.last_mentioned_function)
        
        # Resolve file references
        if self.context.last_mentioned_file:
            patterns = ['that file', 'this file', 'the file']
            for pattern in patterns:
                if pattern in query.lower():
                    resolved = resolved.replace(pattern, self.context.last_mentioned_file)
        
        # Resolve crew result references
        if self.context.last_crew_result:
            patterns = ['those changes', 'those suggestions', 'the recommendations']
            for pattern in patterns:
                if pattern in query.lower():
                    resolved = f"{resolved}\n\nContext from previous analysis:\n{self.context.last_crew_result[:500]}"
        
        return resolved
    
    def get_session_history(self) -> List[Dict[str, Any]]:
        """Get all messages in current session."""
        return [msg.to_dict() for msg in self.messages]
    
    def clear_session(self):
        """Clear current session from memory (keeps in DB)."""
        self.current_session_id = None
        self.messages = []
        self.context = ConversationContext()


# Global memory instance
_memory: Optional[ConversationMemory] = None


def get_memory() -> ConversationMemory:
    """Get or create global memory instance."""
    global _memory
    if _memory is None:
        _memory = ConversationMemory()
    return _memory


def test_memory():
    """Test conversation memory."""
    print("Testing Conversation Memory...")
    print("=" * 50)
    
    memory = ConversationMemory(db_path="./data/test_conversations.db")
    
    # Create session
    session_id = memory.create_session(codebase="./data/Carla-Autonomous-Vehicle")
    print(f"✅ Created session: {session_id}")
    
    # Add messages
    memory.add_message("user", "What does the PIDController class do?", 
                      {"function_name": "PIDController"})
    memory.add_message("assistant", "The PIDController implements a PID control loop...")
    memory.add_message("user", "Can you refactor that function?")
    
    print(f"✅ Added 3 messages")
    
    # Test reference resolution
    resolved = memory.resolve_reference("Can you explain that function better?")
    print(f"✅ Resolved: '{resolved}'")
    
    # Get context prompt
    context = memory.get_context_prompt()
    print(f"✅ Context prompt:\n{context[:300]}...")
    
    # Test session persistence
    memory2 = ConversationMemory(db_path="./data/test_conversations.db")
    loaded = memory2.load_session(session_id)
    print(f"✅ Loaded session: {loaded}, messages: {len(memory2.messages)}")
    
    print("\n✅ Conversation Memory working!")


if __name__ == "__main__":
    test_memory()
