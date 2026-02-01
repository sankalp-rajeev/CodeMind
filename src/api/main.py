"""
FastAPI Backend for CodeMind AI

Provides:
- WebSocket endpoint for streaming chat
- REST endpoints for indexing and status
- Integration with OrchestratorAgent
"""

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from contextlib import asynccontextmanager, nullcontext

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules (will be initialized lazily)
retriever = None
orchestrator = None
refactoring_crew = None


def get_tracker_safe():
    """Get MLflow tracker (may not be installed)."""
    try:
        from src.observability.tracker import get_tracker, CrewMetrics
        return get_tracker(), CrewMetrics
    except Exception:
        return None, None


def clean_llm_output(text: str) -> str:
    """Remove LLM special tokens (DeepSeek, etc.) from output."""
    if not text:
        return text
    # DeepSeek tokens
    tokens_to_remove = [
        '<｜begin▁of▁sentence｜>',
        '<｜end▁of▁sentence｜>',
        '<|im_start|>',
        '<|im_end|>',
        '<|endoftext|>',
    ]
    for token in tokens_to_remove:
        text = text.replace(token, '')
    # Also clean any remaining special token patterns like <｜...｜>
    text = re.sub(r'<｜[^｜]+｜>', '', text)
    return text.strip()


def get_memory_safe():
    """Get conversation memory."""
    try:
        from src.memory.conversation import ConversationMemory
        return ConversationMemory()
    except Exception:
        return None


class IndexRequest(BaseModel):
    """Request to index a codebase."""
    directory: str
    force_reindex: bool = False


class ChatMessage(BaseModel):
    """Chat message from client."""
    message: str
    use_rag: bool = True


class IndexStatus(BaseModel):
    """Status of the indexer."""
    indexed: bool
    chunks: int
    directory: Optional[str] = None


class RefactorRequest(BaseModel):
    """Request to run the RefactoringCrew."""
    target: str  # Code target (function name, file, or description)
    focus: Optional[str] = None  # Optional: security, performance, tests, docs


# Global state
app_state = {
    "indexed_directory": None,
    "chunks_count": 0,
    "is_ready": False
}


def get_retriever():
    """Lazy load the retriever."""
    global retriever
    if retriever is None:
        from src.rag.retriever import HybridRetriever
        retriever = HybridRetriever(
            persist_directory="./data/chroma_db",
            use_reranker=False
        )
    return retriever


def get_orchestrator():
    """Lazy load the orchestrator with agents."""
    global orchestrator
    if orchestrator is None:
        from src.agents.orchestrator import OrchestratorAgent
        from src.agents.code_explorer import CodeExplorerAgent
        
        ret = get_retriever()
        explorer = CodeExplorerAgent(retriever=ret, model="deepseek-coder:6.7b")
        
        orchestrator = OrchestratorAgent(model="qwen2.5:7b")
        orchestrator.register_agent("code_explorer", explorer)
    
    return orchestrator


def get_refactoring_crew():
    """Lazy load the RefactoringCrew."""
    global refactoring_crew
    if refactoring_crew is None:
        from src.crews.refactoring_crew import RefactoringCrew
        ret = get_retriever()
        refactoring_crew = RefactoringCrew(retriever=ret)
    return refactoring_crew


# Additional crews for Week 3
testing_crew = None
review_crew = None


def get_testing_crew():
    """Lazy load the TestingCrew."""
    global testing_crew
    if testing_crew is None:
        from src.crews.testing_crew import TestingCrew
        testing_crew = TestingCrew()
    return testing_crew


def get_review_crew():
    """Lazy load the CodeReviewCrew."""
    global review_crew
    if review_crew is None:
        from src.crews.review_crew import CodeReviewCrew
        review_crew = CodeReviewCrew()
    return review_crew


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("[CodeMind] Starting up...")
    
    # Check if we have existing index
    try:
        ret = get_retriever()
        count = ret.indexer.collection.count()
        if count > 0:
            app_state["chunks_count"] = count
            app_state["is_ready"] = True
            print(f"[CodeMind] Loaded existing index with {count} chunks")
    except Exception as e:
        print(f"[CodeMind] No existing index: {e}")
    
    yield
    
    print("[CodeMind] Shutting down...")


# Create FastAPI app
app = FastAPI(
    title="CodeMind AI",
    description="Multi-agent codebase assistant API",
    version="0.1.0",
    lifespan=lifespan
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== REST Endpoints ====================

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "service": "CodeMind AI"}


@app.get("/api/status")
async def get_status() -> Dict[str, Any]:
    """Get system status."""
    status = {
        "indexed": app_state["is_ready"],
        "chunks": app_state["chunks_count"],
        "directory": app_state["indexed_directory"]
    }
    tracker, _ = get_tracker_safe()
    if tracker:
        try:
            status["mlflow"] = tracker.get_summary()
        except Exception:
            status["mlflow"] = {"enabled": False}
    return status


@app.post("/api/index")
async def index_codebase(request: IndexRequest) -> Dict[str, Any]:
    """Index a codebase directory."""
    directory = Path(request.directory)
    
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory}")
    
    try:
        ret = get_retriever()
        # Run in thread pool so we don't block the event loop (indexing is CPU/GPU heavy)
        stats = await asyncio.to_thread(
            ret.index_codebase,
            str(directory),
            force_reindex=request.force_reindex
        )
        
        app_state["indexed_directory"] = str(directory)
        app_state["chunks_count"] = stats.get("chunks_indexed", 0)
        app_state["is_ready"] = True
        
        return {
            "status": "success",
            "chunks_indexed": stats.get("chunks_indexed", 0),
            "files_processed": stats.get("files_processed", 0),
            "duration_seconds": stats.get("duration_seconds", 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Code file extensions (matches indexer)
_CODE_EXTENSIONS = {'.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs'}


def _resolve_code_from_query(query: str) -> Tuple[Optional[str], str]:
    """
    Extract file path from user query and read file content.
    Used by review_crew and testing_crew which need actual code, not a search target.

    Supports: backtick paths (`path/to/file.py`), quoted paths, regex extraction.
    Falls back to RAG search if no path found or file not readable.

    Returns:
        (content, file_path) - content is None if unresolvable; file_path for display.
    """
    query = query.strip()
    file_path_candidate = None

    # 1. Backtick-wrapped path: `path/to/file.py` or `data/Carla.../lane_detection_final.py`
    backtick_match = re.search(r'`([^`]+\.(?:py|js|jsx|ts|tsx|mjs|cjs))`', query)
    if backtick_match:
        file_path_candidate = backtick_match.group(1).strip()

    # 2. Quoted path: "path/to/file.py"
    if not file_path_candidate:
        quoted_match = re.search(r'["\']([^"\']+\.(?:py|js|jsx|ts|tsx|mjs|cjs))["\']', query)
        if quoted_match:
            file_path_candidate = quoted_match.group(1).strip()

    # 3. Regex: any path-like string ending in code extension
    if not file_path_candidate:
        path_match = re.search(r'[\w./\\\-]+\.[a-z]{2,4}\b', query)
        if path_match:
            cand = path_match.group(0)
            if any(cand.lower().endswith(ext) for ext in ('.py', '.js', '.jsx', '.ts', '.tsx', '.mjs', '.cjs')):
                file_path_candidate = cand

    proj_root = Path(__file__).parent.parent.parent
    bases = [Path.cwd(), proj_root]
    if app_state.get("indexed_directory"):
        idx_dir = Path(app_state["indexed_directory"])
        if not idx_dir.is_absolute():
            idx_dir = proj_root / idx_dir
        if idx_dir.exists():
            bases.insert(0, idx_dir)
    bases.extend([
        proj_root / "data" / "Carla-Autonomous-Vehicle",
        proj_root / "data" / "Carla-Autonomous-Vehicle" / "carla_simulation code",
    ])

    if file_path_candidate:
        fp = file_path_candidate.replace("/", os.sep).replace("\\", os.sep)
        for base in bases:
            candidate = (base / fp).resolve() if not Path(fp).is_absolute() else Path(fp).resolve()
            if candidate.exists() and candidate.is_file():
                try:
                    content = candidate.read_text(encoding='utf-8', errors='ignore')
                    try:
                        display_path = str(candidate.relative_to(proj_root))
                    except ValueError:
                        display_path = str(candidate)
                    return content, display_path.replace("\\", "/")
                except Exception:
                    pass

    # Fallback: RAG search - use query to find relevant file, then read it
    if app_state.get("is_ready"):
        try:
            ret = get_retriever()
            results = ret.search(query, n_results=3)
            for r in results:
                fp = r.get("metadata", {}).get("filepath") or r.get("filepath")
                if not fp:
                    continue
                path = Path(fp)
                if not path.is_absolute():
                    path = proj_root / fp
                if path.exists() and path.is_file():
                    try:
                        content = path.read_text(encoding='utf-8', errors='ignore')
                        try:
                            display = str(path.relative_to(proj_root))
                        except ValueError:
                            display = str(path)
                        return content, display.replace("\\", "/")
                    except Exception:
                        continue
        except Exception:
            pass

    return None, query[:80]


def _path_for_api(entry: Path, proj_root: Path) -> str:
    """Get path string for read-file API (relative to proj_root or absolute)."""
    try:
        return str(entry.relative_to(proj_root)).replace('\\', '/')
    except ValueError:
        return str(entry).replace('\\', '/')


def _build_file_tree(root: Path, base: Path, proj_root: Path) -> Optional[Dict[str, Any]]:
    """Build nested file tree. Returns None for non-directories."""
    if not root.is_dir():
        return None
    children: List[Dict[str, Any]] = []
    try:
        entries = sorted(root.iterdir(), key=lambda e: (e.is_file(), e.name.lower()))
        for entry in entries:
            if entry.name.startswith('.'):
                continue
            if entry.is_dir():
                child_tree = _build_file_tree(entry, base, proj_root)
                if child_tree:
                    children.append(child_tree)
            elif entry.suffix.lower() in _CODE_EXTENSIONS:
                path_str = _path_for_api(entry, proj_root)
                children.append({"name": entry.name, "path": path_str, "type": "file"})
    except PermissionError:
        pass
    rel = root.relative_to(base) if root != base else Path(".")
    path_str = _path_for_api(root, proj_root) if str(rel) != '.' else ''
    return {
        "name": root.name if root != base else base.name,
        "path": path_str,
        "type": "dir",
        "children": children
    }


@app.get("/api/file-tree")
async def get_file_tree() -> Dict[str, Any]:
    """Get file tree of the indexed codebase. Returns empty if none indexed."""
    if not app_state["is_ready"] or not app_state["indexed_directory"]:
        return {"root": None, "tree": None}
    try:
        proj_root = Path(__file__).parent.parent.parent
        idx_dir = Path(app_state["indexed_directory"])
        if not idx_dir.is_absolute():
            idx_dir = proj_root / idx_dir
        if not idx_dir.exists() or not idx_dir.is_dir():
            return {"root": app_state["indexed_directory"], "tree": None}
        tree = _build_file_tree(idx_dir, idx_dir, proj_root)
        return {
            "root": str(idx_dir),
            "tree": tree,
            "chunks": app_state["chunks_count"]
        }
    except Exception as e:
        return {"root": None, "tree": None, "error": str(e)}


@app.get("/api/read-file")
async def read_file(path: str) -> Dict[str, Any]:
    """Read file contents for Test/Review panels (relative to project root)."""
    try:
        p = Path(path)
        if not p.is_absolute():
            p = Path(__file__).parent.parent.parent / path
        if not p.exists():
            raise HTTPException(status_code=404, detail=f"File not found: {path}")
        content = p.read_text(encoding='utf-8', errors='ignore')
        return {"path": str(p), "content": content}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search")
async def search(query: str, n_results: int = 5) -> Dict[str, Any]:
    """Search the indexed codebase."""
    if not app_state["is_ready"]:
        raise HTTPException(status_code=400, detail="No codebase indexed")
    
    ret = get_retriever()
    results = ret.search(query, n_results=n_results)
    
    return {
        "query": query,
        "results": [
            {
                "name": r.get("metadata", {}).get("name") or r.get("name"),
                "type": r.get("metadata", {}).get("type") or r.get("type"),
                "filepath": r.get("metadata", {}).get("filepath") or r.get("filepath"),
                "score": r.get("rrf_score") or r.get("score", 0)
            }
            for r in results
        ]
    }


@app.post("/api/refactor")
async def refactor_code(request: RefactorRequest) -> Dict[str, Any]:
    """
    Run the RefactoringCrew on a code target.
    
    This uses 5 specialized agents to analyze and suggest improvements:
    - Code Explorer: Understands structure
    - Security Analyst: Finds vulnerabilities
    - Algorithm Optimizer: Finds performance issues
    - Test Engineer: Suggests test cases
    - Documentation Writer: Creates docstrings
    """
    if not app_state["is_ready"]:
        raise HTTPException(status_code=400, detail="No codebase indexed. Please index first.")
    
    valid_focus = [None, "security", "performance", "tests", "docs"]
    if request.focus and request.focus not in valid_focus:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid focus. Must be one of: {valid_focus}"
        )
    
    try:
        crew = get_refactoring_crew()
        result = crew.refactor(target=request.target, focus=request.focus)
        return {
            "status": "success",
            "target": result.get("target"),
            "focus": result.get("focus"),
            "tasks_completed": result.get("tasks_completed"),
            "result": result.get("result")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== WebSocket Endpoint ====================


async def process_crew_request(websocket: WebSocket, crew_name: str, query: str):
    """
    Execute a crew and stream output to chat websocket.
    """
    # Get crew
    crew = None
    if crew_name == "refactoring_crew":
        crew = get_refactoring_crew()
    elif crew_name == "testing_crew":
        crew = get_testing_crew()
    elif crew_name == "review_crew":
        crew = get_review_crew()
    
    if not crew:
         await websocket.send_json({"type": "error", "content": f"Crew {crew_name} not found"})
         return

    await websocket.send_json({
        "type": "routing_step",
        "step": "crew_start",
        "title": f"{crew_name.replace('_', ' ').title()} Started",
        "detail": "Initializing multi-agent workflow..."
    })

    title = crew_name.replace("_", " ").title()
    await websocket.send_json({"type": "token", "content": f"**Starting {title}...**\n\n"})
    
    crew_start_time = time.time()
    agents_used = []
    
    try:
        # Parse focus from query (e.g. "refactor X for performance" -> focus="performance")
        focus = None
        query_lower = query.lower()
        if "performance" in query_lower or "optimize" in query_lower:
            focus = "performance"
        elif "security" in query_lower or "vulnerability" in query_lower:
            focus = "security"
        elif "test" in query_lower or "coverage" in query_lower:
            focus = "tests"
        elif "doc" in query_lower or "document" in query_lower:
            focus = "docs"
        
        # review_crew and testing_crew need (code, file_path); refactoring_crew needs (target, focus)
        if crew_name in ("review_crew", "testing_crew"):
            code, file_path = _resolve_code_from_query(query)
            if not code:
                await websocket.send_json({
                    "type": "token",
                    "content": f"\n\n**Error:** Could not find or read the file. Please specify a file path in your message, e.g. `data/path/to/file.py` or \"path/to/file.py\". The codebase must be indexed first.\n\n"
                })
                await websocket.send_json({"type": "done"})
                return
            crew_input = (code, file_path)
        else:
            crew_input = (query, focus)
        
        for update in crew.run(*crew_input):
            msg = ""
            detail = ""
            
            if update["type"] == "crew_start":
                agents = update.get('agents', [])
                agents_used.extend(agents)
                if agents:
                    msg = f"*Activated agents: {', '.join(agents)}*\n"
                else:
                    msg = "*Initializing crew agents...*\n"
                detail = "Agents activated"
                
            elif update["type"] == "iteration_start":
                msg = f"\n**Iteration {update.get('iteration')}**...\n"
                detail = f"Iteration {update.get('iteration')}"
                
            elif update["type"] == "phase":
                msg = f"\n**Phase: {update.get('phase')}**...\n"
                detail = f"Phase: {update.get('phase')}"
                
            elif update["type"] == "crew_complete":
                msg = "\n\n**Execution Complete.**\n\n---\n\n"
                
                # Check for result content
                result_text = ""
                if update.get("result"):
                     result_text = update["result"]
                elif update.get("final_tests"):
                     result_text = f"```python\n{update['final_tests']}\n```"
                elif update.get("synthesis"):
                     result_text = update["synthesis"]
                     if update.get("reviews"):
                         # Append reviews if available? Maybe too long.
                         pass
                
                msg += result_text
                detail = "Finished"
                
                # MLflow crew metrics
                tracker, CrewMetrics = get_tracker_safe()
                if tracker and CrewMetrics:
                    try:
                        tracker.log_crew_metrics(CrewMetrics(
                            crew_name=crew_name,
                            agents_involved=agents_used or [crew_name],
                            total_time_s=time.time() - crew_start_time,
                            tasks_completed=update.get("tasks_completed", 0),
                            success=True
                        ))
                    except Exception:
                        pass
            
            # Send content if any (clean LLM tokens)
            if msg:
                await websocket.send_json({"type": "token", "content": clean_llm_output(msg)})
                
                # Update UI status
                if detail:
                    await websocket.send_json({
                        "type": "routing_step",
                        "step": "working",
                        "title": title,
                        "detail": detail
                    })
                
            await asyncio.sleep(0.05)

    except Exception as e:
         await websocket.send_json({"type": "token", "content": f"\n\n**Error:** Failed executing crew: {str(e)}"})

    # Done
    await websocket.send_json({"type": "done"})


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.
    
    Client sends: {"message": "query text", "use_rag": true}
    Server sends: {"type": "token", "content": "..."} for each token
    Server sends: {"type": "done"} when complete
    """
    await websocket.accept()
    print("[WS] Chat client connected")
    
    # Per-connection state: conversation memory
    memory = get_memory_safe()
    if memory:
        memory.create_session()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            use_rag = data.get("use_rag", True)
            
            if not message:
                await websocket.send_json({"type": "error", "content": "Empty message"})
                continue
            
            # Check if ready
            if not app_state["is_ready"]:
                await websocket.send_json({
                    "type": "error", 
                    "content": "No codebase indexed. Please index a codebase first."
                })
                continue
            
            # Get orchestrator and process
            try:
                orch = get_orchestrator()
                
                # Step 1: Received query
                await websocket.send_json({
                    "type": "routing_step",
                    "step": "received",
                    "title": "Query Received",
                    "detail": message[:50] + ("..." if len(message) > 50 else "")
                })
                await asyncio.sleep(0.3)
                
                # Step 2: Orchestrator classifying
                await websocket.send_json({
                    "type": "routing_step",
                    "step": "classifying",
                    "title": "Orchestrator",
                    "detail": "Classifying intent..."
                })
                
                # Send intent classification
                intent = orch.classify_intent(message)
                routing = orch._get_routing(intent)
                
                await websocket.send_json({
                    "type": "routing_step",
                    "step": "classified",
                    "title": f"Intent: {intent.name}",
                    "detail": routing.get("description", "")
                })
                await asyncio.sleep(0.3)
                
                # Step 3: Routing to agent
                agent_name = routing.get("agent", "code_explorer")
                await websocket.send_json({
                    "type": "routing_step",
                    "step": "routing",
                    "title": "Routing",
                    "detail": f"Agent: {agent_name}"
                })
                await asyncio.sleep(0.3)
                
                # Step 4: Agent processing
                if routing.get("use_crew"):
                    # Process via Crew
                    await process_crew_request(websocket, agent_name, message)
                else:
                    # Single Agent processing
                    explorer = orch.agents.get("code_explorer")
                    if explorer:
                        # Get conversation context for multi-turn (token-aware, with summarization)
                        conv_context = None
                        if memory:
                            conv_context, _ = memory.get_context_for_llm(system_prompt="")
                            if not conv_context or len(conv_context.strip()) < 20:
                                conv_context = None
                            memory.add_message("user", message)
                        
                        await websocket.send_json({
                            "type": "routing_step",
                            "step": "rag",
                            "title": "RAG Search",
                            "detail": "Finding relevant code..."
                        })
                        await asyncio.sleep(0.2)
                        
                        await websocket.send_json({
                            "type": "routing_step",
                            "step": "generating",
                            "title": "Code Explorer",
                            "detail": "Generating response..."
                        })
                        
                        await websocket.send_json({
                            "type": "intent",
                            "content": intent.name
                        })
                        
                        tracker, _ = get_tracker_safe()
                        token_count = 0
                        
                        with (tracker.track_query(message, intent.name, "code_explorer") if tracker else nullcontext()) as log_tokens:
                            response_gen = explorer.answer(
                                message, use_rag=use_rag, stream=True,
                                conversation_history=conv_context
                            )
                            
                            full_response = ""
                            for token in response_gen:
                                full_response += token
                                token_count += 1
                                await websocket.send_json({
                                    "type": "token",
                                    "content": token
                                })
                                await asyncio.sleep(0.01)
                            
                            if memory:
                                memory.add_message("assistant", full_response)
                            if tracker and log_tokens:
                                log_tokens(token_count)
                        
                        await websocket.send_json({"type": "done"})
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "content": "Agent not available"
                        })

                    
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("[WS] Chat client disconnected")


# ==================== Crew WebSocket Endpoint ====================

@app.websocket("/ws/refactor")
async def websocket_refactor(websocket: WebSocket):
    """
    WebSocket endpoint for streaming crew refactoring progress.
    
    Client sends: {"target": "code target", "focus": "security"|"performance"|"tests"|"docs"|null}
    Server sends progress updates:
      {"type": "crew_start", "agents": [...], "total": 5}
      {"type": "agent_start", "agent": "...", "index": 0}
      {"type": "agent_output", "agent": "...", "content": "..."}
      {"type": "agent_done", "agent": "...", "summary": "..."}
      {"type": "crew_done", "result": "..."}
    """
    await websocket.accept()
    print("[WS] Crew client connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            target = data.get("target", "")
            focus = data.get("focus")
            
            if not target:
                await websocket.send_json({"type": "error", "content": "No target specified"})
                continue
            
            if not app_state["is_ready"]:
                await websocket.send_json({"type": "error", "content": "No codebase indexed"})
                continue
            
            try:
                # Get the crew
                crew = get_refactoring_crew()
                
                # Define agent info
                all_agents = [
                    {"name": "Code Explorer", "role": "explorer"},
                    {"name": "Security Analyst", "role": "security"},
                    {"name": "Algorithm Optimizer", "role": "algorithm"},
                    {"name": "Test Engineer", "role": "tester"},
                    {"name": "Documentation Writer", "role": "documenter"}
                ]
                
                # Filter agents based on focus
                if focus == "security":
                    agents = [all_agents[0], all_agents[1]]
                elif focus == "performance":
                    agents = [all_agents[0], all_agents[2]]
                elif focus == "tests":
                    agents = [all_agents[0], all_agents[3]]
                elif focus == "docs":
                    agents = [all_agents[0], all_agents[4]]
                else:
                    agents = all_agents
                
                # Send crew start
                await websocket.send_json({
                    "type": "crew_start",
                    "target": target,
                    "focus": focus,
                    "agents": [a["name"] for a in agents],
                    "total": len(agents)
                })
                
                # Simulate streaming progress for each agent
                # (In production, this would hook into CrewAI's callbacks)
                import time
                
                for i, agent in enumerate(agents):
                    # Agent start
                    await websocket.send_json({
                        "type": "agent_start",
                        "agent": agent["name"],
                        "index": i,
                        "total": len(agents)
                    })
                    await asyncio.sleep(0.5)
                    
                    # Simulate agent working with progress updates
                    work_messages = [
                        f"Searching codebase for '{target}'...",
                        "Analyzing code structure...",
                        f"Checking {agent['role']} aspects...",
                        "Generating recommendations..."
                    ]
                    
                    for msg in work_messages:
                        await websocket.send_json({
                            "type": "agent_output",
                            "agent": agent["name"],
                            "content": msg
                        })
                        await asyncio.sleep(0.8)
                    
                    # Agent done
                    await websocket.send_json({
                        "type": "agent_done",
                        "agent": agent["name"],
                        "index": i,
                        "summary": f"{agent['name']} analysis complete"
                    })
                    await asyncio.sleep(0.3)
                
                # Actually run the crew and get result
                result = crew.refactor(target=target, focus=focus)
                
                # Send individual agent outputs (cleaned of LLM tokens)
                agent_outputs = result.get("agent_outputs", [])
                cleaned_outputs = []
                for agent_result in agent_outputs:
                    cleaned_output = {
                        "agent": agent_result.get("agent", "Unknown"),
                        "output": clean_llm_output(agent_result.get("output", ""))
                    }
                    cleaned_outputs.append(cleaned_output)
                    await websocket.send_json({
                        "type": "agent_result",
                        **cleaned_output
                    })
                    await asyncio.sleep(0.1)
                
                # Send final result
                await websocket.send_json({
                    "type": "crew_done",
                    "target": target,
                    "focus": focus,
                    "tasks_completed": result.get("tasks_completed", 0),
                    "result": clean_llm_output(result.get("result", "")),
                    "agent_outputs": cleaned_outputs
                })
                
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"Crew error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("🚀 Crew WebSocket client disconnected")


# ==================== Week 3: Testing & Review Crews ====================

@app.websocket("/ws/test")
async def websocket_testing_crew(websocket: WebSocket):
    """WebSocket endpoint for TestingCrew (iterative testing workflow)."""
    await websocket.accept()
    print("[WS] TestingCrew connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            target_code = data.get("code", "")
            target_file = data.get("file", "test_target.py")
            
            if not target_code:
                await websocket.send_json({
                    "type": "error",
                    "content": "No code provided"
                })
                continue
            
            try:
                crew = get_testing_crew()
                
                # Stream progress updates (clean LLM tokens)
                for progress in crew.run(target_code, target_file):
                    # Clean any string fields that might contain LLM tokens
                    cleaned = {k: clean_llm_output(v) if isinstance(v, str) else v for k, v in progress.items()}
                    await websocket.send_json(cleaned)
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"TestingCrew error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("[WS] TestingCrew disconnected")


@app.websocket("/ws/review")
async def websocket_review_crew(websocket: WebSocket):
    """WebSocket endpoint for CodeReviewCrew (parallel review workflow)."""
    await websocket.accept()
    print("[WS] CodeReviewCrew connected")
    
    try:
        while True:
            data = await websocket.receive_json()
            code = data.get("code", "")
            file_path = data.get("file", "review_target.py")
            
            if not code:
                await websocket.send_json({
                    "type": "error",
                    "content": "No code provided"
                })
                continue
            
            try:
                crew = get_review_crew()
                
                # Stream progress updates (clean LLM tokens)
                for progress in crew.run(code, file_path):
                    # Clean any string fields that might contain LLM tokens
                    cleaned = {k: clean_llm_output(v) if isinstance(v, str) else v for k, v in progress.items()}
                    await websocket.send_json(cleaned)
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                await websocket.send_json({
                    "type": "error",
                    "content": f"CodeReviewCrew error: {str(e)}"
                })
                
    except WebSocketDisconnect:
        print("[WS] CodeReviewCrew disconnected")


# ==================== Dev Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
