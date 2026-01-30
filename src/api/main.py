"""
FastAPI Backend for CodeMind AI

Provides:
- WebSocket endpoint for streaming chat
- REST endpoints for indexing and status
- Integration with OrchestratorAgent
"""

import asyncio
import json
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import our modules (will be initialized lazily)
retriever = None
orchestrator = None


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


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    print("🚀 CodeMind AI starting up...")
    
    # Check if we have existing index
    try:
        ret = get_retriever()
        count = ret.indexer.collection.count()
        if count > 0:
            app_state["chunks_count"] = count
            app_state["is_ready"] = True
            print(f"✅ Loaded existing index with {count} chunks")
    except Exception as e:
        print(f"⚠️ No existing index: {e}")
    
    yield
    
    print("👋 CodeMind AI shutting down...")


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
    return {
        "indexed": app_state["is_ready"],
        "chunks": app_state["chunks_count"],
        "directory": app_state["indexed_directory"]
    }


@app.post("/api/index")
async def index_codebase(request: IndexRequest) -> Dict[str, Any]:
    """Index a codebase directory."""
    directory = Path(request.directory)
    
    if not directory.exists():
        raise HTTPException(status_code=400, detail=f"Directory not found: {request.directory}")
    
    try:
        ret = get_retriever()
        stats = ret.index_codebase(
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


# ==================== WebSocket Endpoint ====================

@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat.
    
    Client sends: {"message": "query text", "use_rag": true}
    Server sends: {"type": "token", "content": "..."} for each token
    Server sends: {"type": "done"} when complete
    """
    await websocket.accept()
    print("🔌 WebSocket client connected")
    
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
                
                # Send intent classification
                intent = orch.classify_intent(message)
                await websocket.send_json({
                    "type": "intent",
                    "content": intent.name
                })
                
                # Get the agent and generate response
                explorer = orch.agents.get("code_explorer")
                if explorer:
                    # Stream the response
                    response_gen = explorer.answer(message, use_rag=use_rag, stream=True)
                    
                    full_response = ""
                    for token in response_gen:
                        full_response += token
                        await websocket.send_json({
                            "type": "token",
                            "content": token
                        })
                        # Small delay to prevent overwhelming the client
                        await asyncio.sleep(0.01)
                    
                    # Send done signal
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
        print("🔌 WebSocket client disconnected")


# ==================== Dev Server ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
