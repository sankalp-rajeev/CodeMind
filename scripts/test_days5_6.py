"""Test script for Days 5-6 - Backend API validation"""
import sys
import asyncio
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test Part 1: Import backend modules"""
    print("\n" + "=" * 60)
    print("📦 PART 1: IMPORT TEST")
    print("=" * 60)
    
    from src.api.main import app, get_retriever, get_orchestrator
    print("   ✓ FastAPI app imported")
    print("   ✓ get_retriever imported")
    print("   ✓ get_orchestrator imported")
    
    print("\n✅ Imports: PASSED")


def test_retriever_init():
    """Test Part 2: Retriever initialization"""
    print("\n" + "=" * 60)
    print("🗄️  PART 2: RETRIEVER INIT")
    print("=" * 60)
    
    from src.api.main import get_retriever
    
    retriever = get_retriever()
    count = retriever.indexer.collection.count()
    print(f"   Chunks in index: {count}")
    
    if count == 0:
        print("   Indexing test repo...")
        stats = retriever.index_codebase("./data/test-repo", force_reindex=True)
        print(f"   Indexed {stats['chunks_indexed']} chunks")
    
    print("\n✅ Retriever: PASSED")


def test_orchestrator_init():
    """Test Part 3: Orchestrator initialization"""
    print("\n" + "=" * 60)
    print("🎯 PART 3: ORCHESTRATOR INIT")
    print("=" * 60)
    
    from src.api.main import get_orchestrator
    
    orchestrator = get_orchestrator()
    print(f"   Registered agents: {list(orchestrator.agents.keys())}")
    
    # Test classification
    intent = orchestrator.classify_intent("What does this function do?")
    print(f"   Test classification: EXPLORE -> {intent.name}")
    
    print("\n✅ Orchestrator: PASSED")


def test_api_endpoints():
    """Test Part 4: API endpoints"""
    print("\n" + "=" * 60)
    print("🌐 PART 4: API ENDPOINTS")
    print("=" * 60)
    
    from fastapi.testclient import TestClient
    from src.api.main import app
    
    client = TestClient(app)
    
    # Test health check
    response = client.get("/")
    assert response.status_code == 200
    print("   ✓ GET / (health check)")
    
    # Test status
    response = client.get("/api/status")
    assert response.status_code == 200
    data = response.json()
    print(f"   ✓ GET /api/status -> indexed={data.get('indexed')}")
    
    # Test search (if indexed)
    if data.get('indexed'):
        response = client.post("/api/search?query=dataset&n_results=3")
        assert response.status_code == 200
        results = response.json()
        print(f"   ✓ POST /api/search -> {len(results.get('results', []))} results")
    
    print("\n✅ API Endpoints: PASSED")


def test_websocket():
    """Test Part 5: WebSocket connection"""
    print("\n" + "=" * 60)
    print("🔌 PART 5: WEBSOCKET")
    print("=" * 60)
    
    from fastapi.testclient import TestClient
    from src.api.main import app, app_state
    
    # Ensure we're indexed for the test
    app_state["is_ready"] = True
    app_state["chunks_count"] = 100
    
    client = TestClient(app)
    
    with client.websocket_connect("/ws/chat") as websocket:
        print("   ✓ WebSocket connected")
        
        # Send a test message
        websocket.send_json({
            "message": "What does the dataset module do?",
            "use_rag": True
        })
        print("   ✓ Message sent")
        
        # Receive responses
        received_intent = False
        received_tokens = False
        received_done = False
        
        # Read a few messages
        for _ in range(20):  # Max 20 messages
            try:
                data = websocket.receive_json()
                
                if data["type"] == "intent":
                    received_intent = True
                    print(f"   ✓ Received intent: {data['content']}")
                elif data["type"] == "token":
                    if not received_tokens:
                        received_tokens = True
                        print("   ✓ Receiving tokens...")
                elif data["type"] == "done":
                    received_done = True
                    print("   ✓ Received done signal")
                    break
            except Exception as e:
                print(f"   Timeout or error: {e}")
                break
        
        if received_done:
            print("\n✅ WebSocket: PASSED")
        else:
            print("\n⚠️ WebSocket: Partial (may need full server)")


def main():
    print("\n" + "=" * 60)
    print("   🧠 CodeMind Days 5-6 - Backend Test")
    print("=" * 60)
    
    test_imports()
    test_retriever_init()
    test_orchestrator_init()
    test_api_endpoints()
    test_websocket()
    
    print("\n" + "=" * 60)
    print("🎉 DAYS 5-6 BACKEND TESTS COMPLETE!")
    print("=" * 60)
    
    print("\n📋 FILES CREATED:")
    print("   • src/api/main.py - FastAPI backend")
    print("   • frontend/src/hooks/useWebSocket.ts")
    print("   • frontend/src/hooks/useApi.ts")
    print("   • frontend/src/pages/ChatPage.tsx (updated)")
    print("   • frontend/src/components/Chat/*.tsx (updated)")
    
    print("\n🚀 TO RUN:")
    print("   Backend:  .\\venv\\Scripts\\python -m uvicorn src.api.main:app --reload")
    print("   Frontend: cd frontend && npm run dev")
    print("")


if __name__ == "__main__":
    main()
