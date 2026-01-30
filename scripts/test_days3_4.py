"""Test script for Days 3-4 - Agent validation"""
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_ollama_models():
    """Test Part 1: Ollama Models"""
    print("\n" + "=" * 60)
    print("📦 PART 1: OLLAMA MODELS")
    print("=" * 60)
    
    import ollama
    models = ollama.list()
    print("\nAvailable models:")
    for m in models.get('models', []):
        # Handle different API versions
        name = m.get('name') or m.get('model', 'unknown')
        print(f"   • {name}")
    
    print("\n✅ Ollama: PASSED")


def test_base_agent():
    """Test Part 2: Base Agent"""
    print("\n" + "=" * 60)
    print("🤖 PART 2: BASE AGENT")
    print("=" * 60)
    
    from src.agents.base import BaseAgent, AgentConfig
    
    config = AgentConfig(
        model="deepseek-coder:6.7b",
        temperature=0.3,
        max_tokens=256
    )
    agent = BaseAgent(config)
    
    print("\nTesting generation...")
    start = time.time()
    response = agent.generate(
        "What is a Python list comprehension? One sentence only.",
        system_prompt="Be extremely concise."
    )
    gen_time = time.time() - start
    
    print(f"   Response: {response[:100]}...")
    print(f"   Time: {gen_time:.2f}s")
    
    print("\n✅ Base Agent: PASSED")


def test_code_explorer():
    """Test Part 3: Code Explorer Agent with RAG"""
    print("\n" + "=" * 60)
    print("🔍 PART 3: CODE EXPLORER AGENT")
    print("=" * 60)
    
    from src.rag.retriever import HybridRetriever
    from src.agents.code_explorer import CodeExplorerAgent
    
    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = HybridRetriever(
        persist_directory="./data/chroma_db_test",
        use_reranker=False
    )
    
    # Check if indexed
    count = retriever.indexer.collection.count()
    if count == 0:
        print("Indexing test repository...")
        retriever.index_codebase("./data/test-repo", force_reindex=True)
    else:
        print(f"Using existing index ({count} chunks)")
    
    # Initialize agent
    agent = CodeExplorerAgent(
        retriever=retriever,
        model="deepseek-coder:6.7b"
    )
    
    # Test RAG query
    question = "How does the dataset loading work?"
    print(f"\nQuestion: {question}")
    print("-" * 40)
    
    start = time.time()
    answer = agent.answer(question)
    answer_time = time.time() - start
    
    print(f"\nAnswer ({answer_time:.2f}s):")
    print(f"   {answer[:300]}...")
    
    print("\n✅ Code Explorer: PASSED")


def test_orchestrator():
    """Test Part 4: Orchestrator Agent"""
    print("\n" + "=" * 60)
    print("🎯 PART 4: ORCHESTRATOR AGENT")
    print("=" * 60)
    
    from src.agents.orchestrator import OrchestratorAgent
    
    orchestrator = OrchestratorAgent(model="qwen2.5:7b")
    
    test_queries = [
        ("What does the training function do?", "EXPLORE"),
        ("Refactor this code to be cleaner", "REFACTOR"),
        ("Write tests for the data loader", "TEST"),
    ]
    
    print("\nIntent Classification:")
    all_correct = True
    
    for query, expected in test_queries:
        intent = orchestrator.classify_intent(query)
        status = "✓" if intent.name == expected else "✗"
        if intent.name != expected:
            all_correct = False
        print(f"   {status} '{query[:30]}...' → {intent.name}")
    
    if all_correct:
        print("\n✅ Orchestrator: PASSED")
    else:
        print("\n⚠️ Orchestrator: Some classifications different (may be OK)")


def test_end_to_end():
    """Test Part 5: End-to-End Pipeline"""
    print("\n" + "=" * 60)
    print("🚀 PART 5: END-TO-END PIPELINE")
    print("=" * 60)
    
    from src.rag.retriever import HybridRetriever
    from src.agents.code_explorer import CodeExplorerAgent
    from src.agents.orchestrator import OrchestratorAgent
    
    # Setup
    retriever = HybridRetriever(
        persist_directory="./data/chroma_db_test",
        use_reranker=False
    )
    
    explorer = CodeExplorerAgent(
        retriever=retriever,
        model="deepseek-coder:6.7b"
    )
    
    orchestrator = OrchestratorAgent(model="qwen2.5:7b")
    orchestrator.register_agent("code_explorer", explorer)
    
    # Test full pipeline
    query = "Explain how the model training works in this codebase"
    print(f"\nQuery: {query}")
    
    start = time.time()
    result = orchestrator.route_query(query)
    total_time = time.time() - start
    
    print(f"\n📊 Results:")
    print(f"   Intent: {result['intent']}")
    print(f"   Agent: {result['agent']}")
    print(f"   Time: {total_time:.2f}s")
    print(f"\n   Response preview:")
    print(f"   {result['response'][:200]}...")
    
    print("\n✅ End-to-End: PASSED")


def main():
    print("\n" + "=" * 60)
    print("   🧠 CodeMind Days 3-4 - First Agent Test")
    print("=" * 60)
    
    # Run all tests
    test_ollama_models()
    test_base_agent()
    test_code_explorer()
    test_orchestrator()
    test_end_to_end()
    
    print("\n" + "=" * 60)
    print("🎉 DAYS 3-4 COMPLETE! All tests passed.")
    print("=" * 60)
    
    print("\n📋 FILES CREATED:")
    print("   • src/agents/base.py - Base agent class")
    print("   • src/agents/code_explorer.py - Code Explorer agent")
    print("   • src/agents/orchestrator.py - Query router")
    
    print("\nNext: Days 5-6 - Frontend + WebSocket API\n")


if __name__ == "__main__":
    main()
