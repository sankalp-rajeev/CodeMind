# CodeMind AI

**🤖 A local, multi-agent AI system for understanding and improving your codebase.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CrewAI](https://img.shields.io/badge/agents-CrewAI-green.svg)](https://crewai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Features

- **🔍 Semantic Code Search** - Ask questions in natural language, get relevant code
- **🤝 Multi-Agent Collaboration** - 5 specialized agents work together on complex tasks
- **🔒 100% Local** - Runs entirely on your hardware, no API costs
- **⚡ Fast** - RAG retrieval <500ms, simple queries <2s

### Agent Crews

| Crew | Description |
|------|-------------|
| **RefactoringCrew** | Collaborative refactoring with security + performance analysis |
| **TestingCrew** | Iterative test generation until coverage targets met |
| **CodeReviewCrew** | Parallel analysis by multiple specialists |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- RTX 3080+ (12GB+ VRAM)
- [Ollama](https://ollama.ai) installed

### GPU Setup (RTX 4080 / NVIDIA)

CodeMind uses your GPU in two places:

1. **Ollama (LLM inference)** – Ollama auto-detects NVIDIA GPUs. No config needed. Verify with:
   ```bash
   ollama run deepseek-coder:6.7b
   # In another terminal: nvidia-smi  # Should show ollama using VRAM
   ```

2. **RAG embeddings** – Uses PyTorch + sentence-transformers. Install CUDA-enabled PyTorch first:
   ```bash
   # CUDA 12.1 (RTX 40-series)
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```
   Then run `python scripts/verify_gpu.py` to confirm GPU is used.

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/codemind-ai.git
cd codemind-ai

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Pull required models
ollama pull qwen2.5-coder:7b
ollama pull qwen2.5-coder:1.5b
```

### Run

```bash
# Start backend
uvicorn src.api.main:app --reload

# In another terminal, start frontend
cd frontend
npm install
npm run dev
```

Open http://localhost:5173 and start asking questions!

---

## 📖 Documentation

- [Product Requirements Document](docs/PRD.md)
- [Folder Structure](docs/FOLDER_STRUCTURE.md)
- [Architecture](docs/ARCHITECTURE.md) *(coming soon)*
- [API Reference](docs/API.md) *(coming soon)*

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React 18, TypeScript, Vite, TailwindCSS |
| Backend | FastAPI, WebSockets |
| Agents | CrewAI, LangChain |
| LLMs | Ollama (local models) |
| RAG | ChromaDB, sentence-transformers, BM25 |
| Observability | MLflow |

---

## Performance (Benchmarked)

| Metric | Target | Actual |
|--------|--------|--------|
| RAG retrieval | <500ms | ~47ms |
| Indexing speed | - | 8.9 chunks/sec |
| Simple query response | <3s | ~2-3s |
| Codebase indexing (100 files) | <30s | ~11s |

---

## Roadmap

- [x] Week 1: Foundation (RAG + Single Agent + UI) ✅
- [ ] Week 2: CrewAI + RefactoringCrew
- [ ] Week 3: More Crews + JS/TS Support
- [ ] Week 4: Polish + MLflow + Documentation

---

## 📝 License

MIT License - see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [CrewAI](https://crewai.com/) for the multi-agent framework
- [Ollama](https://ollama.ai/) for local LLM inference
- [ChromaDB](https://www.trychroma.com/) for vector storage
