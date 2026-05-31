# RAGOps: Enterprise-Grade RAG Platform

> **Admin-Controlled, Project-Based Hybrid Retrieval Augmented Generation System**
>
> *Built with Next.js, FastAPI, LangChain, pgvector, FAISS, and BGE Cross-Encoder Reranker.*

---

## 🚀 The RAGOps Paradigm

In the modern enterprise, deploying simple vector-search chatbots is insufficient. Organizations encounter three fundamental challenges:

1. **Security & Data Isolation**: Sensitive HR documents must not mingle with general internal wikis. RAGOps introduces strict **Project-Based Isolation** with robust role-based access controls (RBAC).
2. **Lexical vs. Semantic Precision**: Pure semantic vector searches fail on specific keyword lookups (like invoice numbers or specialized codes), while pure keyword search misses conceptual context. RAGOps solves this using **Sparse/Dense Hybrid Search (FAISS + BM25)** fused via **Reciprocal Rank Fusion (RRF)**.
3. **Context Noise & Cost**: Feeding massive, redundant chunks to LLMs wastes tokens and dilutes generation quality. RAGOps incorporates a **Deep Cross-Encoder Reranker (BGE-Reranker)** and a **TF-IDF Cosine Similarity Context Pruner** to keep only the highest relevance data.

---

## 🌟 Advanced Production Upgrades (May 2026)

RAGOps has been enhanced with five advanced production-grade components designed to optimize retrieval quality, control LLM spend, and supply granular pipeline diagnostics:

1. **Adaptive Chunking Engine (Ekimetrics LREC 2026)**: Rather than naive fixed-size chunking, RAGOps evaluates four candidate chunking strategies (Fixed, Paragraph, Semantic, and Recursive) against 5 intrinsic metrics (size compliance, cohesive density, boundary flow, blocks integrity, and pronoun antecedent coherence) to select the mathematically optimal chunk configuration per-document.
2. **Pre-Retrieval Query Understanding**: Performs complexity classification (Factoid, Analytical, and Multi-Hop), automatic sub-query decomposition for multi-hop queries, and domain-specific query expansion to prepare optimal search candidates.
3. **Source Confidence & Hallucination Gate**: Formulates a weighted retrieve-relevance confidence score based on source type authority, chunk-query agreement, and document freshness decay. Generation only proceeds if confidence is $\ge 0.65$, preventing hallucinations by returning a graceful refusal.
4. **Cost Control Layer**:
   * **Semantic Query Cache**: A fast TF-IDF cosine-similarity cache that intercepts duplicate/highly similar queries to serve answers instantly.
   * **Heuristic Heuristic Router**: Smart, low-overhead query routing matching complexity classifications to optimal model sizes (e.g. routing simple factoids to `gemini-1.5-flash` and complex analytical queries to `gemini-1.5-pro`).
   * **Spend Circuit Breaker**: Implements hourly/daily USD spending caps to safeguard against budget overruns and run-away agents.
5. **E2E Pipeline Tracing & Timing Metrics**: Deep pipeline tracers track and record execution timing (ms) and success/failure statuses for every retrieval, pruning, and generation stage—viewable inside the interactive Trace Playground.

---

## ✨ Upgraded Enterprise Features

| Capability | Admin | Client |
|------------|:-----:|:------:|
| **Sparse/Dense Hybrid Search** (FAISS + BM25 + RRF) | ✅ | ❌ (View Config) |
| **Deep Reranking** (`BAAI/bge-reranker-base`) | ✅ | ❌ (View Config) |
| **Context Pruning** (TF-IDF Similarity Filter) | ✅ | ❌ (View Config) |
| Dynamic Semantic-vs-Lexical Weight Configuration | ✅ | ❌ |
| Switch embeddings (Google Cloud / Local HF MiniLM) | ✅ | ❌ |
| Live parallel model comparison & chat panel | ✅ | ❌ |
| Glowing Indigo Analytics Panel (Pruning Savings, Rerank KPIs) | ✅ | ❌ |
| Chat with project-isolated knowledge bases | ✅ | ✅ |
| Citation click analytics & source validation | ✅ | ✅ |

---

## 🔮 Retrieval Architecture

RAGOps features a state-of-the-art multi-stage hybrid search, reranking, and context filtering workflow:

```mermaid
graph TD
    Query(["User Query"]) --> Hybrid{"Hybrid Search Trigger"}
    Hybrid -->|Dense Semantic| Semantic["FAISS Vector Store (all-MiniLM-L6-v2)"]
    Hybrid -->|Sparse Lexical| Lexical["BM25 Lexical Index (Project-Isolated)"]
    Semantic --> RRF["Reciprocal Rank Fusion (RRF)"]
    Lexical --> RRF
    RRF --> Rerank["BGE Cross-Encoder Reranker (BAAI/bge-reranker-base)"]
    Rerank --> Prune["Context Pruner (TF-IDF Cosine Similarity Filter)"]
    Prune --> Output(["Pruned High-Relevance Chunks (Max Context Window)"])
```

### Retrieval & Pruning Stages

1. **Sparse Lexical Indexing**: Uses rank-bm25 index isolating indexes per-project and caching indexes on-disk for lightning-fast reads.
2. **Reciprocal Rank Fusion (RRF)**: Merges dense FAISS vector rankings with sparse BM25 keyword rankings based on adjustable weights.
3. **BGE Reranking**: Feeds query and top-fused chunks into `BAAI/bge-reranker-base` to compute precise cross-attention relevance scores, reranking chunks for maximum quality.
4. **TF-IDF Context Pruning**: Employs scikit-learn cosine-similarity vectorizers to filter out duplicate, redundant, or low-scoring context chunks.

---

## ⚡ Technical Stack

### Frontend (Next.js Standalone)
* **Framework**: Next.js 15 (App Router, TypeScript)
* **Styling**: Tailwind CSS + Shadcn UI (Glassmorphic dark design)
* **Visualizations**: Recharts + Glowing Indigo KPI dashboards
* **State Management**: React Context API + Custom Hooks

### Backend (FastAPI Enterprise)
* **Framework**: FastAPI (Python 3.11)
* **Database**: PostgreSQL (pgvector enabled) with robust SQLModel layers
* **AI Orchestration**: LangChain, Groq (Llama 3.3), Google Gemini
* **Models Cache**: On-build HuggingFace model cache warming (zero first-request cold-start latency)
* **Search Engines**: Local FAISS on-disk indexes + Project-isolated BM25 indexes

---

## 🐳 Docker Production Setup (Recommended)

RAGOps is fully containerized using Docker and Docker Compose, enabling instant, single-command production deployment with database schema synchronization and warm model downloads pre-cached into backend images.

**Launch the entire RAGOps platform:**
```bash
docker-compose up --build
```

This starts:
1. **Frontend**: Standalone Next.js production build served on `http://localhost:3000`
2. **Backend**: FastAPI Uvicorn ASGI server served on `http://localhost:8000`
3. **Database**: PostgreSQL container equipped with `pgvector` and standard schema migrations triggers

---

## 🛠️ Local Manual Setup

### 1. Environment Configurations
Rename `.env.example` to `backend/.env` and `frontend/.env.local` and add your LLM API keys:
```env
DATABASE_URL=postgresql://neondb_owner:npg_v94oVqhCKZrw@...
SECRET_KEY=your_secret_key
GEMINI_API_KEY=AIzaSy...
GROQ_API_KEY=gsk_...
```

### 2. Backend Server Installation
```bash
cd backend
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On Unix
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Frontend Dashboard Installation
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to interact with the premium enterprise interface!

---

## 🧪 Integration Verification Suite

To guarantee 100% runtime safety, RAGOps includes an end-to-end integration verification suite. It dynamically creates test projects, verifies role-based permissions (blocking unauthorized users with `403 Forbidden` and welcoming admins with `200 OK`), exercises chat session pipelines, and executes clean database cascade deletes.

**Run the verification suite:**
```bash
cd backend
python verify_backend.py
```

**Verification Results:**
```text
[PASS] API Health 
[PASS] Admin Login Status: 200
[PASS] Client Login Status: 200
[PASS] Create Verification Project Status: 200
[PASS] RBAC Enforcement (Client Blocked) Status: 403
[PASS] RBAC Enforcement (Admin Allowed) Status: 200
[PASS] Chat Endpoint Schema Integration Status: 200
Chat response content successfully returned!
[PASS] Cleanup Verification Project Status: 200
```

---

**Author**: Amritanshu Yadav  
**License**: MIT

