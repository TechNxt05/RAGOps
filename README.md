# RAGOps: Enterprise-Grade RAG Platform

> **Admin-Controlled, Project-Based Hybrid Retrieval-Augmented Generation (RAG) System**
>
> *Built with Next.js 15, FastAPI, LangChain, pgvector, FAISS, BM25, and BGE Cross-Encoder Reranker.*

---

## 🚀 The RAGOps Paradigm

In the modern enterprise, deploying simple vector-search chatbots is insufficient. Organizations encounter three fundamental challenges:

1. **Security & Data Isolation**: Sensitive HR documents must not mingle with general internal wikis. RAGOps introduces strict **Project-Based Isolation** with robust role-based access controls (RBAC).
2. **Lexical vs. Semantic Precision**: Pure semantic vector searches fail on specific keyword lookups (like invoice numbers or specialized codes), while pure keyword search misses conceptual context. RAGOps solves this using **Sparse/Dense Hybrid Search (FAISS + BM25)** fused via **Reciprocal Rank Fusion (RRF)**.
3. **Context Noise & Cost**: Feeding massive, redundant chunks to LLMs wastes tokens and dilutes generation quality. RAGOps incorporates a **Deep Cross-Encoder Reranker (BGE-Reranker)** and a **Sentence-Level Contextual Compressor** to keep only the highest relevance data.

---

## 🌟 Advanced Production Upgrades (Phase 1, 2, & 3)

RAGOps has been enhanced with production-grade components designed to optimize retrieval quality, control LLM spend, and supply granular pipeline diagnostics.

### Phase 1: Query Intelligence Layer
* **Turn-Type Router**: Session-aware gating that classifies query intents into `CHIT_CHAT`, `RETRIEVAL`, or `FOLLOW_UP`. It bypasses the retrieval pipeline for chit-chat or fetches directly from cache for session follow-ups.
* **Pre-Retrieval Query Rewriter**: History-aware query rewriter that leverages conversation history to expand search query candidates, solving pronoun reference dilution.
* **Computation Router**: Directly translates analytical database questions into SQL query statements, bypassing standard vector document retrieval and returning exact database calculations.
* **Constraint Extractor**: Identifies and extracts hard constraints from queries (e.g. excluded terms, date boundaries, file types, source patterns) and applies them dynamically to context chunks.
* **Session Context Cache**: Implements an active memory bank caching raw context chunks for follow-up questions, preventing redundant vector search retrievals.

### Phase 2: Ingestion and Retrieval Core
* **DeltaIndexer**: Hashes document chunk payloads during ingestion, computing differentials to add, update, or prune only modified chunks in FAISS indices.
* **DoclingParser**: Advanced layout-aware document parser targeting PDFs and Word files to extract tables, captions, headers, and semantic structures natively.
* **MultiQueryRetriever**: Employs query-variant generation pipelines, executing parallel searches and deduplicating results to maximize semantic context coverage.
* **CrossReferenceResolver**: Scans chunks for internal references (e.g., "see Section 4.2", "detailed in Appendix B") and resolves them by retrieving the parent/child chunks.
* **ConflictDetector**: Compares facts across retrieved chunks, highlighting contradictory numbers, dates, or statements from different files to alert users.
* **AbsenceProver**: Handles low-confidence queries by checking if information is truly absent in the database, yielding verified refusals.
* **IngestionScanner**: Periodic local directory scanner that automatically discovers, extracts, and indexes documents in the background.
* **VersionedSemanticCache**: Semantic cache that maps questions to answers while respecting document versions and hashing signatures.

### Phase 3: Evaluation and Output Layer
* **SemanticRouter**: Classifies queries into `FACTOID`, `ANALYTICAL`, `COMPARATIVE`, `PROCEDURAL`, or `DEFINITIONAL`, dynamically choosing search strategies (`FAST`, `STANDARD`, `DEEP`, `PARALLEL`) to override retrieval top_k, multi-query, and reranking parameters.
* **ContextualCompressor**: A query-aware sentence-level compressor that strips filler text while preserving key entities, currencies, codes, and numerical values verbatim.
* **RAGASEvaluator**: Purely deterministic mapping of TF-IDF quality signals to the standard RAGAS triad:
  * *Context Relevance*: How relevant retrieved chunks are to the user's query.
  * *Faithfulness*: If response statements are fully supported by context.
  * *Answer Relevance*: How directly the generated answer addresses the question.
  * *Groundedness*: Fraction of generated response tokens present in source documents.
* **OutputContract**: Generates and enforces formatting constraints (`CONCISE`, `STRUCTURED`, `DETAILED`, `TABLE`, or `CODE`) with post-generation compliance checks.
* **PromptCacheManager**: Structures system prompts to enable prefix caching for Anthropic (using `cache_control` markers) and OpenAI, logging cost-savings estimations.

---

## ✨ Upgraded Enterprise Features

| Capability | Admin | Client |
|------------|:-----:|:------:|
| **Sparse/Dense Hybrid Search** (FAISS + BM25 + RRF) | ✅ | ❌ (View Config) |
| **Deep Reranking** (`BAAI/bge-reranker-base`) | ✅ | ❌ (View Config) |
| **Contextual Compression** (Sentence-Level Filter) | ✅ | ❌ (View Config) |
| **RAGAS Quality Telemetry** (Triad + Groundedness) | ✅ | ✅ |
| **Output Contract Constraints** (Concise, Code, Table, etc.) | ✅ | ✅ |
| **Prompt Caching Telemetry** (Savings estimations) | ✅ | ❌ |
| Dynamic Semantic-vs-Lexical Weight Configuration | ✅ | ❌ |
| Switch embeddings (Google Cloud / Local HF MiniLM) | ✅ | ❌ |
| Live parallel model comparison & chat panel | ✅ | ❌ |
| Glowing Indigo Analytics Panel (Daily RAGAS Trends, Cache Savings) | ✅ | ❌ |
| Chat with project-isolated knowledge bases | ✅ | ✅ |
| Citation click analytics & source validation | ✅ | ✅ |

---

## 🔮 Pipeline Architecture

RAGOps features a state-of-the-art multi-stage hybrid search, reranking, and context filtering workflow:

```mermaid
graph TD
    Query(["User Query"]) --> TurnType{"TurnTypeRouter"}
    
    TurnType -->|Chit-Chat| ChitChat["LLM Simple Response"]
    TurnType -->|Follow-Up| SessionCache["Session Context Cache (Get Chunks)"]
    TurnType -->|Retrieval / New Turn| CompRouter{"Computation Router"}
    
    CompRouter -->|SQL Aggregation| CompEngine["DB Direct Execution (No Retrieval)"]
    CompRouter -->|Standard RAG| QueryRewriter["Query Rewriter & Expander"]
    
    QueryRewriter --> ConstraintExtractor["Constraint Extractor (Extract Constraints)"]
    ConstraintExtractor --> SemanticCache{"Versioned Semantic Cache"}
    
    SemanticCache -->|Hit| CacheReturn["Instant Answer Delivery"]
    SemanticCache -->|Miss| SemanticRouter["Semantic Query Router (Router Strategy)"]
    
    SemanticRouter --> MultiQuery["Multi-Query Retrieval Engine (FAISS + BM25)"]
    MultiQuery --> XRef["Cross-Reference Resolver (Parent/Child Resolution)"]
    
    XRef --> ConstraintFilter["Constraint Extractor (Apply Filters to Chunks)"]
    ConstraintFilter --> Rerank["BGE Cross-Encoder Reranker (top_n)"]
    
    Rerank --> Conflict["Conflict Detector (Fact Contradiction Check)"]
    Conflict --> Compressor["Sentence-Level Contextual Compressor"]
    
    Compressor --> Absence{"Absence Prover (Low Confidence Path)"}
    Absence -->|Absence Proven| Refusal["Refusal to Generate"]
    Absence -->|Chunks Found/Retry| ConfGate{"Confidence Gate (>= 0.65)"}
    
    ConfGate -->|Passed| PromptCache["Prompt Cache Manager (Anthropic/OpenAI Prefix Cache)"]
    PromptCache --> OutputContract["Output Contract Builder (System Prompt Directives)"]
    OutputContract --> LLM["LLM Generation"]
    
    LLM --> PostVerify["Post-Generation Output Verifier & RAGAS Evaluator"]
    PostVerify --> DB["Persist to Database (Message / QueryLog)"]
```

---

## 🔮 RAG Architecture Support

| Architecture | Status | Implementation |
|---|---|---|
| Naive RAG | ✅ | Baseline semantic retrieval |
| Advanced RAG | ✅ | Query rewriting + BGE reranker + adaptive chunking |
| Hybrid RAG | ✅ | BM25 + FAISS + RRF fusion |
| Corrective RAG | ✅ | Confidence gate + pre-generation validation |
| Agentic RAG | ✅ | LangGraph retrieval agent with replanning |
| Evaluation Layer | ✅ | Inline RAGAS Triad + Groundedness Metrics |

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
* **AI Orchestration**: LangChain, Groq (Llama 3.3), Google Gemini, Langchain Anthropic, Langchain OpenAI
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
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
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
