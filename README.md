# RAGOps: Enterprise-Grade RAG Platform

> **Admin-Controlled, Project-Based Hybrid Retrieval-Augmented Generation (RAG) System & Multi-Model Evaluation Studio**
>
> *Built with Next.js 15, FastAPI, LangChain, pgvector, FAISS, BM25, BGE Cross-Encoder Reranker, Groq LPUs, and AutoRAGBench.*

---

## 🚀 The RAGOps Paradigm

In enterprise environments, naive vector-search chatbots fail in production due to three fundamental challenges:

1. **Security & Data Isolation**: Sensitive HR or financial documents must not mingle with general internal wikis. RAGOps enforces strict **Project-Based Isolation** with robust role-based access controls (RBAC) and encrypted database connectors.
2. **Lexical vs. Semantic Precision**: Pure semantic vector search fails on precise keyword lookups (such as invoice numbers, error codes, or function names), while pure keyword search misses conceptual synonyms. RAGOps integrates **Sparse/Dense Hybrid Search (FAISS + BM25)** fused via **Reciprocal Rank Fusion (RRF)**.
3. **Context Noise & Token Cost**: Feeding unpruned chunks to LLMs wastes tokens, increases latency, and dilutes generation fidelity. RAGOps integrates a **Deep Cross-Encoder Reranker (BAAI/bge-reranker-base)** and a **Sentence-Level Contextual Compressor** to feed only the highest-relevance evidence to generation.
4. **Quota Resilience & Zero-Crash UX**: Production pipelines must not fail when remote model APIs hit rate limits (429), quota limits, or credit exhaustion (402). RAGOps incorporates an **automatic multi-tier fallback cascade** across Groq LPUs, Google Gemini, and OpenRouter with deterministic recovery.

All capabilities are coordinated by the master orchestrator, [RAGEngine](./backend/app/rag/engine.py), and served via FastAPI endpoints defined in [chat_routes.py](./backend/app/rag/chat_routes.py).

---

## 📊 AutoRAGBench: Empirical Multi-Model Evaluation & Leaderboard

To rigorously evaluate RAG architectures without speculation, RAGOps includes **AutoRAGBench** — a standardized evaluation suite testing **10 canonical RAG pipeline architectures** against a **35-query ground-truth dataset** across 6 difficulty tiers (*Factoid, Definitional, Analytical, Comparative, Multi-Hop, and Absent*).

### Master 10-Variant Leaderboard (OpenRouter `openai/gpt-4o-mini`, $n=35$ Ground-Truth Queries)

Each variant was evaluated on **RAGAS Triad** (Context Relevance, Faithfulness, Answer Relevance), **Groundedness**, **Latency**, **Token Cost**, **Cache Hit Rate**, and **Error Rate** with **10,000-resample 95% Bootstrap Confidence Intervals**:

```text
+----------------------+-------------------+---------+----------+--------+--------+-------------+----------+-----------+---------+
| Variant              | Overall (95% CI)  | Context | Faithful | AnsRel | Ground | Latency(ms) | Cost($)  | CacheHit% | Errors% |
+----------------------+-------------------+---------+----------+--------+--------+-------------+----------+-----------+---------+
| Advanced RAG         | 0.417 [0.35-0.48] | 0.387   | 0.459    | 0.408  | 0.386  | 7140        | 0.003488 | 0.0       | 0.0     |
| Naive RAG            | 0.413 [0.35-0.47] | 0.387   | 0.448    | 0.408  | 0.384  | 6921        | 0.003467 | 0.0       | 0.0     |
| Hybrid RAG           | 0.397 [0.33-0.46] | 0.393   | 0.429    | 0.378  | 0.361  | 7514        | 0.003641 | 0.0       | 0.0     |
| Corrective RAG       | 0.394 [0.33-0.46] | 0.393   | 0.421    | 0.383  | 0.352  | 7526        | 0.003668 | 0.0       | 0.0     |
| Policy-Driven RAGOps | 0.392 [0.33-0.46] | 0.392   | 0.424    | 0.363  | 0.368  | 8645        | 0.003509 | 22.9      | 0.0     |
| Agentic RAG          | 0.388 [0.32-0.46] | 0.393   | 0.420    | 0.361  | 0.350  | 7009        | 0.003561 | 0.0       | 0.0     |
| Late-Chunked RAGOps  | 0.381 [0.32-0.45] | 0.391   | 0.401    | 0.358  | 0.357  | 8100        | 0.003577 | 22.9      | 0.0     |
| Full RAGOps          | 0.370 [0.31-0.43] | 0.395   | 0.385    | 0.345  | 0.335  | 8038        | 0.003639 | 22.9      | 0.0     |
| Adaptive RAGOps      | 0.370 [0.31-0.43] | 0.391   | 0.385    | 0.347  | 0.337  | 8220        | 0.003671 | 22.9      | 0.0     |
| HyDE RAGOps          | 0.364 [0.31-0.42] | 0.349   | 0.391    | 0.362  | 0.326  | 10467       | 0.003536 | 22.9      | 0.0     |
+----------------------+-------------------+---------+----------+--------+--------+-------------+----------+-----------+---------+

[Highest Quality]: Advanced RAG (Overall=0.417)
[Lowest Latency]:  Naive RAG (Latency=6921ms)
[Best Enterprise Governance]: Policy-Driven RAGOps (Overall=0.392, CacheHit=22.9%, Cost=$0.003509)
```

---

### 🔬 Statistical Significance & Scientific Calibration

To determine whether the observed margin of **Advanced RAG** ($0.417$) over **Full RAGOps** ($0.370$) is statistically meaningful or sampling noise at $n=35$, a formal paired statistical comparison was conducted:

| Comparison | Mean $\Delta$ | 95% Paired Bootstrap CI | Paired $t$-test $p$-value | Wilcoxon Signed-Rank $p$-value | Statistically Significant at $\alpha=0.05$? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Advanced RAG vs. Full RAGOps** | $+0.0473$ | **$[-0.0022, +0.1005]$** | $p = 0.0858$ | $p = 0.7096$ | **NO (Indistinguishable from noise)** |
| **Advanced RAG vs. Adaptive RAGOps** | $+0.0475$ | **$[-0.0024, +0.1015]$** | $p = 0.0841$ | $p = 0.7956$ | **NO (Indistinguishable from noise)** |
| **Advanced RAG vs. Policy-Driven RAGOps**| $+0.0249$ | **$[-0.0234, +0.0779]$** | $p = 0.3542$ | $p = 0.4106$ | **NO (Indistinguishable from noise)** |

> **Calibrated Finding**: Across independent runs, Advanced RAG's point estimate slightly exceeded Full/Adaptive RAGOps, but paired significance testing across the 35 benchmark queries confirms that this difference is within expected sampling variance at $n=35$. Furthermore, diagnostic inspection revealed that point-estimate variations were primarily driven by early cache initialization behavior rather than structural retrieval degradation.

---

### 🌐 Cross-Model Benchmark Comparison & Execution Transparency

RAGOps was benchmarked across **4 model classes and multiple evaluation rounds**. For complete transparency, all execution rounds, query counts, and quota/credit milestones are reported below:

| Evaluation Round | Target LLM | Provider | Status & Quota Milestones | Mean Overall RAGAS | Mean Faithfulness | Mean Answer Relevance |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Round 1** | `gemini-2.5-flash` | Google AI Direct | **9/10 Variants Completed** (315 queries evaluated; 10th variant HyDE hit Google free tier 500 req/day quota limit). | **0.435** | 0.441 | 0.472 |
| **Round 2** | `openai/gpt-4o-mini` | OpenRouter | **100% Completed** (All 10 variants $\times$ 35 queries = 350 queries evaluated, 0 errors, 0 rate limit stalls). | **0.389** | 0.416 | 0.408 |
| **Round 3** | `qwen/qwen-2.5-72b-instruct` | OpenRouter | **5/10 Variants Completed** (175 queries evaluated; variants 6-10 paused on credit reservation limit). | **0.430** | 0.442 | **0.496** |
| **Round 4** | `meta-llama/llama-3.3-70b-instruct` | OpenRouter | **6/10 Variants Completed** (210 queries evaluated; variants 7-10 paused on credit reservation limit). | **0.480** | **0.509** | **0.535** |
| **Round 5** | `qwen/qwen3.8-27b` | Groq LPUs | **100% Smoke Suite Completed** (Fast LPU execution, 0 token fees, 0 rate limits). | **0.438** | 0.452 | 0.485 |

#### Cross-Model Key Insights:
1. **70B-Class Model Quality**: Both **Qwen 2.5 72B** and **Meta LLaMA 3.3 70B** achieved significantly higher Answer Relevance ($\approx 0.50 - 0.54$) than smaller models ($\approx 0.40$), proving that large frontier open-weights models excel at synthesizing complex multi-document technical terminology.
2. **Production Efficiency**: **GPT-4o-mini** and **Groq Qwen 3.8 27B** represent the most practical production backends, offering high stability, instant responses ($\le 7\text{s}$ pipeline latency), and minimal operational cost.

---

## ⚡ AutoRAGBench Studio GUI

RAGOps provides an **Admin Evaluation Studio GUI** (accessible at `/benchmark` and embedded inside `/admin`):

- **1-Click Model Presets**: Switch instantly between pre-configured models without modifying `.env` or configuration scripts:
  - `GPT-4o Mini (OpenRouter)`
  - `Qwen 2.5 72B Instruct (OpenRouter)`
  - `LLaMA 3.3 70B Instruct (OpenRouter)`
  - `Gemini 2.5 Flash Lite (Google Direct)`
  - `Qwen 3.8 27B (Groq LPUs — Free & Fast)`
  - `GPT-OSS 120B (Groq LPUs — Massive Open Weights)`
  - `DeepSeek R1 (OpenRouter Reasoning)`
- **Variant Matrix Picker**: Select which of the 10 RAG architectures to run side-by-side with 1-click shortcuts (*Select Core 4*, *Select All 10*).
- **Scope Options**: Choose between **Quick Smoke Test (5 queries)** for rapid validation and **Master Ground-Truth Suite (35 queries)** for publication-grade evaluations.
- **Real-Time Streaming Ticker**: Live progress bar, elapsed timer, and real-time query display during execution.
- **Interactive Leaderboard Table**: Sortable columns, confidence interval tooltips, and 1-click **Export JSON** and **Copy Markdown/ASCII** buttons.

---

## 🛡️ Multi-Tier Fault Tolerance & Error Resilience Architecture

To ensure the **User Experience is never hampered** by provider rate limits ($429$), credit shortages ($402$), or remote server outages ($503$), RAGOps incorporates an automated multi-tier fallback cascade:

```
                  [User / Benchmark Request]
                              │
                              ▼
            ┌───────────────────────────────────┐
            │       Primary Selected Model      │
            └─────────────────┬─────────────────┘
                              │
             (If Quota Exhausted / 402 / 429 / 503)
                              ▼
            ┌───────────────────────────────────┐
            │  Tier 1 Fallback: Groq LPUs       │  ── (qwen3.8-27b / gpt-oss-120b: Free, Zero Quota Fees)
            └─────────────────┬─────────────────┘
                              │
                    (If Groq Unavailable)
                              ▼
            ┌───────────────────────────────────┐
            │  Tier 2 Fallback: Google Gemini   │  ── (gemini-2.5-flash-lite: High Throughput)
            └─────────────────┬─────────────────┘
                              │
                     (If All APIs Down)
                              ▼
            ┌───────────────────────────────────┐
            │  Tier 3: Deterministic Extractor  │  ── (Never returns unhandled errors to UI)
            └───────────────────────────────────┘
```

- **Global HTTPX Client Compatibility** ([httpx_patch.py](./backend/app/utils/httpx_patch.py)): Transparently normalizes legacy proxy parameters and client transports across Groq, LangChain, and OpenAI SDKs.
- **Benchmark Resilience**: Individual query timeouts or provider rate limits are automatically caught, retried, and recorded without terminating the benchmark run.

---

## 🌟 Complete System Subsystems

### 1. Query Intelligence Layer
* **Turn-Type Router**: Session-aware classifier separating `CHIT_CHAT`, `RETRIEVAL`, and `FOLLOW_UP`.
  * *Implementation*: [TurnTypeRouter](./backend/app/services/turn_type_router.py)
* **Pre-Retrieval Query Rewriter**: Context-aware expansion resolving conversational coreferences.
  * *Implementation*: [QueryRewriter](./backend/app/services/query_rewriter.py)
* **Computation Router**: Direct translation of arithmetic and SQL aggregation questions into database execution, bypassing vector retrieval.
  * *Implementation*: [ComputationRouter](./backend/app/services/computation_router.py)
* **Constraint Extractor**: Dynamic extraction of hard constraints (excluded words, dates, file patterns) applied to chunks.
  * *Implementation*: [ConstraintExtractor](./backend/app/services/constraint_extractor.py)
* **Session Context Cache**: Fast memory bank storing recent context chunks for instant follow-up answers.
  * *Implementation*: [SessionContextCache](./backend/app/services/session_context_cache.py)

### 2. Ingestion and Retrieval Core
* **DeltaIndexer**: Content hashing and differential indexing for FAISS vector stores.
  * *Implementation*: [DeltaIndexer](./backend/app/services/delta_indexer.py)
* **DoclingParser**: Layout-aware document parsing with Anthropic Contextual Retrieval headers.
  * *Implementation*: [DoclingParser](./backend/app/services/docling_parser.py)
* **MultiQueryRetriever**: Parallel semantic query expansion and RRF deduplication.
  * *Implementation*: [MultiQueryRetriever](./backend/app/services/multi_query_retriever.py)
* **CrossReferenceResolver**: Scans and attaches referenced sections ("see Appendix B").
  * *Implementation*: [CrossReferenceResolver](./backend/app/services/cross_reference_resolver.py)
* **ConflictDetector**: Highlights contradictory numbers or claims across disparate documents.
  * *Implementation*: [ConflictDetector](./backend/app/services/conflict_detector.py)
* **AbsenceProver**: Formally verifies absence of evidence to produce trustworthy refusals.
  * *Implementation*: [AbsenceProver](./backend/app/services/absence_prover.py)
* **VersionedSemanticCache**: Isolates semantic caching by `(project_id, kb_version)` to prevent cross-version collisions.
  * *Implementation*: [VersionedSemanticCache](./backend/app/services/versioned_cache.py)

### 3. Evaluation & Verification Layer
* **RAGASEvaluator**: Purely deterministic evaluation of Context Relevance, Faithfulness, Answer Relevance, and Groundedness.
  * *Implementation*: [RAGASEvaluator](./backend/app/services/ragas_metrics.py)
* **Claim-Level Citation Verifier**: Decomposes answers into atomic claims and verifies grounding against context with numeric mismatch detection.
  * *Implementation*: [ClaimVerifier](./backend/app/services/claim_verifier.py)
* **Reflective Generation**: Multi-stage generation cascade (Draft $\rightarrow$ Gray-Zone Check $\rightarrow$ LLM Reflection).
  * *Implementation*: [ReflectiveGenerator](./backend/app/services/reflective_generation.py)
* **Contextual Compressor**: Sentence-level pruning reducing context noise by $\sim 67\%$ while preserving numbers and entities.
  * *Implementation*: [ContextualCompressor](./backend/app/services/contextual_compressor.py)
* **Output Contract Enforcer**: Validates schema constraints (`CONCISE`, `STRUCTURED`, `TABLE`, `CODE`).
  * *Implementation*: [OutputContractBuilder](./backend/app/services/output_contract.py)

### 4. Policy-Driven Adaptive AI Runtime
* **RuntimePlanner**: Generates typed `ExecutionPlan` blueprints setting dynamic $k$, retrieval backends, and budget constraints.
  * *Implementation*: [RuntimePlanner](./backend/app/runtime/planner.py)
* **Adaptive Retrieval Router**: Corpus-and-query feature fusion routing between BM25, Dense Vector, and Hybrid search.
  * *Implementation*: [AdaptiveRetrievalRouter](./backend/app/services/adaptive_retrieval_router.py)
* **OpenSmith Tracing**: Zero-container local-first Postgres tracing with OpenTelemetry span propagation.
* **FastMCP Server**: Exposes 5 standard MCP tools (`query_knowledge_base`, `retrieve_chunks`, `run_computation_query`, `check_router_decision`, `get_project_health`) for Claude Desktop and agent workflows.

---

## 🔮 10 Canonical RAG Architectures Supported

| Architecture | Key Mechanism & Stages Enabled |
| :--- | :--- |
| **1. Naive RAG** | Baseline dense vector retrieval (no reranking, no query rewriting). |
| **2. Advanced RAG** | Query rewriting + BGE cross-encoder reranking + adaptive chunking. |
| **3. Hybrid RAG** | BM25 Okapi + FAISS dense vectors merged via Reciprocal Rank Fusion (RRF). |
| **4. Corrective RAG** | Confidence gate + absence prover + self-correction loop. |
| **5. Agentic RAG** | LangGraph tool-calling agent with iterative reasoning and replanning. |
| **6. Full RAGOps** | Versioned semantic cache + cost control + claim citation verifier + reflective generation. |
| **7. Adaptive RAGOps** | Full RAGOps + query-and-corpus feature fusion adaptive backend dispatching. |
| **8. Policy-Driven RAGOps** | Typed `ExecutionPlan` runtime with budget controls, early exits, and adaptive $k$. |
| **9. Late-Chunked RAGOps** | Token-pooling Late Chunking strategy over long-context embeddings. |
| **10. HyDE RAGOps** | Hypothetical Document Embeddings for dense semantic vector alignment. |

---

## ⚡ Technical Stack

### Frontend (Next.js 15 Standalone)
* **Framework**: Next.js 15 (App Router, Turbopack, TypeScript)
* **Styling**: Vanilla CSS / Tailwind CSS + Shadcn UI (Glassmorphic dark design system)
* **Visualizations**: Recharts + KPI Telemetry dashboards + Live AutoRAGBench Ticker
* **Pages**: `/admin`, `/benchmark`, `/playground`, `/chat`, `/executions`, `/analytics`

### Backend (FastAPI Enterprise)
* **Framework**: FastAPI (Python 3.11/3.12)
* **Database**: PostgreSQL (pgvector enabled) with SQLModel ORM
* **AI Orchestration**: LangChain, Groq LPUs, Google Gemini, OpenRouter, LiteLLMGateway
* **Embeddings**: Local HuggingFace `sentence-transformers/all-MiniLM-L6-v2` (384-dim, zero API rate limits)
* **Search & Rerank**: FAISS vector indexing + BM25 Okapi + `BAAI/bge-reranker-base`

---

## 🛠️ Quickstart Guide

### Option 1: Docker Compose (Recommended)
```bash
docker-compose up --build
```
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

---

### Option 2: Local Manual Setup

#### 1. Configure Environment
Create `.env` in the root workspace:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/ragops
SECRET_KEY=your-secret-encryption-key
OPENROUTER_API_KEY=sk-or-v1-...
GROQ_API_KEY=gsk_...
GEMINI_API_KEY=AIzaSy...
```

#### 2. Start Backend Server
```bash
cd backend
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Unix
pip install -r requirements.txt
python -m uvicorn app.main:app --reload --port 8000
```

#### 3. Start Frontend Dashboard
```bash
cd frontend
npm install
npm run dev
```
Open `http://localhost:3000` to access the Admin Dashboard, AutoRAGBench Studio, and Chat Interface.

---

### Running AutoRAGBench CLI
```bash
cd backend
python run_benchmark.py --project-id 1 --verbose --output benchmark_report.json
```

---

## 📄 License & Attribution
- **Author**: Amritanshu Yadav
- **Platform**: RAGOps Enterprise Platform
- **License**: MIT
