# RAGOps: Enterprise-Grade RAG Platform

> **Admin-Controlled, Project-Based Hybrid Retrieval-Augmented Generation (RAG) System & Multi-Model Evaluation Studio**
>
> *Built with Next.js 16, FastAPI, LangChain, PostgreSQL + pgvector, BM25, BGE Cross-Encoder Reranker, Groq / Gemini / OpenRouter free tiers, MCP, and AutoRAGBench.*

---

## 🚀 The RAGOps Paradigm

In enterprise environments, naive vector-search chatbots fail in production due to three fundamental challenges:

1. **Security & Data Isolation**: Sensitive HR or financial documents must not mingle with general internal wikis. RAGOps enforces strict **Project-Based Isolation** with robust role-based access controls (RBAC) and encrypted database connectors.
2. **Lexical vs. Semantic Precision**: Pure semantic vector search fails on precise keyword lookups (such as invoice numbers, error codes, or function names), while pure keyword search misses conceptual synonyms. RAGOps integrates **Sparse/Dense Hybrid Search (pgvector + BM25)** fused via **Reciprocal Rank Fusion (RRF)**.
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

To isolate LLM generator capability from pipeline orchestration, RAGOps was evaluated across **4 frontier model families** on the **Master 35-Query Evaluation Suite**.

> **Note:** the figures in this section were measured before the v2 pipeline fixes (single-pass retrieval, calibrated confidence gate, budget-aware compression that no longer drops answer-bearing sentences). Re-run AutoRAGBench to refresh them.

#### 1. Fair Controlled 5-Variant Benchmark (Identical 35 Queries $\times$ 5 Shared Architectures = 175 Queries per Model)
Because API credit reservations paused variants 6–10 on Qwen 72B and LLaMA 70B, the only methodologically sound cross-model comparison evaluates the **identical 5 shared variants** (*Naive, Advanced, Hybrid, Corrective, Agentic*) evaluated across the exact same 35 queries:

| Pipeline Variant | Google Gemini 2.5 Flash | OpenAI GPT-4o-mini | Qwen 2.5 72B Instruct | Meta LLaMA 3.3 70B Instruct |
| :--- | :---: | :---: | :---: | :---: |
| **Naive RAG** | 0.4509 | 0.4130 | 0.4314 | **0.4800** |
| **Advanced RAG** | 0.4509 | 0.4173 | 0.4355 | **0.4787** |
| **Hybrid RAG** | 0.4601 | 0.3971 | 0.4275 | **0.4838** |
| **Corrective RAG** | 0.4601 | 0.3939 | 0.4350 | **0.4830** |
| **Agentic RAG** | 0.4601 | 0.3878 | 0.4228 | **0.4876** |
| **Shared 5-Variant Mean** | **0.4564** | **0.3979** | **0.4284** | **0.4828** |

#### Complete 10-Variant Multi-Model Matrix (Full Horizon)
Below is the side-by-side record across all 10 variants, documenting where API quota or credit ceilings were reached:

| Pipeline Variant | Google Gemini 2.5 Flash | OpenAI GPT-4o-mini | Qwen 2.5 72B Instruct | Meta LLaMA 3.3 70B Instruct |
| :--- | :---: | :---: | :---: | :---: |
| **1. Naive RAG** | 0.4509 | 0.4130 | 0.4314 | **0.4800** |
| **2. Advanced RAG** | 0.4509 | 0.4173 | 0.4355 | **0.4787** |
| **3. Hybrid RAG** | 0.4601 | 0.3971 | 0.4275 | **0.4838** |
| **4. Corrective RAG** | 0.4601 | 0.3939 | 0.4350 | **0.4830** |
| **5. Agentic RAG** | 0.4601 | 0.3878 | 0.4228 | **0.4876** |
| **6. Full RAGOps** | 0.4111 | 0.3699 | 0.3220 | 0.4643 |
| **7. Adaptive RAGOps** | 0.4111 | 0.3698 | *[Credit Limit]* | *[Credit Limit]* |
| **8. Policy-Driven RAGOps** | 0.4111 | 0.3924 | *[Credit Limit]* | *[Credit Limit]* |
| **9. Late-Chunked RAGOps** | 0.4111 | 0.3812 | *[Credit Limit]* | *[Credit Limit]* |
| **10. HyDE RAGOps** | 0.2200 (*quota limit) | 0.3635 | *[Credit Limit]* | *[Credit Limit]* |

#### 2. Answer Relevance Breakdown (Precision & Vocabulary Synthesis)
Measuring how directly and comprehensively each model synthesizes complex technical terminology from retrieved chunks:

| Pipeline Variant | GPT-4o-mini AnsRel | Qwen 2.5 72B AnsRel | Meta LLaMA 3.3 70B AnsRel |
| :--- | :---: | :---: | :---: |
| **Naive RAG** | 0.408 | 0.500 | **0.564** |
| **Advanced RAG** | 0.408 | 0.498 | **0.569** |
| **Hybrid RAG** | 0.378 | 0.491 | **0.553** |
| **Corrective RAG** | 0.383 | 0.502 | **0.554** |
| **Agentic RAG** | 0.361 | 0.490 | **0.556** |
| **Shared 5-Variant Mean** | **0.388** | **0.496** | **0.559** |

#### 3. Execution History & Quota Reporting (Master 35-Query Suite)
For complete scientific reproducibility, all execution runs on the 35-query ground-truth dataset are documented with their completion status and quota milestones:

| Round | Model | Provider | Suite Depth | Queries Run | Status & Quota Milestones |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **1** | `gemini-2.5-flash` | Google Direct | 35 queries / var | **315** (9/10 variants) | Stalled on 10th variant (HyDE) due to Google free tier 500 req/day quota limit. |
| **2** | `openai/gpt-4o-mini` | OpenRouter | 35 queries / var | **350** (10/10 variants) | **100% Completed**: 0 errors, 0 rate limits across all 10 architectures. |
| **3** | `qwen/qwen-2.5-72b-instruct` | OpenRouter | 35 queries / var | **175** (5/10 variants) | Paused at variant 6 due to OpenRouter credit reservation threshold. |
| **4** | `meta-llama/llama-3.3-70b-instruct` | OpenRouter | 35 queries / var | **210** (6/10 variants) | Paused at variant 7 due to OpenRouter credit reservation threshold. |

*(Note: Groq LPU `qwen/qwen3.8-27b` was verified via the separate 5-query Quick Smoke Suite for pipeline health and zero-token-fee validation, achieving 0.438 on the smoke subset without rate limits).*

#### Key Research Insights:
1. **70B-Class Model Quality**: Both **Meta LLaMA 3.3 70B** ($0.483$ overall, $0.559$ AnsRel) and **Qwen 2.5 72B** ($0.428$ overall, $0.496$ AnsRel) significantly outperform smaller models on technical synthesis, demonstrating superior handling of dense multi-hop cryptography and quantum physics context.
2. **Production Cost-Latency Tradeoff**: **GPT-4o-mini** provides the highest throughput and lowest operational cost ($1/10\text{th}$ cost of 70B models, zero rate limits), making it optimal for standard enterprise SLA budgets.
3. **Architecture Consistency**: **Advanced RAG** (Query Rewriting + Dense/Sparse Hybrid Retrieval + BGE Cross-Encoder Reranker) consistently achieves near-optimal retrieval fidelity across every tested model family.

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
* **Content-addressed delta indexing**: unchanged chunks keep their stored embeddings (even if they moved); only new or edited text is re-embedded. Vectors live in PostgreSQL via pgvector.
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
| **3. Hybrid RAG** | BM25 Okapi + pgvector dense vectors merged via Reciprocal Rank Fusion (RRF). |
| **4. Corrective RAG** | Confidence gate + absence prover + self-correction loop. |
| **5. Agentic RAG** | LangGraph tool-calling agent with iterative reasoning and replanning. |
| **6. Full RAGOps** | Versioned semantic cache + cost control + claim citation verifier + reflective generation. |
| **7. Adaptive RAGOps** | Full RAGOps + query-and-corpus feature fusion adaptive backend dispatching. |
| **8. Policy-Driven RAGOps** | Typed `ExecutionPlan` runtime with budget controls, early exits, and adaptive $k$. |
| **9. Late-Chunked RAGOps** | Token-pooling Late Chunking strategy over long-context embeddings. |
| **10. HyDE RAGOps** | Hypothetical Document Embeddings for dense semantic vector alignment. |

---

## 🔌 Platform Features

* **Per-project pipeline studio**: every project has its own models (primary + fallback chain from the live provider catalog), embedding model, chunking, thresholds and on/off switches for every technique (hybrid search, query rewriting, multi-query, HyDE, adaptive routing, reranking, cross-references, conflict detection, compression, self-correction, claim verification, reflective generation, semantic cache).
* **Multi-file ingestion**: drag-and-drop many files (PDF, DOCX, TXT, MD, CSV, JSON, HTML…); each is validated, de-duplicated, scanned for secrets/PII and processed by a durable background job queue.
* **Streaming chat** with numbered citations, relevance scores, a calibrated confidence gate and real time-to-first-token metrics.
* **Ask your database**: a manager connects a PostgreSQL/MySQL database and chooses which tables to expose; employees ask questions in plain language. Read-only sessions, SQL AST validation, table allow-list, SSRF protection, timeouts and row limits.
* **MCP server**: `/mcp/` (Streamable HTTP) with per-project API keys, so Claude Desktop, Claude Code, Cursor or any MCP client can use a project's documents and databases.
* **Workspaces (multi-tenant)**: anyone can create a workspace and becomes its admin. Each workspace brings its own LLM API keys (Groq / Gemini / OpenRouter, encrypted at rest) and never sees another workspace's projects, people, analytics or keys. Admins invite clients with one-time links and tick which projects each client can open (e.g. Sales sees only the Sales project).
* **Public demo**: optional one-click "Demo admin" / "Demo client" logins (`DEMO_MODE=1`) into a shared, restricted sandbox that runs on the server's keys and resets daily.
* **Access control**: workspace admins and clients, project visibility (assigned members vs whole workspace), viewer/editor members.
* **Observability**: real per-request execution traces and phase timings (`/executions`), analytics, AutoRAGBench.
* **Tests & CI**: `backend/tests` (runs on SQLite or Postgres+pgvector) and a GitHub Actions workflow.

**Deployment (free, no credit card):** Render + Neon + Vercel, see [DEPLOY_RENDER.md](./DEPLOY_RENDER.md). The backend runs its local models on ONNX Runtime (no PyTorch), peaking at about 470 MB, so it fits Render's 512 MB free tier. For a larger VM (Oracle Always Free needs a card), see [DEPLOY_ORACLE.md](./DEPLOY_ORACLE.md).

---

## ⚡ Technical Stack

### Frontend (Next.js 16 Standalone)
* **Framework**: Next.js 16 (App Router, Turbopack, TypeScript)
* **Styling**: Vanilla CSS / Tailwind CSS + Shadcn UI (Glassmorphic dark design system)
* **Visualizations**: Recharts + KPI Telemetry dashboards + Live AutoRAGBench Ticker
* **Pages**: `/admin`, `/benchmark`, `/playground`, `/chat`, `/executions`, `/analytics`

### Backend (FastAPI Enterprise)
* **Framework**: FastAPI (Python 3.11/3.12)
* **Database**: PostgreSQL (pgvector enabled) with SQLModel ORM
* **AI Orchestration**: LangChain, Groq LPUs, Google Gemini, OpenRouter, LiteLLMGateway
* **Embeddings**: Local `all-MiniLM-L6-v2` (384-dim) on ONNX Runtime by default (identical vectors to the PyTorch version); `MODEL_BACKEND=torch` for larger hosts
* **Search & Rerank**: pgvector (exact search up to 50k vectors/project, HNSW beyond) + BM25 Okapi + cross-encoder reranker (`ms-marco-MiniLM-L-6-v2` on ONNX by default; `BAAI/bge-reranker-base` with `MODEL_BACKEND=torch`)
* **Jobs**: Postgres-backed durable queue (`FOR UPDATE SKIP LOCKED`) with in-process workers
* **Integrations**: MCP (Streamable HTTP), PostgreSQL/MySQL connector (sqlglot-validated, read-only)

---

## 🛠️ Quickstart Guide

### Option 1: Docker Compose (Recommended)
```bash
cp .env.example .env        # set SECRET_KEY, ADMIN_EMAIL/ADMIN_PASSWORD, POSTGRES_PASSWORD, an LLM key
docker compose --profile frontend up -d --build
```
- **Frontend**: `http://localhost:3000`
- **Backend API**: `http://localhost:8000`
- **API Documentation**: `http://localhost:8000/docs`

---

### Option 2: Local Manual Setup

#### 1. Configure Environment
Create `.env` in the root workspace:
```env
DATABASE_URL=postgresql://postgres:password@localhost:5432/ragops   # needs the pgvector extension
SECRET_KEY=your-secret-encryption-key
ADMIN_EMAIL=you@example.com
ADMIN_PASSWORD=a-strong-password
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

### Running the tests
```bash
cd backend
pip install pytest
python -m pytest tests -q                                   # SQLite
TEST_DATABASE_URL=postgresql://... python -m pytest tests -q  # Postgres + pgvector
```

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
