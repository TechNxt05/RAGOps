# RAGOps Architecture

RAGOps is a multi-project RAG platform. Each project has its own documents, connected
databases, members, API keys and a fully configurable retrieval/generation pipeline.

## 1. Deployment topology

```mermaid
graph LR
    U[Browser] --> FE[Next.js frontend<br/>Vercel]
    FE -->|HTTPS| T[Render HTTPS<br/>or Caddy / Cloudflare Tunnel on a VM]
    M[MCP clients<br/>Claude, Cursor] -->|HTTPS + API key| T
    T --> API[FastAPI backend on Render free<br/>1 process: API + ingestion worker + ONNX models]
    API --> PG[(Neon Postgres + pgvector<br/>rows, vectors, jobs, cache)]
    API --> LLM[Groq / Gemini / OpenRouter<br/>free tiers, fallback chain]
    API -->|read-only| EXT[(Company databases<br/>Postgres / MySQL)]
```

One backend process holds the local models (MiniLM embeddings and an ms-marco MiniLM
cross-encoder on ONNX Runtime by default, or bge-reranker-base with `MODEL_BACKEND=torch`) in memory once and
serves requests from a thread pool; background worker threads process the job queue.
All state lives in Postgres, so restarts and redeploys lose nothing.

## 2. Ingestion

```mermaid
graph TD
    UP[POST /rag/ingest/upload<br/>many files] --> V{validate type/size,<br/>dedupe by SHA-256}
    V -->|queued| J[(ingestionjob row<br/>+ raw bytes)]
    J --> W[worker thread<br/>FOR UPDATE SKIP LOCKED]
    W --> P[parse: Docling / pypdf / docx / html / text]
    P --> S{secret & PII scan}
    S -->|critical secret| Q[quarantined, never indexed]
    S --> C[adaptive chunker]
    C --> D[content-addressed delta:<br/>reuse embeddings of unchanged chunks]
    D --> E[embed new chunks in batches]
    E --> X[(chunk + chunkembedding rows,<br/>one transaction, kb_version++)]
```

## 3. Question answering (one chat turn)

```mermaid
graph TD
    Q[Question] --> A{access check +<br/>rate limit}
    A --> SC{semantic cache<br/>same keywords AND cosine >= 0.92}
    SC -->|hit| OUT
    SC --> TT[turn router: chit-chat / follow-up / fresh]
    TT -->|follow-up| RW[rewrite to standalone question]
    TT --> AR[adaptive router + semantic router]
    RW --> AR
    AR -->|count/aggregate| SQL[metadata SQL]
    AR --> R[retrieval ONCE:<br/>pgvector + BM25 -> weighted RRF<br/>optional multi-query / HyDE / cross-refs]
    R --> RR[BGE cross-encoder rerank]
    RR --> G{confidence gate<br/>max of reranker and cosine relevance}
    G -->|weak| SCL[self-correction retry] --> G2{gate}
    G2 -->|still weak| AB[absence proof -> refusal]
    G -->|pass| CMP[compress only if over token budget]
    G2 -->|pass| CMP
    CMP --> GEN[generate with citations<br/>streamed, model fallback chain]
    GEN --> EV[RAGAS-style metrics, output contract,<br/>claim verification]
    EV --> OUT[answer + sources + trace]
```

Key properties:
* The chunks placed in the prompt are exactly the chunks cited, shown as sources and
  scored by every metric.
* Relevance is calibrated: cross-encoder logits are mapped to [0, 1] from measured
  relevant/irrelevant pairs, and combined with bi-encoder cosine (max) so paraphrases
  are not rejected. Default gate threshold 0.40 (per-project override).
* Every stage can be switched on/off per project; the trace records what ran and how long
  each phase took (`/executions`).

## 4. Ask-your-database

Manager connects a database (read-only user recommended) and selects exposed tables.
Questions go: write-intent guard -> table selection (BM25 over schema) -> LLM writes SQL ->
sqlglot AST validation (single SELECT, no writes / `SELECT INTO` / locking / dangerous
functions, allow-listed tables only, enforced LIMIT) -> read-only session with statement
timeout -> one automatic repair attempt on SQL errors -> answer summarised from the rows,
with the SQL and rows shown to the user. Hosts resolving to private/internal addresses are
rejected (SSRF protection).

## 5. Data model (main tables)

| Table | Purpose |
|---|---|
| `project`, `projectmember` | projects, visibility, viewer/editor members |
| `ragconfig` | versioned per-project pipeline configuration |
| `document`, `chunk` | uploaded files and their chunks |
| `chunkembedding` | one vector per (chunk, embedding model), pgvector |
| `ingestionjob` | durable work queue |
| `semanticcacheentry` | answer cache scoped to (project, kb_version, model) |
| `externaldatabase` | connected databases (Fernet-encrypted credentials) |
| `projectapikey` | hashed API keys for MCP / programmatic access |
| `chatsession`, `message`, `querylog`, `tokenusage` | conversations, traces, analytics |

Schema upgrades are additive and automatic on startup (`app/db.py`); one-time data fixes
are tracked in `schema_migrations`.

## 6. Benchmark variants

`run_benchmark.py` / AutoRAGBench ablates the pipeline across 10 variants (naive, advanced,
hybrid, corrective, agentic, full, adaptive, policy-driven, late-chunking, ...) with
RAGAS-style metrics; see the README for results and caveats.
