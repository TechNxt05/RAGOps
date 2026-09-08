# Reproducibility Guide for RAGOps Benchmark

This document provides the exact specifications, datasets, models, and commands required to reproduce the empirical results reported in the associated IEEE Access submission.

## 1. Hardware & Environment

- **OS:** Windows / Linux / macOS (Cross-platform)
- **Python Version:** 3.10+
- **Primary Dependencies:**
  - `langchain==0.1.0`
  - `langchain-google-genai==0.0.5`
  - `datasets==2.16.1` (for RAGRouter-Bench extraction)
  - `scikit-learn==1.4.0`
  - `sqlalchemy`
  - `pydantic`

## 2. API Services & Models
 
To fully reproduce the pipeline, you will need active API keys for:
- **Google Gemini API:** `gemini-3.1-flash-lite` used as the primary default generator for both benchmarking and execution planning.
- **Local Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) used for dense vector embeddings without remote rate limits.
- **Groq API / OpenRouter:** Alternative high-throughput OSS endpoints (e.g., `qwen/qwen3.8-27b`), with LiteLLMGateway automatic provider fallback.

Set your environment variables in `.env`:
```bash
GEMINI_API_KEY=your_key_here
GROQ_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
```

## 3. Datasets

### A. RAGOps Internal Benchmark (Phase 1)
- **Corpus:** A hand-curated subset of Wikipedia articles (e.g., "Theory of Relativity", "Quantum Mechanics").
- **Queries:** 35 hand-labeled queries categorized by intent (`factoid`, `reasoning`, `summary`).
- **Location:** The queries and expected ground truth sources are injected dynamically during the benchmark execution via `scratch/labeled_queries.py`.

### B. RAGRouter-Bench (Phase 2)
- **Source:** HuggingFace `Chaplain0908/RAGRouter`
- **Subsets used:** `musique`, `quality`, `ultraDomain_legal`, `graphragBench_medical`
- **Sampling:** A stratified random sample of 250 queries per domain (1,000 queries total) using seed `42` to evaluate the `AdaptiveRetrievalRouter`.

## 4. Hyperparameters

### Generation
- **Temperature:** `0.0` (Strictly deterministic)
- **Top P:** `1.0`
- **Max Output Tokens:** `1024`

### Adaptive Retrieval Router
- **Heuristic Thresholds:**
  - Exact ID/Code Pattern Match: `> 0.70` $\rightarrow$ BM25
  - Aggregation Keywords: `> 0.65` $\rightarrow$ Computation / Live DB
  - Multi-hop Keywords: `> 0.65` $\rightarrow$ GraphRAG Stage 1
  - Default Fallback $\rightarrow$ Hybrid (FAISS + BM25 + RRF)

### Semantic Cache
- **Embedding:** Pure Python TF-IDF with custom tokenization.
- **Similarity Threshold:** `0.85` Cosine Similarity.
- **Eviction Policy:** LRU with a capacity of 500 items and a 1-hour TTL.

### Cost Circuit Breaker
- **Hourly Limit:** $5.00 USD
- **Daily Limit:** $50.00 USD
- **Cooldown:** 300 seconds

## 5. Execution Commands

### Run Full Pipeline Benchmark (Phase 1)
Executes the full pipeline across all variants (naive, advanced, hyde, etc.) using the 35 ground-truth queries. Resumable via `benchmark_report.partial.json`.
```powershell
$env:PYTHONPATH="backend"; python backend/run_benchmark.py --project-id 1
```

### Run Adaptive Router Evaluation (Phase 2)
Evaluates the `AdaptiveRetrievalRouter` heuristic against the RAGRouter-Bench optimal paradigm mapping and reports the Macro-F1 score.
```powershell
$env:PYTHONPATH="backend"; python scratch/eval_ragrouter.py
```

## 6. Known Limitations to Note for Reviewers
1. **GraphRAG Scope:** The current implementation of `GraphRAGService` is **Stage 1 only**. It performs lightweight chunk-neighbor structural linking based on document boundaries and section headings. It does not perform Stage 2 (RAGA-style typed entity relationship extraction with merge operations). 
2. **Semantic Cache Threshold:** The TF-IDF semantic cache uses a high threshold (`0.85`) which is optimized for longer query texts but may result in false negatives for extremely short semantic variations. 

For full statistical metrics (Recall@5, Precision@5, MRR, nDCG) and 95% Bootstrap Confidence Intervals, refer to the generated `benchmark_report.json` output after running Phase 1.
