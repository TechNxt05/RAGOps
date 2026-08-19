"""
AutoRAGBench CLI Runner

Usage:
    cd backend
    python run_benchmark.py --project-id 1 [--variants naive,hybrid,full_ragops] [--output report.json]

Runs the benchmark suite against the specified project, comparing all 6 pipeline
variants (or a subset), and prints an ASCII leaderboard to stdout.
"""

import asyncio
import argparse
import os
import sys
import logging

# Ensure the backend directory is on the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

# Patch httpx 0.28+ breaking change (proxies argument removed in favor of proxy)
try:
    import httpx
    _orig_client_init = httpx.Client.__init__
    def _patched_client_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            p = kwargs.pop("proxies")
            if p and "proxy" not in kwargs:
                kwargs["proxy"] = p
        return _orig_client_init(self, *args, **kwargs)
    httpx.Client.__init__ = _patched_client_init

    _orig_async_client_init = httpx.AsyncClient.__init__
    def _patched_async_client_init(self, *args, **kwargs):
        if "proxies" in kwargs:
            p = kwargs.pop("proxies")
            if p and "proxy" not in kwargs:
                kwargs["proxy"] = p
        return _orig_async_client_init(self, *args, **kwargs)
    httpx.AsyncClient.__init__ = _patched_async_client_init
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoRAGBench")


async def main():
    parser = argparse.ArgumentParser(description="AutoRAGBench — RAGOps Pipeline Variant Benchmark")
    parser.add_argument("--project-id", type=int, required=True, help="Project ID to benchmark against")
    parser.add_argument("--variants", type=str, default=None, help="Comma-separated list of variants (e.g. naive,hybrid,full_ragops)")
    parser.add_argument("--output", type=str, default="benchmark_report.json", help="Output JSON report path")
    parser.add_argument("--verbose", action="store_true", help="Show per-query detail")
    args = parser.parse_args()

    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    from app.db import get_session, init_db
    from app.benchmark.variants import PipelineVariant
    from app.benchmark.runner import BenchmarkRunner
    from app.benchmark.report import render_ascii_table, render_per_query_detail, save_report

    # Initialize DB columns
    init_db()

    # Parse variant filter
    variant_filter = None
    if args.variants:
        names = [v.strip() for v in args.variants.split(",")]
        variant_filter = []
        for name in names:
            try:
                variant_filter.append(PipelineVariant(name))
            except ValueError:
                logger.error(f"Unknown variant: {name}. Valid: {[v.value for v in PipelineVariant]}")
                sys.exit(1)

    # Get database session
    session_gen = get_session()
    session_db = next(session_gen)

    # Ensure Project and RAGConfig exist in DB
    from sqlmodel import select
    from app.models.rag import Project, RAGConfig, Document, Chunk
    project = session_db.get(Project, args.project_id)
    if not project:
        project = Project(id=args.project_id, name=f"Benchmark Project {args.project_id}", description="Auto-created benchmark project")
        session_db.add(project)
        session_db.commit()
        session_db.refresh(project)

    rag_config = session_db.exec(select(RAGConfig).where(RAGConfig.project_id == args.project_id).where(RAGConfig.is_active == True)).first()
    if not rag_config:
        rag_config = RAGConfig(project_id=args.project_id, is_active=True)
        session_db.add(rag_config)
        session_db.commit()

    # Seed benchmark document and chunks if project has no indexed documents
    existing_doc = session_db.exec(select(Document).where(Document.project_id == args.project_id)).first()
    if not existing_doc:
        doc = Document(
            project_id=args.project_id,
            filename="ragops_enterprise_overview.pdf",
            processed=True,
            content="RAGOps is an enterprise-grade RAG platform created by Amritanshu Yadav. It implements project-based isolation, sparse-dense hybrid retrieval using FAISS and BM25, cross-encoder reranking, and self-correcting retrieval loops."
        )
        session_db.add(doc)
        session_db.commit()
        session_db.refresh(doc)

        c1 = Chunk(document_id=doc.id, content="The main purpose of RAGOps is to provide enterprise-grade hybrid RAG with project isolation, security RBAC, and precision hybrid retrieval.")
        c2 = Chunk(document_id=doc.id, content="RAGOps was created and authored by Amritanshu Yadav as an admin-controlled project-based hybrid RAG system.")
        c3 = Chunk(document_id=doc.id, content="Key approaches mentioned include sparse BM25 keyword lookup, dense vector FAISS retrieval, BGE cross-encoder reranking, contextual compression, and self-correction.")
        c4 = Chunk(document_id=doc.id, content="Step-by-step instructions: 1. Deploy via Docker Compose. 2. Configure RAG parameters in Admin panel. 3. Ingest documents with DeltaIndexer. 4. Query via API or UI.")
        c5 = Chunk(document_id=doc.id, content="Specific stats: 91.2% RAGAS accuracy score, sub-2.0s p95 latency SLA threshold, 50% token cost reduction via contextual compressor.")
        session_db.add_all([c1, c2, c3, c4, c5])
        session_db.commit()

        # Seed FAISS and BM25 index
        from app.rag.engine import RAGEngine
        rag_engine = RAGEngine(session_db)
        rag_engine.rebuild_full_index()

    # Create a synthetic admin user for benchmark (read-only operations)
    class BenchmarkUser:
        id = 1
        role = "admin"

    try:
        runner = BenchmarkRunner(
            project_id=args.project_id,
            session_db=session_db,
            user=BenchmarkUser(),
            variants=variant_filter,
        )

        logger.info(f"Starting AutoRAGBench for project_id={args.project_id}")
        logger.info(f"Variants: {[v.value for v in runner.variants]}")
        logger.info(f"Queries: {len(runner.queries)} benchmark queries")
        logger.info("=" * 60)

        summaries = await runner.run_all()

        # Print ASCII leaderboard
        table = render_ascii_table(summaries)
        print(table)

        # Print per-query detail if verbose
        if args.verbose:
            detail = render_per_query_detail(summaries)
            print(detail)

        # Save JSON report
        report_path = save_report(summaries, args.output)
        logger.info(f"Full report saved to: {report_path}")

    finally:
        session_gen.close()


if __name__ == "__main__":
    asyncio.run(main())
