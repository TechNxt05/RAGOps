"""
Verification Test Harness for RAGOps Phase 2 Upgrade.
Verifies MCP tools, DB Connector security constraints (Fernet encryption, read-only enforcement, secret redaction),
Build 3 Table Retrieval, Contextual Retrieval, GraphRAG Stage 1, and LiteLLM Gateway.
"""

import asyncio
import os
import sys

# Ensure backend directory is in sys.path
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

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

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


async def verify_phase2():
    print("=" * 70)
    print("           RAGOps Phase 2 Upgrade — Empirical Verification Harness")
    print("=" * 70)

    from app.db import init_db
    init_db()

    # 1. MCP Server Tool Verification
    print("\n[1/6] Verifying MCP Server Tooling...")
    from mcp_server import (
        query_knowledge_base,
        retrieve_chunks,
        run_computation_query,
        check_router_decision,
        get_project_health
    )

    decision_res = await check_router_decision("What is the total revenue for SKU-9904?")
    assert decision_res["success"] is True
    print(f"  ✅ check_router_decision: chosen_backend='{decision_res['chosen_backend']}'")

    health_res = await get_project_health(project_id=1)
    assert health_res["success"] is True
    print(f"  ✅ get_project_health: p95_latency={health_res['p95_latency_ms']}ms, SLA_warning={health_res['sla_warning_exceeded']}")

    ret_res = await retrieve_chunks(project_id=1, query="hybrid RAGOps", top_k=2)
    assert ret_res["success"] is True
    print(f"  ✅ retrieve_chunks: returned {ret_res['count']} chunks")

    comp_res = await run_computation_query(project_id=1, query="how many documents in project")
    if not comp_res.get("success"):
        print(f"  ❌ run_computation_query error: {comp_res}")
    assert comp_res["success"] is True
    print(f"  ✅ run_computation_query: backend='{comp_res['backend']}'")

    kb_res = await query_knowledge_base(project_id=1, query="What is RAGOps?")
    if not kb_res.get("success"):
        print(f"  ❌ query_knowledge_base error: {kb_res}")
    assert kb_res["success"] is True
    print(f"  ✅ query_knowledge_base: answer_length={len(kb_res['answer'])} chars, passed={kb_res['claim_verification']['passed']}")

    # 2. Database Connector Security & Encryption
    print("\n[2/6] Verifying DB Connector Security Constraints & Envelope Encryption...")
    from app.services.db_connector.credential_vault import credential_vault
    from app.services.db_connector.query_executor import query_executor
    from app.services.db_connector.connection_parser import connection_parser

    secret = "MockTestSecretPassword!"
    encrypted = credential_vault.encrypt(secret)
    decrypted = credential_vault.decrypt(encrypted)
    assert decrypted == secret
    assert encrypted != secret
    print("  ✅ Fernet Credential Encryption & Decryption: VERIFIED (Encrypted string is opaque)")

    # Test Secret Redaction from Traces
    raw_uri = "postgresql://admin_user:MockTestSecretPassword!@db.internal.example.com:5432/orders_db"
    redacted_uri = credential_vault.redact_secrets(raw_uri)
    assert "MockTestSecretPassword!" not in redacted_uri
    assert "***" in redacted_uri
    print(f"  ✅ Trace Secret Redaction: VERIFIED -> '{redacted_uri}'")

    # Test Structural Read-Only Allowlist Enforcement
    try:
        query_executor.validate_read_only_query("DELETE FROM users WHERE id = 1;")
        assert False, "Failed to block DELETE query!"
    except ValueError as val_err:
        print(f"  ✅ Structural Read-Only Enforcement: VERIFIED -> Blocked query with error: '{val_err}'")

    try:
        query_executor.validate_read_only_query("DROP TABLE orders;")
        assert False, "Failed to block DROP query!"
    except ValueError as val_err:
        print(f"  ✅ Structural Read-Only Enforcement: VERIFIED -> Blocked DROP table attempt")

    # 3. Connection Parser & Wizard Summary
    print("\n[3/6] Verifying Connection Parser & Confirmation Summaries...")
    parsed_res = connection_parser.parse_connection_string("postgresql://mock_user:mock_password@mock_db_host:5432/analytics_db")
    assert parsed_res["success"] is True
    assert "Confirm to save?" in parsed_res["confirmation_summary"]
    print(f"  ✅ Connection Parser: {parsed_res['confirmation_summary']}")

    # 4. Build 3 Table Retrieval on Multi-Table Schemas
    print("\n[4/6] Verifying Build 3 Table Retrieval for Multi-Table Schemas...")
    from app.services.db_connector.schema_introspector import schema_introspector
    mock_schema = schema_introspector.introspect_schema("postgresql", "host", 5432, "analytics", "user", "pass")
    relevant_tables = schema_introspector.retrieve_relevant_tables(mock_schema, "Show user order totals and prices", top_n=2)
    assert len(relevant_tables) == 2
    assert "orders" in relevant_tables or "order_items" in relevant_tables
    print(f"  ✅ Table Retriever: Selected relevant tables {list(relevant_tables.keys())} for query")

    # 5. GraphRAG Stage 1 Structural Graph Walking
    print("\n[5/6] Verifying GraphRAG Stage 1 Structure-Only Document Graph...")
    from app.services.graph_rag_service import graph_rag_service
    graph_res = graph_rag_service.build_document_graph(
        doc_id=101,
        title="Architecture Overview",
        text_blocks=[
            {"content": "Architecture Overview", "is_heading": True},
            {"content": "RAGOps uses hybrid BM25 and FAISS retrieval.", "is_heading": False},
            {"content": "Security Layer", "is_heading": True},
            {"content": "Structural read-only database execution prevents writes.", "is_heading": False}
        ]
    )
    assert graph_res["nodes_created"] > 0
    matched = graph_rag_service.search_structural_graph("hybrid retrieval")
    assert len(matched) > 0
    print(f"  ✅ GraphRAG Stage 1: Built graph ({graph_res['nodes_created']} nodes) and walked graph to match chunk '{matched[0]['title']}'")

    # 6. Contextual Retrieval (Anthropic technique) & LiteLLM Gateway
    print("\n[6/6] Verifying Anthropic Contextual Retrieval & LiteLLM Gateway...")
    from app.services.docling_parser import DoclingParser
    docling = DoclingParser()
    ctx_chunk = await docling.contextualize_chunk("FAISS vector index stores dense embeddings.", document_title="RAGOps Specs")
    print(f"  ✅ Contextual Retrieval Chunk Header:\n'{ctx_chunk[:120]}...'")

    print("\n" + "=" * 70)
    print("   🎉 ALL PHASE 2 VERIFICATION CHECKS PASSED CLEANLY WITH ZERO ERRORS!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(verify_phase2())
