import pytest

from app.services.contextual_compressor import ContextualCompressor
from app.services.db_connector.sql_agent import DatabaseQueryError, ensure_public_host, validate_sql
from app.services.semantic_cache import keywords

ALLOWED = ["customers", "orders"]


@pytest.mark.parametrize("sql", [
    "SELECT * FROM salaries",
    "WITH x AS (SELECT * FROM salaries) SELECT * FROM x",
    "DELETE FROM orders",
    "UPDATE orders SET amount = 0",
    "SELECT 1; DROP TABLE orders",
    "SELECT pg_sleep(30)",
    "SELECT * INTO hack FROM orders",
    "SELECT * FROM orders FOR UPDATE",
    "SELECT pg_read_file('/etc/passwd')",
    "I'm sorry, I can't help with that",
])
def test_sql_validator_blocks(sql):
    with pytest.raises(DatabaseQueryError):
        validate_sql(sql, "postgresql", ALLOWED, 100)


def test_sql_validator_allows_and_limits():
    out = validate_sql("SELECT c.name, SUM(o.amount) FROM customers c JOIN orders o ON o.customer_id = c.id GROUP BY c.name LIMIT 99999",
                       "postgresql", ALLOWED, 100)
    assert out.endswith("LIMIT 100")
    assert validate_sql("SELECT name FROM customers", "mysql", ALLOWED, 50).endswith("LIMIT 50")


def test_ssrf_guard_blocks_internal_hosts(monkeypatch):
    from app.core.config import settings
    monkeypatch.setattr(settings, "db_connector_allow_private_hosts", False)
    for host in ("127.0.0.1", "10.0.0.5", "169.254.169.254", "localhost"):
        with pytest.raises(DatabaseQueryError):
            ensure_public_host(host, 5432)


def test_cache_keywords_distinguish_entities():
    assert keywords("What is the capital of France?") != keywords("What is the capital of Spain?")
    assert keywords("How many vacation days?") == keywords("how many vacation days")


def test_compressor_keeps_context_under_budget_and_relevant_sentence_over_budget():
    c = ContextualCompressor()
    chunk = {"content": "International travel must be approved by a director. Economy class is required on short flights."}
    assert c.compress_chunks("who signs off on overseas trips", [chunk], 2000)["compressed_chunks"][0]["content"] == chunk["content"]
    filler = {"content": " ".join(f"Filler sentence {i} about office plants and coffee." for i in range(80))}
    out = c.compress_chunks("who approves international travel", [chunk, filler], 60)
    assert "approved by a director" in out["compressed_chunks"][0]["content"]
    assert out["compressed_token_estimate"] < out["original_token_estimate"]


def test_rewrites_are_reduced_to_one_query():
    """Small models sometimes return keyword lists; those must never reach the
    cross-encoder as a 'query' (they inflate relevance for unrelated chunks)."""
    from app.services.query_rewriter import sanitize_query
    assert sanitize_query("France capital city?; capital of France; name of France's capital",
                          "What is the capital of France?") == "France capital city?"
    assert sanitize_query("Reset VPN password\nChange VPN login\nVPN recovery", "How do I reset my VPN password?") == "Reset VPN password"
    assert sanitize_query("Standalone query: Can I carry over unused PTO?", "Can I roll them over?") == "Can I carry over unused PTO?"
    assert len(sanitize_query(" ".join(["word"] * 200), "short question").split()) <= 12
    assert sanitize_query("", "original") == "original"
