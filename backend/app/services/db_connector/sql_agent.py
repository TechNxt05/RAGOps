"""
Natural-language questions over a manager-connected SQL database.

Safety layers (all must pass):
  1. Network: the host must resolve to a public address (blocks SSRF into the
     platform's own network / cloud metadata), unless explicitly allowed.
  2. Session: the connection is opened read-only with a statement timeout.
  3. Query: the generated SQL is parsed with sqlglot; exactly one read-only
     SELECT is allowed, no writes / SELECT INTO / dangerous functions.
  4. Scope: every table referenced must be in the manager's allow-list.
  5. Size: a LIMIT is enforced on the outermost query.
Nothing is ever simulated: connection or query errors are reported as errors.
"""

import ipaddress
import json
import logging
import re
import socket
from datetime import date, datetime, time as dtime
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import sqlglot
from sqlglot import exp

from app.core.config import settings

logger = logging.getLogger(__name__)

SUPPORTED_DB_TYPES = ("postgresql", "mysql")
DIALECTS = {"postgresql": "postgres", "mysql": "mysql"}
BLOCKED_FUNCTIONS = {
    "pg_sleep", "pg_sleep_for", "pg_sleep_until", "pg_read_file", "pg_read_binary_file", "pg_ls_dir",
    "pg_stat_file", "lo_import", "lo_export", "lo_get", "dblink", "dblink_exec", "pg_terminate_backend",
    "pg_cancel_backend", "set_config", "pg_reload_conf", "pg_rotate_logfile", "query_to_xml",
    "sleep", "benchmark", "load_file", "get_lock", "release_lock", "sys_exec", "sys_eval",
}
WRITE_NODES = (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create, exp.Alter, exp.Command,
               exp.Merge, exp.TruncateTable, exp.Into, exp.Lock, exp.Copy)


class DatabaseQueryError(Exception):
    pass


_WRITE_INTENT = re.compile(
    r"^\s*(please\s+|can you\s+|could you\s+)?(delete|remove|drop|update|insert|add|create|modify|change|"
    r"set|truncate|alter|rename|grant|revoke|erase|wipe)\b", re.I)
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Network guard
# ---------------------------------------------------------------------------

def ensure_public_host(host: str, port: int) -> None:
    if settings.db_connector_allow_private_hosts:
        return
    try:
        infos = socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        raise DatabaseQueryError(f"Cannot resolve host '{host}'") from exc
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved or ip.is_multicast or ip.is_unspecified:
            raise DatabaseQueryError(
                f"Host '{host}' resolves to a private/internal address ({ip}). Only publicly reachable "
                "databases can be connected (use a tunnel such as Cloudflare Tunnel for private ones)."
            )


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def connect(db_type: str, host: str, port: int, db_name: str, username: str, password: str, ssl_mode: Optional[str] = None):
    if db_type not in SUPPORTED_DB_TYPES:
        raise DatabaseQueryError(f"Unsupported database type '{db_type}'. Supported: {', '.join(SUPPORTED_DB_TYPES)}")
    ensure_public_host(host, port)
    timeout_ms = settings.db_connector_timeout_ms
    try:
        if db_type == "postgresql":
            import psycopg2
            conn = psycopg2.connect(
                host=host, port=port, dbname=db_name, user=username, password=password,
                connect_timeout=10, sslmode=ssl_mode or "prefer",
                options=f"-c statement_timeout={timeout_ms} -c default_transaction_read_only=on",
            )
            conn.set_session(readonly=True, autocommit=False)
            return conn
        import pymysql
        conn = pymysql.connect(
            host=host, port=port, database=db_name, user=username, password=password,
            connect_timeout=10, read_timeout=max(1, timeout_ms // 1000) + 5,
            ssl={"ssl": {}} if ssl_mode in ("require", "verify-ca", "verify-full") else None,
        )
        with conn.cursor() as cur:
            cur.execute("SET SESSION TRANSACTION READ ONLY")
            cur.execute(f"SET SESSION MAX_EXECUTION_TIME = {int(timeout_ms)}")
        return conn
    except DatabaseQueryError:
        raise
    except Exception as exc:
        raise DatabaseQueryError(f"Could not connect: {str(exc).strip()[:300]}") from exc


def introspect(conn, db_type: str) -> Dict[str, List[Dict[str, str]]]:
    with conn.cursor() as cur:
        if db_type == "postgresql":
            cur.execute("""
                SELECT table_schema, table_name, column_name, data_type
                FROM information_schema.columns
                WHERE table_schema NOT IN ('pg_catalog', 'information_schema')
                  AND table_schema NOT LIKE 'pg_toast%%'
                ORDER BY table_schema, table_name, ordinal_position
            """)
            rows = cur.fetchall()
            tables: Dict[str, List[Dict[str, str]]] = {}
            for schema, table, column, dtype in rows:
                name = table if schema == "public" else f"{schema}.{table}"
                tables.setdefault(name, []).append({"column": column, "type": dtype})
        else:
            cur.execute("""
                SELECT table_name, column_name, data_type FROM information_schema.columns
                WHERE table_schema = DATABASE() ORDER BY table_name, ordinal_position
            """)
            tables = {}
            for table, column, dtype in cur.fetchall():
                tables.setdefault(table, []).append({"column": column, "type": dtype})
    conn.rollback()
    return tables


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _table_names(table: exp.Table) -> Tuple[str, str]:
    name = table.name.lower()
    schema = (table.db or "").lower()
    return name, f"{schema}.{name}" if schema else name


def validate_sql(sql: str, db_type: str, allowed_tables: List[str], row_limit: int) -> str:
    dialect = DIALECTS[db_type]
    if not re.match(r"^\s*(\(\s*)*(select|with)\b", sql, re.I):
        raise DatabaseQueryError("The question can't be answered from the tables this database exposes.")
    try:
        statements = [s for s in sqlglot.parse(sql, read=dialect) if s is not None]
    except sqlglot.errors.ParseError as exc:
        raise DatabaseQueryError(f"Generated SQL could not be parsed: {_ANSI.sub('', str(exc)).splitlines()[0]}") from exc
    if len(statements) != 1:
        raise DatabaseQueryError("Exactly one SQL statement is allowed")
    stmt = statements[0]
    if not isinstance(stmt, exp.Query):
        raise DatabaseQueryError("Only read-only SELECT queries are allowed")
    for node in stmt.walk():
        if isinstance(node, WRITE_NODES):
            raise DatabaseQueryError(f"Forbidden SQL construct: {type(node).__name__}")
        if isinstance(node, exp.Func):
            fname = (node.name if isinstance(node, exp.Anonymous) else node.sql_name()).lower()
            if fname in BLOCKED_FUNCTIONS:
                raise DatabaseQueryError(f"Function '{fname}' is not allowed")

    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    allowed = {t.lower() for t in allowed_tables}
    allowed |= {t.split(".", 1)[1] for t in allowed if "." in t and t.startswith("public.")}
    for table in stmt.find_all(exp.Table):
        name, qualified = _table_names(table)
        if not name or name in cte_names:
            continue
        if name not in allowed and qualified not in allowed:
            raise DatabaseQueryError(f"Table '{qualified}' is not exposed by the project manager")

    limit = stmt.args.get("limit")
    current = None
    if limit is not None:
        try:
            current = int(limit.expression.name)
        except (AttributeError, ValueError):
            current = None
    if current is None or current > row_limit:
        stmt = stmt.limit(row_limit)
    return stmt.sql(dialect=dialect)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def _json_value(v: Any) -> Any:
    if isinstance(v, Decimal):
        return float(v)
    if isinstance(v, (datetime, date, dtime)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return f"<{len(bytes(v))} bytes>"
    if isinstance(v, (dict, list)):
        return json.loads(json.dumps(v, default=str))
    return v


def execute(conn, sql: str) -> Tuple[List[str], List[List[Any]]]:
    try:
        with conn.cursor() as cur:
            cur.execute(sql)
            columns = [d[0] for d in cur.description or []]
            rows = [[_json_value(v) for v in row] for row in cur.fetchall()]
        conn.rollback()
        return columns, rows
    except Exception as exc:
        try:
            conn.rollback()
        except Exception:
            pass
        raise DatabaseQueryError(str(exc).strip().splitlines()[0][:400]) from exc


# ---------------------------------------------------------------------------
# NL -> SQL -> answer
# ---------------------------------------------------------------------------

def relevant_tables(schema: Dict[str, List[Dict[str, str]]], question: str, top_n: int = 8) -> Dict[str, List[Dict[str, str]]]:
    if len(schema) <= top_n:
        return schema
    from rank_bm25 import BM25Okapi
    names = list(schema)
    corpus = [re.findall(r"\w+", (n + " " + " ".join(c["column"] for c in schema[n])).lower().replace("_", " ")) for n in names]
    scores = BM25Okapi(corpus).get_scores(re.findall(r"\w+", question.lower()))
    ranked = sorted(range(len(names)), key=lambda i: scores[i], reverse=True)[:top_n]
    return {names[i]: schema[names[i]] for i in ranked}


def _schema_text(tables: Dict[str, List[Dict[str, str]]]) -> str:
    return "\n".join(f"- {t}({', '.join(c['column'] + ' ' + c['type'] for c in cols)})" for t, cols in tables.items())


def _extract_sql(text: str) -> str:
    fence = re.search(r"```(?:sql)?\s*(.*?)```", text, re.S | re.I)
    sql = fence.group(1) if fence else text
    return sql.strip().rstrip(";").strip()


async def ask(llm, db_type: str, conn, schema: Dict[str, List[Dict[str, str]]], allowed_tables: List[str],
              question: str, description: Optional[str] = None) -> Dict[str, Any]:
    from starlette.concurrency import run_in_threadpool
    from app.utils.llm_response import extract_text

    if _WRITE_INTENT.match(question):
        raise DatabaseQueryError("This connection is read-only: I can look up data but never change, add or delete it.")
    exposed = {t: cols for t, cols in schema.items() if t.lower() in {a.lower() for a in allowed_tables}}
    if not exposed:
        raise DatabaseQueryError("No tables are exposed for this database yet. Ask the project manager to select tables.")
    tables = relevant_tables(exposed, question)
    row_limit = settings.db_connector_row_limit
    dialect_name = "PostgreSQL" if db_type == "postgresql" else "MySQL"
    base_prompt = (
        f"You write a single read-only {dialect_name} SELECT query that answers the user's question.\n"
        + (f"About this database: {description}\n" if description else "")
        + f"Available tables (only these may be used):\n{_schema_text(tables)}\n\n"
        f"Rules: one SELECT statement only; use only the tables/columns listed; qualify ambiguous columns; "
        f"add LIMIT {row_limit} or less; no comments. Return only the SQL. If the listed tables cannot "
        f"answer the question, return exactly: NO_ANSWER\n\n"
        f"Question: {question}\nSQL:"
    )
    attempts, last_error, sql = 0, None, ""
    columns: List[str] = []
    rows: List[List[Any]] = []
    while attempts < 2:
        attempts += 1
        prompt = base_prompt if last_error is None else (
            base_prompt + f"\n\nYour previous query:\n{sql}\nfailed with: {last_error}\nWrite a corrected query.\nSQL:"
        )
        raw = extract_text((await llm.ainvoke(prompt)).content)
        try:
            sql = validate_sql(_extract_sql(raw), db_type, list(exposed), row_limit)
            columns, rows = await run_in_threadpool(execute, conn, sql)
            last_error = None
            break
        except DatabaseQueryError as exc:
            last_error = str(exc)
            sql = sql or _extract_sql(raw)
    if last_error:
        raise DatabaseQueryError(f"Could not answer from the database: {last_error}")

    preview = [dict(zip(columns, r)) for r in rows[:50]]
    summary_prompt = (
        "Answer the user's question using ONLY these query results. Be concise and precise with numbers. "
        "If the results are empty, say no matching records were found. You have read-only access: never "
        "claim that anything was changed, deleted, created or sent.\n\n"
        f"Question: {question}\nSQL used: {sql}\nRows returned: {len(rows)}"
        f"{' (truncated to ' + str(row_limit) + ')' if len(rows) >= row_limit else ''}\n"
        f"Results (JSON): {json.dumps(preview, default=str)[:12000]}\n\nAnswer:"
    )
    answer = extract_text((await llm.ainvoke(summary_prompt)).content)
    return {
        "answer": answer,
        "sql": sql,
        "columns": columns,
        "rows": rows,
        "row_count": len(rows),
        "truncated": len(rows) >= row_limit,
        "tables_considered": list(tables),
        "attempts": attempts,
    }
