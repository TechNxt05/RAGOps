from app.db import engine
from sqlalchemy import text

def run_migration():
    print("Starting migrations for RAGOps Upgrades...")
    
    # 1. Add chunking columns to document table
    doc_cols = [
        ("chunking_strategy", "VARCHAR"),
        ("chunking_metrics", "JSON")
    ]
    for col_name, col_type in doc_cols:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE document ADD COLUMN {col_name} {col_type};"))
            print(f"Added column {col_name} ({col_type}) to document table")
        except Exception as e:
            print(f"Skipping column {col_name} addition on document: {e}")

    # 2. Add pipeline_trace column to querylog table
    try:
        with engine.begin() as conn:
            conn.execute(text("ALTER TABLE querylog ADD COLUMN pipeline_trace JSON;"))
        print("Added column pipeline_trace (JSON) to querylog table")
    except Exception as e:
        print(f"Skipping column pipeline_trace addition on querylog: {e}")

    # 3. Add Phase 2 columns to chunk, document, project, and ragconfig tables
    phase2_migrations = [
        ("chunk", "content_hash", "VARCHAR(64)"),
        ("chunk", "chunk_version", "INTEGER DEFAULT 1"),
        ("chunk", "doc_id_version", "VARCHAR(256)"),
        ("chunk", "chunk_index", "INTEGER"),
        ("document", "document_hash", "VARCHAR(64)"),
        ("document", "parsing_method", "VARCHAR(64)"),
        ("document", "redaction_log", "JSON"),
        ("document", "parsed_chunks_json", "JSON"),
        ("project", "kb_version", "INTEGER DEFAULT 1"),
        ("project", "kb_version_updated_at", "TIMESTAMP"),
        ("ragconfig", "use_multi_query", "BOOLEAN DEFAULT TRUE"),
        ("querylog", "ragas_scores", "JSON"),
        ("message", "ragas_scores", "JSON"),
    ]

    for table, col, col_type in phase2_migrations:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type};"))
            print(f"Added column {col} ({col_type}) to {table} table")
        except Exception as e:
            print(f"Skipping column {col} on {table}: {e}")

    print("All migrations complete.")

if __name__ == "__main__":
    run_migration()
