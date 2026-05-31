from app.db import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        print("Starting migrations for RAGOps Upgrades...")
        
        # 1. Add chunking columns to document table
        doc_cols = [
            ("chunking_strategy", "VARCHAR"),
            ("chunking_metrics", "JSON")
        ]
        for col_name, col_type in doc_cols:
            try:
                conn.execute(text(f"ALTER TABLE document ADD COLUMN {col_name} {col_type};"))
                print(f"Added column {col_name} ({col_type}) to document table")
            except Exception as e:
                print(f"Skipping column {col_name} addition on document: {e}")

        # 2. Add pipeline_trace column to querylog table
        try:
            conn.execute(text("ALTER TABLE querylog ADD COLUMN pipeline_trace JSON;"))
            print("Added column pipeline_trace (JSON) to querylog table")
        except Exception as e:
            print(f"Skipping column pipeline_trace addition on querylog: {e}")

        conn.commit()
        print("All migrations complete.")

if __name__ == "__main__":
    run_migration()
