from app.db import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        print("Starting chat & analytics tables migration...")
        
        # 1. Ensure chatsession has all columns
        chatsession_cols = [
            ("project_id", "INTEGER"),
            ("settings", "JSON")
        ]
        for col_name, col_type in chatsession_cols:
            try:
                conn.execute(text(f"ALTER TABLE chatsession ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                print(f"Ensured column {col_name} exists on table chatsession")
            except Exception as e:
                print(f"Error adding {col_name} to chatsession: {e}")

        # 2. Ensure message has all columns
        message_cols = [
            ("usage_metadata", "JSON")
        ]
        for col_name, col_type in message_cols:
            try:
                conn.execute(text(f"ALTER TABLE message ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                print(f"Ensured column {col_name} exists on table message")
            except Exception as e:
                print(f"Error adding {col_name} to message: {e}")

        # 3. Ensure querylog has all columns
        querylog_cols = [
            ("chunks_before_pruning", "INTEGER DEFAULT 0"),
            ("chunks_after_pruning", "INTEGER DEFAULT 0"),
            ("pruning_reduction_pct", "DOUBLE PRECISION DEFAULT 0.0"),
            ("used_hybrid_search", "BOOLEAN DEFAULT FALSE")
        ]
        for col_name, col_type in querylog_cols:
            try:
                conn.execute(text(f"ALTER TABLE querylog ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                print(f"Ensured column {col_name} exists on table querylog")
            except Exception as e:
                print(f"Error adding {col_name} to querylog: {e}")
        
        conn.commit()
        print("Chat & analytics tables migration complete.")

if __name__ == "__main__":
    run_migration()
