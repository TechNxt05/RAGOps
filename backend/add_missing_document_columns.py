from app.db import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        print("Starting document table migration...")
        
        columns_to_add = [
            ("file_size_bytes", "INTEGER"),
            ("page_count", "INTEGER"),
            ("chunk_count", "INTEGER"),
            ("chunk_size_used", "INTEGER"),
            ("embedding_model_used", "VARCHAR"),
            ("version", "INTEGER DEFAULT 1"),
            ("is_active", "BOOLEAN DEFAULT TRUE"),
            ("processing_status", "VARCHAR DEFAULT 'pending'"),
            ("processing_error", "VARCHAR"),
            ("uploaded_by", "INTEGER")
        ]
        
        for col_name, col_type in columns_to_add:
            try:
                conn.execute(text(f"ALTER TABLE document ADD COLUMN IF NOT EXISTS {col_name} {col_type};"))
                print(f"Ensured column {col_name} exists on table document")
            except Exception as e:
                print(f"Error adding {col_name}: {e}")
        
        conn.commit()
        print("Document table migration complete.")

if __name__ == "__main__":
    run_migration()
