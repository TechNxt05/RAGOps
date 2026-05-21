from app.db import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        print("Starting RAGOps V2 Migration...")
        
        # 1. Add fields to ragconfig
        try:
            conn.execute(text("ALTER TABLE ragconfig ADD COLUMN use_hybrid_search BOOLEAN DEFAULT TRUE;"))
            print("Added use_hybrid_search to ragconfig")
        except Exception as e:
            print(f"Skipping use_hybrid_search update (may exist): {e}")

        try:
            conn.execute(text("ALTER TABLE ragconfig ADD COLUMN semantic_weight FLOAT DEFAULT 0.6;"))
            print("Added semantic_weight to ragconfig")
        except Exception as e:
            print(f"Skipping semantic_weight update (may exist): {e}")

        # 2. Add fields to querylog
        try:
            conn.execute(text("ALTER TABLE querylog ADD COLUMN chunks_before_pruning INTEGER DEFAULT 0;"))
            print("Added chunks_before_pruning to querylog")
        except Exception as e:
            print(f"Skipping chunks_before_pruning update (may exist): {e}")

        try:
            conn.execute(text("ALTER TABLE querylog ADD COLUMN chunks_after_pruning INTEGER DEFAULT 0;"))
            print("Added chunks_after_pruning to querylog")
        except Exception as e:
            print(f"Skipping chunks_after_pruning update (may exist): {e}")

        try:
            conn.execute(text("ALTER TABLE querylog ADD COLUMN pruning_reduction_pct FLOAT DEFAULT 0.0;"))
            print("Added pruning_reduction_pct to querylog")
        except Exception as e:
            print(f"Skipping pruning_reduction_pct update (may exist): {e}")

        try:
            conn.execute(text("ALTER TABLE querylog ADD COLUMN used_hybrid_search BOOLEAN DEFAULT FALSE;"))
            print("Added used_hybrid_search to querylog")
        except Exception as e:
            print(f"Skipping used_hybrid_search update (may exist): {e}")

        conn.commit()
        print("V2 Migration complete.")

if __name__ == "__main__":
    run_migration()
