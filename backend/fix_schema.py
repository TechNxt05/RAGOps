from app.db import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        print("Starting migration...")
        
        # 1. Add project_id to ragconfig
        try:
            conn.execute(text("ALTER TABLE ragconfig ADD COLUMN project_id INTEGER;"))
            print("Added project_id to ragconfig")
        except Exception as e:
            print(f"Skipping ragconfig update (maybe exists): {e}")

        # 2. Add project_id to document
        try:
            conn.execute(text("ALTER TABLE document ADD COLUMN project_id INTEGER;"))
            print("Added project_id to document")
        except Exception as e:
            print(f"Skipping document update: {e}")

        # 3. Add project_id to chatsession
        try:
            conn.execute(text("ALTER TABLE chatsession ADD COLUMN project_id INTEGER;"))
            print("Added project_id to chatsession")
        except Exception as e:
            print(f"Skipping chatsession update: {e}")

        conn.commit()
        print("Migration complete.")

if __name__ == "__main__":
    run_migration()
