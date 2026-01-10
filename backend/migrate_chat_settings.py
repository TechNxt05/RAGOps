from app.db import engine
from sqlalchemy import text

def run_migration():
    with engine.connect() as conn:
        print("Starting migration regarding settings...")
        
        # Add settings to chatsession
        try:
            # Check if column exists first (optional, but good practice) or just try-except
            conn.execute(text("ALTER TABLE chatsession ADD COLUMN settings JSON;"))
            print("Added settings to chatsession")
        except Exception as e:
            print(f"Skipping chatsession update (maybe exists?): {e}")

        conn.commit()
        print("Migration complete.")

if __name__ == "__main__":
    run_migration()
