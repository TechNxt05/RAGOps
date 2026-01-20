from sqlmodel import create_engine, text
import os

# Standalone DB connection logic
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ragops.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}, echo=True)

def add_column():
    with engine.connect() as conn:
        try:
            conn.execute(text("ALTER TABLE message ADD COLUMN usage_metadata JSON"))
            print("Successfully added usage_metadata column to message table")
        except Exception as e:
            print(f"Error (column might already exist): {e}")

if __name__ == "__main__":
    add_column()
