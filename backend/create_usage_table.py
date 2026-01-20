from sqlmodel import create_engine, SQLModel
import os
from app.models.usage import TokenUsage  # Import to register with SQLModel.metadata

# Standalone DB connection logic
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./ragops.db")
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}, echo=True)

def create_usage_table():
    print("Creating token_usage table...")
    SQLModel.metadata.create_all(engine)
    print("Done!")

if __name__ == "__main__":
    create_usage_table()
