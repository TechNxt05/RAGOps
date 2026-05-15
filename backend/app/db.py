from sqlmodel import SQLModel, create_engine, Session
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# Import models so SQLModel metadata registers all tables before create_all
from app.models.user import User  # noqa: F401
from app.models.rag import Project, RAGConfig, Document, Chunk  # noqa: F401
from app.models.chat import ChatSession, Message  # noqa: F401
from app.models.usage import TokenUsage  # noqa: F401
from app.models.query_log import QueryLog  # noqa: F401

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Fallback for local development if not set, though Railway provides it.
if not DATABASE_URL:
    # Default to a local sqlite for development since no PG is running
    DATABASE_URL = "sqlite:///./ragops.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {}, echo=True)

def get_session():
    with Session(engine) as session:
        yield session

def init_db():
    SQLModel.metadata.create_all(engine)
