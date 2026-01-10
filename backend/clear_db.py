from sqlmodel import Session, text, create_engine
from app.db import engine
from app.models.rag import RAGConfig, Document, Chunk
from app.models.chat import ChatSession, Message
from app.models.rag import Project

def clear_db():
    with Session(engine) as session:
        print("Creating session...")
        # Order matters due to foreign keys
        print("Deleting Messages...")
        session.exec(text("DELETE FROM message"))
        print("Deleting ChatSessions...")
        session.exec(text("DELETE FROM chatsession"))
        print("Deleting Chunks...")
        session.exec(text("DELETE FROM chunk"))
        print("Deleting Documents...")
        session.exec(text("DELETE FROM document"))
        print("Deleting RAGConfig...")
        session.exec(text("DELETE FROM ragconfig"))
        print("Deleting Projects...")
        session.exec(text("DELETE FROM project"))
        
        session.commit()
        print("✅ Database cleared (Users preserved).")

if __name__ == "__main__":
    clear_db()
