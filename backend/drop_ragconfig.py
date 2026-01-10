from sqlmodel import text
from app.db import engine
from sqlmodel import Session

def drop_ragconfig():
    with Session(engine) as session:
        print("Dropping RAGConfig table...")
        session.exec(text("DROP TABLE IF EXISTS ragconfig CASCADE"))
        session.commit()
        print("✅ RAGConfig table dropped.")

if __name__ == "__main__":
    drop_ragconfig()
