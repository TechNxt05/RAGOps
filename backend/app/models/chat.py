from sqlmodel import SQLModel, Field, Relationship
from typing import Optional, List, Dict, Any
from datetime import datetime
from sqlalchemy import Column, JSON

class ChatSession(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int 
    project_id: Optional[int] = None # Added for Project-based RAG
    created_at: datetime = Field(default_factory=datetime.utcnow)
    title: Optional[str] = None
    settings: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    
    messages: List["Message"] = Relationship(back_populates="session")

class Message(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: Optional[int] = Field(default=None, foreign_key="chatsession.id")
    role: str # user or assistant
    content: str
    sources: Optional[str] = None # JSON string or specific format
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    session: Optional[ChatSession] = Relationship(back_populates="messages")
