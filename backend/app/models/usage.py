from sqlmodel import SQLModel, Field
from typing import Optional
from datetime import datetime

class TokenUsage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    project_id: Optional[int] = Field(default=None, index=True)
    user_id: int = Field(index=True)
    session_id: Optional[int] = Field(default=None)
    
    model: str
    provider: str
    
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    cost: float = 0.0
    timestamp: datetime = Field(default_factory=datetime.utcnow)
