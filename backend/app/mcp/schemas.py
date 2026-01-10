from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union

class MCPToolInput(BaseModel):
    # Base class for tool inputs
    pass

class MCPToolOutput(BaseModel):
    content: Any
    metadata: Optional[Dict[str, Any]] = None

class ToolMetadata(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any] # JSON Schema
    
class MCPContext(BaseModel):
    user_id: int
    user_role: str = "client" # client, admin
    project_id: Optional[int] = None
    session_id: Optional[int] = None
