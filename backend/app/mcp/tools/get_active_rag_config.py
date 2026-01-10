from pydantic import BaseModel
from app.mcp.schemas import MCPContext
from app.db import engine
from sqlmodel import Session, select
from app.models.rag import RAGConfig

class GetRAGConfigInput(BaseModel):
    pass

async def get_active_rag_config_tool(args: GetRAGConfigInput, context: MCPContext):
    if not context.project_id:
        return {"error": "Project ID required"}
        
    with Session(engine) as session:
        config = session.exec(
            select(RAGConfig)
            .where(RAGConfig.project_id == context.project_id)
            .where(RAGConfig.is_active == True)
            .order_by(RAGConfig.created_at.desc())
        ).first()
        
        if not config:
            return {"error": "No active config found"}
            
        return config.dict()
