from langchain_core.pydantic_v1 import BaseModel
from app.mcp.schemas import MCPContext
from app.db import engine
from sqlmodel import Session, select
from app.models.rag import Document

class GetDocumentSourcesInput(BaseModel):
    limit: int = 100

async def get_document_sources_tool(args: GetDocumentSourcesInput, context: MCPContext):
    if not context.project_id:
        return {"error": "Project ID required"}
        
    with Session(engine) as session:
        docs = session.exec(
            select(Document)
            .where(Document.project_id == context.project_id)
            .limit(args.limit)
        ).all()
        
        return {
            "documents": [
                {"id": d.id, "filename": d.filename, "uploaded_at": d.uploaded_at.isoformat()}
                for d in docs
            ]
        }
