from langchain_core.pydantic_v1 import BaseModel, Field
from app.mcp.schemas import MCPContext
from app.rag.engine import RAGEngine
from app.db import engine
from sqlmodel import Session

class SearchDocumentsInput(BaseModel):
    query: str = Field(..., description="The query string to search for")
    top_k: int = Field(default=4, description="Number of results to return")

async def search_documents_tool(args: SearchDocumentsInput, context: MCPContext):
    if not context.project_id:
        return {"error": "Project ID required for search"}
    
    print(f"DEBUG: Entering search_documents_tool with query='{args.query}'")
        
    with Session(engine) as session:
        rag_engine = RAGEngine(session)
        # Get config to check threshold
        try:
            config = rag_engine.get_active_config(context.project_id)
            threshold = config.similarity_threshold
        except:
            threshold = 0.0
            
        results = rag_engine.search(
            query=args.query, 
            project_id=context.project_id, 
            k=args.top_k,
            score_threshold=threshold
        )
        
        # Format results
        return {
            "chunks": [
                {
                    "content": doc.page_content,
                    "score": score,
                    "source": doc.metadata.get("source")
                }
                for doc, score in results
            ]
        }
