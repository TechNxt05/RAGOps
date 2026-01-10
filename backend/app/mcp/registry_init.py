from app.mcp.registry import MCPRegistry
from app.mcp.tools.search_documents import search_documents_tool, SearchDocumentsInput
from app.mcp.tools.get_active_rag_config import get_active_rag_config_tool, GetRAGConfigInput
from app.mcp.tools.get_document_sources import get_document_sources_tool, GetDocumentSourcesInput

def register_tools():
    MCPRegistry.register(
        name="search_documents",
        description="Search for relevant document chunks using Vector RAG. Use this to answer user questions based on knowledge base.",
        input_model=SearchDocumentsInput,
        fn=search_documents_tool
    )
    
    MCPRegistry.register(
        name="get_active_rag_config",
        description="Get the current RAG configuration settings (chunk size, overlap, etc.) for the project.",
        input_model=GetRAGConfigInput,
        fn=get_active_rag_config_tool
    )
    
    MCPRegistry.register(
        name="get_document_sources",
        description="Get a list of all documents uploaded to this project.",
        input_model=GetDocumentSourcesInput,
        fn=get_document_sources_tool
    )
    
    print("✅ MCP Tools Registered")
