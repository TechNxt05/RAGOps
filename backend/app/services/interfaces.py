from typing import Protocol, List, Tuple, Any
from langchain_core.documents import Document as LCDocument

class RetrievalStrategy(Protocol):
    """
    Protocol for defining custom retrieval strategies.
    Any class implementing this protocol can be injected into the RAG engine.
    """
    def retrieve(self, query: str, project_id: int, top_k: int, **kwargs) -> List[Tuple[LCDocument, float]]:
        """
        Execute the retrieval strategy.
        
        Args:
            query: The user's query or rewritten query.
            project_id: The ID of the project to search within.
            top_k: Number of documents to retrieve.
            
        Returns:
            List of tuples containing Langchain Document and its score.
        """
        ...

class ChunkingStrategy(Protocol):
    """
    Protocol for defining custom document chunking strategies.
    Any class implementing this protocol can be used during document ingestion.
    """
    def chunk(self, text: str, document_type: str, **kwargs) -> List[str]:
        """
        Execute the chunking strategy.
        
        Args:
            text: The full text of the document.
            document_type: Type of the document (e.g., 'pdf', 'txt').
            
        Returns:
            List of text chunks.
        """
        ...
