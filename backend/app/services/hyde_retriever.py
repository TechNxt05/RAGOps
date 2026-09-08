"""
HyDE (Hypothetical Document Embeddings) Retriever

Generates a hypothetical answer to the query using a cheap LLM tier,
which can then be embedded to search the vector database.
"""

import logging
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage

logger = logging.getLogger(__name__)

from app.utils.llm_response import extract_text

class HyDERetriever:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm

    def generate_hypothetical_document(self, query: str) -> str:
        """
        Given a query, uses the LLM to generate a hypothetical, factual-sounding
        document that answers the query.
        """
        sys_msg = SystemMessage(
            content="You are a knowledgeable assistant. "
                    "Write a brief, factual, hypothetical document snippet "
                    "that directly answers the user's question. "
                    "Use formal language and domain-specific vocabulary. "
                    "Do NOT include conversational filler like 'Here is a hypothetical document'."
        )
        
        try:
            response = self.llm.invoke([sys_msg, HumanMessage(content=query)])
            hypothetical_doc = extract_text(response.content)
            logger.info(f"Generated HyDE document for query: '{query[:30]}...'")
            return hypothetical_doc
        except Exception as e:
            logger.error(f"HyDE generation failed: {e}")
            # Fallback to the original query if generation fails
            return query
