from typing import List, Optional
from app.utils.llm_response import extract_text

class QueryRewriter:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def rewrite(
        self, 
        query: str, 
        conversation_history: List[dict],
        max_history_turns: int = 3,
        retry_reason: Optional[str] = None
    ) -> dict:
        """
        Returns:
            {
                "original_query": str,
                "rewritten_query": str,
                "was_rewritten": bool,
                "rewrite_reason": str
            }
        """
        if retry_reason:
            prompt = f"""We attempted to retrieve information for the following search query, but the retrieval confidence was low (reason: {retry_reason}).

Please rewrite the search query to make it more semantically diverse. Rephrase, expand, or use synonyms to target alternative phrasings and find relevant documents for this original question.
Return ONLY the new rewritten query, nothing else.

Original query: {query}
Diversified search query:"""
            
            response = await self.llm.ainvoke(prompt)
            rewritten = extract_text(response.content)
            
            return {
                "original_query": query,
                "rewritten_query": rewritten,
                "was_rewritten": True,
                "rewrite_reason": f"diversification retry due to: {retry_reason}"
            }

        # Skip rewrite if query is already standalone (no pronouns, no implicit refs)
        if not self._needs_rewriting(query):
            return {
                "original_query": query,
                "rewritten_query": query,
                "was_rewritten": False,
                "rewrite_reason": "query is already standalone"
            }
        
        # Take last N turns of history
        recent_history = conversation_history[-max_history_turns * 2:]
        
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}" 
            for msg in recent_history
        ])
        
        prompt = f"""Given this conversation history:
{history_text}
 
Rewrite this follow-up query as a fully standalone search query that can be 
understood without any context. Resolve all pronouns and implicit references.
Return ONLY the rewritten query, nothing else.

Follow-up query: {query}
Standalone query:"""
        
        # We assume the llm_client has an ainvoke method like LangChain Chat Models
        response = await self.llm.ainvoke(prompt)
        rewritten = extract_text(response.content)
        
        return {
            "original_query": query,
            "rewritten_query": rewritten,
            "was_rewritten": True,
            "rewrite_reason": "pronoun/reference resolution"
        }
    
    def _needs_rewriting(self, query: str) -> bool:
        pronouns = ["it", "its", "they", "their", "this", "that", "these",
                   "those", "he", "she", "him", "her", "the same", "the above"]
        query_lower = query.lower()
        return any(f" {p} " in f" {query_lower} " for p in pronouns)
