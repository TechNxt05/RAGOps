from typing import List, Optional

class QueryRewriter:
    def __init__(self, llm_client):
        self.llm = llm_client
    
    async def rewrite(
        self, 
        query: str, 
        conversation_history: List[dict],
        max_history_turns: int = 3
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
        rewritten = response.content.strip()
        
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
