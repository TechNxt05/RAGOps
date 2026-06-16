import hashlib
from typing import Optional, List

class PromptCacheManager:
    """
    Manages prompt caching for LLM API calls.
    Supports Anthropic prefix caching and OpenAI cached input tokens.
    Caches: system prompts, project context, few-shot examples.
    """
    
    def __init__(self):
        self._system_prompt_cache: dict = {}
    
    def build_anthropic_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: List[dict],
        retrieved_context: str,
        project_name: str
    ) -> dict:
        """
        Build Anthropic API payload with cache_control markers.
        
        Cache-eligible content (large, stable, repeated):
        - System prompt (same for all queries in a project)
        - Project context/instructions
        
        Not cached (changes per query):
        - Retrieved chunks (different every query)
        - User message
        - Conversation history
        """
        return {
            "system": [
                {
                    "type": "text",
                    "text": system_prompt,
                    "cache_control": {"type": "ephemeral"}
                    # ephemeral = cache for up to 5 minutes
                    # Anthropic charges 25% write, 10% read vs full price
                }
            ],
            "messages": [
                # Conversation history (not cached - changes per session)
                *[
                    {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                    for msg in conversation_history[-6:]  # Last 3 turns
                ],
                # Current query with retrieved context
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": f"Context from {project_name} knowledge base:\n\n{retrieved_context}\n\nQuestion: {user_message}"
                        }
                    ]
                }
            ]
        }
    
    def build_openai_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: List[dict],
        retrieved_context: str
    ) -> List[dict]:
        """
        Build OpenAI API messages.
        OpenAI automatically caches prompts >= 1024 tokens.
        Structure to maximize cache hits: stable content first.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            # Stable few-shot examples would go here if any
        ]
        
        # Add conversation history
        messages.extend([
            {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            for msg in conversation_history[-6:]
        ])
        
        # Current query (changes every request - goes last)
        messages.append({
            "role": "user",
            "content": f"Context:\n{retrieved_context}\n\nQuestion: {user_message}"
        })
        
        return messages
    
    def build_groq_messages(
        self,
        system_prompt: str,
        user_message: str,
        conversation_history: List[dict],
        retrieved_context: str
    ) -> List[dict]:
        """
        Groq does not support prefix caching.
        Standard message format, optimized for minimal tokens.
        """
        return [
            {"role": "system", "content": system_prompt},
            *[
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
                for msg in conversation_history[-4:]
            ],
            {
                "role": "user",
                "content": f"Context:\n{retrieved_context}\n\nQuestion: {user_message}"
            }
        ]
    
    def get_system_prompt_hash(self, system_prompt: str) -> str:
        """Track which system prompts are being cached."""
        return hashlib.sha256(system_prompt.encode()).hexdigest()[:16]
    
    def estimate_cache_savings(
        self,
        system_prompt_tokens: int,
        queries_per_day: int,
        provider: str
    ) -> dict:
        """
        Estimate daily cost savings from prompt caching.
        For display in admin analytics.
        """
        savings_rate = {
            "anthropic": 0.90,   # 90% reduction on cached tokens
            "openai": 0.50,      # 50% reduction on cached input
            "groq": 0.0          # No caching support
        }.get(provider, 0.0)
        
        # Approximate cost per token (Claude Sonnet)
        cost_per_token = 0.000003
        
        daily_uncached = system_prompt_tokens * queries_per_day * cost_per_token
        daily_cached = daily_uncached * (1 - savings_rate)
        daily_savings = daily_uncached - daily_cached
        
        return {
            "provider": provider,
            "system_prompt_tokens": system_prompt_tokens,
            "savings_rate_percent": savings_rate * 100,
            "estimated_daily_savings_usd": round(daily_savings, 4),
            "estimated_monthly_savings_usd": round(daily_savings * 30, 2)
        }
