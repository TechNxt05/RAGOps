import hashlib
from typing import Optional, Any

class VersionedSemanticCache:
    """
    Wraps the existing semantic cache with version-aware keys.
    Cache key = hash(query + project_id + kb_version)
    When kb_version changes, all old cached responses become unreachable.
    """
    def __init__(self, base_cache):
        self.base_cache = base_cache  # Existing SemanticCache instance
        
    def _build_key(self, query: str, project_id: int, kb_version: int) -> str:
        """Build version-aware cache key."""
        key_content = f"{query}::{project_id}::{kb_version}"
        return hashlib.sha256(key_content.encode("utf-8")).hexdigest()
        
    def get(self, query: str, project_id: int, kb_version: int) -> Optional[str]:
        """Lookup query response in versioned cache."""
        versioned_query = self._build_key(query, project_id, kb_version)
        return self.base_cache.get(versioned_query)
        
    def set(self, query: str, response: str, project_id: int, kb_version: int):
        """Cache query response with versioned key."""
        versioned_query = self._build_key(query, project_id, kb_version)
        self.base_cache.set(versioned_query, response)

    @property
    def stats(self) -> dict:
        return self.base_cache.stats
