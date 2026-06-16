from typing import Dict, List, Optional
from datetime import datetime, timedelta

class SessionContextCache:
    def __init__(self, ttl_minutes: int = 30, max_sessions: int = 1000):
        self._cache: Dict[str, dict] = {}
        self.ttl = timedelta(minutes=ttl_minutes)
        self.max_sessions = max_sessions
    
    def store(self, session_id: str, query: str, chunks: List[dict]) -> None:
        """Store retrieved chunks for a session after fresh retrieval."""
        self._evict_expired()
        self._cache[str(session_id)] = {
            "chunks": chunks,
            "last_query": query,
            "stored_at": datetime.utcnow(),
            "hit_count": 0
        }
    
    def get(self, session_id: str) -> Optional[List[dict]]:
        """Get cached chunks for a session."""
        if str(session_id) not in self._cache:
            return None
        
        entry = self._cache[str(session_id)]
        
        # Check TTL
        if datetime.utcnow() - entry["stored_at"] > self.ttl:
            del self._cache[str(session_id)]
            return None
        
        entry["hit_count"] += 1
        return entry["chunks"]
    
    def has_context(self, session_id: str) -> bool:
        """Check if session has cached context (for TurnTypeRouter)."""
        return self.get(session_id) is not None
    
    def invalidate(self, session_id: str) -> None:
        """Invalidate cache when new document is uploaded to project."""
        if str(session_id) in self._cache:
            del self._cache[str(session_id)]
    
    def _evict_expired(self) -> None:
        now = datetime.utcnow()
        expired = [
            k for k, v in self._cache.items() 
            if now - v["stored_at"] > self.ttl
        ]
        for k in expired:
            del self._cache[k]
    
    def get_stats(self) -> dict:
        return {
            "active_sessions": len(self._cache),
            "total_cached_chunks": sum(
                len(v["chunks"]) for v in self._cache.values()
            )
        }

# Global singleton
session_cache = SessionContextCache()
