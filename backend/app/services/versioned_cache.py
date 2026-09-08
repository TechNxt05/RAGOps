from typing import Optional, Dict, Tuple
from app.services.cost_control import SemanticCache

class VersionedSemanticCache:
    """
    Wraps semantic caching with strict project and version namespace partitioning.
    When kb_version changes, queries access an isolated cache partition, guaranteeing
    that no stale entries from previous document versions can match semantically.
    """
    def __init__(self, base_cache: Optional[SemanticCache] = None):
        self.base_cache = base_cache or SemanticCache()
        self._partitions: Dict[Tuple[int, int], SemanticCache] = {}
        
    def _get_partition(self, project_id: int, kb_version: int) -> SemanticCache:
        key = (int(project_id), int(kb_version))
        if key not in self._partitions:
            self._partitions[key] = SemanticCache(
                threshold=self.base_cache.threshold,
                max_size=self.base_cache.max_size,
                ttl_seconds=self.base_cache.ttl_seconds
            )
        return self._partitions[key]
        
    def get(self, query: str, project_id: int, kb_version: int) -> Optional[str]:
        """Lookup query response in isolated version partition."""
        partition = self._get_partition(project_id, kb_version)
        return partition.get(query)
        
    def set(self, query: str, response: str, project_id: int, kb_version: int):
        """Cache query response in isolated version partition."""
        partition = self._get_partition(project_id, kb_version)
        partition.set(query, response)

    def clear(self, project_id: Optional[int] = None, kb_version: Optional[int] = None):
        """Clear cache partitions."""
        if project_id is not None and kb_version is not None:
            self._partitions.pop((int(project_id), int(kb_version)), None)
        elif project_id is not None:
            to_del = [k for k in self._partitions if k[0] == int(project_id)]
            for k in to_del:
                del self._partitions[k]
        else:
            self._partitions.clear()

    @property
    def stats(self) -> dict:
        total_hits = sum(p._hits for p in self._partitions.values()) + self.base_cache._hits
        total_misses = sum(p._misses for p in self._partitions.values()) + self.base_cache._misses
        total_size = sum(len(p._entries) for p in self._partitions.values())
        total = total_hits + total_misses
        return {
            "hits": total_hits,
            "misses": total_misses,
            "hit_rate": round(total_hits / total, 3) if total > 0 else 0.0,
            "size": total_size,
            "active_partitions": len(self._partitions)
        }

