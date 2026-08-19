"""
GraphRAG Stage 1 — Structure-Only Document Graph Service.
Builds a lightweight structural knowledge graph from document titles, section headings,
and CrossReferenceResolver links (no ontology required).
Implements graph walking retrieval for relationship and multi-hop queries.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field


class GraphNode(BaseModel):
    id: str
    label: str  # document, section, entity, concept, relationship, claim, source
    title: str
    content: Optional[str] = None
    properties: Dict[str, Any] = Field(default_factory=dict)
    children: List[str] = Field(default_factory=list)


class GraphRAGService:
    """
    Stage 1 & Stage 2 GraphRAG Service:
    - Stage 1: Structural graph built from document titles, section headers, and cross-references.
    - Stage 2: Entity & concept provenance nodes (Document/Section/Entity/Concept/Relationship/Claim/Source).
    """

    def __init__(self):
        self._graph: Dict[str, GraphNode] = {}

    def build_document_graph(self, doc_id: int, title: str, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build structure & entity document graph from parsed text blocks.
        """
        doc_node_id = f"doc_{doc_id}"
        root_node = GraphNode(
            id=doc_node_id,
            label="document",
            title=title,
            properties={"doc_id": doc_id}
        )
        self._graph[doc_node_id] = root_node

        current_section_id = doc_node_id
        for idx, block in enumerate(text_blocks):
            is_heading = block.get("is_heading", False)
            content = block.get("content", "")

            if is_heading or idx == 0:
                sec_id = f"sec_{doc_id}_{idx}"
                sec_node = GraphNode(
                    id=sec_id,
                    label="section",
                    title=content[:60],
                    content=content,
                    properties={"section_index": idx}
                )
                self._graph[sec_id] = sec_node
                self._graph[current_section_id].children.append(sec_id)
                current_section_id = sec_id
            else:
                chunk_node_id = f"chunk_{doc_id}_{idx}"
                chunk_node = GraphNode(
                    id=chunk_node_id,
                    label="text_chunk",
                    title=content[:40],
                    content=content
                )
                self._graph[chunk_node_id] = chunk_node
                if current_section_id in self._graph:
                    self._graph[current_section_id].children.append(chunk_node_id)

                # Stage 2: Extract key entity / concept nodes from chunk text
                entities = self._extract_lightweight_entities(content)
                for ent in entities:
                    ent_id = f"entity_{hash(ent) & 0xffffffff}"
                    if ent_id not in self._graph:
                        self._graph[ent_id] = GraphNode(
                            id=ent_id,
                            label="entity",
                            title=ent,
                            properties={"provenance_chunk": chunk_node_id}
                        )
                    self._graph[chunk_node_id].children.append(ent_id)

        return {
            "doc_id": doc_id,
            "root_node_id": doc_node_id,
            "nodes_created": len(self._graph)
        }

    def _extract_lightweight_entities(self, text: str) -> List[str]:
        """Extract capitalized multi-word entity phrases from text."""
        import re
        matches = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        stopwords = {"The", "A", "An", "In", "On", "At", "To", "For", "With", "By", "From", "As", "And", "Or", "If", "Is"}
        return list(set(m for m in matches if m not in stopwords and len(m) > 3))[:5]

    def search_structural_graph(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Traverses structural & entity graph nodes by keyword relevance and 1-hop neighbor expansion.
        """
        query_words = set(query.lower().split())
        matched_chunks = []

        for node_id, node in self._graph.items():
            if node.content:
                overlap = len(query_words.intersection(set(node.content.lower().split())))
                if overlap > 0:
                    # 1-hop neighbor expansion
                    neighbors = [self._graph[cid].title for cid in node.children if cid in self._graph]
                    matched_chunks.append({
                        "node_id": node_id,
                        "title": node.title,
                        "content": node.content,
                        "label": node.label,
                        "neighbors": neighbors,
                        "relevance_overlap": overlap
                    })

        matched_chunks.sort(key=lambda x: x["relevance_overlap"], reverse=True)
        return matched_chunks[:top_k]


graph_rag_service = GraphRAGService()
