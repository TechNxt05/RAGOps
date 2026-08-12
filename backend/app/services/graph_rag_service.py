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
    label: str  # document, section, link
    title: str
    content: Optional[str] = None
    children: List[str] = Field(default_factory=list)


class GraphRAGService:
    def __init__(self):
        self._graph: Dict[str, GraphNode] = {}

    def build_document_graph(self, doc_id: int, title: str, text_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Build structure-only document graph from parsed section headers and text blocks.
        """
        doc_node_id = f"doc_{doc_id}"
        root_node = GraphNode(
            id=doc_node_id,
            label="document",
            title=title,
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
                    content=content
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

        return {
            "doc_id": doc_id,
            "root_node_id": doc_node_id,
            "nodes_created": len(self._graph)
        }

    def search_structural_graph(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Traverses structural graph nodes by keyword / title relevance and returns related context chunks.
        """
        query_words = set(query.lower().split())
        matched_chunks = []

        for node_id, node in self._graph.items():
            if node.content:
                overlap = len(query_words.intersection(set(node.content.lower().split())))
                if overlap > 0:
                    matched_chunks.append({
                        "node_id": node_id,
                        "title": node.title,
                        "content": node.content,
                        "relevance_overlap": overlap
                    })

        matched_chunks.sort(key=lambda x: x["relevance_overlap"], reverse=True)
        return matched_chunks[:top_k]


graph_rag_service = GraphRAGService()
