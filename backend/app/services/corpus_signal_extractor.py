"""
CorpusSignalExtractor — RAGOps Phase 3

Extracts corpus-side structural signals latent in DoclingParser outputs and DB chunks:
- Document type distribution (PDF/DOCX/HTML/Markdown ratio)
- Table density (tables per document / table chunk ratio)
- Section hierarchy depth (max and avg heading depth)
- Cross-reference link density (internal links per chunk using CrossReferenceResolver)
- Multi-column layout presence
"""

import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class CorpusFeatureScores(BaseModel):
    table_density: float = Field(default=0.0, description="Ratio of chunks containing structured tables (0.0 - 1.0)")
    section_depth: float = Field(default=1.0, description="Average heading level depth across document sections")
    max_section_depth: int = Field(default=1, description="Maximum heading depth in corpus")
    cross_ref_density: float = Field(default=0.0, description="Average cross-reference links per chunk")
    pdf_ratio: float = Field(default=0.0, description="Proportion of PDF documents in corpus")
    multi_column_present: bool = Field(default=False, description="Whether multi-column layouts were detected")
    code_density: float = Field(default=0.0, description="Ratio of code blocks or structured JSON/XML chunks")
    total_documents: int = Field(default=0)
    total_chunks: int = Field(default=0)


class CorpusSignalExtractor:
    """
    Extracts structural corpus signals to fuse with query-side features in AdaptiveRetrievalRouter.
    """

    def __init__(self):
        from app.services.cross_reference_resolver import CrossReferenceResolver
        self.ref_resolver = CrossReferenceResolver()

    def extract_from_chunks(
        self,
        chunks: List[Any],
        documents: Optional[List[Any]] = None
    ) -> CorpusFeatureScores:
        """
        Compute structural signals over a list of document chunks or raw text strings.
        """
        if not chunks:
            return CorpusFeatureScores()

        total_chunks = len(chunks)
        table_chunk_count = 0
        code_chunk_count = 0
        heading_depths = []
        cross_ref_count = 0
        multi_column_count = 0

        # Document type distribution
        pdf_count = 0
        doc_count = len(documents) if documents else 0
        if documents:
            for doc in documents:
                fname = getattr(doc, "filename", "") or ""
                if fname.lower().endswith(".pdf"):
                    pdf_count += 1

        pdf_ratio = (pdf_count / doc_count) if doc_count > 0 else 0.0

        for chunk in chunks:
            text = ""
            meta = {}

            if hasattr(chunk, "page_content"):
                text = chunk.page_content or ""
                meta = getattr(chunk, "metadata", {}) or {}
            elif isinstance(chunk, dict):
                text = chunk.get("content") or chunk.get("text") or ""
                meta = chunk.get("metadata") or {}
            else:
                text = str(chunk)

            # Table detection
            if meta.get("element_type") == "table" or "|" in text or "<table>" in text.lower():
                table_chunk_count += 1

            # Code/structured detection
            if "```" in text or "{" in text and "}" in text and ":" in text:
                code_chunk_count += 1

            # Section depth detection
            headings = re.findall(r'^(#{1,6})\s+', text, re.MULTILINE)
            if headings:
                depths = [len(h) for h in headings]
                heading_depths.extend(depths)
            elif meta.get("section"):
                # Estimate depth from section numbering (e.g. 1.2.3 -> depth 3)
                sec = str(meta.get("section"))
                dots = sec.count(".") + 1
                heading_depths.append(dots)

            # Cross-reference detection
            refs = self.ref_resolver.detect_references([text])
            if refs:
                cross_ref_count += len(refs)

            # Multi-column layout hint
            if "column_2" in meta or meta.get("layout") == "multi-column":
                multi_column_count += 1

        avg_depth = (sum(heading_depths) / len(heading_depths)) if heading_depths else 1.0
        max_depth = max(heading_depths) if heading_depths else 1

        return CorpusFeatureScores(
            table_density=round(table_chunk_count / total_chunks, 3),
            section_depth=round(avg_depth, 2),
            max_section_depth=max_depth,
            cross_ref_density=round(cross_ref_count / total_chunks, 3),
            pdf_ratio=round(pdf_ratio, 3),
            multi_column_present=(multi_column_count > 0),
            code_density=round(code_chunk_count / total_chunks, 3),
            total_documents=doc_count,
            total_chunks=total_chunks,
        )


# Singleton instance
corpus_signal_extractor = CorpusSignalExtractor()
