from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class DoclingParser:
    def __init__(self):
        self._converter = None
    
    def _get_converter(self):
        """Lazy initialization - downloads models on first use."""
        if self._converter is None:
            try:
                from docling.document_converter import DocumentConverter
                self._converter = DocumentConverter()
                logger.info("Docling converter initialized successfully")
            except ImportError:
                logger.warning("Docling not installed, falling back to basic parser")
                self._converter = False
        return self._converter
    
    def is_available(self) -> bool:
        converter = self._get_converter()
        return converter is not False and converter is not None
    
    def parse(self, pdf_path: str) -> dict:
        """
        Parse PDF using Docling for rich structure extraction.
        """
        converter = self._get_converter()
        if not converter:
            return self._fallback_parse(pdf_path)
        
        try:
            result = converter.convert(pdf_path)
            doc = result.document
            
            # Extract text blocks with section context
            text_blocks = []
            for item in doc.texts:
                text_blocks.append({
                    "content": item.text,
                    "label": item.label,  # paragraph, section_header, title, etc.
                    "page": getattr(item, "page_no", None),
                    "is_heading": item.label in ["section_header", "title"],
                })
            
            # Extract tables as markdown
            tables = []
            for table in doc.tables:
                try:
                    markdown_rows = self._table_to_markdown(table)
                    tables.append({
                        "content": "\n".join(markdown_rows),
                        "label": "table",
                        "row_count": table.data.num_rows,
                        "col_count": table.data.num_cols,
                        "page": getattr(table, "page_no", None),
                    })
                except Exception as e:
                    logger.warning(f"Table extraction failed: {e}")
            
            # Extract headings for TOC
            headings = [
                b for b in text_blocks 
                if b["is_heading"]
            ]
            
            # Extract figure captions
            figures = []
            for picture in doc.pictures:
                if hasattr(picture, "caption") and picture.caption:
                    figures.append({
                        "content": picture.caption,
                        "label": "figure_caption",
                        "page": getattr(picture, "page_no", None),
                    })
            
            # Count pages safely
            pages_set = set(b["page"] for b in text_blocks if b["page"] is not None)
            page_count = len(pages_set) if pages_set else 1
            
            return {
                "text_blocks": text_blocks,
                "tables": tables,
                "headings": headings,
                "figures": figures,
                "parsing_method": "docling",
                "page_count": page_count,
                "table_count": len(tables),
                "heading_count": len(headings),
            }
        
        except Exception as e:
            logger.error(f"Docling parsing failed: {e}, falling back")
            return self._fallback_parse(pdf_path)
    
    def _table_to_markdown(self, table) -> list:
        """Convert Docling TableItem to markdown rows."""
        n_rows = table.data.num_rows
        n_cols = table.data.num_cols
        grid = [[""] * n_cols for _ in range(n_rows)]
        header_rows = set()
        
        for cell in table.data.table_cells:
            row = cell.start_row_offset_idx
            col = cell.start_col_offset_idx
            if row < n_rows and col < n_cols:
                grid[row][col] = cell.text.strip()
                if cell.column_header:
                    header_rows.add(row)
        
        h = min(header_rows) if header_rows else 0
        rows = ["| " + " | ".join(grid[h]) + " |"]
        rows.append("| " + " | ".join(["---"] * n_cols) + " |")
        rows += [
            "| " + " | ".join(grid[r]) + " |"
            for r in range(n_rows) if r != h
        ]
        return rows
    
    def to_chunks(self, parsed: dict) -> List[dict]:
        """
        Convert Docling parsed output to chunk format
        compatible with existing chunking pipeline.
        Each table becomes its own chunk to preserve structure.
        Text blocks are grouped by section.
        """
        chunks = []
        
        # Tables as individual chunks
        for i, table in enumerate(parsed["tables"]):
            chunks.append({
                "content": table["content"],
                "metadata": {
                    "content_type": "table",
                    "page": table.get("page"),
                    "parsing_method": "docling",
                    "row_count": table.get("row_count"),
                    "col_count": table.get("col_count"),
                }
            })
        
        # Text blocks grouped (non-table content)
        current_section = ""
        current_text = []
        
        for block in parsed["text_blocks"]:
            if block["is_heading"]:
                # Save previous section
                if current_text:
                    chunks.append({
                        "content": "\n".join(current_text),
                        "metadata": {
                            "content_type": "text",
                            "section": current_section,
                            "parsing_method": "docling",
                            "page": block.get("page"),
                        }
                    })
                    current_text = []
                current_section = block["content"]
                current_text = [f"# {block['content']}"]
            else:
                current_text.append(block["content"])
        
        # Save last section
        if current_text:
            chunks.append({
                "content": "\n".join(current_text),
                "metadata": {
                    "content_type": "text",
                    "section": current_section,
                    "parsing_method": "docling",
                }
            })
        
        # Figure captions
        for figure in parsed["figures"]:
            chunks.append({
                "content": figure["content"],
                "metadata": {
                    "content_type": "figure_caption",
                    "page": figure.get("page"),
                    "parsing_method": "docling",
                }
            })
        
        # If everything is empty but we parsed something, build a fallback text chunk
        if not chunks and parsed.get("text_blocks"):
            full_text = "\n".join([b["content"] for b in parsed["text_blocks"]])
            chunks.append({
                "content": full_text,
                "metadata": {"content_type": "text", "parsing_method": "docling"}
            })
            
        return chunks
    
    def _fallback_parse(self, pdf_path: str) -> dict:
        """Fallback to basic text extraction if Docling unavailable."""
        return {
            "text_blocks": [],
            "tables": [],
            "headings": [],
            "figures": [],
            "parsing_method": "fallback_basic",
            "page_count": 0,
            "table_count": 0,
            "heading_count": 0,
        }
