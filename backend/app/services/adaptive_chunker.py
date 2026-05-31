"""
Adaptive Chunking Service — Ekimetrics paper implementation
Selects optimal chunking strategy per document using 5 intrinsic metrics.
No ground-truth QA required — purely structural scoring.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple
import re
import math

class ChunkStrategy(Enum):
    PAGE_BASED = "page_based"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"
    SPLIT_THEN_MERGE = "split_then_merge"

@dataclass
class ChunkQualityMetrics:
    """5 intrinsic metrics from Ekimetrics paper"""
    size_compliance: float          # SC: chunks within embedding token limits
    intrachunk_cohesion: float      # ICC: semantic focus within each chunk
    contextual_coherence: float     # DCC: alignment with surrounding context
    block_integrity: float          # BI: tables/code blocks kept whole
    reference_completeness: float   # RC: entity-pronoun pairs intact
    
    @property
    def composite_score(self) -> float:
        """Weighted composite — BI and RC weighted higher (structural integrity)"""
        return (
            self.size_compliance * 0.15 +
            self.intrachunk_cohesion * 0.20 +
            self.contextual_coherence * 0.20 +
            self.block_integrity * 0.25 +
            self.reference_completeness * 0.20
        )

class AdaptiveChunker:
    """
    Selects optimal chunking strategy per document.
    Runs all 4 strategies, scores each with 5 intrinsic metrics,
    returns best strategy + its chunks + metrics for admin display.
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_tokens: int = 512  # embedding model limit
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
    
    def select_and_chunk(
        self, 
        text: str, 
        document_type: str = "unknown"
    ) -> Tuple[List[str], ChunkStrategy, ChunkQualityMetrics, dict]:
        """
        Main entry point. Returns:
        - chunks: List[str]
        - selected_strategy: ChunkStrategy
        - metrics: ChunkQualityMetrics (best strategy's metrics)
        - all_scores: dict (all strategies and their scores, for admin display)
        """
        candidates = {
            ChunkStrategy.PAGE_BASED: self._page_based_chunk(text),
            ChunkStrategy.RECURSIVE: self._recursive_chunk(text),
            ChunkStrategy.SEMANTIC: self._semantic_chunk(text),
            ChunkStrategy.SPLIT_THEN_MERGE: self._split_then_merge_chunk(text),
        }
        
        scores = {}
        for strategy, chunks in candidates.items():
            if chunks:
                metrics = self._score_chunks(chunks, text)
                scores[strategy] = (chunks, metrics)
        
        # Select strategy with highest composite score
        best_strategy = max(scores.keys(), key=lambda s: scores[s][1].composite_score)
        best_chunks, best_metrics = scores[best_strategy]
        
        all_scores = {
            s.value: {
                "composite": m.composite_score,
                "size_compliance": m.size_compliance,
                "intrachunk_cohesion": m.intrachunk_cohesion,
                "contextual_coherence": m.contextual_coherence,
                "block_integrity": m.block_integrity,
                "reference_completeness": m.reference_completeness,
                "chunk_count": len(c)
            }
            for s, (c, m) in scores.items()
        }
        
        return best_chunks, best_strategy, best_metrics, all_scores
    
    def _page_based_chunk(self, text: str) -> List[str]:
        """Split on page boundaries — best for legal/structured docs"""
        pages = re.split(r'\f|\n{3,}|--- ?[Pp]age \d+', text)
        chunks = []
        for page in pages:
            page = page.strip()
            if len(page) > 50:
                # If page too long, split further
                if len(page) > self.chunk_size * 4:
                    chunks.extend(self._recursive_chunk(page))
                else:
                    chunks.append(page)
        return chunks if chunks else self._recursive_chunk(text)
    
    def _recursive_chunk(self, text: str) -> List[str]:
        """LangChain-style recursive character splitting"""
        separators = ["\n\n", "\n", ". ", " ", ""]
        return self._recursive_split(text, separators)
    
    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text] if text.strip() else []
        
        separator = separators[0] if separators else ""
        splits = text.split(separator) if separator else list(text)
        
        chunks = []
        current = ""
        for split in splits:
            if len(current) + len(split) + len(separator) <= self.chunk_size:
                current += (separator if current else "") + split
            else:
                if current:
                    chunks.append(current)
                if len(split) > self.chunk_size and len(separators) > 1:
                    chunks.extend(self._recursive_split(split, separators[1:]))
                else:
                    current = split
        if current:
            chunks.append(current)
        
        # Add overlap
        if self.chunk_overlap > 0:
            overlapped = []
            for i, chunk in enumerate(chunks):
                if i > 0:
                    prev_words = chunks[i-1].split()[-self.chunk_overlap//10:]
                    chunk = " ".join(prev_words) + " " + chunk
                overlapped.append(chunk)
            return overlapped
        return chunks
    
    def _semantic_chunk(self, text: str) -> List[str]:
        """
        Sentence-boundary aware chunking.
        Groups sentences until chunk_size reached, respects paragraph breaks.
        """
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    def _split_then_merge_chunk(self, text: str) -> List[str]:
        """
        Ekimetrics split-then-merge: split aggressively, merge tiny fragments.
        Eliminates context-poor micro-chunks that waste retrieval slots.
        """
        min_chunk_size = self.chunk_size // 4
        
        # Aggressive split first
        raw_chunks = re.split(r'\n\n|\n(?=[A-Z])|(?<=[.!?])\s+(?=[A-Z])', text)
        raw_chunks = [c.strip() for c in raw_chunks if c.strip()]
        
        # Merge tiny fragments with neighbors
        merged = []
        buffer = ""
        for chunk in raw_chunks:
            if len(buffer) + len(chunk) < self.chunk_size:
                buffer += (" " + chunk if buffer else chunk)
                # Merge if buffer still below minimum
                if len(buffer) >= min_chunk_size:
                    merged.append(buffer)
                    buffer = ""
            else:
                if buffer:
                    merged.append(buffer)
                buffer = chunk
        if buffer:
            if merged and len(buffer) < min_chunk_size:
                merged[-1] += " " + buffer  # Merge tiny tail into last chunk
            else:
                merged.append(buffer)
        
        return merged
    
    def _score_chunks(self, chunks: List[str], original_text: str) -> ChunkQualityMetrics:
        """Compute all 5 intrinsic metrics for a chunking strategy"""
        return ChunkQualityMetrics(
            size_compliance=self._score_size_compliance(chunks),
            intrachunk_cohesion=self._score_intrachunk_cohesion(chunks),
            contextual_coherence=self._score_contextual_coherence(chunks),
            block_integrity=self._score_block_integrity(chunks, original_text),
            reference_completeness=self._score_reference_completeness(chunks),
        )
    
    def _score_size_compliance(self, chunks: List[str]) -> float:
        """SC: what fraction of chunks are within embedding token limits"""
        if not chunks:
            return 0.0
        # Estimate tokens: ~4 chars per token
        compliant = sum(1 for c in chunks if len(c) / 4 <= self.max_tokens)
        return compliant / len(chunks)
    
    def _score_intrachunk_cohesion(self, chunks: List[str]) -> float:
        """
        ICC: measure semantic focus within each chunk.
        Proxy: low lexical diversity (high word repetition) = focused topic.
        More rigorous: embedding similarity between sentences in chunk.
        """
        if not chunks:
            return 0.0
        scores = []
        for chunk in chunks:
            words = chunk.lower().split()
            if len(words) < 5:
                scores.append(0.5)
                continue
            # Type-token ratio inverted: lower TTR = more focused = higher cohesion
            unique_words = set(words)
            ttr = len(unique_words) / len(words)
            # Normalize: TTR of 0.3-0.7 is typical for focused text
            cohesion = max(0, 1 - abs(ttr - 0.5) * 2)
            scores.append(cohesion)
        return sum(scores) / len(scores)
    
    def _score_contextual_coherence(self, chunks: List[str]) -> float:
        """
        DCC: do adjacent chunks share context?
        Proxy: word overlap between consecutive chunks.
        Higher overlap = smoother boundaries = better coherence.
        """
        if len(chunks) < 2:
            return 1.0
        
        scores = []
        for i in range(len(chunks) - 1):
            words_a = set(chunks[i].lower().split())
            words_b = set(chunks[i+1].lower().split())
            if not words_a or not words_b:
                continue
            # Jaccard similarity
            intersection = words_a & words_b
            union = words_a | words_b
            overlap = len(intersection) / len(union)
            scores.append(overlap)
        
        return sum(scores) / len(scores) if scores else 0.5
    
    def _score_block_integrity(self, chunks: List[str], original_text: str) -> float:
        """
        BI: are structural blocks (tables, code, lists) kept whole?
        Detect blocks in original, check if they're split across chunks.
        """
        # Detect structured blocks
        block_patterns = [
            r'\|.+\|.+\|',           # Markdown tables
            r'```[\s\S]+?```',        # Code blocks
            r'^\s*[-*]\s.+$',         # List items
            r'^\d+\.\s.+$',           # Numbered lists
        ]
        
        blocks_found = 0
        blocks_intact = 0
        
        for pattern in block_patterns:
            matches = re.findall(pattern, original_text, re.MULTILINE)
            for match in matches:
                blocks_found += 1
                # Check if this block appears intact in any single chunk
                match_clean = match.strip()[:100]  # First 100 chars as fingerprint
                if any(match_clean in chunk for chunk in chunks):
                    blocks_intact += 1
        
        if blocks_found == 0:
            return 1.0  # No blocks to break = perfect score
        return blocks_intact / blocks_found
    
    def _score_reference_completeness(self, chunks: List[str]) -> float:
        """
        RC: are entity-pronoun pairs kept intact?
        Check if pronouns (it, they, this, these) appear without their antecedent
        in the same chunk.
        """
        pronouns = {'it', 'its', 'they', 'their', 'them', 'this', 'these', 'that', 'those'}
        
        if not chunks:
            return 0.0
        
        orphan_pronoun_chunks = 0
        for chunk in chunks:
            words = chunk.lower().split()
            if not words:
                continue
            
            # Check if chunk starts with a pronoun (antecedent likely in prev chunk)
            first_meaningful_word = next(
                (w for w in words[:5] if w not in {'the', 'a', 'an', 'and', 'but', 'or'}),
                words[0] if words else ""
            )
            
            if first_meaningful_word in pronouns:
                orphan_pronoun_chunks += 1
        
        return 1.0 - (orphan_pronoun_chunks / len(chunks))


def get_adaptive_chunker(chunk_size: int = 1000, chunk_overlap: int = 200) -> AdaptiveChunker:
    return AdaptiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
