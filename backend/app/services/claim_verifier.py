"""
Claim-Level Citation Verification Service

Post-generation check that decomposes the LLM answer into atomic claims,
then verifies each claim against the retrieved context chunks.

Each claim is tagged as:
  - SUPPORTED:           Claim is directly supported by at least one chunk.
  - PARTIALLY_SUPPORTED: Claim overlaps with context but key details are missing.
  - UNSUPPORTED:         No evidence for the claim exists in retrieved context.

The verifier outputs an audit trail attached to the output contract,
surfacing `unverified_claims` so the caller can warn the user.
"""

import re
import logging
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logger = logging.getLogger(__name__)


class ClaimStatus(str, Enum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    UNSUPPORTED = "unsupported"


@dataclass
class VerifiedClaim:
    """A single atomic claim extracted from the LLM answer."""
    claim_text: str
    status: ClaimStatus
    confidence: float  # 0.0 – 1.0
    supporting_chunk_indices: List[int] = field(default_factory=list)
    best_overlap_score: float = 0.0
    reason: str = ""


@dataclass
class ClaimVerificationResult:
    """Aggregate verification result for the entire answer."""
    total_claims: int = 0
    supported: int = 0
    partially_supported: int = 0
    unsupported: int = 0
    verification_score: float = 1.0  # weighted fraction of supported claims
    claims: List[VerifiedClaim] = field(default_factory=list)
    unverified_claims: List[str] = field(default_factory=list)
    has_unverified: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_claims": self.total_claims,
            "supported": self.supported,
            "partially_supported": self.partially_supported,
            "unsupported": self.unsupported,
            "verification_score": round(self.verification_score, 3),
            "has_unverified": self.has_unverified,
            "unverified_claims": self.unverified_claims,
            "claims": [asdict(c) for c in self.claims],
        }


class ClaimVerifier:
    """
    Deterministic claim-level citation verifier.

    Pipeline:
      1. Split the generated answer into atomic claims (sentence-level).
      2. For each claim, compute token-overlap similarity against every
         retrieved context chunk.
      3. Classify the claim based on the best overlap score:
         - >= SUPPORTED_THRESHOLD  → SUPPORTED
         - >= PARTIAL_THRESHOLD    → PARTIALLY_SUPPORTED
         - < PARTIAL_THRESHOLD     → UNSUPPORTED
    """

    SUPPORTED_THRESHOLD = 0.45
    PARTIAL_THRESHOLD = 0.20

    # Sentences that are just meta-commentary, not factual claims
    NON_CLAIM_PATTERNS = [
        r"^(here|below|above|following)\s+(is|are)",
        r"^(i|we)\s+(can|cannot|don't|couldn't|do not)",
        r"^(sure|okay|certainly|of course)",
        r"^(let me|i'll|i will)",
        r"^(in summary|to summarize|in conclusion|overall)",
        r"^(based on|according to)\s+(the|my)",
        r"^(note|please note|important)",
    ]

    def __init__(
        self,
        supported_threshold: float = 0.45,
        partial_threshold: float = 0.20,
    ):
        self.supported_threshold = supported_threshold
        self.partial_threshold = partial_threshold

    # ------------------------------------------------------------------
    # Claim extraction
    # ------------------------------------------------------------------

    def _split_into_claims(self, answer: str) -> List[str]:
        """
        Split a generated answer into atomic claim sentences.
        Filters out meta-commentary and trivially short fragments.
        """
        # Split on sentence-ending punctuation
        raw_sentences = re.split(r'(?<=[.!?])\s+', answer.strip())

        claims = []
        for sent in raw_sentences:
            sent = sent.strip()
            if not sent:
                continue

            # Remove markdown artifacts (bullets, numbering)
            sent = re.sub(r'^[\-\*•]\s*', '', sent)
            sent = re.sub(r'^\d+[\.\)]\s*', '', sent)
            sent = sent.strip()

            # Skip very short fragments (< 5 words)
            if len(sent.split()) < 5:
                continue

            # Skip non-claim meta-commentary
            is_meta = False
            for pattern in self.NON_CLAIM_PATTERNS:
                if re.match(pattern, sent.lower()):
                    is_meta = True
                    break
            if is_meta:
                continue

            claims.append(sent)

        return claims

    # ------------------------------------------------------------------
    # Token-level overlap similarity
    # ------------------------------------------------------------------

    def _tokenize(self, text: str) -> set:
        """Lowercase tokenization with stopword removal."""
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "shall", "should", "may", "might", "must", "can",
            "could", "to", "of", "in", "for", "on", "with", "at", "by",
            "from", "as", "into", "through", "during", "before", "after",
            "above", "below", "between", "and", "but", "or", "nor", "not",
            "so", "yet", "both", "either", "neither", "each", "every",
            "all", "any", "few", "more", "most", "other", "some", "such",
            "no", "only", "own", "same", "than", "too", "very", "just",
            "about", "it", "its", "this", "that", "these", "those",
            "i", "me", "my", "we", "our", "you", "your", "he", "him",
            "his", "she", "her", "they", "them", "their", "what", "which",
            "who", "whom", "how", "when", "where", "why",
        }
        tokens = set(re.findall(r'\b\w+\b', text.lower()))
        return tokens - stopwords

    def _compute_overlap(self, claim_tokens: set, chunk_tokens: set) -> float:
        """Jaccard-style overlap biased toward claim coverage."""
        if not claim_tokens:
            return 0.0
        intersection = claim_tokens & chunk_tokens
        # Bias toward claim coverage (what fraction of claim tokens are in the chunk)
        coverage = len(intersection) / len(claim_tokens) if claim_tokens else 0.0
        # Also consider Jaccard for mutual relevance
        union = claim_tokens | chunk_tokens
        jaccard = len(intersection) / len(union) if union else 0.0
        # Weighted blend: 70% claim coverage, 30% Jaccard
        return 0.7 * coverage + 0.3 * jaccard

    # ------------------------------------------------------------------
    # Numeric / entity verification
    # ------------------------------------------------------------------

    def _extract_numbers(self, text: str) -> set:
        """Extract all numeric values from text."""
        return set(re.findall(r'\b\d+(?:\.\d+)?(?:%|st|nd|rd|th)?\b', text))

    def _has_number_mismatch(self, claim: str, best_chunk: str) -> bool:
        """
        Check if a claim contains numbers not present in its
        best-matching chunk — a potential hallucination signal.
        """
        claim_numbers = self._extract_numbers(claim)
        if not claim_numbers:
            return False
        chunk_numbers = self._extract_numbers(best_chunk)
        unmatched = claim_numbers - chunk_numbers
        # If >50% of claim's numbers aren't in the chunk, flag it
        return len(unmatched) / len(claim_numbers) > 0.5 if claim_numbers else False

    # ------------------------------------------------------------------
    # Main verification
    # ------------------------------------------------------------------

    def verify(
        self,
        answer: str,
        context_chunks: List[str],
    ) -> ClaimVerificationResult:
        """
        Verify all claims in the generated answer against context chunks.

        Args:
            answer: The LLM-generated response text.
            context_chunks: List of raw text chunks used as retrieval context.

        Returns:
            ClaimVerificationResult with per-claim audit trail.
        """
        result = ClaimVerificationResult()

        if not answer or not answer.strip():
            return result

        claims = self._split_into_claims(answer)
        if not claims:
            # No extractable claims → treat the whole answer as supported
            result.verification_score = 1.0
            return result

        # Pre-tokenize chunks once
        chunk_token_sets = [self._tokenize(c) for c in context_chunks]

        for claim_text in claims:
            claim_tokens = self._tokenize(claim_text)

            best_score = 0.0
            best_chunk_idx = -1
            supporting_indices = []

            for idx, chunk_tokens in enumerate(chunk_token_sets):
                overlap = self._compute_overlap(claim_tokens, chunk_tokens)
                if overlap > best_score:
                    best_score = overlap
                    best_chunk_idx = idx
                if overlap >= self.partial_threshold:
                    supporting_indices.append(idx)

            # Determine status
            if best_score >= self.supported_threshold:
                status = ClaimStatus.SUPPORTED
                confidence = min(best_score * 1.5, 1.0)
                reason = f"Claim covered by chunk(s) with {best_score:.2f} overlap."
            elif best_score >= self.partial_threshold:
                status = ClaimStatus.PARTIALLY_SUPPORTED
                confidence = best_score
                reason = f"Partial overlap ({best_score:.2f}); some claim details may not be in context."

                # Downgrade further if numbers mismatch
                if best_chunk_idx >= 0 and self._has_number_mismatch(claim_text, context_chunks[best_chunk_idx]):
                    confidence *= 0.5
                    reason += " Numeric mismatch detected."
            else:
                status = ClaimStatus.UNSUPPORTED
                confidence = best_score
                reason = f"No chunk covers this claim (best overlap={best_score:.2f})."

            verified = VerifiedClaim(
                claim_text=claim_text,
                status=status,
                confidence=round(confidence, 3),
                supporting_chunk_indices=supporting_indices,
                best_overlap_score=round(best_score, 3),
                reason=reason,
            )
            result.claims.append(verified)

        # Aggregate
        result.total_claims = len(result.claims)
        result.supported = sum(1 for c in result.claims if c.status == ClaimStatus.SUPPORTED)
        result.partially_supported = sum(1 for c in result.claims if c.status == ClaimStatus.PARTIALLY_SUPPORTED)
        result.unsupported = sum(1 for c in result.claims if c.status == ClaimStatus.UNSUPPORTED)

        # Weighted verification score: supported=1.0, partial=0.5, unsupported=0.0
        if result.total_claims > 0:
            weighted = (
                result.supported * 1.0
                + result.partially_supported * 0.5
                + result.unsupported * 0.0
            )
            result.verification_score = round(weighted / result.total_claims, 3)
        else:
            result.verification_score = 1.0

        # Collect unverified claims for the output contract warning
        result.unverified_claims = [
            c.claim_text for c in result.claims
            if c.status == ClaimStatus.UNSUPPORTED
        ]
        result.has_unverified = len(result.unverified_claims) > 0

        return result
