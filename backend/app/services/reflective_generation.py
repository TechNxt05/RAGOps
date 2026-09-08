"""
Reflective Generation Service (3-Stage Cascade)

Stage 1: Fast deterministic token-overlap check using ClaimVerifier.
Stage 2: Escalate if claims are PARTIALLY_SUPPORTED or UNSUPPORTED.
Stage 3: LLM Critique and single-attempt regeneration.
"""

import logging
from typing import List, Tuple, Optional
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from app.services.claim_verifier import ClaimVerifier, ClaimVerificationResult, ClaimStatus
from app.utils.llm_response import extract_text
logger = logging.getLogger(__name__)

class ReflectiveGenerator:
    def __init__(self, llm: BaseChatModel):
        self.llm = llm
        self.claim_verifier = ClaimVerifier()

    async def generate_with_reflection(
        self,
        draft_answer: str,
        query: str,
        context_chunks: List[str]
    ) -> Tuple[str, ClaimVerificationResult, int, str]:
        """
        Executes the 3-stage reflective generation cascade.
        
        Returns:
            Tuple containing:
            - final_answer (str)
            - verification_result (ClaimVerificationResult)
            - stage_reached (int: 1, 2, or 3)
            - outcome (str)
        """
        if not context_chunks:
            # Nothing to verify against
            return draft_answer, self.claim_verifier.verify(draft_answer, []), 1, "no_context"

        # Stage 1: Deterministic check
        verification_result = self.claim_verifier.verify(draft_answer, context_chunks, query)
        
        if not verification_result.has_unverified and verification_result.partially_supported == 0:
            # All claims are SUPPORTED. We're done.
            return draft_answer, verification_result, 1, "accepted_stage_1"
            
        # Stage 2: We hit the gray zone. Escalate.
        logger.info(f"Reflective Generation escalated to Stage 3 for query: '{query}'")
        
        # Stage 3: LLM Critique and Rewrite (max_regeneration_attempts = 1)
        unverified_text = "\n".join([f"- {c}" for c in verification_result.unverified_claims])
        partial_claims = [c.claim_text for c in verification_result.claims if c.status == ClaimStatus.PARTIALLY_SUPPORTED]
        partial_text = "\n".join([f"- {c}" for c in partial_claims])
        
        sys_msg = SystemMessage(
            content="You are an expert factual editor. Your job is to rewrite a draft answer "
                    "so that it strictly aligns with the provided source context. "
                    "Do not add any external knowledge. If a claim cannot be verified, remove or soften it."
        )
        
        context_str = "\n\n---\n\n".join(context_chunks)
        
        prompt = f"""
        Original Query: {query}
        
        Source Context:
        {context_str}
        
        Draft Answer:
        {draft_answer}
        
        The following claims from the draft are completely UNSUPPORTED by the context:
        {unverified_text}
        
        The following claims are only PARTIALLY_SUPPORTED (may contain number mismatches or missing details):
        {partial_text}
        
        Please rewrite the draft answer. Your rewrite MUST:
        1. Remove the unsupported claims.
        2. Fix or hedge the partially supported claims to match the context exactly.
        3. Directly answer the original query.
        
        Return ONLY the rewritten answer text.
        """
        
        try:
            response = await self.llm.ainvoke([sys_msg, HumanMessage(content=prompt)])
            rewritten_answer = extract_text(response.content)
            
            # Re-verify the new answer to attach the final metrics
            final_verification = self.claim_verifier.verify(rewritten_answer, context_chunks, query)
            return rewritten_answer, final_verification, 3, "rewritten_stage_3"
            
        except Exception as e:
            logger.error(f"LLM rewrite failed during Stage 3 reflection: {e}")
            # Fallback to the original draft if LLM fails
            return draft_answer, verification_result, 3, "rewrite_failed_fallback"
