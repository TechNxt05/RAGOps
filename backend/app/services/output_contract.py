from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class ResponseFormat(Enum):
    CONCISE = "concise"           # 1-3 sentences, factual queries
    STRUCTURED = "structured"     # bullet points, list queries
    DETAILED = "detailed"         # full explanation, analytical queries
    TABLE = "table"               # tabular comparison queries
    CODE = "code"                 # code/technical queries

class OutputContract(BaseModel):
    format: ResponseFormat
    max_tokens: int
    require_citations: bool
    bullet_points: bool
    max_sentences: Optional[int]
    system_prompt_addendum: str

class OutputContractBuilder:
    """
    Builds output contracts based on query type classification.
    Uses query intent from Phase 1 QueryUnderstanding stage.
    """
    
    FORMAT_CONFIGS = {
        ResponseFormat.CONCISE: {
            "max_tokens": 150,
            "max_sentences": 3,
            "bullet_points": False,
            "require_citations": True,
            "instruction": (
                "Answer in 1-3 sentences maximum. Be direct and factual. "
                "No preamble, no elaboration unless directly relevant. "
                "Cite the source document for every fact."
            )
        },
        ResponseFormat.STRUCTURED: {
            "max_tokens": 400,
            "max_sentences": None,
            "bullet_points": True,
            "require_citations": True,
            "instruction": (
                "Structure your answer as a clear bullet point list. "
                "Each bullet should be one complete thought. "
                "Maximum 6 bullets. Cite sources inline."
            )
        },
        ResponseFormat.DETAILED: {
            "max_tokens": 600,
            "max_sentences": None,
            "bullet_points": False,
            "require_citations": True,
            "instruction": (
                "Provide a thorough explanation with supporting context. "
                "Use paragraphs, not bullets. "
                "Cite every claim to its source document."
            )
        },
        ResponseFormat.TABLE: {
            "max_tokens": 500,
            "max_sentences": None,
            "bullet_points": False,
            "require_citations": False,
            "instruction": (
                "Format your answer as a markdown table where appropriate. "
                "Include all relevant columns. "
                "Add a brief summary sentence after the table."
            )
        },
        ResponseFormat.CODE: {
            "max_tokens": 800,
            "max_sentences": None,
            "bullet_points": False,
            "require_citations": False,
            "instruction": (
                "Format code in proper markdown code blocks with language tags. "
                "Add brief inline comments for clarity. "
                "Keep explanation concise - let the code speak."
            )
        },
    }
    
    def classify_format(
        self, 
        query: str,
        query_intent: Optional[str] = None
    ) -> ResponseFormat:
        """
        Classify required output format from query intent and signals.
        """
        query_lower = query.lower()
        
        # Code signals
        if any(kw in query_lower for kw in [
            "code", "function", "script", "implement", "write a",
            "program", "snippet", "syntax", "example of"
        ]):
            return ResponseFormat.CODE
        
        # Table/comparison signals
        if any(kw in query_lower for kw in [
            "compare", "difference between", "vs", "versus",
            "table", "list all", "what are the", "types of"
        ]):
            return ResponseFormat.TABLE
        
        # List/structured signals
        if any(kw in query_lower for kw in [
            "list", "enumerate", "what are", "steps to",
            "how to", "ways to", "features of", "advantages"
        ]):
            return ResponseFormat.STRUCTURED
        
        # Analytical/detailed signals
        if any(kw in query_lower for kw in [
            "explain", "why", "how does", "describe",
            "elaborate", "analyze", "what causes", "impact of"
        ]):
            return ResponseFormat.DETAILED
        
        # Use query_intent from Phase 1 if available
        if query_intent == "analytical":
            return ResponseFormat.DETAILED
        if query_intent == "factoid":
            return ResponseFormat.CONCISE
        if query_intent == "multi_hop":
            return ResponseFormat.DETAILED
        
        # Default: concise for short queries, detailed for long
        return (
            ResponseFormat.CONCISE 
            if len(query.split()) <= 8 
            else ResponseFormat.DETAILED
        )
    
    def build(
        self, 
        query: str,
        query_intent: Optional[str] = None,
        force_format: Optional[ResponseFormat] = None
    ) -> OutputContract:
        """Build output contract for this query."""
        format_type = force_format or self.classify_format(query, query_intent)
        config = self.FORMAT_CONFIGS[format_type]
        
        return OutputContract(
            format=format_type,
            max_tokens=config["max_tokens"],
            require_citations=config["require_citations"],
            bullet_points=config["bullet_points"],
            max_sentences=config.get("max_sentences"),
            system_prompt_addendum=config["instruction"]
        )
    
    def apply_to_system_prompt(
        self, 
        base_system_prompt: str,
        contract: OutputContract
    ) -> str:
        """Inject output contract instructions into system prompt."""
        contract_section = f"""

RESPONSE FORMAT CONTRACT:
{contract.system_prompt_addendum}
Maximum response length: {contract.max_tokens} tokens.
{"You MUST cite the source document for every factual claim." if contract.require_citations else ""}
Do not exceed these constraints under any circumstances.
"""
        return base_system_prompt + contract_section


class OutputVerifier:
    """
    Post-generation verification of output contract compliance.
    Deterministic checks - no LLM calls.
    """
    
    def verify(
        self, 
        response: str, 
        contract: OutputContract,
        retrieved_chunks: List[dict]
    ) -> dict:
        """
        Verify generated response against output contract.
        Returns verification result with pass/fail per check.
        """
        checks = {}
        
        # Length check (approximate token count)
        approx_tokens = len(response.split()) * 1.3
        checks["length_compliant"] = approx_tokens <= contract.max_tokens * 1.2
        
        # Citation check
        if contract.require_citations:
            source_names = [
                c.get("source", "").split("/")[-1] 
                for c in retrieved_chunks 
                if c.get("source")
            ]
            has_citation = any(
                source.lower() in response.lower() 
                for source in source_names 
                if source
            )
            checks["has_citations"] = has_citation
        else:
            checks["has_citations"] = True
        
        # Sentence count check
        if contract.max_sentences:
            sentences = [s.strip() for s in response.split(".") if len(s.strip()) > 5]
            checks["sentence_count_compliant"] = (
                len(sentences) <= contract.max_sentences
            )
        else:
            checks["sentence_count_compliant"] = True
        
        # Format check for structured responses
        if contract.bullet_points:
            checks["has_bullet_structure"] = (
                "- " in response or "* " in response or "\n•" in response
            )
        else:
            checks["has_bullet_structure"] = True
        
        all_passed = all(checks.values())
        
        return {
            "contract_compliant": all_passed,
            "checks": checks,
            "format_used": contract.format.value,
            "approximate_tokens": int(approx_tokens)
        }
