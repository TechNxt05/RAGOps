from enum import Enum
from typing import Optional

class QueryCategory(Enum):
    FACTOID = "factoid"           # Single fact lookup - fast path
    ANALYTICAL = "analytical"     # Reasoning required - deep path
    COMPARATIVE = "comparative"   # Multiple entities - parallel path
    PROCEDURAL = "procedural"     # Step-by-step - sequential path
    DEFINITIONAL = "definitional" # Definition/explanation - direct path

class RetrievalStrategy(Enum):
    FAST = "fast"           # top_k=3, no multi-query, direct BGE rerank
    STANDARD = "standard"   # top_k=6, multi-query on, standard rerank
    DEEP = "deep"           # top_k=10, multi-query on, full pipeline
    PARALLEL = "parallel"   # Run two separate retrievals, merge

class SemanticRouter:
    """
    Routes queries to optimal retrieval strategies based on query semantics.
    Sits between Phase 1 query intelligence and the retrieval pipeline.
    """
    
    CATEGORY_SIGNALS = {
        QueryCategory.FACTOID: [
            "what is", "what was", "when did", "who is", "where is",
            "how much", "how many", "what date", "what time",
            "is it", "does it", "which one"
        ],
        QueryCategory.ANALYTICAL: [
            "why", "how does", "explain", "what causes", "what impact",
            "analyze", "what led to", "reason for", "effect of",
            "what happens when", "how would"
        ],
        QueryCategory.COMPARATIVE: [
            "compare", "difference between", "vs", "versus",
            "better than", "worse than", "similar to",
            "how does x differ", "contrast", "which is better"
        ],
        QueryCategory.PROCEDURAL: [
            "how to", "steps to", "process for", "procedure",
            "guide to", "instructions for", "how do i",
            "what are the steps", "walk me through"
        ],
        QueryCategory.DEFINITIONAL: [
            "what does", "define", "meaning of", "definition",
            "what is meant by", "describe", "tell me about",
            "overview of", "introduction to"
        ]
    }
    
    # Updated mapping to ensure query Category names align with Strategies
    STRATEGY_MAP = {
        QueryCategory.FACTOID: RetrievalStrategy.FAST,
        QueryCategory.ANALYTICAL: RetrievalStrategy.DEEP,
        QueryCategory.COMPARATIVE: RetrievalStrategy.PARALLEL,
        QueryCategory.PROCEDURAL: RetrievalStrategy.STANDARD,
        QueryCategory.DEFINITIONAL: RetrievalStrategy.STANDARD,
    }
    
    STRATEGY_CONFIGS = {
        RetrievalStrategy.FAST: {
            "top_k": 3,
            "use_multi_query": False,
            "rerank_top_n": 3,
            "description": "Fast path: single query, top 3 results"
        },
        RetrievalStrategy.STANDARD: {
            "top_k": 6,
            "use_multi_query": True,
            "rerank_top_n": 5,
            "description": "Standard path: multi-query, top 6 results"
        },
        RetrievalStrategy.DEEP: {
            "top_k": 10,
            "use_multi_query": True,
            "rerank_top_n": 8,
            "description": "Deep path: multi-query, broad retrieval"
        },
        RetrievalStrategy.PARALLEL: {
            "top_k": 5,
            "use_multi_query": True,
            "rerank_top_n": 6,
            "parallel_queries": True,
            "description": "Parallel path: separate retrievals merged"
        }
    }
    
    def classify(
        self, 
        query: str,
        existing_intent: Optional[str] = None
    ) -> dict:
        """
        Classify query into category and select retrieval strategy.
        """
        query_lower = query.lower().strip()
        
        # Check each category's signals
        scores = {}
        for category, signals in self.CATEGORY_SIGNALS.items():
            score = sum(
                1 for signal in signals 
                if signal in query_lower
            )
            scores[category] = score
        
        # Use existing intent from Phase 1 if available
        if existing_intent:
            intent_map = {
                "factoid": QueryCategory.FACTOID,
                "analytical": QueryCategory.ANALYTICAL,
                "multi_hop": QueryCategory.ANALYTICAL,
            }
            # Look up complexity string directly
            intent_key = existing_intent.lower()
            if intent_key in intent_map:
                category = intent_map[intent_key]
            else:
                category = max(scores, key=scores.get)
        else:
            category = max(scores, key=scores.get)
            # Default to STANDARD if no clear signal
            if scores[category] == 0:
                category = QueryCategory.DEFINITIONAL
        
        strategy = self.STRATEGY_MAP[category]
        config = self.STRATEGY_CONFIGS[strategy]
        
        return {
            "query_category": category.value,
            "retrieval_strategy": strategy.value,
            "top_k": config["top_k"],
            "use_multi_query": config["use_multi_query"],
            "rerank_top_n": config["rerank_top_n"],
            "description": config["description"],
            "category_scores": {k.value: v for k, v in scores.items()}
        }
    
    def get_retrieval_config(self, routing_result: dict) -> dict:
        """Extract retrieval config for use in MultiQueryRetriever."""
        return {
            "top_k": routing_result["top_k"],
            "use_multi_query": routing_result["use_multi_query"],
            "rerank_top_n": routing_result["rerank_top_n"],
        }
