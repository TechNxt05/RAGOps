from enum import Enum
from typing import Optional
import re

class TurnType(Enum):
    CHIT_CHAT = "chit_chat"
    FOLLOW_UP = "follow_up"
    FRESH_RETRIEVAL = "fresh_retrieval"

class TurnTypeRouter:
    CHIT_CHAT_PATTERNS = [
        r"^(hi|hello|hey|thanks|thank you|ok|okay|got it|sure|great|perfect)[\s!.]*$",
        r"^(explain|summarize|simplify|rephrase|clarify|elaborate)\s+(that|this|it|more)",
        r"^(what do you mean|can you repeat|say that again)",
        r"^(yes|no|maybe|correct|wrong|right|exactly)[\s!.]*$",
    ]
    
    FOLLOW_UP_PRONOUNS = [
        r"\b(it|its|they|their|them|this|that|these|those|he|she|his|her)\b",
        r"^(what about|how about|and the|tell me more about|what else|also)",
        r"^(can you|could you|would you).*(more|else|also|another|second|third)",
    ]
    
    def classify(
        self, 
        query: str, 
        session_has_context: bool = False,
        conversation_history: Optional[list] = None
    ) -> TurnType:
        query_lower = query.lower().strip()
        
        # Check chit-chat first
        for pattern in self.CHIT_CHAT_PATTERNS:
            if re.search(pattern, query_lower):
                return TurnType.CHIT_CHAT
        
        # Check follow-up (only if session has retrieved context)
        if session_has_context:
            for pattern in self.FOLLOW_UP_PRONOUNS:
                if re.search(pattern, query_lower):
                    return TurnType.FOLLOW_UP
            # Short queries with no new nouns when context exists
            if len(query.split()) <= 6 and session_has_context:
                return TurnType.FOLLOW_UP
        
        return TurnType.FRESH_RETRIEVAL
