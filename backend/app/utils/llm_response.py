"""
Utility to safely extract text from LLM responses.

Some Langchain + Gemini/OpenRouter combinations return response.content
as a list[dict] (e.g. [{'type': 'text', 'text': '...', 'extras': {...}}])
instead of a plain string. This utility normalizes both formats.
"""


def extract_delta(content) -> str:
    """Text of a streaming chunk, WITHOUT stripping: token deltas carry their
    own leading/trailing spaces and newlines."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
            if not isinstance(item, dict) or item.get("type", "text") == "text"
        )
    return str(content)


def extract_text(content) -> str:
    """
    Extract plain text from an LLM response content field.
    
    Handles:
    - str: returned as-is (stripped)
    - list[dict]: extracts 'text' field from each dict, joins with space
    - list[str]: joins with space
    - other: str() conversion
    
    Returns:
        Plain text string, stripped of leading/trailing whitespace.
    """
    if content is None:
        return ""
    
    if isinstance(content, str):
        return content.strip()
    
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict):
                # Common format: {'type': 'text', 'text': 'actual content', ...}
                text_val = item.get("text", "")
                if text_val:
                    texts.append(str(text_val))
            elif isinstance(item, str):
                texts.append(item)
            else:
                texts.append(str(item))
        return " ".join(texts).strip() if texts else str(content)
    
    return str(content).strip()
