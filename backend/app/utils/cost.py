def calculate_cost(model_name: str, provider: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate estimated cost in USD based on model and tokens.
    Pricing is per 1 million tokens (updated Jan 2026 est).
    """
    price_map = {
        "google": {
            "gemini-2.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-2.5-pro": {"input": 3.50, "output": 10.50},
        },
        "groq": {
            "llama3-70b-8192": {"input": 0.59, "output": 0.79}, 
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
            "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
            "openai/gpt-oss-20b": {"input": 0.20, "output": 0.20},
        }
    }

    try:
        provider_pricing = price_map.get(provider, {})
        pricing = provider_pricing.get(model_name)
        if not pricing:
            # Fallback for unknown models (to avoid $0.0000)
            pricing = {"input": 0.15, "output": 0.15}
            
        cost_input = (input_tokens / 1_000_000) * pricing["input"]
        cost_output = (output_tokens / 1_000_000) * pricing["output"]
        
        return round(cost_input + cost_output, 6)
    except:
        return 0.0
