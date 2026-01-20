def calculate_cost(model_name: str, provider: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate estimated cost in USD based on model and tokens.
    Pricing is per 1 million tokens (updated Jan 2026 est).
    """
    price_map = {
        "google": {
            "gemini-1.5-flash": {"input": 0.075, "output": 0.30},
            "gemini-1.5-pro": {"input": 3.50, "output": 10.50},
        },
        "groq": {
            # Groq is currently free/beta, but let's use standard Llama 3 pricing as proxy or 0
            "llama-3.3-70b-versatile": {"input": 0.70, "output": 0.90}, # Approx
            "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
        }
    }

    try:
        pricing = price_map.get(provider, {}).get(model_name)
        if not pricing:
            return 0.0
            
        cost_input = (input_tokens / 1_000_000) * pricing["input"]
        cost_output = (output_tokens / 1_000_000) * pricing["output"]
        
        return round(cost_input + cost_output, 6)
    except:
        return 0.0
