import math
from typing import List

def calculate_recall_at_k(retrieved_ids: List[int], expected_ids: List[int], k: int) -> float:
    if not expected_ids:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for e in expected_ids if e in retrieved_k)
    return hits / len(expected_ids)

def calculate_precision_at_k(retrieved_ids: List[int], expected_ids: List[int], k: int) -> float:
    if not retrieved_ids[:k]:
        return 0.0
    retrieved_k = retrieved_ids[:k]
    hits = sum(1 for e in expected_ids if e in retrieved_k)
    return hits / len(retrieved_k)

def calculate_mrr(retrieved_ids: List[int], expected_ids: List[int]) -> float:
    for i, r_id in enumerate(retrieved_ids):
        if r_id in expected_ids:
            return 1.0 / (i + 1)
    return 0.0

def calculate_ndcg(retrieved_ids: List[int], expected_ids: List[int], k: int) -> float:
    dcg = 0.0
    idcg = 0.0
    
    # Calculate DCG
    for i, r_id in enumerate(retrieved_ids[:k]):
        if r_id in expected_ids:
            # Binary relevance
            rel = 1
            dcg += rel / math.log2(i + 2) # i+2 because i is 0-indexed and formula uses rank+1 (i+1+1)
            
    # Calculate IDCG (Ideal DCG)
    num_relevant = min(len(expected_ids), k)
    for i in range(num_relevant):
        idcg += 1 / math.log2(i + 2)
        
    if idcg == 0.0:
        return 0.0
        
    return dcg / idcg
