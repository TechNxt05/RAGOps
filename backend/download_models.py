import os
print("Pre-downloading HuggingFace Embedding and Reranker models...")

# Download embedding model
from langchain_huggingface import HuggingFaceEmbeddings
print("Downloading sentence-transformers/all-MiniLM-L6-v2...")
HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Download BGE Reranker
print("Downloading BAAI/bge-reranker-base...")
try:
    from FlagEmbedding import FlagReranker
    FlagReranker("BAAI/bge-reranker-base", use_fp16=False)
    print("Models pre-downloaded and cached successfully!")
except Exception as e:
    print(f"Warning: Failed to pre-download FlagReranker: {e}")
