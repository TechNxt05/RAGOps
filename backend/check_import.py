try:
    from langchain_huggingface import HuggingFaceEmbeddings
    print("✅ SUCCESS: langchain_huggingface imported successfully.")
except ImportError as e:
    print(f"❌ ERROR: {e}")
