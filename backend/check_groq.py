try:
    from langchain_groq import GroqEmbeddings
    print("GroqEmbeddings found!")
except ImportError:
    print("GroqEmbeddings NOT found.")
