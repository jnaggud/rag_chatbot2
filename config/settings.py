"""
Configuration settings for the RAG chatbot.
"""

# Default paths
DEFAULT_PERSIST_DIRECTORY = "./chroma_db"

# Chunking parameters
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
BREAKPOINT_THRESHOLD_AMOUNT = 0.9
BREAKPOINT_THRESHOLD_TYPE = "percentile"

# Retrieval parameters
DEFAULT_TOP_K = 5

# Embedding model
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# LLM Model
DEFAULT_LLM_MODEL = "llama3"

# Logging
LOG_LEVEL = "INFO"
