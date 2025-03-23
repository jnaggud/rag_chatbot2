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

# Chunking parameters
CHUNK_SIZE = 512         # Increased for better context
CHUNK_OVERLAP = 128      # Increased overlap for better continuity
BREAKPOINT_THRESHOLD = 0.15  # Lower threshold is better for this model
BREAKPOINT_THRESHOLD_AMOUNT = 0.15
BREAKPOINT_THRESHOLD_TYPE = "percentile"

# Retrieval parameters
DEFAULT_TOP_K = 7

# Embedding model
#DEFAULT_EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# To
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"

# LLM Model
DEFAULT_LLM_MODEL = "llama3"

# Logging
LOG_LEVEL = "INFO"
