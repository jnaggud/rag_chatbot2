"""
Embedding functions for the RAG chatbot.
"""

import logging
from config.settings import DEFAULT_EMBEDDING_MODEL

logger = logging.getLogger(__name__)

def get_embedding_function(model_name=DEFAULT_EMBEDDING_MODEL):
    """
    Get the embedding function for semantic operations.
    
    Args:
        model_name: Name of the embedding model to use
        
    Returns:
        Embedding function
    """
    try:
        # Update import to use the dedicated package
        from langchain_huggingface import HuggingFaceEmbeddings
        
        embeddings = HuggingFaceEmbeddings(model_name=model_name)
        logger.info(f"Embedding function using {model_name} created successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embedding function: {e}")
        raise