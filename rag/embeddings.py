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
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Check if we're using the all-mpnet-base-v2 model
        if "all-mpnet-base-v2" in model_name:
            # Optimized settings for all-mpnet-base-v2
            model_kwargs = {
                'device': 'cpu'  # Change to 'cuda' if you have a GPU
            }
            encode_kwargs = {
                'normalize_embeddings': True,  # Important for cosine similarity
                'batch_size': 16  # Adjust based on your available RAM
                # Removed 'show_progress_bar': True - this is what caused the conflict
            }
            
            embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs,
                cache_folder="./model_cache"  # Cache the model locally
            )
        else:
            # Default settings for other models
            embeddings = HuggingFaceEmbeddings(model_name=model_name)
        
        logger.info(f"Embedding function using {model_name} created successfully.")
        return embeddings
    except Exception as e:
        logger.error(f"Error creating embedding function: {e}")
        raise