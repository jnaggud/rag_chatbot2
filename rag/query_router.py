"""
Query routing implementation for the RAG chatbot.
"""

import logging
import numpy as np
from typing import List

logger = logging.getLogger(__name__)

class QueryRouter:
    """Routes queries to the most relevant index based on semantic similarity."""
    
    def __init__(self, vector_db_manager, embedding_function):
        """
        Initialize the query router.
        
        Args:
            vector_db_manager: Vector database manager
            embedding_function: Function to generate embeddings
        """
        self.vector_db_manager = vector_db_manager
        self.embedding_function = embedding_function
        
        logger.info("Query Router initialized.")
    
    def get_most_relevant_index(self, query: str, top_k: int = 1) -> List[str]:
        """
        Determine the most relevant index(es) for a given query.
        
        Args:
            query: The user query
            top_k: Number of top indexes to return
            
        Returns:
            List of the top_k most relevant index names
        """
        if not self.vector_db_manager.index_descriptions:
            logger.error("No index descriptions available for routing.")
            return list(self.vector_db_manager.collections.keys())
        
        try:
            # Get query embedding
            query_embedding = self.embedding_function.embed_query(query)
            
            # Calculate similarity with each index description
            similarities = {}
            for index_name, description in self.vector_db_manager.index_descriptions.items():
                description_embedding = self.embedding_function.embed_query(description)
                similarity = self._calculate_cosine_similarity(query_embedding, description_embedding)
                similarities[index_name] = similarity
            
            # Sort indexes by similarity
            sorted_indexes = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Get top-k indexes
            top_indexes = [index_name for index_name, _ in sorted_indexes[:top_k]]
            
            logger.info(f"Query routed to index(es): {', '.join(top_indexes)}")
            return top_indexes
        except Exception as e:
            logger.error(f"Error in query routing: {e}")
            # Fallback to all indexes
            return list(self.vector_db_manager.collections.keys())
    
    def _calculate_cosine_similarity(self, vec1, vec2) -> float:
        """
        Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity value
        """
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)
        return dot_product / (norm_vec1 * norm_vec2)
