"""
Semantic reranking implementation for the RAG chatbot.
"""

import logging
import numpy as np
from typing import List
from langchain_core.documents import Document
from config.settings import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

class SemanticReranker:
    """Reranks retrieved documents based on semantic similarity to the query."""
    
    def __init__(self, embedding_function, top_k: int = DEFAULT_TOP_K):
        """
        Initialize the semantic reranker.
        
        Args:
            embedding_function: Function to generate embeddings
            top_k: Number of top documents to return after reranking
        """
        self.embedding_function = embedding_function
        self.top_k = top_k
        
        logger.info(f"Semantic Reranker initialized with top_k={top_k}")
    
    def rerank(self, query: str, documents: List[Document]) -> List[Document]:
        """
        Rerank documents based on semantic similarity to the query.
        
        Args:
            query: The user query
            documents: List of documents to rerank
            
        Returns:
            Reranked list of documents
        """
        if not documents:
            logger.warning("No documents to rerank.")
            return []
        
        try:
            # Get query embedding
            query_embedding = self.embedding_function.embed_query(query)
            
            # Calculate similarities
            similarities = []
            for i, doc in enumerate(documents):
                doc_text = doc.page_content
                doc_embedding = self.embedding_function.embed_query(doc_text)
                similarity = self._calculate_cosine_similarity(query_embedding, doc_embedding)
                similarities.append((i, similarity))
            
            # Sort by similarity
            sorted_idxs = sorted(similarities, key=lambda x: x[1], reverse=True)
            
            # Take top-k documents
            top_k = min(self.top_k, len(documents))
            top_doc_idxs = [idx for idx, _ in sorted_idxs[:top_k]]
            
            # Return reranked documents
            reranked_docs = [documents[idx] for idx in top_doc_idxs]
            
            logger.info(f"Reranked {len(documents)} documents, returning top {top_k}.")
            return reranked_docs
        except Exception as e:
            logger.error(f"Error in semantic reranking: {e}")
            # Return original documents as fallback
            return documents[:min(self.top_k, len(documents))]
    
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
