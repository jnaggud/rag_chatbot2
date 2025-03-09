"""
Combined retrieval system for the RAG chatbot.
"""

import logging
from typing import List
from langchain_core.documents import Document
from config.settings import DEFAULT_TOP_K

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Combined retriever that routes queries to the correct index,
    retrieves relevant documents, and reranks them.
    """
    
    def __init__(
        self,
        vector_db_manager,
        query_router,
        semantic_reranker,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Initialize the RAG retriever.
        
        Args:
            vector_db_manager: Vector database manager
            query_router: Query router
            semantic_reranker: Semantic reranker
            top_k: Number of documents to retrieve from each index
        """
        self.vector_db_manager = vector_db_manager
        self.query_router = query_router
        self.semantic_reranker = semantic_reranker
        self.top_k = top_k
        
        logger.info("RAG Retriever initialized.")
    
    def retrieve(self, query: str) -> List[Document]:
        """
        Retrieve documents for a query by routing to the most relevant index(es),
        retrieving relevant documents from each index, and reranking all retrieved documents.
        
        Args:
            query: The user query
            
        Returns:
            List of retrieved and reranked documents
        """
        try:
            # Get most relevant index(es)
            relevant_indexes = self.query_router.get_most_relevant_index(query)
            
            # Retrieve documents from each relevant index
            all_docs = []
            for index_name in relevant_indexes:
                index = self.vector_db_manager.get_index(index_name)
                if index:
                    docs = index.similarity_search(query, k=self.top_k)
                    all_docs.extend(docs)
            
            # Rerank all retrieved documents
            reranked_docs = self.semantic_reranker.rerank(query, all_docs)
            
            logger.info(f"Retrieved and reranked {len(reranked_docs)} documents for query: '{query}'")
            return reranked_docs
        except Exception as e:
            logger.error(f"Error in retrieval process: {e}")
            return []
