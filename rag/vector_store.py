"""
Vector database management for the RAG chatbot.
"""

import logging
import os
from typing import List
from langchain_core.documents import Document
from langchain_chroma import Chroma

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """Manager for creating and maintaining multiple indexes in a vector database."""
    
    def __init__(
        self, 
        embedding_function, 
        persist_directory: str = "./chroma_db"
    ):
        """
        Initialize the vector database manager.
        
        Args:
            embedding_function: Function to generate embeddings
            persist_directory: Directory to persist the vector database
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Dictionary to store collections
        self.collections = {}
        
        # Index descriptions for query routing
        self.index_descriptions = {}
        
        logger.info(f"Vector Database Manager initialized with persist directory: {persist_directory}")
    
    def create_index(self, index_name: str, index_description: str) -> None:
        """
        Create a new index in the vector database.
        
        Args:
            index_name: Name of the index
            index_description: Description of the index for query routing
        """
        try:
            # Create a collection directory
            index_directory = os.path.join(self.persist_directory, index_name)
            os.makedirs(index_directory, exist_ok=True)
            
            # Initialize Chroma collection
            collection = Chroma(
                collection_name=index_name,
                embedding_function=self.embedding_function,
                persist_directory=index_directory
            )
            
            self.collections[index_name] = collection
            self.index_descriptions[index_name] = index_description
            
            logger.info(f"Created index '{index_name}' with description: {index_description}")
        except Exception as e:
            logger.error(f"Error creating index '{index_name}': {e}")
            raise
    
    def add_documents_to_index(self, index_name: str, documents: List[Document]) -> None:
        """
        Add documents to an existing index.
        
        Args:
            index_name: Name of the index
            documents: List of documents to add
        """
        if index_name not in self.collections:
            logger.error(f"Index '{index_name}' does not exist.")
            return
        
        try:
            # Extract texts and metadatas
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            
            # Add documents to the collection
            self.collections[index_name].add_texts(texts=texts, metadatas=metadatas)
            
            # Note: persist() is no longer needed as of ChromaDB 0.4.x
            # Documents are automatically persisted
            
            logger.info(f"Added {len(documents)} documents to index '{index_name}'")
        except Exception as e:
            logger.error(f"Error adding documents to index '{index_name}': {e}")
            raise
    
    def get_index(self, index_name: str):
        """
        Get a specific index by name.
        
        Args:
            index_name: Name of the index
            
        Returns:
            The index collection
        """
        if index_name not in self.collections:
            logger.error(f"Index '{index_name}' does not exist.")
            return None
        
        return self.collections[index_name]
    
    def list_indexes(self) -> List[str]:
        """
        List all available indexes.
        
        Returns:
            List of index names
        """
        return list(self.collections.keys())