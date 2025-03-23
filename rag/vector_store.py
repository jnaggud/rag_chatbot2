"""
Vector database management for the RAG chatbot.
"""

import logging
import os
from typing import List, Dict
from langchain_core.documents import Document
from langchain_chroma import Chroma
from utils.persistence import save_index_descriptions, load_index_descriptions

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """Manager for creating and maintaining multiple indexes in a vector database."""
    
    def __init__(
        self, 
        embedding_function, 
        persist_directory: str = "./chroma_db",
        descriptions_file: str = "./index_descriptions.json"
    ):
        """
        Initialize the vector database manager.
        
        Args:
            embedding_function: Function to generate embeddings
            persist_directory: Directory to persist the vector database
            descriptions_file: File to persist index descriptions
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.descriptions_file = descriptions_file
        
        # Create the directory if it doesn't exist
        os.makedirs(persist_directory, exist_ok=True)
        
        # Dictionary to store collections
        self.collections = {}
        
        # Load existing index descriptions
        self.index_descriptions = load_index_descriptions(descriptions_file)
        
        logger.info(f"Vector Database Manager initialized with persist directory: {persist_directory}")
    
    def create_index(self, index_name: str, index_description: str = None) -> None:
        """
        Create a new index in the vector database.
        
        Args:
            index_name: Name of the index
            index_description: Description of the index for query routing
        """
        try:
            # If no description is provided, use existing one if available
            if index_description is None:
                index_description = self.index_descriptions.get(index_name, f"Index for {index_name} data")
            
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
            
            # Save updated index descriptions
            save_index_descriptions(self.index_descriptions, self.descriptions_file)
            
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
    
    def update_index_description(self, index_name: str, new_description: str) -> bool:
        """
        Update the description of an existing index.
        
        Args:
            index_name: Name of the index
            new_description: New description for the index
            
        Returns:
            True if successful, False otherwise
        """
        if index_name not in self.collections:
            logger.error(f"Index '{index_name}' does not exist.")
            return False
        
        try:
            self.index_descriptions[index_name] = new_description
            save_index_descriptions(self.index_descriptions, self.descriptions_file)
            logger.info(f"Updated description for index '{index_name}'")
            return True
        except Exception as e:
            logger.error(f"Error updating index description: {e}")
            return False