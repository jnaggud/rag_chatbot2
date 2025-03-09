"""
Semantic chunking implementation for the RAG chatbot.
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP, BREAKPOINT_THRESHOLD_AMOUNT, BREAKPOINT_THRESHOLD_TYPE

logger = logging.getLogger(__name__)

class SemanticChunkingProcessor:
    """Class for processing documents with semantic chunking."""
    
    def __init__(
        self,
        embedding_function,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
        breakpoint_threshold_amount: float = BREAKPOINT_THRESHOLD_AMOUNT,
        breakpoint_threshold_type: str = BREAKPOINT_THRESHOLD_TYPE
    ):
        """
        Initialize the semantic chunking processor.
        
        Args:
            embedding_function: Function to generate embeddings
            chunk_size: Maximum size of chunks for fallback chunker
            chunk_overlap: Overlap between chunks for fallback chunker
            breakpoint_threshold: Threshold for semantic breakpoints
        """
        self.embedding_function = embedding_function
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.breakpoint_threshold_type = breakpoint_threshold_type
        
        # Initialize chunkers
        self.semantic_chunker = SemanticChunker(
            embedding_function,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            breakpoint_threshold_type=self.breakpoint_threshold_type
        )
        
        self.fallback_chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
    
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """
        Process documents using semantic chunking with fallback to traditional chunking.
        
        Args:
            documents: List of documents to chunk
            
        Returns:
            List of chunked documents
        """
        logger.info(f"Semantically chunking {len(documents)} documents...")
        try:
            # First try semantic chunking
            chunked_docs = self.semantic_chunker.split_documents(documents)
            
            # If semantic chunking produced no chunks or very few chunks, fall back to traditional chunking
            if len(chunked_docs) < len(documents):
                logger.info("Semantic chunking produced fewer chunks than expected. Falling back to traditional chunking.")
                chunked_docs = self.fallback_chunker.split_documents(documents)
            
            logger.info(f"Chunking complete. Created {len(chunked_docs)} chunks.")
            return chunked_docs
        except Exception as e:
            logger.error(f"Error during semantic chunking: {e}")
            logger.info("Falling back to traditional chunking...")
            return self.fallback_chunker.split_documents(documents)
