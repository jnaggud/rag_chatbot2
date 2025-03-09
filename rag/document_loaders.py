"""
Document loading functionality for the RAG chatbot.
"""

import logging
from typing import List
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

logger = logging.getLogger(__name__)

def load_documents(directory_path: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
    """
    Load PDF documents from a directory with a specified glob pattern.
    
    Args:
        directory_path: Path to the directory containing documents
        glob_pattern: Pattern to match document files
        
    Returns:
        List of Document objects
    """
    logger.info(f"Loading PDF documents from {directory_path} with pattern {glob_pattern}...")
    try:
        loader = DirectoryLoader(
            directory_path,
            glob=glob_pattern,
            loader_cls=PyPDFLoader,
            show_progress=True,
            use_multithreading=True
        )
        documents = loader.load()
        logger.info(f"Loaded {len(documents)} PDF documents.")
        return documents
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        return []
