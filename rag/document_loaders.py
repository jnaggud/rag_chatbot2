# rag/document_loaders.py

import os
import glob
import logging
from langchain_community.document_loaders import PyPDFium2Loader, DirectoryLoader

logger = logging.getLogger(__name__)

class DocumentLoader:
    """
    A unified document loader that handles different file types and directories.
    """
    def __init__(self, path):
        self.path = path

    def load_documents(self):
        logger.info(f"Loading documents from: {self.path}")
        loader = DirectoryLoader(
            self.path,
            glob="**/*.pdf",
            loader_cls=PyPDFium2Loader,
            show_progress=True,
            use_multithreading=True,
        )
        documents = loader.load()
        logger.info(f"Successfully loaded {len(documents)} documents.")
        return documents

def load_documents(path):
    """
    Load documents from the specified path.
    
    Args:
        path (str): Path to the directory containing documents or a specific document file
        
    Returns:
        List[Document]: List of loaded documents
    """
    loader = DocumentLoader(path)
    return loader.load_documents()
