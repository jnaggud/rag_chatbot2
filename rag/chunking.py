# rag/chunking.py

import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)

class SemanticChunkingProcessor:
    """
    Heading‑aware chunker: prefer splitting on '##', '#', paragraph breaks,
    falling back to spaces, with tighter chunk sizes.
    """
    def __init__(self):
        self.splitter = RecursiveCharacterTextSplitter(
            separators=["\n## ", "\n# ", "\n\n", " "],
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        logger.info(f"Chunker configured: size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}")

    def chunk_documents(self, documents):
        return self.splitter.split_documents(documents)
