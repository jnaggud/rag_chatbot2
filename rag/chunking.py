# rag/chunking.py

import logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SemanticChunkingProcessor:
    def __init__(self, embedding_fn, chunk_size: int = 512, chunk_overlap: int = 128):
        self.embedding_fn = embedding_fn
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f"Chunking configured: size={chunk_size}, overlap={chunk_overlap}")

    def chunk_documents(self, docs: list[Document]) -> list[Document]:
        logger.info(f"Semantically chunking {len(docs)} documents...")
        all_chunks = []
        for doc in docs:
            pieces = self.splitter.split_text(doc.page_content)
            for i, text in enumerate(pieces):
                chunk_meta = doc.metadata.copy()
                # Unique ID for incremental indexing
                chunk_meta["chunk_id"] = f"{chunk_meta.get('source','doc')}_chunk_{i}"
                all_chunks.append(Document(page_content=text, metadata=chunk_meta))
        logger.info(f"Chunking complete. Created {len(all_chunks)} chunks.")
        return all_chunks
