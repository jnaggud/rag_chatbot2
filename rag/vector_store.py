# rag/vector_store.py

import logging
import os
from typing import List

from langchain_core.documents import Document
from langchain_chroma import Chroma

from utils.persistence import save_index_descriptions, load_index_descriptions

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    def __init__(
        self,
        embedding_function,
        persist_directory: str,
        descriptions_file: str
    ):
        """
        Manage multiple Chroma collections with incremental indexing.

        Args:
            embedding_function: A LangChain embedding function
            persist_directory: Base folder to store each collection
            descriptions_file: JSON file to load/save index descriptions
        """
        self.embedding_function = embedding_function
        self.persist_directory = persist_directory
        self.descriptions_file = descriptions_file

        os.makedirs(persist_directory, exist_ok=True)

        # Load previously‑saved index descriptions (if any)
        self.index_descriptions = load_index_descriptions(descriptions_file)
        self.collections = {}

        logger.info(f"VectorDB Manager initialized at {persist_directory}")

    def create_index(self, name: str, description: str = None):
        """
        Create a new Chroma collection (or open an existing one) for `name`.
        """
        # Fill in description if missing
        if description is None:
            description = self.index_descriptions.get(name, f"Index '{name}'")

        # Ensure on‐disk folder exists
        idx_dir = os.path.join(self.persist_directory, name)
        os.makedirs(idx_dir, exist_ok=True)

        # Initialize Chroma
        col = Chroma(
            collection_name=name,
            embedding_function=self.embedding_function,
            persist_directory=idx_dir,
        )
        self.collections[name] = col

        # Save description
        self.index_descriptions[name] = description
        save_index_descriptions(self.index_descriptions, self.descriptions_file)

        logger.info(f"Created index '{name}' with description: {description}")

    def add_documents_to_index(self, name: str, docs: List[Document]):
        """
        Incrementally add new chunks to the named collection.

        Each Document.metadata must contain a unique "chunk_id" string.
        """
        if name not in self.collections:
            logger.error(f"Index '{name}' not found; cannot add documents.")
            return

        col = self.collections[name]

        # 1) Load all existing chunk_ids
        existing_chunk_ids = set()
        results = col.get(include=["metadatas"])
        for meta in results.get("metadatas", []):
            cid = meta.get("chunk_id")
            if cid is not None:
                existing_chunk_ids.add(cid)

        # 2) Filter out docs we already have
        texts_to_add = []
        metas_to_add = []
        for doc in docs:
            cid = doc.metadata.get("chunk_id")
            if cid is None:
                # Skip any chunk without an ID
                continue
            if cid in existing_chunk_ids:
                # Already indexed
                continue
            texts_to_add.append(doc.page_content)
            metas_to_add.append(doc.metadata)

        if not texts_to_add:
            logger.info(f"No new chunks to index for '{name}'.")
            return

        # 3) Add only the new ones
        col.add_texts(texts=texts_to_add, metadatas=metas_to_add)
        logger.info(f"Indexed {len(texts_to_add)} new chunks into '{name}'.")

    def get_index(self, name: str):
        """Return the Chroma collection for `name` (or None)."""
        return self.collections.get(name)
