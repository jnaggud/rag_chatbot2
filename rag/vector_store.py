# rag/vector_store.py
"""
Persistent Chroma vector‑store helper.
"""

from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Sequence

import chromadb
from chromadb import PersistentClient          # ▼ NEW
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config.settings import PERSIST_DIRECTORY, DESCRIPTIONS_FILE

logger = logging.getLogger(__name__)


def _wrap_embedding(langchain_embeddings):
    """Return a Chroma‑compatible embedding fn."""
    try:
        return embedding_functions.create_langchain_embedding(langchain_embeddings)
    except AttributeError:
        logger.warning(
            "create_langchain_embedding missing – falling back to "
            "DefaultEmbeddingFunction (old Chroma)."
        )
        return embedding_functions.DefaultEmbeddingFunction(langchain_embeddings)


class VectorDatabaseManager:
    """High‑level wrapper that hides Chroma’s collection handling."""

    def __init__(
        self,
        embedding_function,
        persist_directory: str = PERSIST_DIRECTORY,
        descriptions_file: str = DESCRIPTIONS_FILE,
    ) -> None:
        self.persist_directory = persist_directory
        self.descriptions_file = descriptions_file
        self.embed_fn = _wrap_embedding(embedding_function)

        # ---------------- descriptions side‑car -----------------
        try:
            with open(self.descriptions_file, "r") as fh:
                self.index_descriptions: Dict[str, str] = json.load(fh)
        except FileNotFoundError:
            self.index_descriptions = {}

        # ---------------- persistent Chroma client -------------- ▼ CHANGED
        self.client: chromadb.ClientAPI = PersistentClient(
            path=self.persist_directory,
            settings=Settings(anonymized_telemetry=False),
        )

    # ---------- helper & public API sections are unchanged ----------
    # ... _save_descriptions, _collection_exists, _get_collection ...
    # ... get_index_descriptions, create_index, add_documents_to_index ...
    # ... query ...


    # ================================================================== #
    # Internal utilities                                                 #
    # ================================================================== #
    def _save_descriptions(self) -> None:
        with open(self.descriptions_file, "w") as fh:
            json.dump(self.index_descriptions, fh, indent=2)

    def _collection_exists(self, name: str) -> bool:
        exists: Sequence = self.client.list_collections()
        # newer Chroma returns List[str]; older returns List[Collection]
        return (
            name in exists
            if len(exists) and isinstance(exists[0], str)
            else name in [c.name for c in exists]
        )

    def _get_collection(self, name: str):
        """
        Always pass the embed_fn when (re)opening a collection.  Chroma
        caches it internally so we don’t pay a cost each call.
        """
        return self.client.get_collection(name, embedding_function=self.embed_fn)

    # ================================================================== #
    # Public API                                                         #
    # ================================================================== #
    def create_index(self, name: str, description: str) -> None:
        """Create a new collection (if absent) and store its description."""
        self.index_descriptions[name] = description
        self._save_descriptions()

        if not self._collection_exists(name):
            self.client.get_or_create_collection(
                name, embedding_function=self.embed_fn
            )

    # ---------------------------------------------- #
        # -------------------------------------------------------------- #
    def add_documents_to_index(self, name: str, docs: List[Any]) -> None:
        """
        Add a batch of LangChain `Document`s to collection *name*.

        Chroma will embed the texts on the fly via the collection’s
        embedding_function, so we only send:
            • ids          : unique per chunk
            • documents    : chunk text
            • metadatas    : original metadata
        """
        import uuid

        col = self._get_collection(name)

        ids, documents, metadatas = [], [], []
        for doc in docs:
            # build a deterministic – but unique – id
            src = doc.metadata.get("source", "unknown")
            chunk = doc.metadata.get("chunk", 0)
            ids.append(f"{src}-chunk-{chunk}-{uuid.uuid4()}")
            documents.append(doc.page_content)
            metadatas.append(doc.metadata)

        col.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )


    # ---------------------------------------------- #
    def query(self, *, name: str, query: str, k: int):
        """Return top‑*k* matches from collection *name* for *query*."""
        col = self._get_collection(name)
        res = col.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"],
        )

        hits = []
        for text, meta in zip(res["documents"][0], res["metadatas"][0]):
            hits.append({"page_content": text, "metadata": meta})
        return hits

    # =============================================================== #
    # Public helpers                                                  #
    # =============================================================== #
    def get_index_descriptions(self):
        """
        Return the mapping { collection_name : text_description }.

        Required by rag/query_router.py when it decides which domain
        should handle an incoming query.
        """
        return self.index_descriptions

    # # In rag/vector_store.py, in VectorDatabaseManager
    # def query_by_vector(self, name, query_vector, k=10):
    #     collection = self.client.get_collection(name=name)
    #     return collection.query(query_embeddings=[query_vector], n_results=k)
    def query_by_vector(self, name, query_vector, k=10):
        collection = self.client.get_collection(name=name)
        res = collection.query(query_embeddings=[query_vector], n_results=k, include=["documents", "metadatas"])
        hits = []
        for text, meta in zip(res["documents"][0], res["metadatas"][0]):
            hits.append({"page_content": text, "metadata": meta})
        return hits
