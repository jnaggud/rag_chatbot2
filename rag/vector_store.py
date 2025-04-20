# rag/vector_store.py

import json
import logging
from typing import List, Dict, Any

from chromadb import Client
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from config.settings import PERSIST_DIRECTORY, DESCRIPTIONS_FILE

logger = logging.getLogger(__name__)

class VectorDatabaseManager:
    """
    Manages a Chroma-backed vector store with multiple named collections (domains),
    plus a persisted JSON file of index descriptions.
    """

    def __init__(
        self,
        embedding_function,                # Callable that maps text->vector
        persist_directory: str = PERSIST_DIRECTORY,
        descriptions_file: str = DESCRIPTIONS_FILE
    ):
        # Where Chroma will persist its on-disk DB
        self.persist_directory = persist_directory
        # Where we keep our human-written descriptions of each index
        self.descriptions_file = descriptions_file

        # Load or initialize the descriptions JSON
        try:
            with open(self.descriptions_file, "r") as f:
                self.index_descriptions = json.load(f)
        except FileNotFoundError:
            self.index_descriptions = {}

        # Build a Chroma client using the new v0.6+ API
        settings = Settings(
            persist_directory=self.persist_directory,
            anonymized_telemetry=False
        )
        # Wrap your embedding function so it has the signature Chroma expects
        ef = embedding_functions.DefaultEmbeddingFunction(embedding_function)
        self.client = Client(settings=settings, embedding_function=ef)

    def save_descriptions(self) -> None:
        """Persist the index descriptions back to disk."""
        with open(self.descriptions_file, "w") as f:
            json.dump(self.index_descriptions, f, indent=2)

    def create_index(self, name: str, description: str) -> None:
        """
        Create (or load) a named collection in Chroma and record its description.
        """
        # Update and save our descriptions file
        self.index_descriptions[name] = description
        self.save_descriptions()

        # Chroma v0.6: list_collections() returns a list of names
        existing = self.client.list_collections()
        if name not in existing:
            self.client.get_or_create_collection(name)

    def add_documents_to_index(self, name: str, docs: List[Any]) -> None:
        """
        Add a batch of chunked documents to the 'name' collection.
        Each doc must have:
          - doc.page_content (str)
          - doc.embedding     (list[float])
          - doc.metadata      (dict), containing at least 'source' & 'chunk'
        """
        col = self.client.get_collection(name)

        ids, embeddings, documents, metadatas = [], [], [], []
        for doc in docs:
            # Build a unique ID per chunk
            src   = doc.metadata.get("source", "")
            chunk = doc.metadata.get("chunk", 0)
            _id   = f"{src}-chunk-{chunk}"
            ids.append(_id)

            embeddings.append(doc.embedding)
            documents.append(doc.page_content)
            metadatas.append(doc.metadata)

        col.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(self, *, name: str, query: str, k: int) -> List[Dict[str, Any]]:
        """
        Coarsely retrieve up to `k` hits from the named collection for `query`,
        returning a list of {"page_content":..., "metadata":...} dicts.
        """
        col = self.client.get_collection(name)
        results = col.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "metadatas"]
        )
        hits = []
        # results["documents"] is [[doc1,doc2,...]] since we passed a single query_text
        for text, meta in zip(results["documents"][0], results["metadatas"][0]):
            hits.append({
                "page_content": text,
                "metadata": meta
            })
        return hits
