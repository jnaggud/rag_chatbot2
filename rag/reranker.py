# rag/reranker.py

import logging
import numpy as np
from config.settings import RERANK_TOP_K

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Perform a secondary, embedding‑based reranking of retrieved documents.
    """

    def __init__(self, embedding_function, top_k: int = RERANK_TOP_K):
        """
        Args:
            embedding_function: a LangChain Embeddings instance
                                (must implement embed_query & embed_documents)
            top_k: how many to rerank
        """
        self.embedding_function = embedding_function
        self.top_k = top_k
        logger.info(f"Semantic Reranker initialized with top_k={self.top_k}")

    def rerank(self, query: str, docs: list[dict]) -> list[dict]:
        """
        Given the user query and a list of doc dicts (with 'page_content' & metadata),
        compute embeddings and rerank by similarity to the query.
        """
        # 1) Embed the query
        try:
            query_emb = self.embedding_function.embed_query(query)
        except AttributeError:
            # fallback if embed_query isn’t implemented
            query_emb = self.embedding_function.embed_documents([query])[0]

        # 2) Embed each candidate document
        texts = [d["page_content"] for d in docs]
        try:
            doc_embs = self.embedding_function.embed_documents(texts)
        except AttributeError:
            # fallback if only embed_query is available
            doc_embs = [self.embedding_function.embed_query(t) for t in texts]

        # 3) Compute cosine similarities
        sims = [
            float(
                np.dot(query_emb, emb)
                / ((np.linalg.norm(query_emb) * np.linalg.norm(emb)) + 1e-8)
            )
            for emb in doc_embs
        ]

        # 4) Sort by descending sim, take top_k
        ranked = sorted(zip(sims, docs), key=lambda x: x[0], reverse=True)[: self.top_k]

        # 5) Return only the doc dicts, in new order
        return [doc for _, doc in ranked]
