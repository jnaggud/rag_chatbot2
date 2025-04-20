# rag/reranker.py

import logging
from config.settings import RERANK_TOP_K

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Perform a secondary, embedding-based reranking of retrieved documents.
    """

    def __init__(self, embedding_function, top_k: int = RERANK_TOP_K):
        """
        Args:
            embedding_function: a callable that takes a list of texts and returns embeddings
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
        # 1. Embed the query
        query_emb = self.embedding_function([query])[0]

        # 2. Embed each candidate document
        texts    = [d["page_content"] for d in docs]
        # returns list of lists
        doc_embs = self.embedding_function(texts)

        # 3. Compute cosine similarities
        from numpy import dot
        from numpy.linalg import norm

        sims = [
            dot(query_emb, emb) / (norm(query_emb)*norm(emb) + 1e-8)
            for emb in doc_embs
        ]

        # 4. Sort by descending sim, take top_k
        ranked = sorted(
            zip(sims, docs),
            key=lambda x: x[0],
            reverse=True
        )[: self.top_k]

        # 5. Return only the doc dicts, in new order
        return [doc for _, doc in ranked]
