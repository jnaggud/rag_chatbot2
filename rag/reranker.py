# rag/reranker.py

import logging
import numpy as np
from config.settings import RERANK_TOP_K
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class SemanticReranker:
    """
    Given a list of candidate Documents, embed them along with the query,
    compute similarity scores, and return the top_k best matches.
    """
    def __init__(self, embedding_function, top_k: int = RERANK_TOP_K):
        """
        Args:
            embedding_function: A function with .embed_query() and .embed_documents()
            top_k: Number of documents to return after reranking
        """
        self.embedding_function = embedding_function
        self.top_k = top_k
        logger.info(f"Semantic Reranker initialized with top_k={self.top_k}")

    def rerank(self, query: str, docs: list[Document]) -> list[Document]:
        """
        Rerank the given docs for the query.

        1) Embed the query
        2) Embed each document
        3) Compute dot‐product (cosine) similarity
        4) Sort and return top_k
        """
        if not docs:
            return []

        # 1) embed query
        query_emb = self.embedding_function.embed_query(query)

        # 2) embed documents
        doc_texts = [doc.page_content for doc in docs]
        doc_embs = self.embedding_function.embed_documents(doc_texts)

        # 3) compute similarity scores
        #    assuming embeddings are L2-normalized or raw—dot is fine for ranking
        scores = np.dot(doc_embs, query_emb)

        # 4) rank and pick top_k
        ranked_indices = np.argsort(scores)[::-1][: self.top_k]
        reranked_docs = [docs[i] for i in ranked_indices]

        logger.info(f"Reranked {len(docs)} docs down to top {len(reranked_docs)}")
        return reranked_docs
