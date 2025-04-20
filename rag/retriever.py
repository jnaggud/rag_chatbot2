# rag/retriever.py

import logging
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

class RAGRetriever:
    def __init__(self, vector_db_mgr, router, reranker,
                 coarse_top_k: int = 10, rerank_top_k: int = 5):
        self.vdb = vector_db_mgr
        self.router = router
        self.reranker = reranker
        self.coarse_top_k = coarse_top_k
        self.rerank_top_k = rerank_top_k
        logger.info(f"Retriever initialized (coarse_k={coarse_top_k}, rerank_k={rerank_top_k})")

    def retrieve(self, query: str) -> list[Document]:
        # 1) Route to index
        idxs = self.router.get_most_relevant_index(query, top_k=1)
        col = self.vdb.get_index(idxs[0])

        # 2) Coarse retrieval
        coarse = col.similarity_search(query, k=self.coarse_top_k)
        logger.info(f"Coarse retrieved {len(coarse)} docs.")

        # 3) Fine semantic rerank
        reranked = self.reranker.rerank(query, coarse)[:self.rerank_top_k]
        logger.info(f"Returning top {len(reranked)} reranked docs.")
        return reranked
