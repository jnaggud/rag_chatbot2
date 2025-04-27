# rag/retriever.py

import logging
from config.settings import COARSE_TOP_K
from typing import List, Dict, Any
# At the top
from utils.hyde_utils import hyde_embed_query

logger = logging.getLogger(__name__)

class RAGRetriever:
    """
    Document retriever for RAG:
      1) Routes a query to one or more domains/collections
      2) Does a first-pass (coarse) retrieval from Chroma
      3) Reranks those results semantically
    """

    def __init__(
        self,
        vector_db,         # VectorDatabaseManager
        router,            # QueryRouter
        reranker,          # SemanticReranker
        coarse_k: int = COARSE_TOP_K
    ):
        """
        Args:
            vector_db: your Chroma-backed vector store manager
            router:    your QueryRouter (figures out which collections to hit)
            reranker:  your SemanticReranker (rescores top docs)
            coarse_k:  how many initial hits to fetch per domain
        """
        self.vdb = vector_db
        self.router = router
        self.reranker = reranker
        self.coarse_k = coarse_k
        logger.info(f"RAG Retriever initialized with coarse_k={self.coarse_k}")

# In RAGRetriever class, change:
    def retrieve(self, query: str, use_hyde: bool = False) -> List[Dict[str, Any]]:
        """
        Given a user query, return a list of document dicts,
        sorted by semantic relevance after reranking.
        Each dict must have at least 'page_content' (string) and any metadata.
        """
        # 1) decide which domains to search
        domains = self.router.route(query)
        logger.debug(f"Routing query '{query}' to domains: {domains}")

        # 2) gather coarse results
        all_hits = []
        for domain in domains:
            if use_hyde:
                # Use HyDE embedding for retrieval
                query_vec = hyde_embed_query(query)
                hits = self.vdb.query_by_vector(
                    name=domain,
                    query_vector=query_vec,
                    k=self.coarse_k
                )
            else:
                hits = self.vdb.query(
                    name=domain,
                    query=query,
                    k=self.coarse_k
                )
            logger.debug(f"Fetched {len(hits)} hits from domain '{domain}'")
            all_hits.extend(hits)

        if not all_hits:
            logger.warning(f"No hits found for query '{query}'")
            return []

        # 3) rerank globally
        reranked = self.reranker.rerank(query, all_hits)
        logger.debug(f"Reranked to {len(reranked)} docs")
        return reranked
