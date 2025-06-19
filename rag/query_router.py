# rag/query_router.py

import logging
import numpy as np
from numpy.linalg import norm

from config.settings import COARSE_TOP_K

logger = logging.getLogger(__name__)

class QueryRouter:
    """
    Routes a user query to the most relevant domain index(es)
    by comparing the query embedding to each domain’s description embedding.
    """

    def __init__(self, vector_db_manager, embedding_fn, top_k=None):
        """
        Args:
            vector_db_manager: VectorDatabaseManager instance, which knows your indexes
            embedding_fn: an embedding function with `embed_query(text)` -> np.ndarray
            top_k: how many top domains to return (defaults to COARSE_TOP_K)
        """
        self.vdb = vector_db_manager
        self.embedding_fn = embedding_fn
        self.top_k = top_k if top_k is not None else COARSE_TOP_K

    def route(self, query: str):
        """
        Returns a list of index names sorted by relevance (highest cosine
        similarity) between the query and each index’s description.
        """
        # 1) Embed the user’s query
        q_emb = self.embedding_fn.embed_query(query)

        # 2) Load your index descriptions
        #    Expect a dict: { "domain1": "Description text …", ... }
        descriptions = self.vdb.get_index_descriptions()

        logger.info(f"Routing query: '{query}'")
        logger.info(f"Available domain descriptions for routing: {descriptions}")

        scores = []
        for name, desc in descriptions.items():
            # embed each domain description
            d_emb = self.embedding_fn.embed_query(desc)
            # cosine similarity
            sim = float(np.dot(q_emb, d_emb) / ((norm(q_emb) * norm(d_emb)) + 1e-10))
            logger.info(f"  Domain: '{name}', Similarity: {sim:.4f}, Description: '{desc}'")
            scores.append((name, sim))

        # 3) sort by similarity descending, take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        logger.info(f"Sorted routing scores: {scores}")
        
        # Always select only the top 1 domain for routing purposes
        # If scores is empty, top_domains will be empty.
        top_domains = [scores[0][0]] if scores else [] 
        logger.info(f"Selected top 1 domain(s) for query '{query}': {top_domains}")

        # The original log line below might be slightly misleading if top_domains is empty
        # Consider adjusting if it causes confusion, but the core logic is above.
        logger.info(f"Routed query “{query}” → {top_domains}")
        return top_domains
