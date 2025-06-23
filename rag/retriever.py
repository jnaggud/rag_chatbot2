# rag/retriever.py

import logging
import time
from typing import List, Dict, Any, Set, Optional
from config.settings import COARSE_TOP_K
from rank_bm25 import BM25Okapi
import numpy as np
import hashlib
import json
from dataclasses import dataclass, asdict, field
from datetime import datetime
import os
import asyncio
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class RetrievalMetrics:
    query: str
    domain: str
    retrieval_time_ms: float
    num_results: int
    top_score: float
    query_length: int
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

class RetrievalMonitor:
    """Simple monitoring for retrieval performance."""
    
    def __init__(self, log_file: str = "retrieval_metrics.jsonl"):
        self.log_file = log_file
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        
    def log_retrieval(self, metrics: RetrievalMetrics):
        """Log retrieval metrics to a file."""
        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(asdict(metrics)) + "\n")
        except Exception as e:
            logger.error(f"Failed to log metrics: {e}")

class BM25Retriever:
    """BM25 retriever for hybrid search."""
    
    def __init__(self, k: int = 10):
        self.k = k
        self.corpus = []
        self.doc_metadata = []
        self.bm25 = None

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the BM25 index."""
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get('page_content', '')
                metadata = doc.get('metadata', {})
            else:
                text = getattr(doc, 'page_content', '')
                metadata = getattr(doc, 'metadata', {})
            
            if not text.strip():
                continue
                
            tokens = text.lower().split()
            self.corpus.append(tokens)
            self.doc_metadata.append({
                'page_content': text,
                'metadata': metadata
            })
        
        if self.corpus:
            self.bm25 = BM25Okapi(self.corpus)

    def get_relevant_documents(self, query: str, k: int = None) -> List[Dict[str, Any]]:
        """Retrieve documents using BM25."""
        if not self.bm25:
            return []
            
        k = k or self.k
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k documents
        top_indices = np.argsort(scores)[-k:][::-1]
        return [self.doc_metadata[i] for i in top_indices if scores[i] > 0]

class RAGRetriever:
    """
    Enhanced Document retriever for RAG with hybrid search, query understanding,
    and improved duplicate handling.
    """

    def __init__(
        self,
        vector_db,         # VectorDatabaseManager
        router,           # QueryRouter
        reranker,         # SemanticReranker
        llm=None,         # Optional LLM for query understanding
        bm25_retriever: Optional[BM25Retriever] = None,
        hybrid_alpha: float = 0.5,  # Weight for hybrid search (0.0-1.0)
        coarse_k: int = COARSE_TOP_K,
        enable_monitoring: bool = True
    ):
        self.vdb = vector_db
        self.router = router
        self.reranker = reranker
        self.llm = llm
        self.bm25_retriever = bm25_retriever
        self.hybrid_alpha = hybrid_alpha
        self.coarse_k = coarse_k
        self.monitor = RetrievalMonitor() if enable_monitoring else None
        logger.info(f"RAG Retriever initialized with hybrid_alpha={hybrid_alpha}, coarse_k={coarse_k}")

    def _expand_query(self, query: str) -> List[str]:
        """Generate variations of the query to improve retrieval."""
        # Simple query expansion - can be enhanced with LLM later
        query_lower = query.lower()
        expansions = [query]
        
        # Add variations for common question patterns
        if "how to" in query_lower:
            expansions.append(query.replace("how to", "steps to"))
            expansions.append(query.replace("how to", "guide for"))
        
        if "?" in query:
            expansions.append(query.replace("?", ""))
        else:
            expansions.append(f"{query}?")
            
        # Add time-related expansions for watch-related queries
        time_related = ["time", "set time", "adjust time", "change time"]
        if any(term in query_lower for term in ["watch", "clock", "tudor", "rolex"]):
            expansions.extend([f"{query} {term}" for term in time_related])
        
        return list(set(expansions))  # Remove duplicates

    def _get_doc_id(self, doc: Dict) -> str:
        """Generate a unique ID for a document."""
        return hashlib.md5(
            (str(doc.get('metadata', {}).get('source', '')) + 
             str(doc.get('page_content', ''))).encode()
        ).hexdigest()

    def _remove_duplicate_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate or near-duplicate chunks."""
        seen = set()
        unique_chunks = []
        
        for chunk in chunks:
            chunk_id = self._get_doc_id(chunk)
            if chunk_id not in seen:
                seen.add(chunk_id)
                unique_chunks.append(chunk)
        
        return unique_chunks

    def _hybrid_retrieval(self, query: str, domain: str, k: int) -> List[Dict]:
        """Combine vector and BM25 retrieval results."""
        results = []
        
        # Vector search
        vector_results = self.vdb.query(
            name=domain,
            query=query,
            k=k * 2
        ) if self.vdb else []
        
        # BM25 search if available
        bm25_results = self.bm25_retriever.get_relevant_documents(
            query, k=k * 2
        ) if self.bm25_retriever else []
        
        # If we only have one type of results, return them
        if not bm25_results:
            return vector_results[:k]
        if not vector_results:
            return bm25_results[:k]
        
        # Combine and deduplicate
        seen = set()
        combined = []
        
        # Add vector results first
        for doc in vector_results:
            doc_id = self._get_doc_id(doc)
            if doc_id not in seen:
                seen.add(doc_id)
                doc['_score'] = doc.get('_score', 0) * self.hybrid_alpha
                combined.append(doc)
        
        # Add BM25 results with adjusted scores
        for doc in bm25_results:
            doc_id = self._get_doc_id(doc)
            if doc_id in seen:
                # Update score for existing docs
                for existing in combined:
                    if self._get_doc_id(existing) == doc_id:
                        existing['_score'] = existing.get('_score', 0) + (1 - self.hybrid_alpha)
                        break
            else:
                doc['_score'] = (1 - self.hybrid_alpha)
                combined.append(doc)
        
        # Sort by combined score
        combined.sort(key=lambda x: x.get('_score', 0), reverse=True)
        return combined[:k]

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Enhanced retrieval with hybrid search and duplicate removal.
        Maintains backward compatibility with the original interface.
        """
        # Run the async version synchronously
        return asyncio.run(self.aretrieve(query))

    async def aretrieve(self, query: str) -> List[Dict[str, Any]]:
        """
        Async version of retrieve with enhanced functionality.
        """
        start_time = time.time()
        
        try:
            # 1. Get query variations
            query_variations = self._expand_query(query)
            logger.debug(f"Generated query variations: {query_variations}")
            
            # 2. Route queries to appropriate domains
            domains = set()
            for q in query_variations[:1]:  # Only use the first variation for routing
                domains.update(self.router.route(q))
            
            logger.info(f"Routing queries to domains: {domains}")

            # 3. Gather results from all domains using hybrid retrieval
            all_hits = []
            for domain in domains:
                try:
                    hits = self._hybrid_retrieval(
                        query=query_variations[0],
                        domain=domain,
                        k=self.coarse_k * 2  # Get more results for reranking
                    )
                    logger.debug(f"Retrieved {len(hits)} hits from domain '{domain}'")
                    all_hits.extend(hits)
                except Exception as e:
                    logger.error(f"Error querying domain {domain}: {str(e)}", exc_info=True)
            
            # 4. Remove duplicates and low-quality chunks
            unique_hits = self._remove_duplicate_chunks(all_hits)
            logger.info(f"Retrieved {len(unique_hits)} unique chunks from {len(all_hits)} total hits")
            
            if not unique_hits:
                logger.warning(f"No documents found for query: {query}")
                return []
            
            # 5. Rerank the results
            reranked = self.reranker.rerank(
                query=query,
                docs=unique_hits,
            )
            
            # Apply top-k after reranking
            if len(reranked) > 5:  # Limit to top 5 after reranking
                reranked = reranked[:5]
            
            # Log metrics if monitoring is enabled
            if self.monitor:
                self.monitor.log_retrieval(RetrievalMetrics(
                    query=query,
                    domain=",".join(domains) if domains else "unknown",
                    retrieval_time_ms=(time.time() - start_time) * 1000,
                    num_results=len(reranked),
                    top_score=reranked[0].get('_rerank_score', 0) if reranked else 0,
                    query_length=len(query)
                ))
            
            # Track last retrieval for observability
            self.last_query = query
            self.last_retrieved_docs = reranked
            self.last_scores = [doc.get('_rerank_score', 0) for doc in reranked]
            self.last_retrieval_time = time.time() - start_time
            self.last_retrieval_metadata = {
                'index_used': ",".join(domains) if domains else "unknown",
                'num_retrieved': len(reranked),
                'avg_score': sum(self.last_scores) / len(self.last_scores) if self.last_scores else 0,
                'query_length': len(query.split())
            }
            
            logger.info(
                f"Retrieved {len(reranked)} documents for query '{query[:50]}...' "
                f"(avg score: {self.last_retrieval_metadata['avg_score']:.3f}, "
                f"time: {self.last_retrieval_time:.3f}s)"
            )
            
            return reranked
            
        except Exception as e:
            logger.error(f"Error in retrieve: {str(e)}", exc_info=True)
            self.last_retrieval_metadata = {
                'error': str(e),
                'query': query
            }
            return []

    def get_retrieval_metrics(self) -> Dict[str, Any]:
        """
        Get metrics about the last retrieval operation.
        
        Returns:
            Dictionary containing retrieval metrics
        """
        return {
            'query': self.last_query,
            'num_documents': len(self.last_retrieved_docs),
            'scores': self.last_scores,
            'retrieval_time': self.last_retrieval_time,
            'documents': [
                {
                    'page_content': doc.get('page_content', '')[:200] + ('...' if len(doc.get('page_content', '')) > 200 else ''),
                    'metadata': doc.get('metadata', {}),
                    'score': score
                }
                for doc, score in zip(self.last_retrieved_docs, self.last_scores)
            ],
            **self.last_retrieval_metadata
        }