"""
Metrics collection and tracking for RAG system observability.
"""
from typing import Dict, List, Optional, Any
from datetime import datetime
import time
import json
import pandas as pd
from dataclasses import dataclass, asdict, field
from enum import Enum
import tiktoken

class QueryStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"

@dataclass
class QueryMetrics:
    """Class to store metrics for a single query."""
    query_id: str
    timestamp: float
    query: str
    response: str
    status: QueryStatus
    latency_seconds: float
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    error_message: Optional[str] = None
    retrieved_chunks: List[Dict[str, Any]] = field(default_factory=list)
    chunk_scores: List[float] = field(default_factory=list)
    chunk_sources: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for serialization."""
        data = asdict(self)
        data['status'] = self.status.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QueryMetrics':
        """Create QueryMetrics from dictionary."""
        data['status'] = QueryStatus(data['status'])
        return cls(**data)

class MetricsCollector:
    """Collects and manages metrics for the RAG system."""
    
    def __init__(self):
        self.queries: List[QueryMetrics] = []
        self.encoder = tiktoken.get_encoding("cl100k_base")
        
    def start_query(self, query: str, model: str, **metadata) -> str:
        """Start tracking a new query."""
        query_id = f"query_{int(time.time() * 1000)}"
        metrics = QueryMetrics(
            query_id=query_id,
            timestamp=time.time(),
            query=query,
            response="",
            status=QueryStatus.SUCCESS,
            latency_seconds=0,
            model=model,
            input_tokens=len(self.encoder.encode(query)),
            metadata=metadata
        )
        self.queries.append(metrics)
        return query_id
    
    def complete_query(
        self,
        query_id: str,
        response: str,
        status: QueryStatus = QueryStatus.SUCCESS,
        error_message: Optional[str] = None,
        retrieved_chunks: Optional[List[Dict[str, Any]]] = None,
        chunk_scores: Optional[List[float]] = None,
        **metadata
    ) -> None:
        """Complete a query and record metrics."""
        for metrics in reversed(self.queries):
            if metrics.query_id == query_id:
                metrics.response = response
                metrics.status = status
                metrics.error_message = error_message
                metrics.latency_seconds = time.time() - metrics.timestamp
                metrics.output_tokens = len(self.encoder.encode(response))
                metrics.retrieved_chunks = retrieved_chunks or []
                metrics.chunk_scores = chunk_scores or []
                metrics.chunk_sources = [c.get('metadata', {}).get('source', 'unknown') 
                                       for c in (retrieved_chunks or [])]
                metrics.metadata.update(metadata)
                break
    
    def get_metrics_dataframe(self) -> pd.DataFrame:
        """Get all metrics as a pandas DataFrame."""
        if not self.queries:
            return pd.DataFrame()
            
        data = [{
            'timestamp': datetime.fromtimestamp(m.timestamp).strftime('%Y-%m-%d %H:%M:%S'),
            'query': m.query,
            'response': m.response,
            'status': m.status.value,
            'latency_seconds': m.latency_seconds,
            'model': m.model,
            'input_tokens': m.input_tokens,
            'output_tokens': m.output_tokens,
            'total_tokens': m.input_tokens + m.output_tokens,
            'num_chunks': len(m.retrieved_chunks),
            'chunk_sources': ', '.join(set(m.chunk_sources)) if m.chunk_sources else None,
            'error': m.error_message or ''
        } for m in self.queries]
        
        return pd.DataFrame(data)
    
    def get_summary_stats(self) -> Dict[str, Any]:
        """Get summary statistics for all queries."""
        if not self.queries:
            return {}
            
        df = self.get_metrics_dataframe()
        if df.empty:
            return {}
            
        success_queries = df[df['status'] == QueryStatus.SUCCESS.value]
        
        return {
            'total_queries': len(df),
            'success_rate': len(success_queries) / len(df) if len(df) > 0 else 0,
            'avg_latency': df['latency_seconds'].mean(),
            'avg_input_tokens': df['input_tokens'].mean(),
            'avg_output_tokens': df['output_tokens'].mean(),
            'total_tokens_used': df['input_tokens'].sum() + df['output_tokens'].sum(),
            'most_common_sources': df['chunk_sources'].value_counts().head(5).to_dict()
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save metrics to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump([m.to_dict() for m in self.queries], f, indent=2)
    
    def load_from_file(self, filepath: str) -> None:
        """Load metrics from a JSON file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.queries = [QueryMetrics.from_dict(m) for m in data]
        except (FileNotFoundError, json.JSONDecodeError):
            self.queries = []

# Global metrics collector instance
metrics_collector = MetricsCollector()
