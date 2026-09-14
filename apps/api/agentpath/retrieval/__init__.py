"""Retrieval: the port, its RAGFlow implementation, and passage aggregation."""

from agentpath.retrieval.aggregate import aggregate_passages
from agentpath.retrieval.ports import RetrievalError, RetrievalUnavailable, Retriever
from agentpath.retrieval.ragflow_client import RAGFlowClient, RAGFlowError, RAGFlowUnavailable
from agentpath.retrieval.ragflow_retriever import RAGFlowRetriever

__all__ = [
    "RAGFlowClient",
    "RAGFlowError",
    "RAGFlowRetriever",
    "RAGFlowUnavailable",
    "RetrievalError",
    "RetrievalUnavailable",
    "Retriever",
    "aggregate_passages",
]
