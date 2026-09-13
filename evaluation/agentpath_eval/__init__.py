"""Retrieval benchmark for AgentPath AI.

Compares two retrieval modes over the same query set:

* ``baseline`` - exact/lexical matching only. Benchmark reference, never a
  production retrieval path.
* ``hybrid``   - AgentPath AI: query understanding + RAGFlow hybrid retrieval +
  cross-encoder reranking.
"""

__version__ = "0.1.0"
