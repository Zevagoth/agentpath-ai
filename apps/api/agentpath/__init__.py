"""AgentPath AI retrieval orchestration.

Layering (outermost first):

* ``agentpath.api``        - FastAPI transport: schemas, routes, dependencies.
* ``agentpath.query``      - query understanding: language, normalization, lexicon.
* ``agentpath.retrieval``  - the ``Retriever`` port and its RAGFlow implementation.
* ``agentpath.evidence``   - evidence grading and abstention rules.
* ``agentpath.domain``     - typed models shared by every layer.

Nothing below ``agentpath.api`` imports FastAPI, and nothing outside
``agentpath.retrieval.ragflow_*`` knows that RAGFlow exists. That is what makes
the engine replaceable.
"""

__version__ = "0.1.0"
