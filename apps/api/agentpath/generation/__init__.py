"""Local answer generation.

Everything that talks to a language model lives here. Retrieval stays in
``agentpath.retrieval``; this package turns evidence that has already been graded
into words, and it never decides whether there is enough evidence to answer -
:mod:`agentpath.evidence` owns that.

The layers, in the order one request moves through them:

* :mod:`~agentpath.generation.evidence_assembly` - map retrieved chunks to step
  ids, then rebuild each step whole from :mod:`agentpath.procedures`. No
  retrieved text survives this boundary.
* :mod:`~agentpath.generation.prompt` - the prompt, and the per-request schema
  whose ``step_id`` is an ``enum`` over the citations the evidence offers.
* :mod:`~agentpath.generation.ollama_client` - the wire protocol.
* :mod:`~agentpath.generation.answer_service` - the sequence, the validation, and
  the states an agent is allowed to see.
"""

from __future__ import annotations

from agentpath.generation.answer_service import (
    EMPTY_QUERY_MESSAGE,
    GENERATION_FAILED_MESSAGE,
    UNGROUNDED_MESSAGE,
    AnswerResult,
    AnswerService,
    AnswerStatus,
    AnswerStep,
)
from agentpath.generation.evidence_assembly import (
    MAX_EVIDENCE_BLOCKS,
    AssembledEvidence,
    ChunkMapping,
    EvidenceBlock,
    assemble_evidence,
    clarifying_questions,
    evidence_fields,
    map_chunk,
)
from agentpath.generation.ollama_client import (
    ChatCompletion,
    OllamaClient,
    OllamaError,
    OllamaModelMissing,
    OllamaRuntimeOptions,
    OllamaUnavailable,
    StructuredCompletion,
    StructuredOutputError,
)
from agentpath.generation.prompt import (
    SYSTEM_PROMPT,
    DraftAnswer,
    DraftStep,
    build_answer_schema,
    build_prompt,
)

__all__ = [
    "EMPTY_QUERY_MESSAGE",
    "GENERATION_FAILED_MESSAGE",
    "MAX_EVIDENCE_BLOCKS",
    "SYSTEM_PROMPT",
    "UNGROUNDED_MESSAGE",
    "AnswerResult",
    "AnswerService",
    "AnswerStatus",
    "AnswerStep",
    "AssembledEvidence",
    "ChatCompletion",
    "ChunkMapping",
    "DraftAnswer",
    "DraftStep",
    "EvidenceBlock",
    "OllamaClient",
    "OllamaError",
    "OllamaModelMissing",
    "OllamaRuntimeOptions",
    "OllamaUnavailable",
    "StructuredCompletion",
    "StructuredOutputError",
    "assemble_evidence",
    "build_answer_schema",
    "build_prompt",
    "clarifying_questions",
    "evidence_fields",
    "map_chunk",
]
