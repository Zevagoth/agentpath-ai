"""Configuration, loaded from the environment and ``.env.local``.

No secret has a usable default. ``ragflow_api_key`` is a ``SecretStr`` so it
cannot leak into a log line or an exception repr by accident.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentpath.paths import lexicon_path, repo_root


class Settings(BaseSettings):
    """Runtime configuration for the AgentPath AI API."""

    model_config = SettingsConfigDict(
        env_file=(repo_root() / ".env.local", repo_root() / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # -- service ---------------------------------------------------------- #
    api_host: str = Field(default="127.0.0.1", validation_alias="AGENTPATH_API_HOST")
    api_port: int = Field(default=8000, validation_alias="AGENTPATH_API_PORT")
    log_level: str = Field(default="INFO", validation_alias="AGENTPATH_LOG_LEVEL")
    log_query_text: bool = Field(default=False, validation_alias="AGENTPATH_LOG_QUERY_TEXT")

    # -- RAGFlow ---------------------------------------------------------- #
    ragflow_base_url: str = Field(default="http://127.0.0.1:9380", validation_alias="RAGFLOW_BASE_URL")
    ragflow_api_key: SecretStr = Field(default=SecretStr(""), validation_alias="RAGFLOW_API_KEY")
    ragflow_dataset_name: str = Field(
        default="agentpath-northwind-procedures", validation_alias="RAGFLOW_DATASET_NAME"
    )
    # RAGFlow model references are "{model}@{instance}@{provider}".
    ragflow_embedding_model: str = Field(
        default="BAAI/bge-m3@agentpath-embedder@HuggingFace",
        validation_alias="RAGFLOW_EMBEDDING_MODEL",
    )
    ragflow_rerank_model: str = Field(
        default="BAAI/bge-reranker-v2-m3@agentpath-reranker@HuggingFace",
        validation_alias="RAGFLOW_RERANK_MODEL",
    )
    ragflow_timeout_seconds: float = Field(default=30.0, validation_alias="RAGFLOW_TIMEOUT_SECONDS")

    # Hybrid weighting. RAGFlow blends term similarity and vector similarity as
    # ``(1 - w) * term + w * vector``, so this single knob moves the balance
    # between exact policy terms and meaning.
    vector_similarity_weight: float = Field(
        default=0.6, ge=0.0, le=1.0, validation_alias="RAGFLOW_VECTOR_SIMILARITY_WEIGHT"
    )
    # Engine-side floor: chunks below this are not returned at all. Kept low so
    # the abstention decision is made by our own graded thresholds below, where it
    # can be explained to the agent, rather than silently by the engine.
    similarity_threshold: float = Field(
        default=0.15, ge=0.0, le=1.0, validation_alias="RAGFLOW_SIMILARITY_THRESHOLD"
    )
    rerank_candidates: int = Field(default=32, ge=1, validation_alias="RAGFLOW_RERANK_CANDIDATES")
    page_size: int = Field(default=12, ge=1, validation_alias="RAGFLOW_PAGE_SIZE")

    # Feature flag for cross-encoder reranking. Off by default because the only
    # working runtime on the development host is CPU, where reranking measured
    # 20-22 s per query against 0.2 s for hybrid retrieval alone. The model stays
    # registered and the code path stays exercised by the benchmark: reranking
    # earns its cost as the corpus grows, and this flag is how it gets turned on.
    enable_rerank: bool = Field(default=False, validation_alias="AGENTPATH_ENABLE_RERANK")

    # RAGFlow's ``keyword`` flag does NOT enable BM25 (that is always on and
    # weighted by ``vector_similarity_weight``). It asks an LLM to extract
    # keywords and appends them to the question, which costs a chat-model round
    # trip per search. AgentPath does that expansion deterministically from
    # data/synthetic/lexicon.yaml instead, so this stays off.
    ragflow_llm_keyword_expansion: bool = Field(
        default=False, validation_alias="RAGFLOW_LLM_KEYWORD_EXPANSION"
    )
    # Same reasoning: RAGFlow's ``cross_languages`` translates the query with an
    # LLM. Our bilingual lexicon covers the Arabic/English pivot without one.
    ragflow_cross_languages: list[str] = Field(default_factory=list, validation_alias="RAGFLOW_CROSS_LANGUAGES")

    # -- model sidecars ---------------------------------------------------- #
    # Both run inside RAGFlow's Compose network and are published on loopback for
    # verification only. RAGFlow addresses them by service name.
    embedder_url: str = Field(default="http://127.0.0.1:7998", validation_alias="EMBEDDER_URL")
    embedder_model: str = Field(default="BAAI/bge-m3", validation_alias="EMBEDDER_MODEL")
    reranker_url: str = Field(default="http://127.0.0.1:7997", validation_alias="RERANKER_URL")
    reranker_model: str = Field(default="BAAI/bge-reranker-v2-m3", validation_alias="RERANKER_MODEL")

    # -- Ollama (Stage 2 answer generation) -------------------------------- #
    # Called from this process over loopback, so it needs no container network
    # access and stays bound to 127.0.0.1. Unused during the search slice.
    ollama_host_url: str = Field(default="http://127.0.0.1:11434", validation_alias="OLLAMA_HOST_URL")
    ollama_chat_model: str = Field(default="qwen2.5:7b-instruct", validation_alias="OLLAMA_CHAT_MODEL")

    # -- search behaviour -------------------------------------------------- #
    top_procedures: int = Field(default=3, ge=1, le=10, validation_alias="AGENTPATH_TOP_PROCEDURES")
    lexicon_file: Path = Field(default_factory=lexicon_path, validation_alias="AGENTPATH_LEXICON_PATH")

    # Evidence thresholds.
    #
    # Derived from the observed score distribution on the 30-query benchmark set
    # (see EVALUATION.md for the numbers and the caveats):
    #   - lowest-scoring supported query   0.526
    #   - highest-scoring unsupported query 0.493
    # so 0.51 sits in the 0.034-wide gap between "answerable" and "not in the
    # knowledge base". That gap is thin, and it was measured on the same set the
    # report quotes, which is a real methodological weakness: it must be re-derived
    # on a held-out split once the corpus grows past five procedures.
    #
    # Attempts that did *not* separate the two populations, recorded so they are
    # not retried blindly: lexical (BM25) score - a vague but answerable query
    # scored 0.085 against 0.083 for an unanswerable one; and lexicon concept
    # matches - genuine vague queries and reference-number lookups match no
    # concept either.
    evidence_strong_score: float = Field(default=0.60, validation_alias="AGENTPATH_EVIDENCE_STRONG_SCORE")
    evidence_strong_margin: float = Field(default=0.05, validation_alias="AGENTPATH_EVIDENCE_STRONG_MARGIN")
    evidence_partial_score: float = Field(default=0.51, validation_alias="AGENTPATH_EVIDENCE_PARTIAL_SCORE")

    @field_validator("evidence_strong_score")
    @classmethod
    def _tiers_are_ordered(cls, value: float, info) -> float:
        """A strong threshold below the partial threshold would be meaningless."""
        partial = info.data.get("evidence_partial_score")
        if partial is not None and value < partial:
            raise ValueError(
                f"evidence_strong_score ({value}) must be at least evidence_partial_score ({partial})"
            )
        return value

    @field_validator("ragflow_base_url", "ollama_host_url", "reranker_url", "embedder_url")
    @classmethod
    def _strip_trailing_slash(cls, value: str) -> str:
        return value.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def _valid_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @property
    def has_ragflow_credentials(self) -> bool:
        key = self.ragflow_api_key.get_secret_value()
        return bool(key) and not key.startswith("ragflow-REPLACE")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached settings instance.

    Cached because the FastAPI dependency graph and the CLI both ask for it, and
    reading ``.env.local`` repeatedly serves no purpose. Tests clear the cache
    with ``get_settings.cache_clear()``.
    """
    return Settings()
