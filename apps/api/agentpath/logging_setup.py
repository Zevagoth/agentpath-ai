"""Structured logging.

JSON in, JSON out: every log line is machine-readable so latency per stage can be
extracted from logs without parsing prose. Query text is excluded unless
``AGENTPATH_LOG_QUERY_TEXT`` is on, because a customer-service case description
can carry personal detail even when the knowledge base is synthetic.
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(level: str = "INFO", *, json_output: bool = True) -> None:
    """Configure structlog and the standard library to agree on one format."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        _drop_none_values,
    ]

    renderer: structlog.typing.Processor = (
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(numeric_level),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=numeric_level,
        force=True,
    )
    # uvicorn duplicates access logs that our middleware already records.
    logging.getLogger("uvicorn.access").handlers.clear()


def _drop_none_values(_logger: object, _name: str, event_dict: dict) -> dict:
    """Remove ``None`` values.

    Call sites pass ``query=... if settings.log_query_text else None``, and a null
    field in every line is noise.
    """
    return {key: value for key, value in event_dict.items() if value is not None}
