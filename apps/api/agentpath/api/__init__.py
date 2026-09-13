"""HTTP transport layer. Nothing below this package imports FastAPI."""

from agentpath.api.app import create_app

__all__ = ["create_app"]
