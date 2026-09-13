"""Repository paths, resolved once and shared.

Both the API and the ingestion service need to find ``data/synthetic``. Keeping
the resolution here means there is a single definition of "where the repository
is", overridable with ``AGENTPATH_REPO_ROOT`` for containers and tests.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Repository root.

    ``apps/api/agentpath/paths.py`` -> three levels up.
    """
    override = os.environ.get("AGENTPATH_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def data_dir() -> Path:
    return repo_root() / "data" / "synthetic"


def procedures_dir() -> Path:
    return data_dir() / "procedures"


def build_dir() -> Path:
    return data_dir() / "build"


def lexicon_path() -> Path:
    return data_dir() / "lexicon.yaml"


def evaluation_dir() -> Path:
    return repo_root() / "evaluation"


def reports_dir() -> Path:
    return evaluation_dir() / "reports"
