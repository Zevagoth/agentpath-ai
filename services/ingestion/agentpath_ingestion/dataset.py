"""Loading and validating the synthetic procedure dataset.

The dataset lives in ``data/synthetic/procedures/*.yaml``. This module is the
only place that reads those files, so schema failures surface here with the file
name attached instead of as a confusing pydantic traceback.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import ValidationError

from agentpath_ingestion.schema import Procedure

#: Brand names that must never appear in the synthetic dataset. The guard is
#: deliberately broad: this proof of concept must not read as if it reproduced
#: any real company's internal material.
FORBIDDEN_BRAND_TOKENS = (
    "trip.com",
    "trip com",
    "ctrip",
    "skyscanner",
    "booking.com",
    "expedia",
    "agoda",
    "airbnb",
    "hotels.com",
    "makemytrip",
    "almosafer",
    "wego",
)


class DatasetError(RuntimeError):
    """Raised when the dataset on disk is not usable."""


@dataclass(frozen=True)
class LoadedProcedure:
    """A validated procedure together with its origin on disk."""

    path: Path
    procedure: Procedure

    @property
    def procedure_id(self) -> str:
        return self.procedure.procedure_id


def repo_root() -> Path:
    """Locate the repository root from this file's position.

    ``services/ingestion/agentpath_ingestion/dataset.py`` -> three levels up.
    Overridable with ``AGENTPATH_REPO_ROOT`` so tests and containers can point
    somewhere else.
    """
    override = os.environ.get("AGENTPATH_REPO_ROOT")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[3]


def procedures_dir() -> Path:
    return repo_root() / "data" / "synthetic" / "procedures"


def build_dir() -> Path:
    return repo_root() / "data" / "synthetic" / "build"


def lexicon_path() -> Path:
    return repo_root() / "data" / "synthetic" / "lexicon.yaml"


def load_procedure_file(path: Path) -> Procedure:
    """Parse and validate a single procedure YAML file."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetError(f"{path.name}: invalid YAML: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError(f"{path.name}: expected a YAML mapping at the top level")
    try:
        return Procedure.model_validate(raw)
    except ValidationError as exc:
        raise DatasetError(f"{path.name}: schema validation failed:\n{exc}") from exc


def load_procedures(directory: Path | None = None) -> list[LoadedProcedure]:
    """Load every procedure, sorted by ID, with cross-file invariants enforced."""
    directory = directory or procedures_dir()
    if not directory.is_dir():
        raise DatasetError(f"procedure directory not found: {directory}")

    paths = sorted(directory.glob("*.yaml"))
    if not paths:
        raise DatasetError(f"no procedure files found in {directory}")

    loaded = [LoadedProcedure(path=path, procedure=load_procedure_file(path)) for path in paths]

    _assert_unique_ids(loaded)
    _assert_filenames_match_ids(loaded)
    _assert_related_procedures_exist(loaded)
    return sorted(loaded, key=lambda item: item.procedure_id)


def _assert_unique_ids(loaded: list[LoadedProcedure]) -> None:
    seen: dict[str, Path] = {}
    for item in loaded:
        if item.procedure_id in seen:
            raise DatasetError(
                f"duplicate procedure id {item.procedure_id} in {item.path.name} "
                f"and {seen[item.procedure_id].name}"
            )
        seen[item.procedure_id] = item.path


def _assert_filenames_match_ids(loaded: list[LoadedProcedure]) -> None:
    for item in loaded:
        expected = f"{item.procedure_id}.yaml"
        if item.path.name != expected:
            raise DatasetError(f"{item.path.name}: file name should be {expected}")


def _assert_related_procedures_exist(loaded: list[LoadedProcedure]) -> None:
    known = {item.procedure_id for item in loaded}
    for item in loaded:
        for related in item.procedure.related_procedures:
            if related not in known:
                raise DatasetError(
                    f"{item.path.name}: related procedure {related} does not exist in the dataset"
                )


def find_forbidden_brand_tokens(text: str) -> list[str]:
    """Return any real-company brand tokens found in *text* (case-insensitive)."""
    haystack = text.casefold()
    return [token for token in FORBIDDEN_BRAND_TOKENS if token in haystack]
