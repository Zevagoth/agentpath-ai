"""``agentpath-ingest`` command line interface.

Subcommands available in this session:

* ``validate`` - load and validate every procedure without writing anything.
* ``build``    - render the procedures to ``data/synthetic/build/``.

RAGFlow-facing subcommands (``ingest``, ``status``, ``verify``) are added in
Task 5, once the stack is running. They are deliberately absent rather than
stubbed, so the CLI never pretends to do something it cannot.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agentpath_ingestion.dataset import DatasetError, build_dir, load_procedures, procedures_dir
from agentpath_ingestion.render import render, write_build


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        loaded = load_procedures(args.source)
    except DatasetError as exc:
        print(f"dataset invalid: {exc}", file=sys.stderr)
        return 1

    print(f"validated {len(loaded)} procedure(s) in {args.source or procedures_dir()}")
    for item in loaded:
        procedure = item.procedure
        print(
            f"  {procedure.procedure_id}  {procedure.product_line.value:<14} "
            f"v{procedure.version}  {len(procedure.steps)} steps  "
            f"{len(procedure.all_synonyms)} synonyms  "
            f"{len(procedure.screenshot_steps)} screenshot step(s)"
        )
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    try:
        loaded = load_procedures(args.source)
    except DatasetError as exc:
        print(f"dataset invalid: {exc}", file=sys.stderr)
        return 1

    try:
        rendered = [render(item) for item in loaded]
    except ValueError as exc:
        print(f"render failed: {exc}", file=sys.stderr)
        return 1

    target = args.target or build_dir()
    manifest = write_build(rendered, target)

    total_chars = sum(len(item.markdown) for item in rendered)
    print(f"rendered {len(rendered)} document(s) to {target}")
    for item in rendered:
        print(f"  {item.filename:<18} {len(item.markdown):>6} chars  sha256={item.sha256[:12]}")
    print(f"manifest: {manifest}")
    print(f"total: {total_chars} characters")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentpath-ingest",
        description="Render and ingest the synthetic AgentPath AI procedure knowledge base.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate the procedure YAML files")
    validate.add_argument("--source", type=Path, default=None, help="procedure directory")
    validate.set_defaults(func=_cmd_validate)

    build = subparsers.add_parser("build", help="render procedures to Markdown")
    build.add_argument("--source", type=Path, default=None, help="procedure directory")
    build.add_argument("--target", type=Path, default=None, help="output directory")
    build.set_defaults(func=_cmd_build)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
