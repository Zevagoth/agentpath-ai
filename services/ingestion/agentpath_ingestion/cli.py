"""``agentpath-ingest`` command line interface.

* ``validate`` - load and validate every procedure without writing anything.
* ``build``    - render the procedures to ``data/synthetic/build/``.
* ``ingest``   - create the RAGFlow dataset, upload, set metadata, parse, wait.
* ``verify``   - report what RAGFlow actually holds, and run one smoke search.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from agentpath.config import get_settings
from agentpath.retrieval.ragflow_client import RAGFlowClient, RAGFlowError
from agentpath_ingestion.dataset import DatasetError, build_dir, load_procedures, procedures_dir
from agentpath_ingestion.ingest import IngestionError, IngestionReport, IngestionService, smoke_search
from agentpath_ingestion.render import render, write_build
from agentpath_ingestion.visual_chunks import VisualChunkError, VisualChunkReport
from agentpath_ingestion.visual_steps import VisualStepError


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


# --------------------------------------------------------------------------- #
# RAGFlow-facing commands
# --------------------------------------------------------------------------- #


def _require_credentials() -> None:
    settings = get_settings()
    if not settings.has_ragflow_credentials:
        raise SystemExit(
            "RAGFLOW_API_KEY is not set in .env.local.\nRun: make models   (registers the models and stores an API key)"
        )


def _print_report(report: IngestionReport) -> None:
    print(f"dataset : {report.dataset_name}  (id {report.dataset_id})")
    print(f"embedding: {report.embedding_model}")
    print()
    print(f"  {'procedure':<12} {'action':<10} {'status':<8} {'chunks':>6} {'tokens':>8}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 8} {'-' * 6} {'-' * 8}")
    for outcome in sorted(report.outcomes, key=lambda item: item.procedure_id):
        print(
            f"  {outcome.procedure_id:<12} {outcome.action:<10} {outcome.status:<8} "
            f"{outcome.chunk_count:>6} {outcome.token_count:>8}"
        )
    print()
    print(
        f"total: {len(report.outcomes)} document(s), {report.total_chunks} chunk(s)"
        + (f", {report.elapsed_seconds:.1f}s" if report.elapsed_seconds else "")
    )
    if report.visual is not None:
        _print_visual(report.visual)


def _print_visual(visual: VisualChunkReport) -> None:
    """Print the screenshot-chunk state, including everything that is not there.

    A screenshot that never arrived has to look different from a step that has no
    screenshot, so gaps and blocks are printed even when there is nothing to
    celebrate.
    """
    print()
    print("visual chunks (one screenshot per step)")
    if not visual.outcomes:
        print("  none attached")
    else:
        print(f"  {'citation':<16} {'action':<15} {'image':<6} {'chunk id':<18} screenshot")
        print(f"  {'-' * 16} {'-' * 15} {'-' * 6} {'-' * 18} {'-' * 10}")
        for item in sorted(visual.outcomes, key=lambda o: (o.procedure_id, o.step_order)):
            image = "yes" if item.has_image else ("BLOCKED" if item.image_blocked else "no")
            print(f"  {item.citation:<16} {item.action:<15} {image:<6} {item.chunk_id or '-':<18} {item.screenshot_id}")
        summary = ", ".join(f"{action} {count}" for action, count in sorted(visual.actions.items()))
        print(f"  {len(visual.with_image)}/{len(visual.outcomes)} carry an image  ({summary})")

    if visual.gaps:
        print()
        print(f"  screenshots declared but not usable yet: {len(visual.gaps)}")
        for gap in visual.gaps:
            print(f"    {gap.procedure_id}#{gap.step_id} ({gap.screenshot_id}): {gap.problems[0]}")
        print("    these steps are ingested without a screenshot; run: make screenshots")


async def _run_ingest(force: bool) -> int:
    settings = get_settings()
    async with RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=settings.ragflow_timeout_seconds,
    ) as client:
        service = IngestionService(
            client,
            dataset_name=settings.ragflow_dataset_name,
            embedding_model=settings.ragflow_embedding_model,
        )
        report = await service.ingest(force=force)

    _print_report(report)
    if not report.all_parsed:
        unparsed = [o.procedure_id for o in report.outcomes if o.chunk_count == 0]
        print(f"\nnot searchable yet: {', '.join(unparsed)}", file=sys.stderr)
        return 1
    if _report_visual_problems(report.visual) != 0:
        return 1
    print("\nevery procedure is parsed and searchable")
    if report.visual and report.visual.outcomes:
        print(f"{len(report.visual.with_image)} step screenshot(s) attached as image-bearing chunks")
    return 0


def _report_visual_problems(visual: VisualChunkReport | None) -> int:
    """Fail loudly on a screenshot-less visual chunk. Never let it read as success.

    A blocked image is the one outcome that re-running cannot fix, so it names the
    remedy instead of just the symptom.
    """
    if visual is None or visual.ok:
        return 0
    print(file=sys.stderr)
    for item in visual.blocked:
        print(
            f"image NOT stored for {item.citation} (chunk {item.chunk_id}): the chunk text is "
            "there, the screenshot is not",
            file=sys.stderr,
        )
    for item in visual.missing:
        print(f"{item.action}: {item.citation} ({item.screenshot_id})", file=sys.stderr)
    for citation in visual.orphaned:
        print(f"orphaned: {citation} has no ingested document to attach to", file=sys.stderr)
    if visual.remedy:
        print(f"\nremedy: {visual.remedy}", file=sys.stderr)
    return 1


async def _run_verify() -> int:
    settings = get_settings()
    async with RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=settings.ragflow_timeout_seconds,
    ) as client:
        service = IngestionService(
            client,
            dataset_name=settings.ragflow_dataset_name,
            embedding_model=settings.ragflow_embedding_model,
        )
        report = await service.verify()
        _print_report(report)

        problems = [o for o in report.outcomes if o.action in {"missing", "stale", "unexpected"}]
        chunks = await smoke_search(client, report.dataset_id, "hotel cannot accommodate the guest")

    print()
    print(f"smoke search returned {len(chunks)} chunk(s)")
    for chunk in chunks[:3]:
        metadata = chunk.get("document_metadata") or {}
        procedure = metadata.get("procedure_id") or chunk.get("document_keyword", "?")
        print(f"  {chunk.get('similarity', 0):.3f}  {procedure}")

    if problems:
        print(file=sys.stderr)
        for outcome in problems:
            print(f"{outcome.action}: {outcome.procedure_id} ({outcome.filename})", file=sys.stderr)
        return 1
    if not chunks:
        print("\nsmoke search returned nothing; the index is not answering", file=sys.stderr)
        return 1
    return _report_visual_problems(report.visual)


def _cmd_ingest(args: argparse.Namespace) -> int:
    _require_credentials()
    try:
        return asyncio.run(_run_ingest(force=args.force))
    except (IngestionError, DatasetError, VisualChunkError, VisualStepError) as exc:
        print(f"ingestion failed: {exc}", file=sys.stderr)
        return 1
    except RAGFlowError as exc:
        print(f"RAGFlow error: {exc}", file=sys.stderr)
        return 1


def _cmd_verify(args: argparse.Namespace) -> int:
    _require_credentials()
    try:
        return asyncio.run(_run_verify())
    except (IngestionError, DatasetError, VisualChunkError, VisualStepError) as exc:
        print(f"verification failed: {exc}", file=sys.stderr)
        return 1
    except RAGFlowError as exc:
        print(f"RAGFlow error: {exc}", file=sys.stderr)
        return 1


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

    ingest = subparsers.add_parser("ingest", help="ingest the procedures into RAGFlow")
    ingest.add_argument(
        "--force",
        action="store_true",
        help="re-upload every document even when its content hash is unchanged",
    )
    ingest.set_defaults(func=_cmd_ingest)

    verify = subparsers.add_parser("verify", help="report RAGFlow's contents and run a smoke search")
    verify.set_defaults(func=_cmd_verify)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
