"""Fetch and inspect an external corpus.

Two commands, matching the two halves of the adapter protocol:

    uv run python scripts/corpus.py fetch   --corpus mramg-manual
    uv run python scripts/corpus.py inspect --corpus mramg-manual

``fetch`` is the only one that touches the network, and it is safe to re-run: a
second run downloads nothing and says so, which is how idempotence is shown rather
than claimed. ``inspect`` reads what is on disk and prints the counts an
evaluation will be based on, plus anything about the corpus that would invalidate
those numbers.

Registered through :data:`ADAPTERS` rather than imported directly, so adding a
corpus is a line here and a module beside ``mramg.py``.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from agentpath_ingestion.corpora import CorpusAdapter
from agentpath_ingestion.corpora.base import IMAGE_PLACEHOLDER
from agentpath_ingestion.corpora.export import export_bundle
from agentpath_ingestion.corpora.mramg import CORPUS_ID as MRAMG_ID
from agentpath_ingestion.corpora.mramg import MramgManualAdapter
from agentpath_ingestion.corpora.mramg import default_root as mramg_root
from agentpath_ingestion.dataset import repo_root

#: Corpus id to a factory. The synthetic procedures are deliberately absent: they
#: have their own pipeline (`make dataset && make ingest`) and are not fetched.
ADAPTERS: dict[str, Callable[[Path], CorpusAdapter]] = {
    MRAMG_ID: lambda root: MramgManualAdapter(mramg_root(root)),
}

#: Where each corpus's exported bundle goes, and where its image bytes live. Kept
#: beside the adapter registry so adding a corpus is one place.
BUILD_DIRS: dict[str, Callable[[Path], Path]] = {
    MRAMG_ID: lambda root: mramg_root(root) / "build",
}

IMAGE_DIRS: dict[str, Callable[[Path], Path]] = {
    MRAMG_ID: lambda root: mramg_root(root) / "raw" / "images",
}


def human(size: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(size) < 1024 or unit == "GiB":
            return f"{size:,.1f} {unit}" if unit != "B" else f"{size:,} B"
        size /= 1024.0  # type: ignore[assignment]
    return f"{size:,.1f} GiB"


def command_fetch(adapter: CorpusAdapter, args: argparse.Namespace) -> int:
    report = adapter.fetch(force=args.force)

    print(f"corpus   : {report.corpus_id}")
    print(f"downloaded: {human(report.downloaded_bytes)} over the network on this run")
    print(f"reused    : {human(report.reused_bytes)} already on disk")
    print()
    print(f"{'state':<8} {'artefact':<44} {'bytes':>14}  sha256")
    for item in report.artefacts:
        state = "reused" if item.reused else "fetched"
        print(f"{state:<8} {item.name:<44} {item.byte_size:>14,}  {item.sha256[:16]}…")
    print()
    print(f"total on disk: {human(report.total_bytes)}")

    if report.problems:
        print("\nproblems:", file=sys.stderr)
        for problem in report.problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    if report.downloaded_bytes == 0:
        print("\nNothing was downloaded: every artefact was already present and verified.")
    return 0


def command_export(adapter: CorpusAdapter, _args: argparse.Namespace) -> int:
    """Write the normalised bundle the API reads.

    Separate from ``fetch`` because it needs no network, and separate from the API
    because the API may not import this package - see
    ``agentpath_ingestion.corpora.export``.
    """
    bundle = adapter.load()
    build_dir = BUILD_DIRS[bundle.corpus_id](repo_root())
    images_root = IMAGE_DIRS[bundle.corpus_id](repo_root())

    report = export_bundle(bundle, build_dir, images_root=images_root)

    print(f"corpus    : {report.corpus_id}")
    print(f"written to: {report.directory}")
    print(f"documents : {report.documents:,}")
    print(f"chunks    : {report.chunks:,}")
    print(f"questions : {report.questions:,}")
    print(f"images    : {report.images:,} referenced, bytes in {images_root}")
    print(f"bundle    : {human(report.bytes_written)}")
    print("\nThe API reads this directory at startup. Restart it to pick up changes.")
    return 0


def command_ingest(adapter: CorpusAdapter, args: argparse.Namespace) -> int:
    """Index the corpus into its own RAGFlow dataset. Needs the live stack."""
    import asyncio

    from agentpath.config import get_settings
    from agentpath.retrieval.ragflow_client import RAGFlowClient
    from agentpath_ingestion.corpora.ingest import (
        CorpusIngestionService,
        dataset_name_for,
        write_ingest_manifest,
    )

    bundle = adapter.load()
    settings = get_settings()

    async def run() -> int:
        client = RAGFlowClient(
            settings.ragflow_base_url,
            settings.ragflow_api_key.get_secret_value(),
            timeout=settings.ragflow_timeout_seconds,
        )
        try:
            service = CorpusIngestionService(client, embedding_model=settings.ragflow_embedding_model)
            if args.delete:
                removed = await service.delete_corpus(bundle.corpus_id)
                print(f"deleted {removed} documents from {dataset_name_for(bundle.corpus_id)}")
                return 0

            report = await service.ingest(bundle, force=args.force)
            counts = await service.chunk_counts(report)
            path = write_ingest_manifest(report, BUILD_DIRS[bundle.corpus_id](repo_root()), counts)

            print(f"corpus   : {report.corpus_id}")
            print(f"dataset  : {report.dataset_name}  ({report.dataset_id})")
            print(f"embedder : {report.embedding_model}")
            print(f"actions  : {report.actions}")
            print(f"removed  : {len(report.removed)}")
            print(f"seconds  : {report.seconds}")
            total = sum(counts.values())
            print(f"\nRAGFlow produced {total:,} chunks across {len(counts)} documents")
            if counts:
                ordered = sorted(counts.items(), key=lambda item: -item[1])
                print(f"  most chunks : {ordered[0][0]} -> {ordered[0][1]:,}")
                print(f"  fewest      : {ordered[-1][0]} -> {ordered[-1][1]:,}")
            print(f"manifest : {path}")

            if report.problems:
                print("\nproblems:", file=sys.stderr)
                for problem in report.problems:
                    print(f"  - {problem}", file=sys.stderr)
                return 1
            return 0
        finally:
            await client.aclose()

    return asyncio.run(run())


def command_inspect(adapter: CorpusAdapter, _args: argparse.Namespace) -> int:
    bundle = adapter.load()

    print(f"corpus     : {bundle.corpus_id}")
    print(f"title      : {bundle.title}")
    print(f"licence    : {bundle.licence}")
    print(f"attribution: {bundle.attribution}")
    print()
    for key, value in bundle.counts.items():
        print(f"  {key:<24} {value:>8,}")

    lengths = sorted(len(document.text) for document in bundle.documents)
    chunk_lengths = sorted(len(chunk.text) for chunk in bundle.chunks)
    print()
    print(
        f"  document chars           min {lengths[0]:,} / median {lengths[len(lengths) // 2]:,} / max {lengths[-1]:,}"
    )
    print(
        f"  chunk chars              min {chunk_lengths[0]:,} / median "
        f"{chunk_lengths[len(chunk_lengths) // 2]:,} / max {chunk_lengths[-1]:,}"
    )
    with_images = sum(1 for chunk in bundle.chunks if chunk.image_ids)
    print(f"  chunks carrying an image {with_images:,} of {len(bundle.chunks):,}")

    # The two facts that decide how image retrieval has to work.
    print()
    if bundle.counts["images_with_caption"] == 0:
        print("  NOTE: no image in this corpus has a caption, so image ranking uses the")
        print("        text window around each placeholder. See docs/RETRIEVAL_MODES.md.")
    missing_url = bundle.counts["images"] - bundle.counts["images_with_source_url"]
    if missing_url:
        print(f"  NOTE: {missing_url:,} image slots have no upstream URL; those show no original link.")

    missing_files = [image.image_id for image in bundle.images if image.local_path is None]
    if missing_files:
        print(f"\n  {len(missing_files):,} images are referenced but not on disk, e.g. {missing_files[:5]}")
        print("  run: make corpus-fetch")

    print()
    questions = bundle.questions
    with_images = sum(1 for question in questions if question.expected_image_count)
    print(f"  questions expecting images  {with_images:,} of {len(questions):,}")
    print(f"  placeholder token           {IMAGE_PLACEHOLDER!r}")

    problems = bundle.integrity_problems()
    if problems:
        print("\nintegrity problems:", file=sys.stderr)
        for problem in problems[:10]:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\nintegrity: no problems found")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "command",
        choices=("fetch", "export", "ingest", "inspect"),
        help=(
            "fetch: download the raw artefacts. export: write the bundle the API "
            "reads. ingest: index into RAGFlow. inspect: report what is on disk."
        ),
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="With ingest: remove this corpus's documents from its dataset and stop.",
    )
    parser.add_argument(
        "--corpus",
        default=MRAMG_ID,
        choices=sorted(ADAPTERS),
        help="Which corpus to operate on.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when an artefact is already present.",
    )
    args = parser.parse_args(argv)

    adapter = ADAPTERS[args.corpus](repo_root())
    handlers = {
        "fetch": command_fetch,
        "export": command_export,
        "ingest": command_ingest,
        "inspect": command_inspect,
    }
    return handlers[args.command](adapter, args)


if __name__ == "__main__":
    raise SystemExit(main())
