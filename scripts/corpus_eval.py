"""Run the external-corpus retrieval evaluation: three methods, two splits.

    uv run python scripts/corpus_eval.py
    uv run python scripts/corpus_eval.py --splits held_out
    uv run python scripts/corpus_eval.py --methods lexical,multimodal

Reports both splits separately and never pools them, because the whole point of the
split is that the dev numbers were available while decisions were being made and the
held-out numbers were not. A single combined figure would erase that distinction.

Writes JSON and Markdown into ``evaluation/reports/``. The JSON carries every
per-question outcome, so a rate in the table can always be traced to the questions
behind it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentpath.config import get_settings
from agentpath.corpus import CorpusCatalogue, dataset_name_for
from agentpath.corpus_retrieval.ports import RetrievalMethod
from agentpath.corpus_retrieval.service import build_methods
from agentpath.paths import reports_dir
from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath_eval.corpus_eval import (
    DOCUMENT_LIMIT,
    IMAGE_LIMIT,
    MethodReport,
    evaluate_method,
    summarise,
)
from agentpath_eval.corpus_report import render_corpus_markdown
from agentpath_eval.corpus_split import DEV, HELD_OUT, load_split, split_path

DEFAULT_METHODS = ("lexical", "hybrid", "multimodal")
DEFAULT_SPLITS = (DEV, HELD_OUT)


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    catalogue = CorpusCatalogue.load()
    corpus = catalogue.corpus(args.corpus)
    if corpus is None:
        print(f"no bundle for corpus {args.corpus!r}. Run: make corpus-fetch && make corpus-export", file=sys.stderr)
        print(f"available: {list(catalogue.ids)}", file=sys.stderr)
        return 1

    split_file = split_path(corpus.corpus_id)
    if not split_file.is_file():
        print(f"no committed split at {split_file}. Run: make corpus-split", file=sys.stderr)
        return 1
    split = load_split(split_file)

    wanted_methods = [RetrievalMethod(name) for name in args.methods.split(",")]
    wanted_splits = [name for name in args.splits.split(",")]

    print(f"corpus   : {corpus.corpus_id}  ({corpus.title})")
    print(f"licence  : {corpus.licence}")
    print(f"counts   : {corpus.counts}")
    print(f"dataset  : {dataset_name_for(corpus.corpus_id)}")
    print(f"split    : seed {split.seed}, {len(split.dev)} dev / {len(split.held_out)} held out")
    print(f"methods  : {', '.join(method.value for method in wanted_methods)}")
    print(f"depths   : documents @{DOCUMENT_LIMIT}, images @{IMAGE_LIMIT}")
    print(f"images   : {corpus.image_retrieval_mode} (no model reads a pixel)")
    print()

    async with RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=settings.ragflow_timeout_seconds,
    ) as client:
        available = build_methods(corpus, settings=settings, client=client)
        missing = [method.value for method in wanted_methods if method not in available]
        if missing:
            print(f"these methods are not available for this corpus: {', '.join(missing)}", file=sys.stderr)
            print(f"available: {', '.join(method.value for method in available)}", file=sys.stderr)
            return 1

        reports: list[MethodReport] = []
        for side in wanted_splits:
            questions = split.questions_for(side, corpus.questions)
            for method in wanted_methods:
                service = available[method]
                print(f"running {method.value:<11} on {side:<8} ({len(questions)} questions) ...", flush=True)
                report = await evaluate_method(service, questions, split=side)
                reports.append(report)

    metadata = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "corpus_id": corpus.corpus_id,
        "corpus_title": corpus.title,
        "licence": corpus.licence,
        "attribution": corpus.attribution,
        "corpus_counts": corpus.counts,
        "dataset_name": dataset_name_for(corpus.corpus_id),
        "image_retrieval_mode": corpus.image_retrieval_mode,
        "image_context_sizes": list(corpus.image_context_sizes),
        "split": {
            "seed": split.seed,
            "held_out_fraction": split.held_out_fraction,
            "dev": len(split.dev),
            "held_out": len(split.held_out),
            "strata": {name: list(counts) for name, counts in sorted(split.strata.items())},
            "file": str(split_file.relative_to(Path.cwd()))
            if split_file.is_relative_to(Path.cwd())
            else str(split_file),
        },
        "document_limit": DOCUMENT_LIMIT,
        "image_limit": IMAGE_LIMIT,
        "measured_runs_per_question": 1,
        "warmup_queries_discarded_per_method": 1,
        "embedding_model": settings.ragflow_embedding_model,
        # Off for every method here. The reranker was measured to be killed by the
        # kernel on this machine, and a method that dies under load would make the
        # comparison a comparison of what survived.
        "rerank_model": None,
        "vector_similarity_weight": settings.vector_similarity_weight,
        "similarity_threshold": settings.similarity_threshold,
        "platform": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }

    payload = {
        "metadata": metadata,
        "summary": summarise(reports),
        "reports": [report.to_dict() for report in reports],
    }

    target = reports_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    json_path = target / f"corpus-retrieval-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    markdown = render_corpus_markdown(metadata, reports)
    markdown_path = target / f"corpus-retrieval-{stamp}.md"
    markdown_path.write_text(markdown, encoding="utf-8")
    (target / "latest-corpus-retrieval.md").write_text(markdown, encoding="utf-8")

    print_summary(reports)
    print(f"\nwrote {json_path.name}")
    print(f"wrote {markdown_path.name} and latest-corpus-retrieval.md in {target}")
    return 0


def print_summary(reports: list[MethodReport]) -> None:
    print()
    header = (
        f"{'split':<9} {'method':<11} {'images by':<15} "
        f"{'D@1':>6} {'D@3':>6} {'D@5':>6} {'MRR':>6} "
        f"{'I@1':>6} {'I@3':>6} {'I@5':>6} {'IPrec':>6} "
        f"{'cite':>6} {'p50':>7} {'p95':>7}"
    )
    print(header)
    print("-" * len(header))
    for report in reports:
        print(
            f"{report.split:<9} {report.method:<11} {report.image_strategy:<15} "
            f"{report.document_recall_at(1):>6.1%} {report.document_recall_at(3):>6.1%} "
            f"{report.document_recall_at(5):>6.1%} {report.document_mrr:>6.3f} "
            f"{report.image_recall_at(1):>6.1%} {report.image_recall_at(3):>6.1%} "
            f"{report.image_recall_at(5):>6.1%} {report.image_precision:>6.1%} "
            f"{report.citation_validity:>6.1%} "
            f"{report.latency.p50_ms:>6.0f}m {report.latency.p95_ms:>6.0f}m"
        )
    print()
    print("D@k: share of questions whose provenance document is in the top k.")
    print("I@k: mean share of a question's ground-truth images found in the top k (coverage, not hit rate).")
    print("IPrec: precision over the images actually returned. cite: citations with an openable link.")

    print()
    for report in reports:
        failed = [name for name, check in report.target_checks().items() if not check["met"]]
        state = "all targets met" if not failed else f"MISSED: {', '.join(failed)}"
        print(f"{report.split:<9} {report.method:<11} {state}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="mramg-manual", help="which corpus to evaluate")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS), help="comma-separated methods")
    parser.add_argument("--splits", default=",".join(DEFAULT_SPLITS), help="comma-separated splits")
    args = parser.parse_args(argv)
    return asyncio.run(run(args))


if __name__ == "__main__":
    raise SystemExit(main())
