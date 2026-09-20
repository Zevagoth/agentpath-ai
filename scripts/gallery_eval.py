"""Evaluation A: are the right figures actually available, in the right order?

Separate from the retrieval evaluation because it asks a different question. That one asks
"which figure is this question about" and scores precision. This one asks "is the figure
the answer needs on the screen at all", and for a UI screenshot with an arrow on it, that
is the question that matters: once the right procedure is found, the agent wants its
figures in order, and a ranking can only remove one.

Measured over the held-out split, per the targets:

* **Document reached** - was the ground-truth document retrieved at all. The ceiling for
  everything below, since a gallery can only hold figures from documents that were found.
* **Gallery image availability** - of the ground-truth figures, how many are on the screen.
  Compared directly against the ranked selection's coverage, which is the whole argument.
* **Order preserved** - every page in document order.
* **Broken image rate** - figures whose bytes are not on disk, one filesystem call each.
* **Broken source rate** - documents whose citation link does not resolve.
* **Step linking** - figures resolved to the passage containing them, where one exists.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from datetime import UTC, datetime

from agentpath.config import get_settings
from agentpath.corpus import Corpus, CorpusCatalogue
from agentpath.corpus_retrieval.gallery import build_gallery
from agentpath.corpus_retrieval.ports import RetrievalMethod
from agentpath.corpus_retrieval.service import build_methods
from agentpath.paths import reports_dir
from agentpath.provenance import SourceKind
from agentpath.retrieval.ragflow_client import RAGFlowClient
from agentpath_eval.corpus_split import HELD_OUT, load_split, split_path

DOCUMENT_LIMIT = 5
IMAGE_LIMIT = 5
#: Large enough that a document's whole figure set fits on one page, so the numbers below
#: measure the gallery and not the page size.
GALLERY_PAGE_SIZE = 500


async def run(args: argparse.Namespace) -> int:
    settings = get_settings()
    corpus = CorpusCatalogue.load().corpus(args.corpus)
    if corpus is None:
        print(f"no bundle for {args.corpus!r}; run: make corpus-export", file=sys.stderr)
        return 1
    split = load_split(split_path(corpus.corpus_id))
    questions = split.questions_for(args.split, corpus.questions)

    print(f"corpus   : {corpus.corpus_id}  ({corpus.counts['images']} figures)")
    print(f"split    : {args.split}, {len(questions)} questions")
    print(f"roles    : {_role_counts(corpus)}")
    print()

    async with RAGFlowClient(
        settings.ragflow_base_url,
        settings.ragflow_api_key.get_secret_value(),
        timeout=settings.ragflow_timeout_seconds,
    ) as client:
        methods = build_methods(corpus, settings=settings, client=client)
        service = methods[RetrievalMethod(args.method)]

        reached = 0
        gallery_coverage: list[float] = []
        ranked_coverage: list[float] = []
        with_images = 0
        ordered_pages = 0
        total_pages = 0
        broken_images = 0
        shown_images = 0
        broken_sources = 0
        total_sources = 0
        linked = 0
        gallery_sizes: list[int] = []
        rows: list[dict] = []

        for question in questions:
            result = await service.search(question.question, limit=DOCUMENT_LIMIT, image_limit=IMAGE_LIMIT)
            gallery = build_gallery(
                corpus,
                result.documents,
                selected_image_ids=result.image_ids,
                page=1,
                page_size=GALLERY_PAGE_SIZE,
            )

            truth_docs = set(question.provenance_document_ids)
            truth_images = set(question.ground_truth_image_ids)
            document_reached = bool(truth_docs & set(result.document_ids))
            if document_reached:
                reached += 1

            in_gallery = set(gallery.image_ids)
            if truth_images:
                with_images += 1
                gallery_coverage.append(len(truth_images & in_gallery) / len(truth_images))
                ranked_coverage.append(len(truth_images & set(result.image_ids[:IMAGE_LIMIT])) / len(truth_images))

            for item in gallery.documents:
                total_pages += 1
                if item.ordered:
                    ordered_pages += 1
            gallery_sizes.append(gallery.shown_images)

            for item in gallery.documents:
                for image in item.images:
                    shown_images += 1
                    if image.passage_id:
                        linked += 1
                    record = corpus.image(image.image_id)
                    if record is None or not record.path.is_file():
                        broken_images += 1

            for hit in result.documents:
                total_sources += 1
                trail = service.citation_for(hit, result)
                if trail is None or trail.kind is SourceKind.NONE or trail.link() is None:
                    broken_sources += 1
                elif trail.kind is SourceKind.LOCAL_DOCUMENT and corpus.source(trail.source_id) is None:
                    broken_sources += 1

            rows.append(
                {
                    "question_id": question.question_id,
                    "document_reached": document_reached,
                    "expected_images": list(truth_images),
                    "in_gallery": sorted(truth_images & in_gallery),
                    "in_ranked_selection": sorted(truth_images & set(result.image_ids)),
                    "gallery_size": gallery.shown_images,
                }
            )

    total = len(questions)
    summary = {
        "corpus_id": corpus.corpus_id,
        "split": args.split,
        "method": args.method,
        "questions": total,
        "questions_expecting_images": with_images,
        "document_reached_rate": round(reached / total, 4) if total else 0.0,
        "gallery_image_availability": round(statistics.fmean(gallery_coverage), 4) if gallery_coverage else 0.0,
        "ranked_selection_coverage_at_5": round(statistics.fmean(ranked_coverage), 4) if ranked_coverage else 0.0,
        "order_preserved_rate": round(ordered_pages / total_pages, 4) if total_pages else 0.0,
        "figures_shown": shown_images,
        "broken_image_rate": round(broken_images / shown_images, 4) if shown_images else 0.0,
        "broken_images": broken_images,
        "broken_source_rate": round(broken_sources / total_sources, 4) if total_sources else 0.0,
        "broken_sources": broken_sources,
        "figures_linked_to_a_passage": round(linked / shown_images, 4) if shown_images else 0.0,
        "median_figures_per_answer": statistics.median(gallery_sizes) if gallery_sizes else 0,
        "max_figures_per_answer": max(gallery_sizes) if gallery_sizes else 0,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    print(f"{'metric':<38} {'value':>10}   target")
    print("-" * 70)
    print(f"{'document reached (ground truth in top 5)':<38} {summary['document_reached_rate']:>9.1%}   -")
    print(f"{'gallery image availability':<38} {summary['gallery_image_availability']:>9.1%}   100%")
    print(f"{'ranked selection coverage @5':<38} {summary['ranked_selection_coverage_at_5']:>9.1%}   (for comparison)")
    print(f"{'order preserved':<38} {summary['order_preserved_rate']:>9.1%}   100%")
    print(f"{'broken image rate':<38} {summary['broken_image_rate']:>9.1%}   0%")
    print(f"{'broken source rate':<38} {summary['broken_source_rate']:>9.1%}   0%")
    print(f"{'figures linked to a passage':<38} {summary['figures_linked_to_a_passage']:>9.1%}   -")
    print()
    print(f"figures shown across the split : {summary['figures_shown']:,}")
    median_figures = summary["median_figures_per_answer"]
    max_figures = summary["max_figures_per_answer"]
    print(f"figures per answer             : median {median_figures}, max {max_figures}")

    gap = summary["gallery_image_availability"] - summary["ranked_selection_coverage_at_5"]
    print(f"\nwhat the ranking was hiding    : {gap:.1%} of the wanted figures")

    target = reports_dir()
    target.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = target / f"gallery-eval-{stamp}.json"
    path.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {path.name}")
    return 0


def _role_counts(corpus: Corpus) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in corpus.images.values():
        counts[record.role] = counts.get(record.role, 0) + 1
    return dict(sorted(counts.items()))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", default="mramg-manual")
    parser.add_argument("--method", default="multimodal", choices=[m.value for m in RetrievalMethod])
    parser.add_argument("--split", default=HELD_OUT)
    return asyncio.run(run(parser.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
