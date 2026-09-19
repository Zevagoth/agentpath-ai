"""Render the external-corpus retrieval evaluation as Markdown.

Written for a reader who was not in the room, and who is entitled to be sceptical:
this is a system evaluating itself on a corpus, so the report states what was measured,
what each metric means, which targets were missed, and which individual questions
failed. A table of green numbers with no failures listed would be the less useful
document even if the numbers were identical.

Dev and held-out are always separate sections. Pooling them would erase the only thing
the split is for.
"""

from __future__ import annotations

from typing import Any

from agentpath_eval.corpus_eval import (
    DEPTHS,
    TARGET_DOCUMENT_RECALL_AT_3,
    TARGET_IMAGE_RECALL_AT_5,
    TARGET_RETRIEVAL_P95_MS,
    MethodReport,
)
from agentpath_eval.corpus_split import DEV, HELD_OUT

#: Failures listed individually per method. Enough to see a pattern, few enough that the
#: document stays readable; the JSON has every one.
MAX_LISTED_FAILURES = 12


def render_corpus_markdown(metadata: dict[str, Any], reports: list[MethodReport]) -> str:
    counts = metadata["corpus_counts"]
    split = metadata["split"]

    lines: list[str] = [
        f"# External corpus retrieval evaluation - {metadata['corpus_title']}",
        "",
        "> This corpus is **not** ours. It is an external benchmark, used unmodified, to test "
        "whether the architecture works on a corpus we did not author.",
        "",
        f"**Attribution.** {metadata['attribution']}",
        "",
        "## What was measured",
        "",
        f"- Corpus: `{metadata['corpus_id']}` - {counts['documents']} documents, "
        f"{counts['chunks']} chunks, {counts['images']} images, {counts['questions']} questions.",
        f"- Retrieval dataset: `{metadata['dataset_name']}`. One dataset per corpus, and a "
        "retrieval call names exactly one, so results from two corpora cannot be mixed.",
        f"- Split: seed `{split['seed']}`, {split['held_out_fraction']:.0%} held out - "
        f"{split['dev']} dev and {split['held_out']} held-out questions, stratified on how many "
        f"images the answer needs. Recorded in `{split['file']}`.",
        "- **Nothing was tuned on the held-out split.** The split was frozen and committed before "
        "any of these numbers existed.",
        f"- Documents requested per question: {metadata['document_limit']}. Images: {metadata['image_limit']}.",
        f"- Latency: {metadata['measured_runs_per_question']} measured run per question, after "
        f"{metadata['warmup_queries_discarded_per_method']} discarded warm-up query per method. "
        "Percentiles are nearest-rank, not interpolated, and cover the retrieval engine only - "
        "image selection is timed separately.",
        "",
        "### Image retrieval is text retrieval",
        "",
        f"The image retrieval mode is **{metadata['image_retrieval_mode']}**. No model on any path "
        "here reads a pixel. Images are ranked by the document text surrounding their `<PIC>` marker "
        f"({metadata['image_context_sizes'][0]} characters before, {metadata['image_context_sizes'][1]} after), "
        "because every one of this corpus's image captions is an empty string. See "
        "`docs/RETRIEVAL_MODES.md`. Calling this multimodal would be a claim about the wrong thing: "
        "the *result* is text and images together, the *method* is text.",
        "",
        "### Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Embedding model | `{metadata['embedding_model']}` |",
        f"| Vector similarity weight | {metadata['vector_similarity_weight']} "
        f"(term weight {round(1 - float(metadata['vector_similarity_weight']), 2)}) |",
        f"| Similarity threshold | {metadata['similarity_threshold']} |",
        "| Reranker | disabled for every method (measured OOM-killed on this host, exit 137) |",
        f"| Generated | `{metadata['generated_at']}` |",
        f"| Host | {metadata['platform']}, Python {metadata['python']} |",
        "",
        "## Methods compared",
        "",
        "| Method | Documents ranked by | Images chosen by |",
        "| --- | --- | --- |",
        "| `lexical` | in-process BM25 over the exported chunks. No RAGFlow, no embeddings, no model. | "
        "document order (positional) |",
        "| `hybrid` | RAGFlow: BM25 blended with `bge-m3` vector similarity. | document order (positional) |",
        "| `multimodal` | the same RAGFlow retrieval, sharing one retriever instance with `hybrid`. | "
        "context-window ranking |",
        "",
        "`hybrid` and `multimodal` share one retriever object, so any difference between their image "
        "numbers is attributable to image ranking and nothing else. `lexical` exists so the other two "
        'are answerable: "92% recall" means nothing without knowing what keyword matching alone reaches.',
        "",
        "### Metric definitions",
        "",
        "- **Document Recall@k** - share of questions whose provenance document is in the top k.",
        "- **Document MRR** - mean of `1 / rank` of the first correct document; 0 when absent.",
        "- **Image Recall@k** - *coverage*: mean of `|truth ∩ top k| / |truth|`. A question wanting "
        "three images and getting one scores 0.33. This is what the target refers to.",
        '- **Image Hit@k** - mean of "at least one correct image in the top k". Always ≥ Recall@k and '
        "much more forgiving on multi-image questions; reported so the two are not confused.",
        "- **Image precision / F1** - over the images actually returned, averaged across questions that "
        "expect images and received some.",
        "- **Image abstention** - share of the questions expecting *no* image for which none was "
        "returned. Excluded from every other image metric, since recall over an empty truth set is "
        "undefined.",
        "- **Citation validity** - share of returned documents whose provenance names its corpus and "
        "document and offers a link that resolves.",
        "- **Broken source / image rate** - citations with no resolvable link; images whose bytes are "
        "not on disk, checked per image with a filesystem call.",
        "- **Unresolved offsets** - passages whose position in the document could not be recovered. "
        "Those citations open the document at the top rather than at a guessed offset.",
        "",
    ]

    for side, heading in ((DEV, "Dev split"), (HELD_OUT, "Held-out split")):
        side_reports = [report for report in reports if report.split == side]
        if not side_reports:
            continue
        lines.extend(_split_section(heading, side, side_reports))

    lines.extend(_targets_section(reports))
    lines.extend(_failures_section(reports))

    lines.extend(
        [
            "## Reproducing this",
            "",
            "```bash",
            "make corpus-fetch          # 132 MiB, selective range download",
            "make corpus-export         # normalise into the bundle the API reads",
            "make corpus-ingest         # index into its own RAGFlow dataset",
            "make corpus-split          # refuses to change the committed split",
            "make corpus-eval           # this report",
            "```",
            "",
            "`make corpus-eval` also writes a JSON file beside this one carrying every "
            "per-question outcome, so any rate in these tables can be traced to the questions "
            "behind it. That JSON is gitignored - it names RAGFlow ids that mean nothing outside "
            "one local instance - so re-run the command to regenerate it.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _split_section(heading: str, side: str, reports: list[MethodReport]) -> list[str]:
    questions = len(reports[0].outcomes) if reports else 0
    with_images = len(reports[0].with_images) if reports else 0
    no_images = len(reports[0].without_images) if reports else 0

    note = (
        "Decisions were allowed to be informed by these numbers."
        if side == DEV
        else "**These numbers informed no decision.** Nothing was tuned against them."
    )

    lines = [
        f"## {heading}",
        "",
        f"{questions} questions - {with_images} expecting at least one image, {no_images} expecting none. {note}",
        "",
        "### Documents",
        "",
        "| Method | " + " | ".join(f"Recall@{k}" for k in DEPTHS) + " | MRR | No document returned |",
        "| --- | " + " | ".join("---" for _ in DEPTHS) + " | --- | --- |",
    ]
    for report in reports:
        recalls = " | ".join(f"{report.document_recall_at(k):.1%}" for k in DEPTHS)
        lines.append(
            f"| `{report.method}` | {recalls} | {report.document_mrr:.3f} | {report.document_abstention_rate:.1%} |"
        )

    lines.extend(
        [
            "",
            "### Images",
            "",
            "| Method | Chosen by | "
            + " | ".join(f"Recall@{k}" for k in DEPTHS)
            + " | Hit@5 | Precision | F1 | Abstained when none wanted |",
            "| --- | --- | " + " | ".join("---" for _ in DEPTHS) + " | --- | --- | --- | --- |",
        ]
    )
    for report in reports:
        recalls = " | ".join(f"{report.image_recall_at(k):.1%}" for k in DEPTHS)
        lines.append(
            f"| `{report.method}` | {report.image_strategy} | {recalls} | "
            f"{report.image_hit_at(5):.1%} | {report.image_precision:.1%} | {report.image_f1:.3f} | "
            f"{report.image_abstention_accuracy:.1%} |"
        )

    lines.extend(
        [
            "",
            "### Provenance and latency",
            "",
            "| Method | Citation validity | Broken sources | Broken images | Unresolved offsets | p50 | p95 | max |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for report in reports:
        lines.append(
            f"| `{report.method}` | {report.citation_validity:.1%} | {report.broken_source_rate:.1%} | "
            f"{report.broken_image_rate:.1%} ({report.broken_images}) | "
            f"{report.unresolved_offset_rate:.1%} | "
            f"{report.latency.p50_ms:.0f} ms | {report.latency.p95_ms:.0f} ms | {report.latency.max_ms:.0f} ms |"
        )

    lines.extend(["", "#### How citation offsets were established", "", "| Method | Offset sources |", "| --- | --- |"])
    for report in reports:
        sources = ", ".join(f"{name} {count}" for name, count in report.offset_sources.items()) or "none"
        lines.append(f"| `{report.method}` | {sources} |")

    lines.extend(
        [
            "",
            "`bundle` and `exact_match` are exact. `normalised_match` located the passage after "
            "collapsing whitespace, so the span is right and its edges can be off by the whitespace "
            "that was collapsed. `unresolved` means no offset was reported at all.",
            "",
            "#### By how many images the answer needs",
            "",
            "| Method | " + " | ".join(sorted(reports[0].by_stratum())) + " |",
            "| --- | " + " | ".join("---" for _ in reports[0].by_stratum()) + " |",
        ]
    )
    strata = sorted(reports[0].by_stratum())
    for report in reports:
        table = report.by_stratum()
        cells = []
        for stratum in strata:
            row = table.get(stratum, {})
            document = row.get("document_recall_at_3")
            image = row.get("image_recall_at_5")
            image_text = f" / I {image:.0%}" if isinstance(image, float) else ""
            cells.append(f"D {document:.0%}{image_text}" if isinstance(document, float) else "-")
        lines.append(f"| `{report.method}` | " + " | ".join(cells) + " |")
    lines.extend(["", "D = Document Recall@3, I = Image Recall@5 within that stratum.", ""])

    return lines


def _targets_section(reports: list[MethodReport]) -> list[str]:
    lines = [
        "## Targets",
        "",
        f"Stated before the run: Document Recall@3 ≥ {TARGET_DOCUMENT_RECALL_AT_3:.0%}, "
        f"Image Recall@5 ≥ {TARGET_IMAGE_RECALL_AT_5:.0%}, citation validity 100%, zero broken sources "
        f"and images, retrieval p95 < {TARGET_RETRIEVAL_P95_MS / 1000:.0f} s. "
        "A miss is reported as a miss with its real number.",
        "",
        "| Split | Method | Target | Measured | Required | Met |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        for name, check in report.target_checks().items():
            measured = check["measured"]
            target = check["target"]
            unit = " ms" if name.endswith("_ms") else ""
            shown = f"{measured}{unit}" if unit else f"{measured:.1%}" if measured <= 1.0 else f"{measured}"
            required = f"{target}{unit}" if unit else f"{target:.0%}"
            lines.append(
                f"| {report.split} | `{report.method}` | {name} | {shown} | "
                f"{check['direction']} {required} | {'yes' if check['met'] else '**no**'} |"
            )
    lines.append("")
    return lines


def _failures_section(reports: list[MethodReport]) -> list[str]:
    lines = [
        "## Where it failed",
        "",
        "Questions whose provenance document was not in the top 3, worst first. Listed rather than "
        "summarised, because a rate says how often and only the cases say why.",
        "",
    ]
    for report in reports:
        failures = report.failures()
        lines.append(f"### `{report.method}` on {report.split} - {len(failures)} of {len(report.outcomes)}")
        lines.append("")
        if not failures:
            lines.extend(["Every question's document was retrieved in the top 3.", ""])
            continue
        lines.append("| Question | Wanted | Got | RR |")
        lines.append("| --- | --- | --- | --- |")
        for outcome in failures[:MAX_LISTED_FAILURES]:
            wanted = ", ".join(outcome.expected_documents)
            got = ", ".join(outcome.ranked_documents[:3]) or "nothing"
            question = outcome.question.replace("|", "\\|")[:90]
            lines.append(
                f"| {outcome.question_id}: {question} | {wanted} | {got} | {outcome.document_reciprocal_rank:.2f} |"
            )
        if len(failures) > MAX_LISTED_FAILURES:
            lines.append(f"| … and {len(failures) - MAX_LISTED_FAILURES} more, in the JSON | | | |")
        lines.append("")
    return lines


__all__ = ["MAX_LISTED_FAILURES", "render_corpus_markdown"]
