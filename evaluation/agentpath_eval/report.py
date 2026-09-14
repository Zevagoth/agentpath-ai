"""Markdown rendering for the benchmark report.

Written for a reader who was not in the room: it states the method, separates the
two scoring populations, and lists the individual failures rather than only the
aggregate, because the failures are where the next improvement comes from.
"""

from __future__ import annotations

from typing import Any

from agentpath_eval.metrics import ModeReport


def render_markdown(metadata: dict[str, Any], reports: list[ModeReport]) -> str:
    lines: list[str] = [
        "# AgentPath AI - retrieval benchmark",
        "",
        "> All knowledge-base content is **fictional and synthetic**.",
        "",
        "## Method",
        "",
        f"- Query set: **{metadata['query_set_size']} queries** - "
        f"{metadata['supported_queries']} supported (a procedure answers them) and "
        f"{metadata['unsupported_queries']} unsupported (nothing in the knowledge base answers them).",
        "- Retrieval quality (Recall@1, Recall@3, MRR) is measured on **supported queries only**.",
        "- Abstention accuracy is measured on **unsupported queries only**.",
        "- False abstention rate is the share of *supported* queries the system refused to answer, "
        "reported alongside recall so a cautious threshold cannot flatter the result.",
        f"- Latency: {metadata['repeats']} measured runs per query, after "
        f"{metadata['warmup_runs_discarded_per_query']} discarded warm-up run per query. "
        "Percentiles are nearest-rank, not interpolated.",
        f"- Top-k: {metadata['top_k']}.",
        f"- Reranking: **{'enabled' if metadata['reranking_enabled'] else 'disabled'}**"
        + (f" ({metadata['rerank_model']})" if metadata["reranking_enabled"] else "")
        + ".",
        "",
        "### Configuration",
        "",
        "| Setting | Value |",
        "| --- | --- |",
        f"| Embedding model | `{metadata['embedding_model']}` |",
        f"| Vector similarity weight | {metadata['vector_similarity_weight']} "
        f"(term weight {round(1 - float(metadata['vector_similarity_weight']), 2)}) |",
        f"| Similarity threshold | {metadata['similarity_threshold']} |",
        f"| Evidence thresholds | strong ≥ {metadata['evidence_thresholds']['strong_score']}, "
        f"margin ≥ {metadata['evidence_thresholds']['strong_margin']}, "
        f"partial ≥ {metadata['evidence_thresholds']['partial_score']} |",
        f"| Generated | `{metadata['generated_at']}` |",
        f"| Host | {metadata['platform']}, Python {metadata['python']} |",
        "",
        "## Modes compared",
        "",
    ]

    for report in reports:
        lines.append(f"- **`{report.mode}`** - {report.description}")
    lines += ["", "## Results", ""]

    lines += [
        "| Mode | Recall@1 | Recall@3 | MRR | Abstention | False abstention | p50 | p95 | mean |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for report in reports:
        lines.append(
            f"| `{report.mode}` | {report.recall_at_1:.0%} | {report.recall_at_3:.0%} | "
            f"{report.mrr:.3f} | {report.abstention_accuracy:.0%} | "
            f"{report.false_abstention_rate:.0%} | "
            f"{report.latency.p50_ms:.0f} ms | {report.latency.p95_ms:.0f} ms | "
            f"{report.latency.mean_ms:.0f} ms |"
        )

    lines += [
        "",
        "Recall, MRR and false abstention are over supported queries; abstention accuracy is over unsupported queries.",
        "",
    ]

    # -- per-type breakdown ------------------------------------------------- #
    lines += ["## By query type", ""]
    all_types: list[str] = []
    for report in reports:
        for query_type in report.by_type():
            if query_type not in all_types:
                all_types.append(query_type)

    header = "| Query type | n | " + " | ".join(f"`{r.mode}` R@3" for r in reports) + " |"
    lines.append(header)
    lines.append("| --- | --- | " + " | ".join("---" for _ in reports) + " |")
    for query_type in all_types:
        counts = next(
            (r.by_type()[query_type]["queries"] for r in reports if query_type in r.by_type()),
            0,
        )
        cells = []
        for report in reports:
            stats = report.by_type().get(query_type, {})
            if "recall_at_3" in stats:
                cells.append(f"{float(stats['recall_at_3']):.0%}")
            elif "abstention_accuracy" in stats:
                cells.append(f"abstain {float(stats['abstention_accuracy']):.0%}")
            else:
                cells.append("-")
        lines.append(f"| {query_type} | {counts} | " + " | ".join(cells) + " |")
    lines.append("")

    # -- failures ----------------------------------------------------------- #
    lines += ["## Misses", "", "Supported queries whose expected procedure was not in the top 3.", ""]
    any_failures = False
    for report in reports:
        failures = report.failures()
        lines.append(f"### `{report.mode}`")
        lines.append("")
        if not failures:
            lines += ["None.", ""]
            continue
        any_failures = True
        lines += ["| Query | Type | Expected | Returned | Abstained |", "| --- | --- | --- | --- | --- |"]
        for outcome in failures:
            returned = ", ".join(outcome.ranked) or "_nothing_"
            lines.append(
                f"| {outcome.query} | {outcome.type} | {', '.join(outcome.expected)} | "
                f"{returned} | {'yes' if outcome.abstained else 'no'} |"
            )
        lines.append("")

    if not any_failures:
        lines += ["", "_Every supported query was answered within the top 3 by every mode._", ""]

    # -- unsupported behaviour ---------------------------------------------- #
    lines += [
        "## Unsupported queries",
        "",
        "These have no answer in the knowledge base. Abstaining is the correct behaviour; "
        "returning a procedure means inventing policy.",
        "",
        "| Query | " + " | ".join(f"`{r.mode}`" for r in reports) + " |",
        "| --- | " + " | ".join("---" for _ in reports) + " |",
    ]
    unsupported_ids = [o.query_id for o in (reports[0].unsupported if reports else [])]
    for query_id in unsupported_ids:
        cells = []
        query_text = ""
        for report in reports:
            outcome = next((o for o in report.unsupported if o.query_id == query_id), None)
            if outcome is None:
                cells.append("-")
                continue
            query_text = outcome.query
            if outcome.abstained:
                cells.append("abstained")
            else:
                cells.append(f"answered: {', '.join(outcome.ranked[:2])}")
        lines.append(f"| {query_text} | " + " | ".join(cells) + " |")

    lines += ["", "## Full per-query results", "", "See the JSON file next to this report.", ""]
    return "\n".join(lines)
