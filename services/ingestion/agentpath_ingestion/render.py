"""Render procedures to retrieval-friendly Markdown.

Why Markdown at all, when the source is already structured YAML? Because
RAGFlow's parser, chunker, embedder and citation highlighter all work on
documents. Rendering gives us one artefact that is simultaneously:

* what RAGFlow indexes,
* what a human reviewer reads when verifying an answer,
* and what the UI links to as "the original source".

The layout is written for retrieval rather than for beauty: every section starts
with the words an agent would search for, alternative phrasings are inlined near
the top so lexical search can hit them, and screen and button names stay in the
step text because agents search for those too.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from agentpath_ingestion.dataset import LoadedProcedure, find_forbidden_brand_tokens
from agentpath_ingestion.schema import FICTIONAL_BANNER, Procedure

BANNER_LINE = f"> **{FICTIONAL_BANNER}** · Fictional agency *Northwind Travel* · Fictional console *AgentDesk*"

DISCLAIMER = (
    "> This document is invented for a proof of concept. It is not a real policy, "
    "it describes no real company, and it must not be used to serve real customers."
)


@dataclass(frozen=True)
class RenderedProcedure:
    """A rendered document plus the metadata RAGFlow needs alongside it."""

    procedure_id: str
    filename: str
    markdown: str
    metadata: dict[str, str]

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.markdown.encode("utf-8")).hexdigest()


def _bullets(items: list[str]) -> list[str]:
    return [f"- {item.strip()}" for item in items if item and item.strip()]


def _metadata_table(procedure: Procedure) -> list[str]:
    rows = [
        ("Procedure ID", procedure.procedure_id),
        ("Product line", procedure.product_line.value),
        ("Scenario tags", ", ".join(procedure.scenario_tags)),
        ("Severity", procedure.severity.value),
        ("Version", procedure.version),
        ("Effective date", procedure.effective_date.isoformat()),
        ("Owner team", procedure.owner_team),
    ]
    lines = ["| Field | Value |", "| --- | --- |"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return lines


def _render_steps(procedure: Procedure) -> list[str]:
    lines: list[str] = ["## Steps"]
    for step in procedure.steps:
        heading = f"### Step {step.number}"
        context = []
        if step.screen:
            context.append(f"Screen: {step.screen}")
        if step.ui_element:
            context.append(f"Control: {step.ui_element}")
        if context:
            heading += " — " + " — ".join(context)
        lines += ["", heading, "", step.instruction.strip()]
        if step.instruction_ar:
            lines += ["", f"**عربي:** {step.instruction_ar.strip()}"]
        if step.expected_result:
            lines += ["", f"**Expected result:** {step.expected_result.strip()}"]
        if step.screenshot:
            lines += [
                "",
                f"**Screenshot reference:** `{step.screenshot}` "
                "(original screenshot is attached to this step in the knowledge base)",
            ]
    return lines


def _render_branches(procedure: Procedure) -> list[str]:
    if not procedure.decision_branches:
        return []
    lines = ["## Decision branches"]
    for branch in procedure.decision_branches:
        text = f"- **If** {branch.condition.strip()} **then** {branch.then.strip()}"
        if branch.otherwise:
            text += f" **Otherwise** {branch.otherwise.strip()}"
        if branch.goto_step is not None:
            text += f" (continue at step {branch.goto_step})"
        lines.append(text)
    return lines


def _render_escalation(procedure: Procedure) -> list[str]:
    esc = procedure.escalation
    lines = [
        "## Escalation",
        "",
        f"- **Escalate when:** {esc.trigger.strip()}",
        f"- **Queue:** {esc.queue}",
        f"- **Channel:** {esc.contact_channel}",
        f"- **Response target:** {esc.response_target}",
        f"- **Approval required:** {'yes' if esc.approval_required else 'no'}",
    ]
    if esc.approver_role:
        lines.append(f"- **Approver:** {esc.approver_role}")
    return lines


def render_markdown(procedure: Procedure) -> str:
    """Render one procedure as a Markdown document."""
    lines: list[str] = [
        BANNER_LINE,
        ">",
        DISCLAIMER,
        "",
        f"# {procedure.procedure_id} — {procedure.title}",
        "",
        f"**العنوان بالعربية:** {procedure.title_ar}",
        "",
        *_metadata_table(procedure),
        "",
        "## Summary",
        "",
        procedure.summary.en.strip(),
        "",
        f"**ملخص:** {procedure.summary.ar.strip()}",
        "",
        "## Also known as (search terms)",
        "",
        "Agents rarely type the official title. These are the phrasings this "
        "procedure should also be found by:",
        "",
        f"- **English:** {'; '.join(procedure.synonyms.en)}",
        f"- **العربية:** {'؛ '.join(procedure.synonyms.ar)}",
        "",
        "## When this procedure applies",
        "",
        *_bullets(procedure.applies_when),
    ]

    if procedure.does_not_apply_when:
        lines += ["", "## When this procedure does not apply", "", *_bullets(procedure.does_not_apply_when)]

    if procedure.required_information:
        lines += [
            "",
            "## Information to collect before acting",
            "",
            *_bullets(procedure.required_information),
        ]

    lines += ["", *_render_steps(procedure)]

    branches = _render_branches(procedure)
    if branches:
        lines += ["", *branches]

    if procedure.do:
        lines += ["", "## Do", "", *_bullets(procedure.do)]
    if procedure.do_not:
        lines += ["", "## Do not", "", *_bullets(procedure.do_not)]
    if procedure.exceptions:
        lines += ["", "## Exceptions and special conditions", "", *_bullets(procedure.exceptions)]

    lines += ["", *_render_escalation(procedure)]

    lines += [
        "",
        "## Suggested customer wording",
        "",
        "Generated wording for the agent to adapt. It is not part of the procedure "
        "itself and carries no policy weight.",
        "",
        f"- **English:** {procedure.customer_wording.en.strip()}",
        f"- **العربية:** {procedure.customer_wording.ar.strip()}",
    ]

    if procedure.related_procedures:
        lines += [
            "",
            "## Related procedures",
            "",
            *_bullets(procedure.related_procedures),
        ]

    lines += [
        "",
        "---",
        "",
        f"{FICTIONAL_BANNER} · {procedure.procedure_id} · version {procedure.version} "
        f"· effective {procedure.effective_date.isoformat()}",
        "",
    ]
    return "\n".join(lines)


def build_metadata(procedure: Procedure) -> dict[str, str]:
    """Metadata pushed to RAGFlow so retrieval can filter on it.

    RAGFlow metadata conditions compare strings, so every value is stringified
    here rather than at the call site.
    """
    return {
        "procedure_id": procedure.procedure_id,
        "product_line": procedure.product_line.value,
        "scenario_tags": ",".join(procedure.scenario_tags),
        "severity": procedure.severity.value,
        "version": procedure.version,
        "effective_date": procedure.effective_date.isoformat(),
        "owner_team": procedure.owner_team,
        "doc_kind": "procedure",
        "language": "en+ar",
        "synthetic": "true",
    }


def render(loaded: LoadedProcedure) -> RenderedProcedure:
    """Render a loaded procedure and assert it is safe to publish."""
    procedure = loaded.procedure
    markdown = render_markdown(procedure)

    if FICTIONAL_BANNER not in markdown:
        raise ValueError(f"{procedure.procedure_id}: rendered document is missing the fictional-data banner")

    found = find_forbidden_brand_tokens(markdown)
    if found:
        raise ValueError(f"{procedure.procedure_id}: rendered document mentions real brands: {', '.join(found)}")

    return RenderedProcedure(
        procedure_id=procedure.procedure_id,
        filename=f"{procedure.procedure_id}.md",
        markdown=markdown,
        metadata=build_metadata(procedure),
    )


def write_build(
    rendered: list[RenderedProcedure],
    target_dir: Path,
) -> Path:
    """Write rendered documents plus a manifest, and return the manifest path."""
    target_dir.mkdir(parents=True, exist_ok=True)

    for item in rendered:
        (target_dir / item.filename).write_text(item.markdown, encoding="utf-8")

    manifest = {
        "banner": FICTIONAL_BANNER,
        "document_count": len(rendered),
        "documents": [
            {
                "procedure_id": item.procedure_id,
                "filename": item.filename,
                "sha256": item.sha256,
                "metadata": item.metadata,
            }
            for item in rendered
        ],
    }
    manifest_path = target_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifest_path
