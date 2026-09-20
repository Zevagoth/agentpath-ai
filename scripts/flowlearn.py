"""FlowLearn: fetch, split, and measure whether the model can read a flowchart.

    uv run python scripts/flowlearn.py fetch
    uv run python scripts/flowlearn.py split
    uv run python scripts/flowlearn.py eval --split dev
    uv run python scripts/flowlearn.py eval --split held_out

CC BY-NC-SA 4.0. Non-commercial evaluation only, nothing committed, no training. See
``agentpath_ingestion.corpora.flowlearn`` for the full licence note.

The split is written before any model runs, and ``eval --split held_out`` is meant to be run
once, after the dev numbers are settled. Nothing here tunes anything.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime

from agentpath.paths import repo_root, reports_dir
from agentpath_eval.corpus_split import DEV, HELD_OUT, load_split, make_split, split_path, write_split
from agentpath_eval.workflow_eval import evaluate
from agentpath_ingestion.corpora.flowlearn import (
    CORPUS_ATTRIBUTION,
    CORPUS_ID,
    CORPUS_LICENCE,
    CORPUS_TITLE,
    SAMPLE_SEED,
    SAMPLE_SIZE,
    SOURCE_URL,
    FlowchartRecord,
    corpus_fingerprint,
    default_root,
    fetch,
    load_corpus,
)
from agentpath_ingestion.corpora.workflow import WorkflowExtractor


class _SplitShim:
    """Adapts a flowchart to the split's ``QuestionLike`` protocol.

    The split code is reused rather than reimplemented so both corpora get the same seeded,
    order-independent, stratum-local algorithm and the same guarantees. Stratified on node
    count, which is this corpus's difficulty axis: a 3-node chart and a 10-node chart are not
    the same task, and an unstratified shuffle could load one side with the easy ones.
    """

    __slots__ = ("difficulty", "ground_truth_image_ids", "question_id")

    def __init__(self, record: FlowchartRecord) -> None:
        self.question_id = record.flowchart_id
        self.ground_truth_image_ids: tuple[str, ...] = ()
        self.difficulty = f"nodes={record.expected_nodes}"


def command_fetch(args: argparse.Namespace) -> int:
    report = fetch(default_root(repo_root()), sample=args.sample, force=args.force)
    print(f"corpus     : {CORPUS_ID}  ({CORPUS_TITLE})")
    print(f"licence    : {CORPUS_LICENCE}  - non-commercial evaluation only")
    print(f"source     : {SOURCE_URL}")
    print(f"images     : {report.images}")
    print(f"downloaded : {report.downloaded_bytes / 1e6:.1f} MB")
    print(f"reused     : {report.reused_bytes / 1e6:.1f} MB")
    print(f"seconds    : {report.seconds}")
    for problem in report.problems[:10]:
        print(f"  problem: {problem}", file=sys.stderr)
    return 0 if not report.problems else 1


def command_split(args: argparse.Namespace) -> int:
    records = load_corpus(default_root(repo_root()), sample=args.sample)
    if not records:
        print("no flowcharts on disk. Run: make flowlearn-fetch", file=sys.stderr)
        return 1

    split = make_split([_SplitShim(record) for record in records], corpus_id=CORPUS_ID)
    target = split_path(CORPUS_ID)

    if target.exists() and not args.force:
        existing = load_split(target)
        if existing.dev == split.dev and existing.held_out == split.held_out:
            print(f"split unchanged: {target}")
        else:
            print(f"REFUSED: {target} exists and differs from the computed split.", file=sys.stderr)
            print("Every number reported against it would become incomparable.", file=sys.stderr)
            return 1
    else:
        write_split(split, target)
        print(f"written : {target}")

    print(f"corpus      : {split.corpus_id}")
    print(f"sample seed : {SAMPLE_SEED} (which flowcharts were downloaded)")
    print(f"split seed  : {split.seed} (how they were divided)")
    print(f"fingerprint : {corpus_fingerprint(records)}")
    print(f"total       : {split.total}  dev {len(split.dev)}  held-out {len(split.held_out)}")
    print()
    print(f"{'stratum':<14} {'dev':>5} {'held-out':>9}")
    for stratum, (dev_count, held_count) in sorted(split.strata.items()):
        print(f"{stratum:<14} {dev_count:>5} {held_count:>9}")
    if set(split.dev) & set(split.held_out):
        print("\nBROKEN: a flowchart is on both sides", file=sys.stderr)
        return 1
    print("\nevery flowchart is on exactly one side")
    return 0


def command_eval(args: argparse.Namespace) -> int:
    root = default_root(repo_root())
    records = load_corpus(root, sample=args.sample)
    if not records:
        print("no flowcharts on disk. Run: make flowlearn-fetch", file=sys.stderr)
        return 1

    target = split_path(CORPUS_ID)
    if not target.is_file():
        print(f"no split at {target}. Run: make flowlearn-split", file=sys.stderr)
        return 1
    split = load_split(target)

    by_id = {record.flowchart_id: record for record in records}
    wanted = split.dev if args.split == DEV else split.held_out
    chosen = [by_id[flowchart_id] for flowchart_id in wanted if flowchart_id in by_id]
    if args.limit:
        chosen = chosen[: args.limit]

    print(f"corpus   : {CORPUS_ID}  (licence {CORPUS_LICENCE}, non-commercial evaluation)")
    print(f"split    : {args.split}, {len(chosen)} flowcharts")
    print(f"model    : {args.model}")
    print(f"seeds    : sample {SAMPLE_SEED}, split {split.seed}")
    if args.split == HELD_OUT:
        print("NOTE     : held-out. Nothing here may be used to tune anything.")
    print()

    extractor = WorkflowExtractor(model=args.model)

    def progress(index: int, total: int, outcome) -> None:
        if index % 10 == 0 or index == total:
            print(
                f"  {index:>4}/{total}  last: nodes {outcome.predicted_nodes}/{outcome.expected_nodes} "
                f"edges {outcome.predicted_edges}/{outcome.expected_edges} "
                f"{'review' if outcome.needs_review else 'ok':<6} {outcome.seconds:.1f}s",
                flush=True,
            )

    report = evaluate(chosen, extractor, corpus_id=CORPUS_ID, split=args.split, on_progress=progress)
    metrics = report.metrics

    print()
    print("=" * 74)
    print(f"{'metric':<52} {'value':>10}")
    print("=" * 74)
    labels = {
        "1_text": "1. Text (OCR)",
        "2_nodes": "2. Node detection",
        "3_edges_undirected": "3. Edge detection (undirected)",
        "4_arrow_direction": "4. Arrow direction",
        "5_branches": "5. Branch / decision detection",
        "6_path_reachability": "6. Path reachability",
        "operational": "Operational",
    }
    for group, values in metrics.items():
        print(f"\n{labels.get(group, group)}")
        for name, value in values.items():
            shown = f"{value:.1%}" if isinstance(value, float) and value <= 1.0 else str(value)
            print(f"  {name:<50} {shown:>10}")

    reasons = report.review_reason_counts()
    if reasons:
        print(f"\nwhy the validator sent extractions to review (top {min(6, len(reasons))}):")
        for reason, count in list(reasons.items())[:6]:
            print(f"  {count:>4}  {reason}")

    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out / f"flowlearn-{args.split}-{stamp}.json"
    path.write_text(
        json.dumps(
            {
                "metadata": {
                    "corpus_id": CORPUS_ID,
                    "corpus_title": CORPUS_TITLE,
                    "licence": CORPUS_LICENCE,
                    "attribution": CORPUS_ATTRIBUTION,
                    "source": SOURCE_URL,
                    "non_commercial_use_only": True,
                    "used_for_training": False,
                    "sample_seed": SAMPLE_SEED,
                    "sample_size": args.sample,
                    "split_seed": split.seed,
                    "split": args.split,
                    "fingerprint": corpus_fingerprint(records),
                    "model": args.model,
                    "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
                },
                **report.to_dict(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nwrote {path.name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("fetch", "split", "eval"))
    parser.add_argument("--sample", type=int, default=SAMPLE_SIZE, help="how many flowcharts to use")
    parser.add_argument("--split", default=DEV, choices=(DEV, HELD_OUT))
    parser.add_argument("--model", default="qwen3.5:4b", help="the vision model to test")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N of the split")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    handlers = {"fetch": command_fetch, "split": command_split, "eval": command_eval}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
