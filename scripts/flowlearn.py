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
from pathlib import Path

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
from agentpath_ingestion.corpora.workflow import DEFAULT_NUM_CTX, WorkflowExtractor


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


def stratified_sample(records: list[FlowchartRecord], size: int) -> list[FlowchartRecord]:
    """A fixed sample spread evenly over node counts, for comparing two models fairly.

    Deterministic and derived only from the split's own order, so two models get **the same
    flowcharts** - which is the whole point of a comparison. Spread over complexity because a
    sample drawn from the head of the split would be whatever complexity happened to sort
    first, and a model that is good at small charts and bad at large ones would look uniformly
    good or uniformly bad depending on the draw.
    """
    buckets: dict[int, list[FlowchartRecord]] = {}
    for record in records:
        buckets.setdefault(record.expected_nodes, []).append(record)

    chosen: list[FlowchartRecord] = []
    order = sorted(buckets)
    index = 0
    # Round-robin across strata, so a shortfall in one is made up by the others rather than
    # silently skewing the sample.
    while len(chosen) < size and any(index < len(buckets[key]) for key in order):
        for key in order:
            if len(chosen) >= size:
                break
            if index < len(buckets[key]):
                chosen.append(buckets[key][index])
        index += 1
    return chosen


def resource_snapshot() -> dict[str, object]:
    """VRAM, RAM and whether the model is running on the GPU or spilling to the CPU.

    ``size`` against ``size_vram`` from Ollama is the only reliable CPU-offload signal: when
    they are equal the whole model is resident on the GPU, and any shortfall is running on the
    CPU at a fraction of the speed.
    """
    import subprocess

    snapshot: dict[str, object] = {}
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        ).stdout.strip()
        used, total, util = (part.strip() for part in output.split(","))
        snapshot["vram_used_mib"] = int(used)
        snapshot["vram_total_mib"] = int(total)
        snapshot["gpu_utilisation_percent"] = int(util)
    except (OSError, ValueError, subprocess.SubprocessError):
        snapshot["vram"] = "unavailable"

    try:
        import httpx

        loaded = httpx.get("http://127.0.0.1:11434/api/ps", timeout=10.0).json().get("models") or []
        models = []
        for entry in loaded:
            size = int(entry.get("size") or 0)
            in_vram = int(entry.get("size_vram") or 0)
            models.append(
                {
                    "name": entry.get("name"),
                    "size_gb": round(size / 1e9, 2),
                    "size_vram_gb": round(in_vram / 1e9, 2),
                    "fully_on_gpu": size > 0 and size == in_vram,
                    "cpu_offload_percent": round(100 * (size - in_vram) / size, 1) if size else 0.0,
                }
            )
        snapshot["loaded_models"] = models
    except Exception:
        snapshot["loaded_models"] = "unavailable"

    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
        values = {}
        for line in meminfo.splitlines():
            key, _, rest = line.partition(":")
            values[key] = int(rest.strip().split()[0])
        snapshot["ram_used_gb"] = round((values["MemTotal"] - values["MemAvailable"]) / 1e6, 1)
        snapshot["ram_total_gb"] = round(values["MemTotal"] / 1e6, 1)
    except (OSError, KeyError, ValueError, IndexError):
        snapshot["ram"] = "unavailable"
    return snapshot


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
    if args.stratified:
        chosen = stratified_sample(chosen, args.stratified)
    elif args.limit:
        chosen = chosen[: args.limit]

    print(f"corpus   : {CORPUS_ID}  (licence {CORPUS_LICENCE}, non-commercial evaluation)")
    print(f"split    : {args.split}, {len(chosen)} flowcharts")
    print(f"sampling : {'stratified over node count' if args.stratified else 'split order'}")
    print(f"model    : {args.model}  (num_ctx {args.num_ctx}, temperature 0, think off)")
    print(f"seeds    : sample {SAMPLE_SEED}, split {split.seed}")
    if args.split == HELD_OUT:
        print("NOTE     : held-out. Nothing here may be used to tune anything.")
    before = resource_snapshot()
    print(f"resources: {before.get('vram_used_mib')} MiB VRAM used, {before.get('ram_used_gb')} GB RAM")
    print()

    extractor = WorkflowExtractor(model=args.model, num_ctx=args.num_ctx)

    def progress(index: int, total: int, outcome) -> None:
        if index % 10 == 0 or index == total:
            print(
                f"  {index:>4}/{total}  last: nodes {outcome.predicted_nodes}/{outcome.expected_nodes} "
                f"edges {outcome.predicted_edges}/{outcome.expected_edges} "
                f"{'review' if outcome.needs_review else 'ok':<6} {outcome.seconds:.1f}s",
                flush=True,
            )

    peak: dict[str, object] = {}

    def progress_and_sample(index: int, total: int, outcome) -> None:
        progress(index, total, outcome)
        # Sampled mid-run rather than only at the end: the model is resident then, so this is
        # what actually measures VRAM and CPU offload.
        if index == min(3, total) or index == total // 2:
            snapshot = resource_snapshot()
            if not peak or int(snapshot.get("vram_used_mib") or 0) > int(peak.get("vram_used_mib") or 0):
                peak.clear()
                peak.update(snapshot)

    report = evaluate(chosen, extractor, corpus_id=CORPUS_ID, split=args.split, on_progress=progress_and_sample)
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
        "7_acceptance_gate": "7. Acceptance gate - does it separate right from wrong?",
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

    after = resource_snapshot()
    print("\nresources")
    for name, snapshot in (("before", before), ("during", peak or after), ("after", after)):
        vram = snapshot.get("vram_used_mib")
        models = snapshot.get("loaded_models")
        detail = ""
        if isinstance(models, list) and models:
            entry = models[0]
            detail = (
                f"  {entry['name']} {entry['size_gb']} GB, {entry['size_vram_gb']} GB in VRAM, "
                f"cpu offload {entry['cpu_offload_percent']}%"
            )
        print(f"  {name:<7} VRAM {vram} MiB   RAM {snapshot.get('ram_used_gb')} GB{detail}")

    out = reports_dir()
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    tag = f"-s{len(chosen)}" if args.stratified else ""
    path = out / f"flowlearn-{args.split}-{args.model.replace(':', '_')}{tag}-{stamp}.json"
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
                    "num_ctx": args.num_ctx,
                    "temperature": 0.0,
                    "think": False,
                    "sampling": "stratified" if args.stratified else "split_order",
                    "flowcharts_scored": len(chosen),
                    "flowchart_ids": [record.flowchart_id for record in chosen],
                    "resources": {"before": before, "during": peak or after, "after": after},
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
    parser.add_argument("--num-ctx", type=int, default=DEFAULT_NUM_CTX, help="context window")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N of the split")
    parser.add_argument(
        "--stratified",
        type=int,
        default=0,
        help="score a fixed N spread evenly over node counts. Use this to compare two models.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    handlers = {"fetch": command_fetch, "split": command_split, "eval": command_eval}
    return handlers[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
