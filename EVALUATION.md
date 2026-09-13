# Evaluation

How AgentPath AI is measured, what the current numbers are, and what they do not
prove. Regenerate everything with:

```bash
make api      # in one terminal, so the full HTTP path can be measured
make eval     # writes JSON + Markdown into evaluation/reports/
```

> All knowledge-base content is **fictional and synthetic**.

## Method

### Two populations, scored separately

`evaluation/queries.yaml` holds **30 queries**:

| Population | Count | What "correct" means |
| --- | --- | --- |
| **supported** | 26 | The expected procedure is retrieved. Scored with Recall@1, Recall@3, MRR. |
| **unsupported** | 4 | Nothing in the knowledge base answers it, so the system must abstain. Scored with abstention accuracy. |

Merging the two into one accuracy figure would hide the behaviour that matters
most: refusing to answer is correct for one population and a failure for the
other. So:

- **Recall@k, MRR** are computed over supported queries only.
- **Abstention accuracy** is computed over unsupported queries only.
- **False abstention rate** is the share of *supported* queries the system refused
  to answer. It is reported next to recall on purpose: without it, a system could
  score well on abstention by refusing everything.

A supported query may list more than one acceptable procedure when a case is
genuinely served by either; matching any one of them counts as a hit.

### Query mix

| Type | n | Stresses |
| --- | --- | --- |
| `exact_title` | 5 | The easy case: official wording |
| `synonym` | 6 | What agents actually type |
| `arabic` | 6 | Arabic query against an English procedure |
| `mixed` | 1 | Arabic and English in one query |
| `typo` | 3 | Misspellings |
| `abbreviation` | 2 | Agent shorthand (`nonref`, `cxl`, `dup bkg`) |
| `vague` | 2 | Under-specified but answerable |
| `procedure_id` | 1 | Lookup by reference |
| `unsupported` | 4 | No answer exists |

### Latency

Each query gets one **warm-up run whose latency is discarded**, then 3 measured
runs. Percentiles are **nearest-rank**, not interpolated, so every reported
percentile is a latency some request actually experienced.

### Three modes

| Mode | What it is | Why it is here |
| --- | --- | --- |
| `lexical_baseline` | Keyword overlap over the procedure corpus. No embeddings, no lexicon, no typo correction. In process. | The reference point: what a conventional keyword index does. |
| `ragflow_hybrid` | RAGFlow hybrid retrieval (BM25 + vector) on the **raw** query. | Isolates what the engine contributes on its own. |
| `agentpath_full` | The shipping path over HTTP: query understanding, hybrid retrieval, aggregation, evidence grading. | The product. Measured through `POST /api/search`, so latency includes the hop an agent pays. |

Comparing `ragflow_hybrid` with `agentpath_full` is the only way to say whether
our own layer earns its place.

## Results

Reranking **disabled** (`AGENTPATH_ENABLE_RERANK=false`); see
[the reranking note](#reranking-is-off-and-why) below.

| Mode | Recall@1 | Recall@3 | MRR | Abstention | False abstention | p50 | p95 | mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical_baseline` | 85% | 92% | 0.885 | 50% | 8% | <1 ms | <1 ms | <1 ms |
| `ragflow_hybrid` | 92% | 96% | 0.942 | 100% | 12% | 114 ms | 154 ms | 114 ms |
| `agentpath_full` | **100%** | **100%** | **1.000** | **100%** | **0%** | 136 ms | 169 ms | 135 ms |

Recall, MRR and false abstention are over the 26 supported queries; abstention
accuracy is over the 4 unsupported ones.

### What each layer adds

**Keyword search → engine.** The baseline cannot handle misspellings at all:
`overboking at the hotle` and `dupliacte bookign for same guset` return *nothing*,
because no index term matches. Hybrid retrieval recovers one of the two through
vector similarity.

**Engine → full pipeline.** Two effects, both measured:

1. `overboking at the hotle` still returns nothing from the raw engine. Through
   `/api/search` the lexicon corrects both words and the correct procedure comes
   back first, in 136 ms. This single query is the clearest evidence in the whole
   benchmark that the query-understanding layer is doing real work.
2. False abstention drops from 12% to 0%. On raw queries, three supported cases
   score below the abstention threshold and get refused; after normalization,
   synonym expansion and typo correction, their scores rise above it. The engine
   alone is not just less accurate, it is more likely to give an agent nothing.

**Arabic, stated carefully.** All 6 Arabic queries resolve to the right English
procedure in the full pipeline — but so do they in the lexical baseline, which
scores Recall@1 100% on that group too. The reason is the dataset, not the
retrieval method: every procedure is authored with its Arabic synonyms, so those
Arabic strings are literally in the indexed text and plain term overlap finds them.

Claiming "keyword search cannot handle Arabic" would therefore be false here. What
this benchmark does show about Arabic is narrower and worth keeping separate:

- The **mixed-language** query (`overbooking بالفندق والضيف موجود`) is ranked
  correctly at position 1 only by the full pipeline; the baseline and the raw engine
  both put the wrong hotel procedure first (Recall@1 0%, Recall@3 100%).
- Arabic **spelling variation** is handled by normalization rather than by the
  index: `إلغاء`, `الغاء`, `إلغــاء` and `إِلْغَاء` all collapse to one form before
  matching, which is what keeps Arabic recall from depending on how the agent typed
  the word.

Whether semantic retrieval is what carries Arabic in a knowledge base where Arabic
synonyms are *not* hand-authored is a reasonable expectation, but this benchmark
does not test it. Doing so needs a procedure set written English-only, which is a
Stage 2 experiment.

## Abstention, and the threshold behind it

The evidence thresholds are:

| Level | Rule | Meaning shown to the agent |
| --- | --- | --- |
| `strong` | top score ≥ 0.60 and ≥ 0.05 clear of the runner-up | One procedure is clearly the match |
| `partial` | top score ≥ 0.51 | Relevant, but verify before acting |
| `insufficient` | below 0.51 | Abstain: "No confirmed procedure was found in the available knowledge base." |

**How 0.51 was chosen.** From the observed score distribution on this query set:

- lowest-scoring **supported** query: **0.526** (a vague case)
- highest-scoring **unsupported** query: **0.493** ("passenger name correction on
  a flight ticket", a deliberate near-miss)

0.51 sits in the gap. **The gap is 0.034 wide.** That is thin, and it was measured
on the same 30 queries this report quotes, which is a genuine methodological
weakness: the threshold is fitted to the set it is evaluated on. Treat the 100%
abstention accuracy as "separable on this set", not as a general guarantee.

**Signals that did not work**, recorded so they are not retried blindly:

- **Lexical (BM25) grounding.** The hope was that a semantic false positive would
  have no term overlap. Measured: an answerable but vague query scored a lexical
  0.085, while an unanswerable one scored 0.083. No separation.
- **Lexicon concept matches.** Requiring a recognised concept would abstain on
  genuine vague queries and on reference-number lookups, both of which match no
  concept either.

**What to do about it in Stage 2**, once there are 12+ procedures and 50+ queries:
split the set, tune on one half, report on the other, and revisit whether a
learned combination of signals separates the populations more robustly than a
single threshold.

## Reranking is off, and why

`bge-reranker-v2-m3` is registered in RAGFlow and the code path is live behind
`AGENTPATH_ENABLE_RERANK`. It is off because:

- On **CPU** it cost **20-22 s per query** for ~30 candidates, against 0.2 s for
  hybrid retrieval alone.
- On **GPU** it does not start: an unresolved CUDA initialization failure inside
  Docker (`CUDA_ERROR_UNKNOWN`), reproduced with two different image tags, while
  `nvidia-smi` works inside the same container and host-side CUDA works. See
  [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

With five procedures the pipeline reaches 100% Recall@1 without it, so nothing is
blocked. That is **not** evidence that reranking is unnecessary in general: with
hundreds of procedures, more candidates will be near-ties and the cross-encoder is
exactly the tool for that. The flag exists so it can be switched on and measured
the moment a working runtime is available.

## Targets

| Target | Status |
| --- | --- |
| Correct procedure in the top 3 for ≥ 90% of the benchmark | **Met** (100% for the full pipeline; 96% for the raw engine) |
| First results within ~2 s | **Met** (p95 169 ms end to end, including the HTTP hop) |
| Unsupported questions trigger abstention | **Met on this set** (4/4), with the threshold caveat above |
| Every procedural recommendation carries citations | **Met** - the API cannot return a procedure without its citations; enforced by test |
| Complete concise answers within ~10 s | Not applicable yet: answer generation is Stage 2 |

## Honest limitations

1. **Corpus size.** Five procedures. Retrieval over five documents is a much
   easier problem than over five hundred; these numbers will move.
2. **The threshold is fitted to the reported set.** See above.
3. **Single machine, single run.** One host (RTX 4060, 20-thread CPU), 3 repeats
   per query. No concurrency testing, so p95 under load is unknown.
4. **The query set is written by the same person who wrote the procedures.** That
   biases phrasing toward what the corpus contains. Queries from real agents would
   be harsher.
5. **Arabic recall is helped by bilingual authoring.** Because Arabic synonyms are
   written into every procedure, even keyword overlap finds them, so this set does
   not isolate what cross-lingual embeddings contribute. A procedure set authored
   English-only would.
6. **No citation-correctness scoring yet.** The API guarantees citations *exist*
   and point at the retrieved chunks; whether each cited chunk truly supports a
   specific claim only becomes measurable once answer generation exists.
7. **Reranking unmeasured on GPU**, so the accuracy/latency trade-off of the full
   architecture is not yet known.

## Files

| Path | Contents |
| --- | --- |
| `evaluation/queries.yaml` | The query set, typed and with expectations |
| `evaluation/agentpath_eval/baseline.py` | Lexical baseline (benchmark only) |
| `evaluation/agentpath_eval/modes.py` | The three measured modes |
| `evaluation/agentpath_eval/metrics.py` | Metric definitions |
| `evaluation/reports/latest.md` | Newest run, human readable |
| `evaluation/reports/benchmark-*.json` | Newest run, per-query detail |
| `scripts/measure_latency.py` | Engine-only latency probe |
| `scripts/inspect_benchmark.py` | Prints misses and abstention behaviour from a report |
