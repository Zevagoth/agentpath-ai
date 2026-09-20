# Can the model read a flowchart? Measured on FlowLearn

Task 9B built a workflow extractor and had nothing to measure it against: MRAMG-Manual
contains no decision workflows (`docs/IMAGE_ROLES.md`). This is the measurement.

**Short answer: no, not well enough to route a case.** `qwen3.5:4b` reads the *boxes* well
(90%) and the *arrows* poorly (65%), and the graph it produces answers a directed
reachability question correctly 56.7% of the time — barely above a coin flip. Nothing in the
product claims otherwise, and the validator refuses 93% of extractions so no unreliable path
is ever shown.

## Licence and the limits on this corpus

**FlowLearn, CC BY-NC-SA 4.0.** [jopan/FlowLearn](https://huggingface.co/datasets/jopan/FlowLearn) ·
Pan, Zhang, Caragea, Dragut and Latecki, *FlowLearn: Evaluating Large Vision-Language Models
on Flowchart Understanding*, ECAI 2024, [arXiv:2407.05183](https://arxiv.org/abs/2407.05183).

- **Non-commercial evaluation and research only.** If AgentPath becomes a commercial project
  this corpus must be removed or separately licensed from the authors.
- **Nothing is committed.** Images and annotations live in `data/external/`, which is
  gitignored. Only aggregate metrics reach this repository.
- **Not used for training or fine-tuning**, and not shipped in the product or any install
  package.
- 600 flowcharts, **16.2 MB** downloaded from the SimFlowchart word test split — 0.5% of the
  3.39 GB archive.

**FlowVQA was excluded entirely.** Its `LICENSE` is MIT, but its README states *"Under
Construction. Please refrain from using the data or code until further notice."* That notice
is unretracted, so neither its data nor its code was downloaded or used.

### Why the simulated half, not the scientific half

The scientific half was inspected first and rejected for this purpose. Its annotations are
OCR boxes, true/false statements and captions — no node, edge or direction ground truth — so
it supports one of the six ordered metrics and not the other five. The simulated half ships
the Mermaid source, which is what makes the structural metrics computable at all.

## Method

**The split was written before any model ran.** Sample seed `20260920` decides which 600
flowcharts were downloaded; split seed `20260919` divides them, stratified on node count
(3–10), giving **302 dev / 298 held-out** balanced within one per stratum. Recorded in
`evaluation/splits/flowlearn-sim.json`. Corpus fingerprint `c8066d6064eeb6af`.

**The ground truth is verified before use.** Our Mermaid parser is checked against the
corpus's own `Num_Nodes` and `Num_Arrows` for every flowchart; a disagreement excludes that
flowchart with the reason recorded, because scoring against a graph *we* mis-parsed would
report our bug as the model's. **600 of 600 parsed cleanly (100%).**

**Alignment is on labels, and two numbers are reported.** The model can only report the words
in a box; the corpus calls that box `entity3`. Strict matching normalises case, punctuation and
whitespace and requires equality. Relaxed matching allows containment with a 4-character floor.
The gap between them is how much of the error is *reading* rather than *structure* — here it is
under half a point, so the structural failures are structural.

Prompt unchanged from the one written for customer-service diagrams. No prompt was tuned
against either split.

## Results — held-out, 120 flowcharts, `qwen3.5:4b`

| # | Metric | Held-out | Dev (30) |
| --- | --- | --- | --- |
| 1 | **Text (OCR) label recall** | **89.9%** | 92.0% |
| 2 | **Node F1** | **89.9%** | 91.7% |
| | node count exactly right | **100%** | 100% |
| | whole node set exactly right | 55.8% | 50.0% |
| 3 | **Edge F1 (undirected)** | **70.4%** | 65.8% |
| | edge count exactly right | 59.2% | 23.3% |
| 4 | **Directed edge F1** | **65.0%** | 54.5% |
| | direction right *among edges found* | 91.0% | 79.4% |
| | corpus ordered-pair probe | **60.0%** | 40.0% |
| 5 | **Branch / decision F1** | **53.1%** | 61.4% |
| 6 | **Path reachability accuracy** | **56.7%** | 53.3% |

Operational: 0 extraction failures, median 9.0 s per flowchart, 1,417 s total, `needs_review`
on 93.3%.

### What the numbers say, in order

**It reads the text.** 89.9% label recall and a node count that is exactly right on
**every single flowchart** — the model sees all the boxes and misreads about one label in ten.

**It loses arrows.** Edge F1 70.4%, and the errors are asymmetric: of 120 flowcharts it
**undercounted edges 42 times and overcounted 7**. It drops connections rather than
inventing them.

**Direction is the interesting one.** Among edges it *did* find, 91.0% point the right way —
that looks good until you notice the corpus's own probe, which asks whether one specific
ordered pair is drawn and another is not, is answered correctly only **60.0%** of the time.
The two are consistent: per-edge direction is mostly right, but a single reversal anywhere
breaks the pair question, and errors compound across a graph.

**Branches are the worst result, and they matter most.** F1 53.1% means about half the
decision points are missed — and a decision point is the entire reason to extract a workflow
rather than show it.

**Paths are a coin flip.** 56.7%. A system routing a customer case on this graph would be
wrong nearly half the time.

**Complexity is not the problem.** Edge F1 by size: 78.1% at 3 nodes, 67.5% at 4, 71.5% at 5,
67.4% at 10. It does not degrade with graph size — it is uniformly mediocre, which points at
arrow perception rather than at reasoning load.

### The validator is blunt, not calibrated — an honest negative result

The 93.3% `needs_review` rate is high enough to be safe. It is **not** well aimed:

| | n | Directed edge F1 | Path reachability |
| --- | --- | --- | --- |
| Sent to review | 112 | 64.2% | 58.0% |
| Passed as usable | 8 | 76.1% | **37.5%** |

The eight extractions the validator *accepted* have **worse** reachability than the 112 it
refused. It is a blunt filter that happens to catch nearly everything, not a signal that
separates good extractions from bad ones. Anyone reading "93% refused" as evidence of a
well-calibrated guard would be reading it wrong.

Two of its rules also fire for reasons specific to this corpus rather than to quality: 84 of
120 tripped "at least one node's purpose could not be determined" and 71 tripped the
confidence floor. FlowLearn's boxes are abstract word pairs with no start/question/action
semantics, so the node-kind requirement cannot be satisfied here. That inflates the refusal
rate but does **not** touch metrics 1–6, which are scored from node text and edges directly.

### Examples

Best: `9168.jpeg` and `8204.jpeg` — 10 of 10 nodes, 10 of 10 edges, 100% directed F1. It can
do this. Worst: `9722.jpeg`, `8427.jpeg`, `9575.jpeg` — every node and every edge found, and
**0% directed F1**: the whole graph read backwards. All six were sent to review.

## Verdict

**The architecture is sound and the model is not good enough.** The pipeline — classify role,
extract, validate, refuse, show the original — works end to end, produced no failures over
120 flowcharts, and never showed an unvalidated path. What it cannot do is read arrows
reliably enough for the extracted graph to be trusted.

Against the rule that no workflow-understanding success may be claimed without held-out
numbers: **no such claim is made.** Branch detection at 53% and path reachability at 57% are
not a working feature.

### Behaviour this implies, and which is implemented

- A workflow diagram is **always shown as the original image**.
- A failed or unvalidated extraction shows **`Needs manual review`** with the reason, and no
  inferred path.
- A path summary appears above the image **only** when the validator passed the extraction —
  which on this evidence will be rare, and should be.

## Limitations

- 120 of 298 held-out flowcharts were scored. Standard error on a proportion near 55% is about
  4.5 points, so the differences between metrics 3–6 are real but their absolute values are
  ±5.
- Extraction is not perfectly reproducible at temperature 0; repeated runs differ slightly.
- FlowLearn's charts are synthetic, with random word-pair labels and no semantic roles. A real
  customer-service decision chart has meaningful text that a model can use as a prior, so
  these numbers are plausibly a **lower** bound for the real task and a fair upper bound for
  pure arrow perception.
- `qwen3.5:4b` only. A larger model has not been tried; this report is the evidence for
  deciding whether to.

*Content from the linked paper and dataset card was summarised for this document; see the
sources for the originals.*

---

# Comparison: `qwen3.5:4b` against `qwen3.5:9b`

A limited comparison, run on dev only, on the **same 40 flowcharts** with the same prompt, the
same schema, the same metrics and the same settings (`temperature=0`, `think=false`,
`num_ctx=8192`). The 40 are a fixed sample spread evenly over node count, so neither model
gets an easier draw. 4B was unloaded from VRAM before 9B ran; the two were never resident
together.

## Result: 9B is better at structure and still fails the bar

| # | Metric | `qwen3.5:4b` | `qwen3.5:9b` | Δ |
| --- | --- | --- | --- | --- |
| 1 | Text (OCR) label recall | 89.2% | **93.8%** | +4.6 |
| 2 | Node F1 | 89.2% | **93.8%** | +4.6 |
| | whole node set exactly right | 45.0% | **72.5%** | +27.5 |
| | node count exactly right | 100% | 100% | — |
| 3 | Edge F1 (undirected) | 68.9% | **79.7%** | +10.8 |
| 4 | Directed edge F1 | 63.5% | **77.1%** | +13.6 |
| | direction right among edges found | 92.9% | **96.4%** | +3.5 |
| | corpus ordered-pair probe | 60.0% | 62.5% | +2.5 |
| 5 | **Branch / decision F1** | 58.4% | **69.6%** | **+11.2** |
| 6 | **Path reachability** | 52.5% | **55.0%** | **+2.5** |

### Against the promotion criteria

| Criterion | Required | Measured | |
| --- | --- | --- | --- |
| Branch F1 gain | ≥ +15 points | **+11.2** | **fails** |
| Path reachability gain | ≥ +15 points | **+2.5** | **fails** |
| No material OCR / Node F1 regression | no regression | +4.6 / +4.6 | passes |
| No wrong path accepted as usable | zero | **0** | passes |

**Two of four fail, including the one that decides the feature.** Path reachability moved 2.5
points and sits at 55.0% — still a coin flip. So 9B is **not promising** by the stated bar, and
per the instruction no quantizations or further models were tried and 9B was **not run on
held-out**.

### The interesting part: why structure improved and paths did not

9B is clearly the better reader. It gets the whole node set exactly right on 72.5% of
flowcharts against 45.0%, and its per-edge direction accuracy is 96.4%.

And yet path reachability barely moves. The reason is compounding: reachability needs *every*
edge along the relevant path to be right, so a per-edge accuracy of 96.4% over a 7-edge median
graph still leaves a large chance of at least one break, and one break changes the answer.
Per-edge gains do not multiply into whole-graph correctness — which is why per-edge numbers are
the wrong thing to promote a feature on, and why metric 6 exists.

This is also evidence about *what kind* of improvement would help. Scaling the general model
2.4× bought +13.6 on directed edges and +2.5 on paths. Reaching a usable path accuracy this way
would need a model far larger than this hardware can hold, which points at a different approach
- a diagram-specific model, or an intermediate structured text representation - rather than at
more parameters.

### The acceptance gate under 9B

| | `qwen3.5:4b` | `qwen3.5:9b` |
| --- | --- | --- |
| Accepted as usable | 5 | 1 |
| **Falsely accepted (wrong path shown as usable)** | **4** | **0** |
| False-accept rate of accepted | **80.0%** | 0.0% |
| Reachability when accepted | 20.0% | 100% |
| Reachability when refused | 57.1% | 53.8% |
| Gate separates right from wrong | **No** | Yes |

The 4B row is the worst number in this whole report: of five extractions the validator passed
as usable, **four had the wrong path**, and accepted extractions were *less* accurate than
refused ones. The gate was actively anti-correlated with correctness.

9B's row looks clean but rests on **a single accepted extraction out of 40**, which is not
evidence of a calibrated gate — it is a gate that refused 97.5% and happened to be right about
the one it let through. Neither model justifies trusting the acceptance decision.

### Cost

| | `qwen3.5:4b` | `qwen3.5:9b` |
| --- | --- | --- |
| Median seconds per flowchart | 13.0 | **40.5** (3.1×) |
| Peak VRAM during run | 6,615 MiB | 7,208 MiB |
| Model resident in VRAM | 3.27 of 3.27 GB | **3.6 of 6.28 GB** |
| **CPU offload** | **0%** | **42.7%** |
| Peak RAM | 19.8 GB | 20.6 GB |

9B does not fit in 8 GB of VRAM beside the embedder: 42.7% of it runs on the CPU, which is
where the 3.1× latency comes from. A 40-second extraction is acceptable at ingestion time and
would not be at request time.

## Decision

**`qwen3.5:4b` stays as the product model.** 9B was a comparison, not a candidate, and it was
not promoted. Workflow extraction is **not enabled automatically** anywhere in the product.

The shipped behaviour for a workflow diagram is therefore:

1. **Always show the original image.** It is the source of truth.
2. **`Needs manual review`**, with the validator's reasons, whenever the extraction did not
   pass — which on this evidence is 87.5% to 97.5% of the time.
3. **Never show an inferred path** that the validator did not accept.

A path summary above the image remains implemented and remains gated on acceptance. Given that
the gate does not demonstrably separate correct from wrong, it stays off by default and no
automatic routing is claimed.

### What would change this

Path reachability well above 90%, *and* an acceptance gate whose accepted set measurably beats
its refused set on a sample large enough to mean something. Neither model reaches either.
