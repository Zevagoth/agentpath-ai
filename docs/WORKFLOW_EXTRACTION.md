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
