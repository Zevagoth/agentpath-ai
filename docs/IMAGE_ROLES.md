# Image roles: two kinds of figure, two different jobs

Stage 2B1's headline image number — Image Recall@5 of 40.7% against an 80% target — was
measuring one task when a manual actually contains two, and conflating them made the
product problem look worse than it is.

| Kind | What it is | What the system should do |
| --- | --- | --- |
| **UI screenshot** | An interface with an arrow showing where to press | Show **all** of the procedure's figures, in order. Never rank. |
| **Workflow diagram** | A decision flowchart with questions, branches, arrows | Understand it **once**, at ingestion, so the agent is told their path |

Ranking is the wrong tool for the first and the only tool for the second. The rest of this
document is what was built, what was measured, and the honest finding that the benchmark
corpus contains none of the second kind.

## The role schema

`agentpath_ingestion.corpora.roles` assigns every image a role, written to
`roles.jsonl` in the bundle and read back by the API into `CorpusImageRecord.role`.

```
image_id, document_id
role         = workflow_diagram | ui_screenshot | other
method       = corpus_metadata | vision_model | prefilter
confidence, reason, warnings[]
width, height, has_arrows, has_branches, visible_text, model, seconds
```

`method` matters more than `role`. **`prefilter` means no model examined the image** — it is
not a claim that the image is not a diagram, it is a statement that nothing checked. The API
exposes it so a UI cannot render a role badge that overstates what is known.

### Why the vision model runs on 67 images and not 2,599

The only decision the model's answer changes is whether an image goes to workflow
extraction. UI screenshots and `other` are displayed identically, so paying 2,599 model
calls to separate two roles that behave the same would spend an hour to change nothing.

A cheap prefilter picks the candidates on two signals that fail in different directions:

- **Geometry** — a decision diagram is page-scale. Parsed from JPEG headers in pure Python
  (Pillow is not installed); 86 of the corpus's images are not parseable and are simply not
  size-filtered.
- **Vocabulary near the marker** — "flowchart", "chart", "diagram", or a question with
  yes/no branches.

Vocabulary alone is not enough: 155 windows mention flow vocabulary and exactly one image
turned out to be a diagram, because "troubleshoot" sits beside tables constantly. Size alone
is not enough either, or every product photo would be sent.

**The acknowledged weakness:** a diagram that is both small *and* has no flow vocabulary near
it would be missed. That was probed rather than argued away — see below.

## Measured on MRAMG-Manual

`make corpus-roles` — 2,599 images, 67 candidates, 64 answered, 263 s, no GPU pressure
beyond the 4.7B model already loaded.

| role | count |
| --- | --- |
| `other` | 2,579 |
| `ui_screenshot` | 19 |
| `workflow_diagram` | **1** |

Nine examined images produced warnings worth keeping: eight where the model answered outside
the schema (`table` ×7, `product_photo` ×1) and one where the `has_branches` guard fired.

### The corpus contains no decision workflows

138 of the 2,599 images were examined by the vision model, chosen to be the most likely
candidates in the corpus:

- all **4** whose surrounding text asks a question *and* offers yes/no
- the **60** with flow vocabulary at ≥0.25 MP
- the **15** largest images in the corpus
- **60** drawn at random (seed `20260919`) from the 2,521 the prefilter rejected — this is
  what bounds the prefilter's blind spot

Three were labelled `workflow_diagram`. All three were then opened and looked at by hand:

| Image | What it actually is |
| --- | --- |
| `Manual35_45` | A symptom × check **matrix**: 8 TV symptoms as columns, 9 checks as rows, a filled cell meaning "this check applies". Arrows, but no branching edges, no start, no end. |
| `Manual21_14` | A **linear** keycap assembly illustration with motion arrows. |
| `Manual25_12` | A **linear** numbered CPU-install sequence (steps 4–8). |

None branches. None has a question, an option or a condition. **MRAMG-Manual is a corpus of
appliance manuals, and appliance manuals express troubleshooting as tables, not flowcharts.**

The `has_branches` guard is what keeps this from quietly going wrong: a figure the model
itself says does not branch is recorded as `other` regardless of the label it was given.

### What the extractor did with the one survivor

`Manual35_45` went through `WorkflowExtractor`. The result is the clearest evidence in this
whole task, in both directions:

**The text extraction was excellent.** Every symptom and every check was read correctly and
verbatim — "No power", "Poor picture but good sound", "Is power cord plugged in?", all nine
checks. OCR coverage on this figure is effectively complete.

**The structure extraction was entirely wrong.** The model chained the eight symptoms into a
line, `n1 → n2 → … → n8`, and labelled all eight `start`. The real figure is an 8×9 matrix.
Nothing about that structure is in the picture.

**The validator refused it.** Status `needs_review`, reason: *"no node has more than one
outgoing edge, so nothing branches; this is a linear sequence"*. `usable without review: 0 of
1`, and the original image is what gets shown.

That is the rule working as specified: the image is the source of truth, the model's reading
is not, and a reading that cannot be structurally justified is declined rather than rendered.

## The gallery: the half that does work

`agentpath.corpus_retrieval.gallery` returns **every** figure of the retrieved documents, in
document order, filtered by nothing. It sits beside the ranked selection in
`/api/corpus-search`, and the two mean different things:

- `images` — the ranked **selection**. Which figure is this question about? May be empty. What
  Image Recall is measured on.
- `gallery` — the **complete set**, in order. Recall 1.0 by construction. Paginated, never
  filtered; `omitted_by_cap` is the only way a figure can be absent and it is counted.

### Evaluation A, held-out split, 194 questions

`make gallery-eval`

| metric | measured | target |
| --- | --- | --- |
| Ground-truth document reached (top 5) | 99.0% | — |
| **Gallery image availability** | **98.9%** | 100% |
| Ranked selection coverage @5 | 40.8% | *for comparison* |
| Order preserved | **100%** | 100% |
| Broken image rate | **0%** | 0% |
| Broken source rate | **0%** | 0% |
| Figures linked to a passage | 100% | — |

**The ranking was hiding 58.1% of the wanted figures.** Availability is 98.9% against a
ceiling of 99.0% — a gallery cannot hold figures from a document that was never retrieved, so
it is at 99.9% of what is achievable.

One real bug surfaced here. Order was initially 88.4%, not 100%: MRAMG-Manual carries 2,607
`<PIC>` slots for 2,599 distinct ids, so five documents place the same image at two positions.
The registry holds one record per id with its first position, so walking the raw slot list
returned it again at a position already passed and the sequence ran backwards for 4 of 40
documents. `Corpus.images_for` now sorts on position and shows a reused figure once.

### Step linking, honestly

These are prose manuals, not the step-structured procedure YAML of Stage 2A, so there are no
numbered steps to link to. A figure is linked to the **passage** whose character span contains
it, and to that passage's heading — and `in_matched_passage` says whether retrieval matched
that passage. 100% of shown figures resolve to a passage. A figure that did not resolve would
still be shown: an unlinked correct figure is worth more than a hidden one.

## Limitations

- **No accuracy numbers for workflow extraction.** There is no corpus to measure against. The
  schema, the validator and the refusal path are implemented and tested (58 tests); node,
  edge, direction, branch and needs-review accuracy are unmeasured.
- **Vision classification is not perfectly reproducible** at temperature 0. `Manual35_45`
  reported `has_branches: false` on one run and `true` on another. The guard is therefore a
  filter, not a guarantee.
- **The prefilter's recall is bounded, not known.** 60 random rejected images contained no
  decision workflow; that bounds the miss rate loosely, it does not prove zero.
- **The frontend is not built.** The gallery contract, pagination, role badges, `selected`
  flags, image ids and source links are all in the API response. Zoom, the collapsible
  gallery and the role badge rendering are UI work and are not done.
- **Median 430 figures per answer** when 5 documents are retrieved with a large page size.
  The API default of 24 per document per page is what a client should use; the evaluation
  raised it to measure availability rather than to propose it as a display setting.

## To test workflow extraction properly, a corpus is needed

Two candidates, neither downloaded — approval required first.

**FlowVQA** — 2,272 human-verified flowchart images, 22,413 QA pairs, built specifically for
information localisation, decision-making and logical progression.
[flowvqa.github.io](https://flowvqa.github.io/) ·
[github.com/flowvqa/flowvqa](https://github.com/flowvqa/flowvqa) ·
[paper](https://arxiv.org/abs/2406.19237). The QA pairs are the closest match to the product
task: "given this case, which path applies".

**FlowLearn** — 3,858 flowcharts from scientific literature plus 10,000 script-generated ones,
annotated with visual components, OCR text, **Mermaid code** and VQA pairs.
[github.com/Jo-Pan/FlowLearn](https://github.com/Jo-Pan/FlowLearn) ·
[paper](https://arxiv.org/html/2407.05183v2). The Mermaid representation is node-and-edge
ground truth, which is what node accuracy, edge accuracy and arrow-direction accuracy actually
need — FlowVQA's QA pairs cannot measure those directly.

Both are well under the 2 GB limit for the image subsets, and both would use the existing
`CorpusAdapter` interface and land in their own RAGFlow dataset, so isolation is unaffected.
**Licence terms for each need confirming from the repository before any download** — that check
has not been done, and it is a condition of proceeding, not an afterthought.

Recommendation: **FlowLearn scientific subset for structural accuracy, FlowVQA for path
selection.** If only one, FlowLearn — without node/edge ground truth the structural metrics
this task asks for cannot be computed at all.

*Content of the linked papers and repositories was summarised; see the sources for the
originals.*
