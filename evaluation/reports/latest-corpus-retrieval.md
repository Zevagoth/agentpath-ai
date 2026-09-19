# External corpus retrieval evaluation - MRAMG-Bench - Manual

> This corpus is **not** ours. It is an external benchmark, used unmodified, to test whether the architecture works on a corpus we did not author.

**Attribution.** MRAMG-Bench (MRAMG/MRAMG-Bench on Hugging Face), Manual subset. Licensed Apache-2.0. Text, images, questions and ground truth are the benchmark authors' work and are used unmodified.

## What was measured

- Corpus: `mramg-manual` - 40 documents, 1506 chunks, 2599 images, 390 questions.
- Retrieval dataset: `agentpath-mramg-manual`. One dataset per corpus, and a retrieval call names exactly one, so results from two corpora cannot be mixed.
- Split: seed `20260919`, 50% held out - 196 dev and 194 held-out questions, stratified on how many images the answer needs. Recorded in `evaluation/splits/mramg-manual.json`.
- **Nothing was tuned on the held-out split.** The split was frozen and committed before any of these numbers existed.
- Documents requested per question: 5. Images: 5.
- Latency: 1 measured run per question, after 1 discarded warm-up query per method. Percentiles are nearest-rank, not interpolated, and cover the retrieval engine only - image selection is timed separately.

### Image retrieval is text retrieval

The image retrieval mode is **context-window**. No model on any path here reads a pixel. Images are ranked by the document text surrounding their `<PIC>` marker (600 characters before, 300 after), because every one of this corpus's image captions is an empty string. See `docs/RETRIEVAL_MODES.md`. Calling this multimodal would be a claim about the wrong thing: the *result* is text and images together, the *method* is text.

### Configuration

| Setting | Value |
| --- | --- |
| Embedding model | `BAAI/bge-m3@agentpath-embedder@HuggingFace` |
| Vector similarity weight | 0.6 (term weight 0.4) |
| Similarity threshold | 0.15 |
| Reranker | disabled for every method (measured OOM-killed on this host, exit 137) |
| Generated | `2026-09-19T20:33:06+00:00` |
| Host | Linux 7.2.6-1-cachyos, Python 3.12.13 |

## Methods compared

| Method | Documents ranked by | Images chosen by |
| --- | --- | --- |
| `lexical` | in-process BM25 over the exported chunks. No RAGFlow, no embeddings, no model. | document order (positional) |
| `hybrid` | RAGFlow: BM25 blended with `bge-m3` vector similarity. | document order (positional) |
| `multimodal` | the same RAGFlow retrieval, sharing one retriever instance with `hybrid`. | context-window ranking |

`hybrid` and `multimodal` share one retriever object, so any difference between their image numbers is attributable to image ranking and nothing else. `lexical` exists so the other two are answerable: "92% recall" means nothing without knowing what keyword matching alone reaches.

### Metric definitions

- **Document Recall@k** - share of questions whose provenance document is in the top k.
- **Document MRR** - mean of `1 / rank` of the first correct document; 0 when absent.
- **Image Recall@k** - *coverage*: mean of `|truth ∩ top k| / |truth|`. A question wanting three images and getting one scores 0.33. This is what the target refers to.
- **Image Hit@k** - mean of "at least one correct image in the top k". Always ≥ Recall@k and much more forgiving on multi-image questions; reported so the two are not confused.
- **Image precision / F1** - over the images actually returned, averaged across questions that expect images and received some.
- **Image abstention** - share of the questions expecting *no* image for which none was returned. Excluded from every other image metric, since recall over an empty truth set is undefined.
- **Citation validity** - share of returned documents whose provenance names its corpus and document and offers a link that resolves.
- **Broken source / image rate** - citations with no resolvable link; images whose bytes are not on disk, checked per image with a filesystem call.
- **Unresolved offsets** - passages whose position in the document could not be recovered. Those citations open the document at the top rather than at a guessed offset.

## Dev split

196 questions - 176 expecting at least one image, 20 expecting none. Decisions were allowed to be informed by these numbers.

### Documents

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | No document returned |
| --- | --- | --- | --- | --- | --- |
| `lexical` | 79.6% | 94.9% | 96.9% | 0.869 | 0.0% |
| `hybrid` | 92.9% | 99.0% | 99.0% | 0.959 | 0.0% |
| `multimodal` | 92.9% | 99.0% | 99.0% | 0.959 | 0.0% |

### Images

| Method | Chosen by | Recall@1 | Recall@3 | Recall@5 | Hit@5 | Precision | F1 | Abstained when none wanted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical` | document-order | 1.5% | 3.5% | 6.8% | 7.4% | 3.3% | 0.044 | 0.0% |
| `hybrid` | document-order | 1.7% | 4.9% | 9.0% | 9.7% | 4.3% | 0.058 | 0.0% |
| `multimodal` | context-window | 12.0% | 33.3% | 44.5% | 55.1% | 22.2% | 0.296 | 0.0% |

### Provenance and latency

| Method | Citation validity | Broken sources | Broken images | Unresolved offsets | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical` | 100.0% | 0.0% | 0.0% (0) | 0.0% | 3 ms | 5 ms | 15 ms |
| `hybrid` | 100.0% | 0.0% | 0.0% (0) | 0.0% | 173 ms | 275 ms | 328 ms |
| `multimodal` | 100.0% | 0.0% | 0.0% (0) | 0.0% | 177 ms | 278 ms | 367 ms |

#### How citation offsets were established

| Method | Offset sources |
| --- | --- |
| `lexical` | bundle 5655 |
| `hybrid` | exact_match 2667, normalised_match 1890 |
| `multimodal` | exact_match 2667, normalised_match 1890 |

`bundle` and `exact_match` are exact. `normalised_match` located the passage after collapsing whitespace, so the span is right and its edges can be off by the whitespace that was collapsed. `unresolved` means no offset was reported at all.

#### By how many images the answer needs

| Method | images=0 | images=1 | images=2 | images=3 | images=4 | images=5 | images=6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical` | D 95% | D 96% / I 9% | D 94% / I 3% | D 95% / I 13% | D 100% / I 0% | D 100% / I 7% | D 80% / I 17% |
| `hybrid` | D 95% | D 100% / I 13% | D 98% / I 3% | D 100% / I 15% | D 100% / I 5% | D 100% / I 7% | D 100% / I 17% |
| `multimodal` | D 95% | D 100% / I 61% | D 98% / I 38% | D 100% / I 42% | D 100% / I 38% | D 100% / I 33% | D 100% / I 27% |

D = Document Recall@3, I = Image Recall@5 within that stratum.

## Held-out split

194 questions - 174 expecting at least one image, 20 expecting none. **These numbers informed no decision.** Nothing was tuned against them.

### Documents

| Method | Recall@1 | Recall@3 | Recall@5 | MRR | No document returned |
| --- | --- | --- | --- | --- | --- |
| `lexical` | 79.9% | 92.8% | 94.8% | 0.859 | 0.0% |
| `hybrid` | 86.6% | 98.5% | 99.0% | 0.921 | 0.0% |
| `multimodal` | 86.6% | 98.5% | 99.0% | 0.921 | 0.0% |

### Images

| Method | Chosen by | Recall@1 | Recall@3 | Recall@5 | Hit@5 | Precision | F1 | Abstained when none wanted |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical` | document-order | 0.1% | 1.2% | 2.9% | 4.0% | 1.6% | 0.021 | 0.0% |
| `hybrid` | document-order | 0.8% | 2.3% | 4.6% | 5.7% | 2.3% | 0.031 | 0.0% |
| `multimodal` | context-window | 9.9% | 30.1% | 40.7% | 50.0% | 18.7% | 0.256 | 0.0% |

### Provenance and latency

| Method | Citation validity | Broken sources | Broken images | Unresolved offsets | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical` | 100.0% | 0.0% | 0.0% (0) | 0.0% | 3 ms | 5 ms | 7 ms |
| `hybrid` | 100.0% | 0.0% | 0.0% (0) | 0.0% | 172 ms | 273 ms | 336 ms |
| `multimodal` | 100.0% | 0.0% | 0.0% (0) | 0.0% | 177 ms | 272 ms | 338 ms |

#### How citation offsets were established

| Method | Offset sources |
| --- | --- |
| `lexical` | bundle 5351 |
| `hybrid` | exact_match 2643, normalised_match 1793 |
| `multimodal` | exact_match 2643, normalised_match 1793 |

`bundle` and `exact_match` are exact. `normalised_match` located the passage after collapsing whitespace, so the span is right and its edges can be off by the whitespace that was collapsed. `unresolved` means no offset was reported at all.

#### By how many images the answer needs

| Method | images=0 | images=1 | images=2 | images=3 | images=4 | images=5 | images=6 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `lexical` | D 95% | D 91% / I 2% | D 94% / I 3% | D 92% / I 3% | D 94% / I 0% | D 83% / I 0% | D 100% / I 25% |
| `hybrid` | D 100% | D 96% / I 7% | D 100% / I 3% | D 97% / I 3% | D 100% / I 6% | D 100% / I 0% | D 100% / I 25% |
| `multimodal` | D 100% | D 96% / I 50% | D 100% / I 44% | D 97% / I 35% | D 100% / I 28% | D 100% / I 23% | D 100% / I 12% |

D = Document Recall@3, I = Image Recall@5 within that stratum.

## Targets

Stated before the run: Document Recall@3 ≥ 90%, Image Recall@5 ≥ 80%, citation validity 100%, zero broken sources and images, retrieval p95 < 2 s. A miss is reported as a miss with its real number.

| Split | Method | Target | Measured | Required | Met |
| --- | --- | --- | --- | --- | --- |
| dev | `lexical` | document_recall_at_3 | 94.9% | at least 90% | yes |
| dev | `lexical` | image_recall_at_5 | 6.8% | at least 80% | **no** |
| dev | `lexical` | citation_validity | 100.0% | at least 100% | yes |
| dev | `lexical` | broken_source_rate | 0.0% | at most 0% | yes |
| dev | `lexical` | broken_image_rate | 0.0% | at most 0% | yes |
| dev | `lexical` | retrieval_p95_ms | 5.012 ms | at most 2000.0 ms | yes |
| dev | `hybrid` | document_recall_at_3 | 99.0% | at least 90% | yes |
| dev | `hybrid` | image_recall_at_5 | 9.0% | at least 80% | **no** |
| dev | `hybrid` | citation_validity | 100.0% | at least 100% | yes |
| dev | `hybrid` | broken_source_rate | 0.0% | at most 0% | yes |
| dev | `hybrid` | broken_image_rate | 0.0% | at most 0% | yes |
| dev | `hybrid` | retrieval_p95_ms | 274.5845 ms | at most 2000.0 ms | yes |
| dev | `multimodal` | document_recall_at_3 | 99.0% | at least 90% | yes |
| dev | `multimodal` | image_recall_at_5 | 44.5% | at least 80% | **no** |
| dev | `multimodal` | citation_validity | 100.0% | at least 100% | yes |
| dev | `multimodal` | broken_source_rate | 0.0% | at most 0% | yes |
| dev | `multimodal` | broken_image_rate | 0.0% | at most 0% | yes |
| dev | `multimodal` | retrieval_p95_ms | 278.4508 ms | at most 2000.0 ms | yes |
| held_out | `lexical` | document_recall_at_3 | 92.8% | at least 90% | yes |
| held_out | `lexical` | image_recall_at_5 | 2.9% | at least 80% | **no** |
| held_out | `lexical` | citation_validity | 100.0% | at least 100% | yes |
| held_out | `lexical` | broken_source_rate | 0.0% | at most 0% | yes |
| held_out | `lexical` | broken_image_rate | 0.0% | at most 0% | yes |
| held_out | `lexical` | retrieval_p95_ms | 5.0849 ms | at most 2000.0 ms | yes |
| held_out | `hybrid` | document_recall_at_3 | 98.5% | at least 90% | yes |
| held_out | `hybrid` | image_recall_at_5 | 4.6% | at least 80% | **no** |
| held_out | `hybrid` | citation_validity | 100.0% | at least 100% | yes |
| held_out | `hybrid` | broken_source_rate | 0.0% | at most 0% | yes |
| held_out | `hybrid` | broken_image_rate | 0.0% | at most 0% | yes |
| held_out | `hybrid` | retrieval_p95_ms | 272.8359 ms | at most 2000.0 ms | yes |
| held_out | `multimodal` | document_recall_at_3 | 98.5% | at least 90% | yes |
| held_out | `multimodal` | image_recall_at_5 | 40.8% | at least 80% | **no** |
| held_out | `multimodal` | citation_validity | 100.0% | at least 100% | yes |
| held_out | `multimodal` | broken_source_rate | 0.0% | at most 0% | yes |
| held_out | `multimodal` | broken_image_rate | 0.0% | at most 0% | yes |
| held_out | `multimodal` | retrieval_p95_ms | 271.8533 ms | at most 2000.0 ms | yes |

## Where it failed

Questions whose provenance document was not in the top 3, worst first. Listed rather than summarised, because a rate says how often and only the cases say why.

### `lexical` on dev - 10 of 196

| Question | Wanted | Got | RR |
| --- | --- | --- | --- |
| MANUAL_119: From three different observing views, what buttons and interfaces should this eReader have | 50017 | 50010, 50037, 50009 | 0.00 |
| MANUAL_21: What modes do air purifiers usually have? How do you set them up? What features do these m | 50007 | 50009, 50018, 50030 | 0.00 |
| MANUAL_346: What safety precautions should be followed to ensure safe operation during this process? | 50022 | 50016, 50013, 50014 | 0.00 |
| MANUAL_376: What are the proper steps to clean and maintain a washing machine to ensure optimal perfor | 50029 | 50002, 50036, 50005 | 0.00 |
| MANUAL_58: How the ship steers? | 50009 | 50033, 50025, 50031 | 0.00 |
| MANUAL_60: If I want to listen to music on my phone, how do I turn on the sound system? | 50009 | 50035, 50014, 50022 | 0.00 |
| MANUAL_127: How can I know the specifications of this exercise bikes? | 50010 | 50021, 50015, 50001 | 0.20 |
| MANUAL_104: If I want to charge my drill, what steps should I follow? | 50012 | 50035, 50018, 50001 | 0.25 |
| MANUAL_317: What is the correct procedure for cleaning the vacuum cleaner filter? | 50025 | 50002, 50007, 50003 | 0.25 |
| MANUAL_323: What is the correct procedure for cleaning the side brush of a vacuum cleaner? | 50025 | 50007, 50003, 50002 | 0.25 |

### `hybrid` on dev - 2 of 196

| Question | Wanted | Got | RR |
| --- | --- | --- | --- |
| MANUAL_346: What safety precautions should be followed to ensure safe operation during this process? | 50022 | 50006, 50036, 50000 | 0.00 |
| MANUAL_60: If I want to listen to music on my phone, how do I turn on the sound system? | 50009 | 50022, 50035, 50014 | 0.00 |

### `multimodal` on dev - 2 of 196

| Question | Wanted | Got | RR |
| --- | --- | --- | --- |
| MANUAL_346: What safety precautions should be followed to ensure safe operation during this process? | 50022 | 50006, 50036, 50000 | 0.00 |
| MANUAL_60: If I want to listen to music on my phone, how do I turn on the sound system? | 50009 | 50022, 50035, 50014 | 0.00 |

### `lexical` on held_out - 14 of 194

| Question | Wanted | Got | RR |
| --- | --- | --- | --- |
| MANUAL_0: What are the important components in the air conditiioner? | 50002 | 50030, 50039, 50023 | 0.00 |
| MANUAL_110: After purchasing the drill, what are the accessories I should have? | 50012 | 50030, 50035, 50021 | 0.00 |
| MANUAL_140: I'm about to move this fax. Is there anything I should pay attention to? | 50013 | 50009, 50025, 50001 | 0.00 |
| MANUAL_176: What are the last three steps to start the engine from the manual? | 50018 | 50005, 50009, 50032 | 0.00 |
| MANUAL_181: I just made mistakes on air filter of the geenrator. Please tell me the right steps. | 50018 | 50039, 50007, 50001 | 0.00 |
| MANUAL_195: I'd like to remove and install the seat of my jetski, what steps should I follow? | 50011 | 50035, 50018, 50036 | 0.00 |
| MANUAL_27: What should you do if need long term storage? | 50007 | 50012, 50020, 50034 | 0.00 |
| MANUAL_309: What are the proper steps to safely stop the engine of a pump? | 50023 | 50009, 50024, 50020 | 0.00 |
| MANUAL_345: What are the key factors to check when troubleshooting technical issues or malfunctions? | 50022 | 50014, 50024, 50017 | 0.00 |
| MANUAL_37: What are the parts of a dishwaher? | 50005 | 50019, 50030, 50018 | 0.00 |
| MANUAL_122: If I want to listen to music using this eReader, what should I do? | 50017 | 50035, 50001, 50022 | 0.20 |
| MANUAL_134: If I want to edit my user profie of exercise bikes, what steps should I follow? | 50010 | 50035, 50018, 50001 | 0.25 |
| … and 2 more, in the JSON | | | |

### `hybrid` on held_out - 3 of 194

| Question | Wanted | Got | RR |
| --- | --- | --- | --- |
| MANUAL_311: What is the detailed procedure for cleaning the fuel tank filter? | 50023 | 50000, 50036, 50005 | 0.00 |
| MANUAL_345: What are the key factors to check when troubleshooting technical issues or malfunctions? | 50022 | 50034, 50000, 50024 | 0.00 |
| MANUAL_181: I just made mistakes on air filter of the geenrator. Please tell me the right steps. | 50018 | 50007, 50000, 50002 | 0.20 |

### `multimodal` on held_out - 3 of 194

| Question | Wanted | Got | RR |
| --- | --- | --- | --- |
| MANUAL_311: What is the detailed procedure for cleaning the fuel tank filter? | 50023 | 50000, 50036, 50005 | 0.00 |
| MANUAL_345: What are the key factors to check when troubleshooting technical issues or malfunctions? | 50022 | 50034, 50000, 50024 | 0.00 |
| MANUAL_181: I just made mistakes on air filter of the geenrator. Please tell me the right steps. | 50018 | 50007, 50000, 50002 | 0.20 |

## Reproducing this

```bash
make corpus-fetch          # 132 MiB, selective range download
make corpus-export         # normalise into the bundle the API reads
make corpus-ingest         # index into its own RAGFlow dataset
make corpus-split          # refuses to change the committed split
make corpus-eval           # this report
```

`make corpus-eval` also writes a JSON file beside this one carrying every per-question outcome, so any rate in these tables can be traced to the questions behind it. That JSON is gitignored - it names RAGFlow ids that mean nothing outside one local instance - so re-run the command to regenerate it.

