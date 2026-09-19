# What "image search" means in AgentPath AI

Written for Stage 2B1, which asks a question that deserves a straight answer
before any new corpus is added: **does this system search inside image content, or
does it search text and then show whichever image is attached to the match?**

## The answer: it searches text about images, never the images

Nothing in this system looks at pixels. An image becomes findable only through
text that a human or a generator wrote about it, and that text is what the
embedder and the lexical scorer see.

Four independent confirmations, all checkable:

1. **The ingestion code says so.** `agentpath_ingestion/visual_chunks.py`, in the
   docstring for `build_visual_chunk_content`, on the screenshot description:
   *"which is what makes the image itself searchable — the picture contributes no
   tokens."*

2. **There is no vision model in the repository.** A search for `clip`, `CLIP`,
   `vision`, `image_embed`, `visual_embed`, `multimodal`, `siglip` and `blip`
   across `apps/api/agentpath/**/*.py` returns nothing.

3. **The retrieval request carries text only.** `RAGFlowClient.retrieve` posts
   `{"question": <string>, "vector_similarity_weight": 0.6, …}`. There is no image
   input, no image embedding and no image-similarity term.

4. **The embedding model is text-only.** `BAAI/bge-m3`, served by
   `ghcr.io/huggingface/text-embeddings-inference`. It has no image tower.

The image bytes *are* stored — attached to a RAGFlow chunk as `image_base64`, and
served by `GET /api/screenshots/{id}`. That exists so the console can display the
capture beside its step. It plays no part in ranking.

## So what actually makes a screenshot findable

One RAGFlow chunk per visual step, whose text is assembled in
`build_visual_chunk_content` from:

| Part | Where it comes from | Derived from the image? |
|---|---|---|
| Citation token, procedure id, title, version | procedure YAML | no |
| Step order, screen name, control name | procedure YAML | no |
| Instruction, English and Arabic | procedure YAML | no |
| Expected result | procedure YAML | no |
| Screenshot description, English and Arabic | screenshot sidecar JSON, hand-authored | no |
| Bounded on-screen text excerpt | the rendered page's **DOM**, at capture time | no — read from HTML, not from the PNG |

The DOM excerpt is the closest thing to "what the image says", and even that is
extracted from the HTML that produced the screenshot rather than from the
screenshot. Stage 2A chose this deliberately: the user's instruction was *"بدون
OCR. اجعل نص DOM والـscreenshot schema هما المصدر الحقيقي للنص والوصف"* — no OCR,
the DOM text and the screenshot schema are the source of truth.

## The honest label

For the synthetic procedures corpus:

> **Text retrieval over authored image descriptions and captured DOM text.**

Not visual retrieval. Not caption-based retrieval in the machine-learning sense
either, because the descriptions are structured sidecar fields written by the
screenshot generator, not captions predicted from an image.

## What this means for an external corpus

MRAMG-Manual has no authored descriptions at all. Measured on the official
`IMAGE/images_info/manual_imgs_collection.json`:

```
entries                        2,599
entries with an image_caption      0
caption length            min 0, median 0, max 0
entries with an image_url      2,300  (299 have none)
```

Every caption is an empty string. So the "use the official captions as a starting
point" plan cannot be executed for this subset — there is nothing to use.

What Manual *does* provide is position. Each document's text carries `<PIC>`
markers, and the document's image id list is in the same order:

```
<PIC> markers across 40 documents   2,607
image ids across 40 documents       2,607
documents where the two align       40 / 40
```

So an image can be located exactly in its document, and the text on either side
of its marker is the only semantic signal available about it. Stage 2B1 therefore
implements image ranking over that surrounding text window, and names it:

> **Context-window image retrieval.**

That name is chosen so it cannot be mistaken for either of the two things it is
not. It is not visual retrieval: no model sees the image. It is not caption-based
retrieval: there are no captions. It ranks an image by the words the document puts
around it.

Adding real visual retrieval would mean introducing an image-text embedding model
and embedding 2,599 images at ingestion time. That is a separate decision with a
download and a VRAM cost attached, and it is not taken here.

## How well does it work? Measured

The above was written before the evaluation ran. These are the numbers it produced,
on the held-out half of MRAMG-Manual (194 questions, split seed `20260919`). Full
report in `evaluation/reports/latest-corpus-retrieval.md`.

Context-window ranking is roughly **nine times** better at finding the right figure
than taking the document's images in order — and still falls far short of the 80%
target:

| Images chosen by | Recall@1 | Recall@3 | Recall@5 | Precision |
| --- | --- | --- | --- | --- |
| document order (positional) | 0.8% | 2.3% | 4.6% | 2.3% |
| context window | 9.9% | 30.1% | **40.7%** | 18.7% |

Both rows ran over *identical* document retrieval — the same retriever instance, so
the difference is image ranking and nothing else. Document Recall@3 for both was
98.5%.

### Why it stops at ~41%: adjacent figures are textually identical

The limit is not that the right document is missed. Of the 87 held-out questions
that scored zero image coverage, **74 still returned an image from the correct
document** — the method found the right place and picked the wrong figure. It is
often very nearly right: 19% of those misses returned a figure within one position
of the wanted one, 43% within three.

The cause is measurable. Taking every pair of consecutive figures in the corpus and
comparing the word sets of their context windows:

```
consecutive figure pairs                    2,567
Jaccard overlap of their windows   median 0.70, mean 0.65
pairs overlapping above 0.5                   71%
pairs overlapping above 0.8                   37%
```

A manual that shows ten steps in a row surrounds all ten figures with nearly the
same words. No amount of better text scoring separates figure 137 from figure 138
when 70% of their surrounding vocabulary is shared. **This is the ceiling of
text-only image retrieval on this corpus, and clearing it requires a model that
reads the picture.**

### And it cannot decline to show an image

40 of the 390 questions expect no image at all; their authored answers contain no
`<PIC>`. The system returns images for every one of them, so image abstention
accuracy is **0%**.

That is not an oversight, it is a measured impossibility for this method. On the dev
split, the top image's score for questions that want an image and for questions that
want none are the same distribution:

```
top image score, questions wanting images   n=176  median 1.115  range 0.218–2.856
top image score, questions wanting none     n= 20  median 1.027  range 0.253–2.843
```

The ranges overlap over almost their entire extent. A floor at 1.0 would silence
42.6% of the questions that *need* an image in order to correctly silence 45% of
those that do not; at 2.0 it damages the first population more than it helps the
second. So no absolute score threshold was added, and the 0% is reported as 0%
rather than approximated by a coin toss.

A *relative* floor was added — an image must score at least half the best image's
score to be shown — because that trade is genuinely favourable: on dev it keeps
89.7% of correct images while dropping 47.5% of wrong ones. It raised held-out
precision from 17.1% to 18.7% and left recall unchanged. It cannot produce
abstention, because the best image is at ratio 1.0 by construction.

### Summary

The honest answer to "does the system search inside image content?" is still **no**,
and the evaluation now says what that costs: excellent document retrieval (98.5%
Recall@3), usable-but-weak figure selection (40.7% Recall@5), and no ability to
withhold a figure. The first is architecture working on an external corpus. The
second and third are the boundary of text.
