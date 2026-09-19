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
