# Troubleshooting

Problems actually hit while building AgentPath AI, with what was measured and
what fixed them. Entries stay here after being fixed, because the next person on
a different machine will meet the same ones.

## Open: unresolved CUDA initialization failure inside Docker

**Symptom.** The reranker container (Text Embeddings Inference, `89-1.8` and
`89-latest`, both Ada/sm_89 builds) logs:

```
Could not find a compatible CUDA device on host: CUDA is not available
    DriverError(CUDA_ERROR_UNKNOWN, "unknown error")
Using CPU instead
Starting Bert model on Cpu
```

**What was verified alongside it**

- `nvidia-smi` runs *inside the same container* and reports the GPU correctly
  (Ada-generation GeForce, compute capability sm_89, driver branch `615.x`,
  CUDA UMD 13.x).
- `/dev/nvidia0`, `/dev/nvidiactl`, `/dev/nvidia-uvm`, `/dev/nvidia-uvm-tools`
  are all present inside the container.
- The host driver libraries are injected: `ldconfig -p` resolves
  `libcuda.so.1` to the host driver build, not to the image's bundled `compat`
  copy.
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` and `NVIDIA_VISIBLE_DEVICES=all`
  are set in the container environment.
- Host-side CUDA works: Ollama runs GPU inference on the same machine.
- Both image tags fail identically, which rules out the TEI build.

**Cause: not established.** In particular this is *not* attributed to the image's
CUDA 12.2 runtime meeting a newer driver. NVIDIA documents newer drivers as
supporting older CUDA runtimes
([CUDA compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/latest/why-cuda-compatibility.html)),
so a version gap alone does not explain the failure.

**Impact.** None on the current search path. Reranking is disabled behind
`AGENTPATH_ENABLE_RERANK=false`, and hybrid retrieval alone measures p95 212 ms
with Recall@1 89% on the benchmark set. CPU reranking was measured at 20-22 s per
query, which is why it is not simply left on.

**Next things to try**, in order of cost:

1. CDI device injection instead of the legacy device-request path:
   `sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`, then request the
   device as `nvidia.com/gpu=all`. This is a host-level change and needs the
   machine owner's agreement.
2. Run the reranker on the host driver rather than in Docker, as a small local
   service. Costs a PyTorch download; the host driver path is known to work.
3. Re-test after the next driver update. `615.x` is a very new branch.

## Fixed: RAGFlow reports a missing dataset as a permission error

Looking up a dataset by name returns business code `102` with a message about
lacking permission for that *name* when the dataset simply does not exist. Taken
at face value it makes first-run ingestion impossible.

`RAGFlowClient.find_dataset_id` treats codes 102 and 108 on the name-filtered
lookup as a miss and falls through to the unfiltered listing.

## Fixed: uploads rejected with "No file part!"

The HTTP client set `Content-Type: application/json` as a default header, which
overrode the multipart boundary httpx generates for file uploads, so RAGFlow saw
no file field. Only authentication belongs in the client defaults; JSON calls set
the content type per request.

## Fixed: model reference must name the instance

RAGFlow resolves a model as `{model}@{instance}@{provider}`. The two-part form
`{model}@{provider}` implies an instance literally named `default`, and RAGFlow
refuses to create an instance with that name, so the two-part form can never
resolve for a self-hosted provider. Correct values here:

```
BAAI/bge-m3@agentpath-embedder@HuggingFace
BAAI/bge-reranker-v2-m3@agentpath-reranker@HuggingFace
```

## Fixed: `keyword: true` does not enable keyword search

In `POST /api/v1/retrieval`, `keyword` does not switch on lexical matching. BM25
term scoring is always active, weighted by `1 - vector_similarity_weight`. The
flag asks a chat model to extract keywords and append them to the question, which
adds an LLM round trip per search. AgentPath expands deterministically from
`data/synthetic/lexicon.yaml` instead and leaves the flag off. `cross_languages`
behaves the same way (LLM translation), and is likewise unused.

## Fixed: embedder restart loop under a memory cap

Text Embeddings Inference sizes its ONNX arena from `--max-batch-tokens`. At
TEI's defaults the bge-m3 container settled at **7.5 GB** resident. Capping the
container at 4 GB without lowering that flag made warm-up fail and the container
restart-looped 93 times, exiting 0 each time (so it did not even look like an OOM
kill).

Settled configuration: `--max-batch-tokens 2048`, `--max-client-batch-size 8`,
`OMP_NUM_THREADS=8`, `mem_limit: 5g`. Warm-up then takes about 3 seconds.

## Fixed: healthy RAGFlow reported as unreachable

`GET /api/v1/system/healthz` answers outside the usual `{code, data}` envelope,
returning the component map directly:

```json
{"db":"ok","doc_engine":"ok","redis":"ok","status":"ok","storage":"ok"}
```

Parsing it as wrapped made a healthy service look unreachable in `/health`.

## Environment notes from the reference machine

- **ufw is active**, and it drops traffic from Docker's user-defined bridge to
  host addresses. A host-side model daemon (for example Ollama on
  `127.0.0.1:11434`) is therefore unreachable from RAGFlow's containers, and a
  bridge relay does not help either. The model services run inside RAGFlow's own
  network instead, which needs no firewall change and publishes nothing.
- **Port 8080 is occupied**, so the RAGFlow UI is published on `8081`.
- **Only CPython 3.14 is installed system-wide.** The workspace pins 3.12 through
  uv, because parts of the stack have no 3.14 wheels yet.
- `vm.max_map_count` is already `1048576`, so no `sysctl` change is needed here.
  On a host where it is lower, Elasticsearch will refuse to start; the command is
  printed by `make check`.

## Useful commands

```bash
make check              # environment gates, read-only
make ragflow-health     # per-service health plus the loopback-only audit
make ragflow-logs       # RAGFlow server logs
docker logs agentpath-embedder --tail 20
uv run python scripts/measure_latency.py            # engine-only latency
uv run python scripts/measure_latency.py --rerank   # with the cross-encoder
uv run agentpath-ingest verify                      # what RAGFlow actually holds
```
