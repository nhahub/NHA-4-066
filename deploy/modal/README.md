# Modal.com Deployment

Serverless deployment of the Customer Support RAG chatbot on
[Modal](https://modal.com). Two services scale independently (and to zero
when idle):

- **`RAGService`** (CPU) — BGE-base embedder + MongoDB vector search +
  cross-encoder reranker. Reuses [`src/vector_store/`](../../src/vector_store/)
  unchanged.
- **`OllamaMistral`** (GPU, T4) — Mistral 7B served locally via
  [Ollama](https://ollama.com), reusing
  [`src/rag/generator.py`](../../src/rag/generator.py) unchanged. Model
  weights are cached on a Modal Volume so they only get pulled once.

Both are wired together by three Modal-native web endpoints
(`@modal.fastapi_endpoint` — no FastAPI app to maintain):

| Endpoint | Method | Auth | Description |
|---|---|---|---|
| `/health` | GET | none | Liveness check |
| `/search` | POST | `X-API-Key` | Retrieval only — chunks + formatted context |
| `/chat` | POST | `X-API-Key` | Full RAG answer (retrieval + Mistral generation) |

## 1. Prerequisites

```bash
pip install modal
modal setup        # one-time browser auth
```

## 2. MongoDB Atlas

Modal doesn't host databases. Create a free **MongoDB Atlas M0** cluster and
note its connection string — it replaces the local
`mongodb://localhost:27017` from [`config/config.yaml`](../../config/config.yaml).
The existing in-Python cosine-similarity search in
[`mongo_store.py`](../../src/vector_store/mongo_store.py) works against Atlas
as-is, no `$vectorSearch` index required.

## 3. Create Secrets

```bash
modal secret create rag-mongo MONGO_URI="mongodb+srv://user:pass@cluster.../support_rag?retryWrites=true&w=majority"
modal secret create rag-api-key API_KEY="<choose-a-long-random-string>"
```

`API_KEY` gates `/search` and `/chat`. If you never create the
`rag-api-key` secret with that key, the auth check is skipped (open access) —
not recommended outside of local testing.

## 4. Populate the Vector Store (one-time)

Embeds `data/preprocessed/rag_chunks.jsonl` + `faq_chunks.jsonl` with
BGE-base and upserts them into the `chunk_embeddings` collection in Atlas:

```bash
modal run deploy/modal/app.py::build_vector_store
```

Pass `--drop` to wipe and rebuild from scratch:

```bash
modal run deploy/modal/app.py::build_vector_store --drop
```

## 5. Local Smoke Test (no deploy needed)

Runs one query through both services using ephemeral containers:

```bash
modal run deploy/modal/app.py --query "How do I cancel my order?"
```

First run pulls the `mistral` model (~4GB) into the `ollama-models` Volume —
expect this run to take a few minutes. Subsequent runs reuse the cached model.

## 6. Deploy

```bash
modal deploy deploy/modal/app.py
```

This prints the public URLs for `health`, `search`, and `chat`.

## 7. Usage

```bash
curl https://<your-workspace>--support-rag-chatbot-health.modal.run

curl -X POST https://<your-workspace>--support-rag-chatbot-search.modal.run \
  -H "X-API-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I cancel my order?", "top_k": 5}'

curl -X POST https://<your-workspace>--support-rag-chatbot-chat.modal.run \
  -H "X-API-Key: <your-api-key>" \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I cancel my order?"}'
```

`/chat` response shape:

```json
{
  "query": "how do I cancel my order?",
  "answer": "Orders can be cancelled within 24 hours of purchase before shipping...",
  "context": "[CANCEL_ORDER]\nQ: ...\nA: ...",
  "retrieved_chunks": [ { "chunk_id": "...", "score": 0.91, "intent": "cancel_order", "...": "..." } ],
  "top_intents": ["cancel_order", "cancel_order", "..."]
}
```

## Cost notes

- `RAGService` (CPU) and `OllamaMistral` (GPU T4) both scale to zero after
  5 minutes idle (`scaledown_window=300`).
- First request after idle pays a cold-start cost: ~10-20s for the CPU
  container (loading BGE + reranker), and longer for the GPU container while
  `ollama serve` starts and (on the very first deploy) pulls the model.
- New Modal accounts include $30/month in free credit, which comfortably
  covers light testing/demo traffic on a T4.
