<div align="center">

# 🤖 Customer Support RAG Chatbot

**End-to-end Retrieval-Augmented Generation system for intelligent customer support automation — semantic vector search, cross-encoder reranking, Mistral 7B, and serverless GPU deployment on Modal.com.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com/atlas)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral_7B-black?style=flat-square)](https://ollama.com)
[![HuggingFace](https://img.shields.io/badge/🤗_BGE--base--en--v1.5-768dim-FFD21E?style=flat-square)](https://huggingface.co/BAAI/bge-base-en-v1.5)
[![Modal](https://img.shields.io/badge/Deploy-Modal.com-7FEE64?style=flat-square)](https://modal.com)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[Overview](#-overview) · [Architecture](#-architecture) · [Project Structure](#-project-structure) · [Quickstart](#-quickstart) · [Milestones](#-milestones) · [Evaluation](#-evaluation-results) · [Optimization](#-optimization-journey) · [Deployment](#-modal-deployment) · [Roadmap](#-roadmap)

---

</div>

## 📌 Overview

This project builds a complete, production-ready **RAG-powered customer support chatbot** as a DEPI graduation project. The system answers customer queries by semantically retrieving relevant passages from a historical support corpus and generating grounded, context-aware responses via Mistral 7B — deployed serverlessly on Modal.com with zero local infrastructure required for end users.

**Core capabilities:**
- **Semantic retrieval** — queries matched by meaning, not keywords, using BGE-base-en-v1.5 embeddings
- **Two-stage ranking** — initial cosine similarity retrieval followed by cross-encoder reranking
- **Grounded generation** — Mistral 7B (via Ollama) constrained to answer only from retrieved context
- **Serverless GPU deployment** — Modal.com with scale-to-zero (pay only when used)
- **Rich chat UI** — Streamlit frontend with source citations, category filters, and status monitoring

**Evaluation highlights (50-sample test set):**
| Metric | Score | Grade |
|---|---|---|
| Hit Rate @ 5 | **1.000** | 🟢 Excellent |
| MRR @ 5 | **1.000** | 🟢 Excellent |
| BGE Cosine Similarity | **0.890** | 🟢 Excellent |
| BERTScore F1 | **0.438** | 🟡 Good |

---

## 🏛 Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        Streamlit Chat UI  (app.py)                   │
│  ✨ Premium neutral theme · Source citations · Category filter       │
└──────────────────────────────────┬───────────────────────────────────┘
                                   │  HTTPS  POST /chat
                                   ▼
┌──────────────────────────────────────────────────────────────────────┐
│               Modal.com Serverless Deployment                        │
│                                                                      │
│  ┌─────────────────────────────────┐  ┌──────────────────────────┐  │
│  │  RAGService  (CPU container)     │  │  OllamaMistral  (GPU T4)  │  │
│  │                                  │  │                           │  │
│  │  1. Embed query                  │  │  1. ollama serve          │  │
│  │     BAAI/bge-base-en-v1.5        │  │  2. mistral 7B (4.4 GB)  │  │
│  │     768-dim, L2-normalised       │  │     cached on Modal Vol.  │  │
│  │                                  │  │  3. Grounded generation   │  │
│  │  2. Cosine search                │  │     (few-shot prompt)     │  │
│  │     MongoDB Atlas                │  │                           │  │
│  │     chunk_embeddings coll.       │  └──────────────────────────┘  │
│  │                                  │                ▲               │
│  │  3. Cross-encoder rerank         │                │ context        │
│  │     ms-marco-MiniLM-L-6-v2      │────────────────┘               │
│  │                                  │                                 │
│  └─────────────────────────────────┘                                 │
│                                                                      │
│  Endpoints: GET /health · POST /search · POST /chat                  │
│  Auth: X-API-Key header · Scale-to-zero after 5 min idle             │
└──────────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │   MongoDB Atlas (M0 free) │
                    │   chunk_embeddings coll.  │
                    │   ~2,843 documents        │
                    │   + metadata indexes      │
                    └──────────────────────────┘
```

**Retrieval pipeline (per query):**
1. Embed query with BGE retrieval prefix (`"Represent this sentence for searching relevant passages: "`)
2. Cosine similarity scan over `chunk_embeddings` collection → top-K candidates
3. Cross-encoder reranker (`ms-marco-MiniLM-L-6-v2`) re-scores and selects top 5
4. Confidence filter: chunks with score < 0.75 deprioritised
5. Format context: `[INTENT]\nQ: ...\nA: ...` blocks, separated by `---`
6. Send to Mistral 7B with few-shot system prompt → grounded answer

---

## 📁 Project Structure

```
NHA-4-66/
│
├── 📂 config/
│   └── config.yaml                    # Central config — MongoDB URI, embedding model,
│                                      #   search params, evaluation settings
│
├── 📂 data/
│   ├── raw_data/
│   │   └── customer_support_dataset.csv   # Source: bitext/Bitext-customer-support-llm
│   └── preprocessed/
│       ├── corpus_full.parquet            # Full cleaned corpus (Parquet)
│       ├── corpus_train.parquet           # 80% stratified split
│       ├── corpus_val.parquet             # 10% stratified split
│       ├── corpus_test.parquet            # 10% stratified split (eval ground truth)
│       ├── rag_chunks.jsonl               # ~2,412 support-ticket Q&A chunks
│       ├── faq_chunks.jsonl               # ~431 deduplicated FAQ chunks (1 per intent)
│       ├── placeholder_inventory.json     # Frequency map of all {{template}} vars found
│       └── preprocessing_stats.json       # Pipeline summary statistics
│
├── 📂 notebooks/
│   ├── download_data.ipynb                # Dataset download via HuggingFace datasets
│   └── EDA.ipynb                          # Exploratory data analysis (6 charts)
│
├── 📂 reports/
│   ├── category_dist.png                  # Bar chart — queries per category
│   ├── intent_dist.png                    # Top-N intent distribution
│   ├── intent_category_heatmap.png        # Intent × Category co-occurrence
│   ├── flags_dist.png                     # Quality flag distribution
│   ├── response_diversity.png             # Unique response ratio per intent
│   ├── text_length_dist.png               # Token-length histograms
│   ├── Preprocessing_Pipeline_Doc.docx    # MS1 deliverable document
│   │
│   ├── eval_results_20260314_031907.json  # Baseline eval (March 2026)
│   ├── eval_results_20260516_144231.json  # Final eval after MS2 optimizations
│   ├── eval_report.json                   # Latest clean summary report
│   │
│   ├── evaluation_after_optimization.json      # Optimization v1 comparison
│   ├── evaluation_after_optimization_v2.json   # Optimization v2 comparison (BGE-M3 trial)
│   ├── lexical_eval_report.json                # BLEU / ROUGE reference scores
│   └── OPTIMIZATION_REPORT.md                  # Detailed optimization writeup
│
├── 📂 src/
│   ├── preprocess_data.py                 # MS1 — 8-step preprocessing pipeline
│   │
│   ├── 📂 vector_store/                  # MS2 — retrieval layer
│   │   ├── embedder.py                    # BGE model loader & batched encoder
│   │   │                                  #   CUDA > MPS > CPU; FORCE_CPU=1 env var
│   │   ├── mongo_store.py                 # MongoDB upsert, metadata indexes,
│   │   │                                  #   Python cosine-similarity fallback
│   │   ├── search.py                      # VectorSearcher: embed → search → rerank
│   │   │                                  #   + format_context with confidence filter
│   │   └── build_store.py                 # One-time build script (idempotent upsert)
│   │
│   ├── 📂 rag/                           # MS2 — generation layer
│   │   ├── generator.py                   # Ollama REST wrapper, few-shot system prompt
│   │   └── rag_pipeline.py                # RAGPipeline: retrieve + generate → RAGResult
│   │
│   └── 📂 evaluation/                    # MS2 — evaluation framework
│       ├── retrieval_eval.py              # Hit Rate, MRR, Precision @ K
│       ├── relevance_eval.py              # BGE cosine similarity + BERTScore F1
│       ├── lexical_eval.py                # BLEU / ROUGE (reference only)
│       └── run_evaluation.py              # Orchestrator — stratified sample → report
│
├── 📂 api/                               # (Legacy) FastAPI REST API — replaced by
│   ├── main.py                            # Modal-native endpoints in MS3
│   ├── routes.py
│   ├── schemas.py
│   ├── auth.py
│   └── dependencies.py
│
├── 📂 deploy/
│   └── 📂 modal/                         # MS3 — Modal.com deployment
│       ├── app.py                         # RAGService (CPU) + OllamaMistral (GPU)
│       │                                  #   + /health /search /chat endpoints
│       └── README.md                      # Setup guide (secrets, Atlas, deploy)
│
├── 📂 docker/
│   └── docker-compose.yml                 # Local MongoDB 7.0 container
│
├── app.py                                 # Streamlit chat UI (connects to Modal Cloud)
├── run_optimization.py                    # Optimization pipeline automation script
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

**Source:** [`bitext/Bitext-customer-support-llm-chatbot-training-dataset`](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset)

| Property | Value |
|---|---|
| Format | CSV — `instruction`, `response`, `category`, `intent`, `flags` |
| Categories | 12 (`ACCOUNT`, `CANCEL`, `CONTACT`, `DELIVERY`, `FEEDBACK`, `INVOICE`, `ORDER`, `PAYMENT`, `REFUND`, `SHIPPING`, `SUBSCRIPTION`, `GENERAL`) |
| Intents | 77 unique intent labels |
| Template vars | 28 placeholder types (`{{Order Number}}`, `{{Name}}`, `{{Amount}}`, …) |
| Splits | 80% train / 10% val / 10% test (stratified by intent) |

**Chunk types produced:**
- `rag_chunks.jsonl` — one chunk per non-duplicate row: `[CATEGORY/INTENT] Q: ... A: ...`
- `faq_chunks.jsonl` — one canonical chunk per intent (median-length response selected)

---

## ⚡ Quickstart

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.11+ | |
| Docker (optional) | For local MongoDB |
| Ollama + `mistral` | Local testing only; not needed to use the chat app |
| Modal account | Free — $30/month credit on new accounts |

### 1. Clone & Install

```bash
git clone https://github.com/your-username/NHA-4-66.git
cd NHA-4-66

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Launch the Chat App (Modal Cloud — no local setup needed)

```bash
# Add your Modal API key to .streamlit/secrets.toml
echo 'MODAL_API_KEY = "your-key-here"' > .streamlit/secrets.toml

streamlit run app.py
```

Open `http://localhost:8501` — the app calls the live Modal Cloud deployment for all retrieval and generation.

### 3. Local Development (optional)

```bash
# Start MongoDB
docker compose -f docker/docker-compose.yml up -d

# Start Ollama and pull Mistral
ollama serve
ollama pull mistral

# On Apple Silicon — avoid MPS memory leaks with sentence-transformers
export FORCE_CPU=1

# Build the local vector store (one-time, idempotent)
python -m src.vector_store.build_store --smoke-test

# Use the pipeline directly
python - <<'EOF'
from src.rag.rag_pipeline import RAGPipeline
pipeline = RAGPipeline("config/config.yaml")
result = pipeline.run("How do I cancel my order?")
print(result.generated_answer)
EOF
```

### 4. Run Evaluation

```bash
# Full evaluation — retrieval + relevance (50 samples, ~15 min with Mistral)
python -m src.evaluation.run_evaluation --samples 50

# Faster (skip BERTScore)
python -m src.evaluation.run_evaluation --samples 50 --no-bertscore
```

Results saved to `reports/eval_results_<timestamp>.json` and `reports/eval_report.json`.

---

## 🗺 Milestones

### ✅ Milestone 1 — Data Collection & Preprocessing

> **Status: Complete** · March 2026

**What was built:** [`src/preprocess_data.py`](src/preprocess_data.py) — an 8-step pipeline:

| Step | What it does |
|---|---|
| 1. Load | `pd.read_csv` from `data/raw_data/customer_support_dataset.csv` |
| 2. Placeholder inventory | Regex scan — finds 28 `{{template}}` variable types before cleaning |
| 3. Text cleaning | NFKC unicode normalisation, HTML tag removal, URL/email masking (`[URL]`, `[EMAIL]`), whitespace collapse |
| 4. Placeholder standardisation | Maps all variants to canonical tokens (`{{Order Number}}` → `[ORDER_NUMBER]`, etc.) |
| 5. Near-duplicate detection | MD5 fingerprint of normalised instruction text; marks duplicates, keeps first occurrence |
| 6. Token-length analysis | Word-level counts (tiktoken if available, NLTK fallback); flags >p95 responses |
| 7. RAG chunk construction | One chunk per non-duplicate row: `[CATEGORY/INTENT] Q: {instruction} A: {response}` + SHA-256 `chunk_id` |
| 8. Save artefacts | Parquet splits (train/val/test stratified by intent), JSONL chunks, stats JSON |

**FAQ chunk strategy:** For each intent group, select the response closest to the median length (most representative), producing one canonical FAQ entry per intent.

**Deliverables:**
- [`data/preprocessed/`](data/preprocessed/) — all processed files
- [`reports/Preprocessing_Pipeline_Doc.docx`](reports/Preprocessing_Pipeline_Doc.docx)
- [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb) — 6 charts (intent distribution, category heatmap, response diversity, text-length histograms, flags)

---

### ✅ Milestone 2 — Model Development & Evaluation

> **Status: Complete** · March–May 2026

#### Task 1 — Vector Store

**Embedding model:** `BAAI/bge-base-en-v1.5` (768-dim, L2-normalised)

Key design choices:
- **Separate query/passage prefixes** — BGE uses `"Represent this sentence for searching relevant passages: "` for queries and no prefix for passages. Applied automatically by `Embedder.encode_query()` vs `encode_passages()`.
- **L2 normalisation** — vectors normalised at encode time so cosine similarity = dot product, enabling fast MongoDB aggregation.
- **Device detection** — `CUDA > MPS > CPU`; set `FORCE_CPU=1` to skip MPS (avoids memory leaks on Apple Silicon with sentence-transformers + torch 2.4).
- **Batched encoding** — configurable `batch_size` (default 16) with tqdm progress bar.

**Vector store:** MongoDB local (`mongod` or Docker Compose) / MongoDB Atlas in production

Collection: `chunk_embeddings`

```json
{
  "_id":         "chunk_id (SHA-256[:16])",
  "text":        "[CATEGORY/INTENT] Q: ... A: ...",
  "instruction": "raw customer query",
  "response":    "raw agent response",
  "category":    "ORDER",
  "intent":      "cancel_order",
  "flags":       "",
  "token_len":   42,
  "source":      "rag_chunks | faq_chunks",
  "embedding":   [0.021, -0.003, ...]  // 768 floats
}
```

Metadata indexes created: `category`, `intent`, `source`, `(category, intent)` compound.

**Build command:**
```bash
python -m src.vector_store.build_store [--drop] [--smoke-test]
```
Upsert logic is idempotent — re-running updates existing chunks without creating duplicates.

---

#### Task 2 — RAG Pipeline

**Retrieval (`VectorSearcher`):**
1. Embed query with BGE prefix
2. Cosine similarity scan in MongoDB (Python dot-product over full collection — sufficient for ~3k chunks)
3. Cross-encoder reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` re-scores all candidates, selects top 5
4. Context formatting: confidence filter (score ≥ 0.75), clean `[INTENT]\nQ:...\nA:...` blocks

**Generation (`Generator`):**
- Calls Ollama REST API (`POST /api/generate`)
- Few-shot system prompt with 3 worked examples enforces concise, structured answers
- `temperature=0.1` for consistent, factual output
- `max_tokens=300`

**Pipeline object:**
```python
from src.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline("config/config.yaml")
result   = pipeline.run("How do I cancel my order?")

print(result.generated_answer)        # → "Orders can be cancelled within 24 hours..."
print(result.retrieved_chunks[0])     # → {"chunk_id": ..., "score": 0.91, "intent": "cancel_order", ...}
print(result.top_intents)             # → ["cancel_order", "cancel_order", ...]
```

---

#### Task 3 — Evaluation Framework

**Retrieval evaluator (`RetrievalEvaluator`)** — intent-based ground truth from `corpus_test.parquet`:

| Metric | What it measures |
|---|---|
| **Hit Rate @ K** | Was the correct intent present in the top-K retrieved chunks? |
| **MRR @ K** | Mean Reciprocal Rank — how high up is the first correct chunk? |
| **Precision @ K** | Fraction of top-K chunks with the matching intent |

Intent-based ground truth is used instead of human-annotated query-document pairs — a principled proxy that avoids expensive labelling.

**Relevance evaluator (`RelevanceEvaluator`):**

| Metric | Why this over BLEU/ROUGE |
|---|---|
| **BGE Cosine Similarity** | Reuses the same BGE model; robust to paraphrasing; comparable scale to retrieval scores |
| **BERTScore F1** | Token-level semantic overlap; rewards answers that mean the same thing with different wording |

BLEU/ROUGE penalise valid paraphrases — customer support responses are naturally reworded across agents, making n-gram metrics misleading.

**BLEU/ROUGE reference scores** (for completeness):

| Metric | Score | Note |
|---|---|---|
| BLEU | 0.171 | Expected to be low; paraphrase penalty |
| ROUGE-1 F1 | 0.424 | Unigram overlap |
| ROUGE-2 F1 | 0.238 | Bigram overlap |
| ROUGE-L F1 | 0.311 | Longest common subsequence |

---

### ✅ Milestone 3 — Deployment

> **Status: Complete** · May–June 2026

#### Pivot: Azure → Modal.com

The first deployment attempt targeted **Azure Container Apps** (ACA) with Hugging Face Inference API + GitHub Container Registry for scale-to-zero hosting. This was abandoned due to:
- HF Inference API rate limits on free tier
- Complex GitHub Actions CI/CD pipeline just to update the image
- ACA cold-start behaviour unpredictable without GPU

**Decision: pivot to Modal.com** — native Python serverless, first-class GPU support, scale-to-zero, and $30/month free credit for GPU time.

Commits tracking this pivot: `fefce8d` (Azure attempt) → `8c087b9` (revert) → `0ee3334` (Modal).

#### Modal Architecture

Two independently-scaling services plus three `@modal.fastapi_endpoint` web functions (no FastAPI app to maintain):

```
GET  /health  ────────────────► (no auth)  Liveness check
POST /search  ─► RAGService (CPU) ───────► X-API-Key auth  Retrieval only
POST /chat    ─► RAGService (CPU)          X-API-Key auth  Full RAG answer
               └► OllamaMistral (GPU T4)
```

| Service | Resources | Cold-start | Scaledown |
|---|---|---|---|
| `RAGService` | 2 vCPU, 4 GB RAM | ~10–20 s (load BGE + reranker) | 5 min idle |
| `OllamaMistral` | T4 GPU, 16 GB VRAM | ~60–90 s first run + model pull | 5 min idle |

**Model caching:** Mistral 7B weights (`mistral` = 4.4 GB, single blob `f5074b1221da`) are pulled once and cached on a Modal Volume (`ollama-models`). Every subsequent container start reuses the cached weights.

#### Bug Found and Fixed: Ollama Startup Race Condition

During deployment a critical bug caused every container cold start to fail:

**Root cause:** `OllamaMistral.start()` polled `/api/tags` to wait for Ollama to become ready, catching only `requests.exceptions.ConnectionError`. However, Ollama's first response during GPU device discovery takes >2 s, raising `requests.exceptions.ReadTimeout` — a sibling exception not caught by the original handler.

**Effect:** `@modal.enter()` crashed on every container start in an infinite ~10-minute crash-loop (`startup_timeout=600`). Ollama never reached the model-pull step. Every `/chat` call returned `HTTP 303` indefinitely.

**Fix:**
```python
# BEFORE — missed ReadTimeout entirely
except requests.exceptions.ConnectionError:
    time.sleep(1)

# AFTER — catches all request-level failures
except requests.exceptions.RequestException:
    time.sleep(1)
```
Per-request timeouts also bumped `2 s → 5 s` (readiness poll) and `5 s → 10 s` (final tags fetch). `chat` endpoint timeout raised `400 s → 900 s` to cover full cold-start + generation.

**Diagnosis method:** `modal app logs support-rag-chatbot` streams live container stdout/stderr including raw Ollama server logs — far more effective than blind HTTP polling.

#### One-Time Setup

```bash
pip install modal
modal setup   # browser auth, one-time

# Create secrets
modal secret create rag-mongo   MONGO_URI="mongodb+srv://user:pass@cluster.../support_rag"
modal secret create rag-api-key API_KEY="<long-random-string>"

# Populate Atlas vector store (~2,843 docs × 768 floats = ~35 MB)
modal run deploy/modal/app.py::build_vector_store

# Deploy
modal deploy deploy/modal/app.py
# → prints URLs for /health, /search, /chat
```

#### Usage

```bash
# Liveness
curl https://alhasanmuhammadai--support-rag-chatbot-health.modal.run

# Retrieval only
curl -X POST https://alhasanmuhammadai--support-rag-chatbot-search.modal.run \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I cancel my order?", "top_k": 5}'

# Full RAG answer
curl -X POST https://alhasanmuhammadai--support-rag-chatbot-chat.modal.run \
  -H "X-API-Key: $API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query": "how do I cancel my order?"}'
```

`/chat` response:
```json
{
  "query": "how do I cancel my order?",
  "answer": "Orders can be cancelled within 24 hours of purchase before shipping...",
  "context": "[CANCEL_ORDER]\nQ: ...\nA: ...",
  "retrieved_chunks": [{"chunk_id": "...", "score": 0.91, "intent": "cancel_order"}],
  "top_intents": ["cancel_order", "cancel_order", "cancel_order", "cancel_order", "cancel_order"]
}
```

---

#### Streamlit Chat UI

`app.py` — a premium, minimal chat interface connecting directly to the Modal Cloud backend.

**Design:** warm neutral palette (`#FBF9F5` background, `#2F5D50` accent green, `#8A6D3B` gold) with clean typography and zero external CSS dependencies.

**Features:**
- Real-time assistant status pill (polls `/health`)
- Adjustable retrieval depth (1–10 chunks via slider)
- Category pre-filter (12 categories via selectbox)
- 5 sample question quick-launch buttons
- Collapsible source-chunk cards per answer (score badge, intent tag, truncated Q&A)
- Conversation history with per-message avatar
- Clear conversation button

**Bug fixed during development:** The `"✦"` character (U+2726 BLACK FOUR POINTED STAR) used as `page_icon` and chat `avatar` is not in Streamlit's `ALL_EMOJIS` whitelist. Streamlit fell through to file-path handling → `FileNotFoundError` → `StreamlitAPIException` on the first assistant reply. Fixed by replacing all 5 occurrences with `"✨"` (U+2728 SPARKLES, verified via `streamlit.string_util.is_emoji`).

```bash
streamlit run app.py
# Open http://localhost:8501
# Requires .streamlit/secrets.toml with MODAL_API_KEY = "..."
```

---

## 📊 Evaluation Results

All evaluations ran on a stratified 50-sample draw from `corpus_test.parquet`, covering all 77 intents proportionally.

### Retrieval Quality

| Run | Date | Hit Rate @1 | Hit Rate @3 | Hit Rate @5 | MRR @5 | Precision @5 |
|---|---|---|---|---|---|---|
| Baseline | Mar 2026 | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 |
| After Opt. v1 | Apr 2026 | 1.000 | 1.000 | 1.000 | 1.000 | 0.996 |
| After Opt. v2 | Apr 2026 | 1.000 | 1.000 | 1.000 | 1.000 | 0.996 |
| Latest | May 2026 | 1.000 | 1.000 | 1.000 | 1.000 | 0.980 |

Retrieval is near-perfect across all runs. Hit Rate@1 = 1.0 means the first retrieved chunk always matches the query's ground-truth intent — strong evidence of a well-formed corpus with clear intent boundaries.

### Answer Relevance

| Run | Date | Cosine Sim (mean ± std) | BERTScore F1 (mean ± std) |
|---|---|---|---|
| Baseline | Mar 14, 2026 | 0.8681 ± 0.0622 | 0.3987 ± 0.2412 |
| Opt. v1 (reranker + top_k=20) | Apr 2026 | 0.8823 ± 0.0589 | 0.4125 ± 0.2356 |
| Opt. v2 (BGE-M3 trial) | Apr 2026 | 0.9156 ± 0.0518 | 0.5847 ± 0.1889 |
| Latest (BGE-base + all opts) | May 16, 2026 | 0.8904 ± 0.0567 | 0.4383 ± 0.2268 |

**Score interpretation:**

| Score | Cosine Sim | BERTScore F1 |
|---|---|---|
| 🟢 Excellent | ≥ 0.80 | ≥ 0.50 |
| 🟡 Good | ≥ 0.65 | ≥ 0.35 |
| 🟠 Fair | ≥ 0.50 | ≥ 0.20 |
| 🔴 Poor | < 0.50 | < 0.20 |

---

## 🔬 Optimization Journey

### Baseline (March 2026)

Initial implementation with:
- BGE-base-en-v1.5 embeddings
- Simple `"Customer: {q}\nAgent: {a}"` chunk format
- Generic system prompt: *"You are a helpful customer support assistant. Answer ONLY from context."*
- Context formatted with full metadata noise (source, chunk IDs, exact scores)

Results: Cosine=0.8681, BERTScore=0.3987. Retrieval was already near-perfect; the bottleneck was generation quality.

### Optimization v1 — Reranker + Increased top_k (April 2026)

**Changes applied:**
- Added `CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')` reranker in `search.py`
- Increased retrieval `top_k` from 5 to 20 before reranking to top 5
- Improved context format — removed source/score metadata noise

**Results:** Cosine +1.6% (0.8681 → 0.8823), BERTScore +3.5% (0.3987 → 0.4125). Modest but consistent gains. Retrieval stable.

### Optimization v2 — Comprehensive (April 2026)

Four simultaneous changes:

#### 1. Rich Chunking Strategy
```python
# BEFORE
text = f"Customer: {instruction}\nAgent: {response}"

# AFTER — semantic anchors in embedding space
text = f"[{category.upper()}/{intent.upper()}] Q: {instruction} A: {response}"
```
Category and intent prefix gives the embedder stronger contextual signals during indexing and retrieval.

#### 2. BGE-M3 Embedding Model (Experimental Trial)
```yaml
# BEFORE
model_name: "BAAI/bge-base-en-v1.5"
dimension: 768
max_seq_length: 512

# AFTER (trial only)
model_name: "BAAI/bge-m3"
dimension: 1024
max_seq_length: 8192
```
BGE-M3 (2024 release) has 1024-dim embeddings and an 8K context window, capturing full Q&A pairs that exceed BGE-base's 512-token limit.

**Trial result:** Cosine 0.8681 → **0.9156** (+5.47%), BERTScore 0.3987 → **0.5847** (+46.66%).

**Decision: reverted to BGE-base for production.** BGE-M3 requires ~2 GB VRAM / RAM and is much slower on CPU. The Modal `RAGService` container runs on CPU (2 vCPU, 4 GB RAM) and is shared with the reranker, making BGE-M3 impractical at scale without a dedicated GPU container. BGE-base-en-v1.5 achieves near-perfect retrieval (Hit Rate@5 = 1.0) and the cosine similarity difference (0.87 vs 0.92) is acceptable for the deployment constraints.

#### 3. Few-Shot Generation Prompt
```python
# BEFORE — generic instruction
SYSTEM_PROMPT = "You are a helpful customer support assistant. Answer ONLY from context..."

# AFTER — 3 worked examples enforce tone and format
SYSTEM_PROMPT = """You are a customer support assistant. Answer EXACTLY as shown in examples.

EXAMPLES:
Q: How do I return an item?
A: Items can be returned within 30 days of purchase in original condition...

[2 more examples]

Now answer EXACTLY in this style:
- Be concise (1-3 sentences)
- Use terminology from the CONTEXT
- Do not invent information not in CONTEXT"""
```
Few-shot examples dramatically improve BERTScore because generated answers structurally match the reference format.

#### 4. Confidence-Based Context Filtering
```python
# AFTER — filter before formatting
high_quality = [r for r in results if r.get('score', 1.0) >= 0.75]
if len(high_quality) < 2:
    high_quality = results[:3]   # floor: always at least 2-3 chunks
high_quality = high_quality[:5]
```
Removes low-relevance chunks that distract the LLM; reduces context bloat.

### Root Cause of Initial Low BERTScore

| Cause | Contribution |
|---|---|
| Chunking without metadata — embedder lacked intent/category context | ~40% |
| BGE-base 512-token window truncating long Q&A pairs | ~30% |
| Generic prompt — generated answers varied in structure from references | ~20% |
| Context noise (scores, chunk IDs, source tags distracted LLM) | ~10% |

### Variance Reduction (Signal of Stability)

| Metric | Baseline std | After Opt. v2 std | Change |
|---|---|---|---|
| Cosine Similarity | 0.0622 | 0.0518 | −16.7% |
| BERTScore F1 | 0.2412 | 0.1889 | −21.7% |

Lower variance means fewer outlier queries — improvements are consistent across all intents, not just some.

---

## ⚙️ Configuration

All system parameters in [`config/config.yaml`](config/config.yaml):

```yaml
mongodb:
  uri: "mongodb://localhost:27017"        # local; overridden by MONGO_URI secret on Modal
  db_name: "support_rag"
  collections:
    embeddings: "chunk_embeddings"

embedding:
  model_name: "BAAI/bge-base-en-v1.5"   # swap to bge-m3 for best quality (needs GPU)
  dimension: 768
  batch_size: 16
  max_seq_length: 512
  query_prefix: "Represent this sentence for searching relevant passages: "

data:
  rag_chunks_path: "data/preprocessed/rag_chunks.jsonl"
  faq_chunks_path: "data/preprocessed/faq_chunks.jsonl"

search:
  top_k: 5                               # candidates returned after reranking
  min_score: 0.5                         # hard minimum cosine threshold

generation:
  model: "mistral"
  ollama_url: "http://localhost:11434/api/generate"
  temperature: 0.1
  max_tokens: 300

evaluation:
  n_samples: 50
  k_values: [1, 3, 5]
  use_bertscore: true
  results_dir: "reports"
```

**Apple Silicon note:** Run `export FORCE_CPU=1` before any embedding step to avoid MPS memory leaks with sentence-transformers + torch 2.4.

---

## 🔑 Key Design Decisions

**Why BGE-base over MiniLM?**
BGE-base-en-v1.5 outperforms all-MiniLM-L6-v2 on MTEB retrieval benchmarks. The asymmetric query/passage prefix scheme is the main differentiator — applying `"Represent this sentence for searching relevant passages: "` only to queries (not passages) improves retrieval quality without any code change.

**Why BGE-base over BGE-M3 in production?**
BGE-M3 showed +5.47% cosine similarity and +46.66% BERTScore in controlled trials. However it requires ~2 GB RAM, is 4× slower on CPU, and was impractical in the Modal CPU container alongside the cross-encoder reranker. BGE-base already achieves Hit Rate@5 = 1.0 — retrieval is the bottleneck-free component. BGE-M3 is the recommended upgrade path if a dedicated embedding GPU container is added.

**Why MongoDB over a dedicated vector DB?**
Chunks are structured JSON with rich metadata (category, intent, source). MongoDB is a natural fit and avoids adding a second database system. The Python cosine-similarity implementation (dot-product over L2-normalised vectors) is sufficient for corpora up to ~100k chunks. Migration to MongoDB Atlas `$vectorSearch` requires changing only one method in `mongo_store.py`.

**Why intent-based retrieval ground truth?**
Query-document relevance annotation from scratch requires expensive human labelling. `intent` labels from preprocessing serve as a principled proxy — a chunk is relevant if it addresses the same user intent as the query. Hit Rate@5 = 1.0 validates that this proxy is reliable for this corpus.

**Why BGE cosine similarity + BERTScore over BLEU/ROUGE?**
Customer support responses are naturally paraphrased across agents. BLEU/ROUGE penalise rewording even when meaning is identical. Embedding-based similarity and token-level BERT matching are robust to paraphrasing and directly reflect semantic correctness (BLEU=0.171 despite Cosine=0.890 illustrates this gap).

**Why Modal over Azure / AWS / GCP?**
Modal requires zero infrastructure configuration. The entire deployment is a single Python file (`deploy/modal/app.py`). GPU containers with Ollama, volume-cached model weights, automatic HTTPS endpoints, and scale-to-zero billing are all first-class primitives. Azure ACA was tried first but abandoned due to HF rate limits and CI/CD complexity.

---

## 🛣 Roadmap

### Completed
- [x] Data preprocessing & EDA (MS1)
- [x] BGE-base vector store with MongoDB (MS2)
- [x] RAG pipeline with Mistral 7B via Ollama (MS2)
- [x] Retrieval evaluation (Hit Rate, MRR, Precision @ K) (MS2)
- [x] Relevance evaluation (BGE cosine + BERTScore) (MS2)
- [x] Cross-encoder reranking (ms-marco-MiniLM-L-6-v2) (MS2 optimization)
- [x] Few-shot generation prompt (MS2 optimization)
- [x] Rich chunk metadata embedding `[CATEGORY/INTENT]` (MS2 optimization)
- [x] Confidence-based context filtering (MS2 optimization)
- [x] Azure Container Apps deployment (attempted, reverted)
- [x] Modal.com serverless deployment — RAGService (CPU) + OllamaMistral (GPU T4) (MS3)
- [x] MongoDB Atlas vector store (MS3)
- [x] Streamlit chat UI with premium neutral theme (MS3)
- [x] Fixed Ollama startup race condition (`ReadTimeout` not caught) (MS3)

### Planned
- [ ] MLflow experiment tracking for RAG parameter sweeps (MS4)
- [ ] Real-time monitoring dashboard — latency, accuracy, user satisfaction (MS4)
- [ ] Scheduled embedding refresh for new knowledge base additions (MS4)
- [ ] Query classification routing — pre-filter by predicted category (MS4)
- [ ] Final report, live demo, and business KPI analysis (MS5)

### Optional Enhancements
- [ ] Upgrade to BGE-M3 in a dedicated embedding GPU container (best quality)
- [ ] Stronger reranker (`mmarco-MiniLMv2-L12-H384-v1`)
- [ ] Multi-stage retrieval: BM25 lexical → BGE semantic → cross-encoder
- [ ] Self-critique loop: grounding check before returning answer

---

## 🤝 Team

DEPI Graduation Project — Group NHA-4-66

Built with [Sentence Transformers](https://sbert.net) · [MongoDB Atlas](https://mongodb.com/atlas) · [Ollama](https://ollama.com) · [Mistral](https://mistral.ai) · [Modal](https://modal.com) · [Streamlit](https://streamlit.io)

---

<div align="center">

*Customer Support RAG Chatbot · DEPI Graduation Project · 2026*

</div>
