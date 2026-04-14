<div align="center">

# 🤖 Customer Support RAG Chatbot

**A production-ready intelligent support automation system powered by Retrieval-Augmented Generation, vector search, and local LLM inference.**

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![MongoDB](https://img.shields.io/badge/MongoDB-7.0-47A248?style=flat-square&logo=mongodb&logoColor=white)](https://mongodb.com)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral_7B-black?style=flat-square)](https://ollama.com)
[![HuggingFace](https://img.shields.io/badge/🤗_Embeddings-BGE_base_v1.5-FFD21E?style=flat-square)](https://huggingface.co/BAAI/bge-base-en-v1.5)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

[Overview](#-overview) · [Architecture](#-architecture) · [Project Structure](#-project-structure) · [Quickstart](#-quickstart) · [Milestones](#-milestones) · [Evaluation](#-evaluation-results) · [Roadmap](#-roadmap)

---

</div>

## 📌 Overview

This project builds a complete, end-to-end **RAG-powered customer support chatbot** that can answer user queries by retrieving relevant knowledge from a historical support corpus and generating grounded, context-aware responses — all running **100% locally** with no external API costs.

The system is designed around production principles:

- **No hallucination** — the LLM is constrained to answer only from retrieved context
- **Semantic retrieval** — queries are matched by meaning, not keywords
- **Fully offline** — BGE embeddings + Mistral 7B via Ollama, no cloud dependency
- **Idempotent pipelines** — every build and eval script is safe to re-run
- **MLOps-ready** — structured for MLflow tracking, monitoring, and retraining (MS4)

---

## 🏛 Architecture

```
                        ┌─────────────────────────────────────────┐
                        │             User Query                   │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │          BGE Embedder (local)            │
                        │    BAAI/bge-base-en-v1.5  (768-dim)      │
                        └──────────────────┬──────────────────────┘
                                           │  query vector
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │         MongoDB Vector Store             │
                        │  cosine similarity search over chunks    │
                        │  (rag_chunks + faq_chunks collections)   │
                        └──────────────────┬──────────────────────┘
                                           │  top-K relevant chunks
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │         Prompt Builder                   │
                        │  system prompt + context + user query    │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │     Mistral 7B via Ollama (local)        │
                        │       grounded answer generation         │
                        └──────────────────┬──────────────────────┘
                                           │
                                           ▼
                        ┌─────────────────────────────────────────┐
                        │            Final Answer                  │
                        └─────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
customer-support-rag/
│
├── 📂 config/
│   └── config.yaml                   # Central config — MongoDB, model, paths, eval settings
│
├── 📂 data/
│   ├── raw_data/
│   │   └── customer_support_dataset.csv
│   └── preprocessed/
│       ├── corpus_full.parquet       # Full cleaned corpus
│       ├── corpus_train.parquet      # Training split
│       ├── corpus_val.parquet        # Validation split
│       ├── corpus_test.parquet       # Test split (used for evaluation)
│       ├── rag_chunks.jsonl          # Chunked support ticket exchanges
│       ├── faq_chunks.jsonl          # Deduplicated FAQ Q&A pairs
│       ├── placeholder_inventory.json
│       └── preprocessing_stats.json
│
├── 📂 notebooks/
│   ├── download_data.ipynb
│   └── EDA.ipynb                     # Exploratory Data Analysis
│
├── 📂 reports/
│   ├── category_dist.png
│   ├── flags_dist.png
│   ├── intent_category_heatmap.png
│   ├── intent_dist.png
│   ├── response_diversity.png
│   ├── text_length_dist.png
│   ├── Preprocessing_Pipeline_Doc.docx
│   ├── eval_report.json              # ← generated after MS2 evaluation
│   └── eval_results_<timestamp>.json # ← generated after MS2 evaluation
│
├── 📂 src/
│   ├── preprocess_data.py            # MS1 — preprocessing pipeline
│   │
│   ├── 📂 vector_store/              # MS2 Task 1 — vector store
│   │   ├── embedder.py               # BGE model loader & batch encoder
│   │   ├── mongo_store.py            # MongoDB upsert, index, cosine search
│   │   ├── search.py                 # Clean retrieval interface for RAG
│   │   └── build_vector_store.py     # One-time build script
│   │
│   ├── 📂 rag/                       # MS2 Task 2 — generation pipeline
│   │   ├── generator.py              # Ollama/Mistral REST API wrapper
│   │   └── rag_pipeline.py           # End-to-end retrieve + generate
│   │
│   └── 📂 evaluation/                # MS2 Task 2 — evaluation
│       ├── retrieval_eval.py         # Hit Rate, MRR, Precision@K
│       ├── relevance_eval.py         # BGE cosine similarity + BERTScore
│       └── run_evaluation.py         # Evaluation orchestrator
│
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart

### Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.11+ | |
| MongoDB | 7.0 (local) | `mongod` must be running |
| Ollama | latest | [Install here](https://ollama.com/download) |
| Mistral model | 7B | `ollama pull mistral` |
| RAM | 8 GB+ | For BGE + Mistral inference |

### 1. Clone & Install

```bash
git clone https://github.com/your-username/customer-support-rag.git
cd customer-support-rag

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Start Services

```bash
# Terminal 1 — MongoDB
mongod --dbpath /data/db

# Terminal 2 — Ollama
ollama serve
ollama pull mistral              # first time only
```

### 3. Build the Vector Store

```bash
# Embeds all chunks and loads them into MongoDB (run once)
python -m src.vector_store.build_vector_store --smoke-test
```

Expected output:
```
✅ Vector store build complete.
Total documents: 2,843  (RAG: 2,412 | FAQ: 431)

SMOKE TEST – Verifying retrieval
Query: 'how do I cancel my order?'
  [0.9123] (faq_chunks | cancel_order) how do I cancel purchase [ORDER_NUMBER]...
  [0.8891] (rag_chunks | cancel_order) question about cancelling order [ORDER_NUMBER]...
✅ Smoke test passed.
```

### 4. Run the Full Evaluation

```bash
python -m src.evaluation.run_evaluation --samples 50
```

### 5. Use the Pipeline Directly

```python
from src.rag.rag_pipeline import RAGPipeline

pipeline = RAGPipeline("config/config.yaml")
result   = pipeline.run("How do I change an item in my order?")

print(result.generated_answer)
# → "To change an item in your order, please navigate to My Orders..."
```

---

## 🗺 Milestones

### ✅ Milestone 1 — Data Collection & Preprocessing

> **Status: Complete**

- Collected and unified historical support tickets and FAQ entries
- Cleaned text, handled placeholders (`[ORDER_NUMBER]`, `[PLACEHOLDER]`), removed noise
- Stratified train/val/test splits with intent-balanced sampling
- Produced `rag_chunks.jsonl` (ticket exchanges) and `faq_chunks.jsonl` (FAQ pairs)
- Full EDA: intent distribution, category heatmaps, response diversity, text length analysis

**Deliverables:** [`data/preprocessed/`](data/preprocessed/) · [`reports/Preprocessing_Pipeline_Doc.docx`](reports/Preprocessing_Pipeline_Doc.docx) · [`notebooks/EDA.ipynb`](notebooks/EDA.ipynb)

---

### ✅ Milestone 2 — Model Development & Evaluation

> **Status: Complete**

#### Task 1 — Vector Store

Built a local MongoDB-backed semantic search engine:

| Component | Detail |
|---|---|
| Embedding model | `BAAI/bge-base-en-v1.5` (768-dim, L2-normalised) |
| Vector store | MongoDB local — cosine similarity via dot product |
| Chunk sources | `rag_chunks.jsonl` + `faq_chunks.jsonl` |
| Metadata indexes | `category`, `intent`, `source` |
| Query prefix | BGE retrieval prefix applied automatically |

```bash
python -m src.vector_store.build_vector_store [--drop] [--smoke-test]
```

#### Task 2 — Evaluation Pipeline

End-to-end evaluation over `corpus_test.parquet`:

| Metric | What it measures |
|---|---|
| **Hit Rate @ K** | Was the correct intent in top-K retrieved chunks? |
| **MRR @ K** | Mean Reciprocal Rank of the first correct chunk |
| **Precision @ K** | Fraction of top-K chunks with the correct intent |
| **BGE Cosine Sim** | Semantic similarity: generated answer vs ground truth |
| **BERTScore F1** | Token-level semantic overlap (paraphrase-robust) |

```bash
python -m src.evaluation.run_evaluation [--samples N] [--no-bertscore]
```

**Deliverables:** [`reports/eval_report.json`](reports/eval_report.json) · [`src/vector_store/`](src/vector_store/) · [`src/evaluation/`](src/evaluation/)

---

### 🔲 Milestone 3 — Azure Deployment

> **Status: Planned**

- Deploy RAG service via Azure Machine Learning / Azure App Service
- Build REST API with FastAPI, integrate with support portal
- Secure endpoints with Azure AD / API key authentication

---

### 🔲 Milestone 4 — MLOps & Monitoring

> **Status: Planned**

- MLflow experiment tracking for RAG parameter variations
- Real-time monitoring: latency, accuracy, user satisfaction
- Scheduled embedding refresh and model weight retraining

---

### 🔲 Milestone 5 — Final Presentation

> **Status: Planned**

- Final report, live demo, and business KPI impact analysis

---

## 📊 Evaluation Results

> Results are generated by running `src/evaluation/run_evaluation.py` and saved to `reports/eval_report.json`.

| Metric | Score | Interpretation |
|---|---|---|
| Hit Rate @ 5 | — | Run eval to populate |
| MRR @ 5 | — | Run eval to populate |
| Precision @ 5 | — | Run eval to populate |
| BGE Cosine Similarity | — | Run eval to populate |
| BERTScore F1 | — | Run eval to populate |

**Score interpretation guide:**

| Score | Retrieval (Hit Rate / MRR) | Relevance (Cosine / BERTScore) |
|---|---|---|
| ≥ 0.85 / 0.75 | 🟢 Excellent | 🟢 Excellent |
| ≥ 0.70 / 0.55 | 🟡 Good | 🟡 Good |
| ≥ 0.50 / 0.35 | 🟠 Fair | 🟠 Fair |
| < 0.50 / 0.35 | 🔴 Poor | 🔴 Poor |

---

## ⚙️ Configuration

All system parameters are controlled from a single file: [`config/config.yaml`](config/config.yaml)

```yaml
mongodb:
  uri: "mongodb://localhost:27017"
  db_name: "support_rag"

embedding:
  model_name: "BAAI/bge-base-en-v1.5"
  dimension: 768
  batch_size: 64

generation:
  model: "mistral"
  temperature: 0.1
  max_tokens: 300

search:
  top_k: 5
  min_score: 0.5

evaluation:
  n_samples: 50
  k_values: [1, 3, 5]
  use_bertscore: true
```

---

## 🔑 Key Design Decisions

**Why BGE over MiniLM?**
`BAAI/bge-base-en-v1.5` consistently outperforms `all-MiniLM-L6-v2` on retrieval benchmarks (MTEB). BGE also uses separate query/passage prefixes, improving retrieval quality without any extra complexity.

**Why local MongoDB instead of a dedicated vector DB?**
Your chunks are structured JSON with rich metadata — MongoDB is a natural fit. It avoids introducing a second database system, and the cosine similarity implementation in Python is fully sufficient for corpora up to ~100k chunks. Migration to MongoDB Atlas Vector Search (production) requires changing only one method in `mongo_store.py`.

**Why intent-based retrieval ground truth?**
Constructing query-document relevance annotations from scratch would require expensive human labeling. The `intent` labels from preprocessing serve as a principled proxy — a chunk is relevant if it addresses the same user intent as the query.

**Why BGE cosine similarity for relevance scoring instead of BLEU/ROUGE?**
Customer support responses are naturally paraphrased. BLEU/ROUGE penalise rewording even when the meaning is identical. Embedding-based similarity is robust to this and directly reflects semantic correctness.

---

## 🛣 Roadmap

- [x] Data preprocessing & EDA (MS1)
- [x] BGE vector store with MongoDB (MS2 T1)
- [x] RAG pipeline with Mistral via Ollama (MS2 T2)
- [x] Retrieval & relevance evaluation framework (MS2 T2)
- [ ] RAG optimization — reranking, chunk size tuning (MS2 T3)
- [ ] FastAPI REST service (MS3)
- [ ] Azure deployment (MS3)
- [ ] MLflow experiment tracking (MS4)
- [ ] Monitoring dashboard (MS4)
- [ ] Retraining pipeline (MS4)
- [ ] Final report & presentation (MS5)

---

## 🤝 Contributing

This is an academic project. Issues and suggestions are welcome via GitHub Issues.

---

<div align="center">

Built with 🧠 using [Sentence Transformers](https://sbert.net) · [MongoDB](https://mongodb.com) · [Ollama](https://ollama.com) · [Mistral](https://mistral.ai)

</div>