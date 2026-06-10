"""
deploy/modal/app.py
────────────────────
Modal.com deployment for the Customer Support RAG chatbot.

Two services, each scaling independently (and to zero when idle):

  - RAGService    (CPU)  : BGE-base embedder + MongoDB vector search +
                            cross-encoder reranker  (src/vector_store/*)
  - OllamaMistral (GPU)  : Mistral 7B served locally via Ollama
                            (reuses src/rag/generator.py as-is)

Web endpoints (Modal-native `@modal.fastapi_endpoint`, no FastAPI app needed):

  GET  /health  - liveness check
  POST /search  - retrieval only   {"query": "...", "top_k": 5, "category": "ORDER"}
  POST /chat    - full RAG answer  {"query": "...", "top_k": 5, "category": "ORDER"}

Both /search and /chat require an `X-API-Key` header matching the
`API_KEY` secret (skip the check by simply not setting that secret key).

──────────────────────────────────────────────────────────────────────────
One-time setup
──────────────────────────────────────────────────────────────────────────
1. Create a MongoDB Atlas (free M0) cluster and load it with the same
   `chunk_embeddings` collection used locally (see build_vector_store below).

2. Create the Modal secrets:
     modal secret create rag-mongo MONGO_URI="mongodb+srv://user:pass@cluster.../support_rag"
     modal secret create rag-api-key API_KEY="<choose-a-long-random-string>"

3. Populate the vector store in Atlas (one-time, ~35MB of chunks):
     modal run deploy/modal/app.py::build_vector_store

──────────────────────────────────────────────────────────────────────────
Deploy
──────────────────────────────────────────────────────────────────────────
     modal deploy deploy/modal/app.py

Quick local smoke test (no web endpoints needed):
     modal run deploy/modal/app.py --query "How do I cancel my order?"
"""

import os
from typing import Optional

import modal
from fastapi import Header, HTTPException

APP_NAME = "support-rag-chatbot"
OLLAMA_MODEL = "mistral"
GPU_TYPE = "T4"  # 16GB VRAM is plenty for a Q4 quantized 7B model

app = modal.App(APP_NAME)

# ── Secrets ──────────────────────────────────────────────────────────────────
# modal secret create rag-mongo MONGO_URI="mongodb+srv://..."
# modal secret create rag-api-key API_KEY="..."
mongo_secret = modal.Secret.from_name("rag-mongo")
api_key_secret = modal.Secret.from_name("rag-api-key")

# ── Persistent storage for pulled Ollama models ─────────────────────────────
ollama_volume = modal.Volume.from_name("ollama-models", create_if_missing=True)


# ── Images ───────────────────────────────────────────────────────────────────

def _preload_retrieval_models():
    """Bake the embedding + reranker weights into the image (faster cold starts)."""
    from sentence_transformers import CrossEncoder, SentenceTransformer

    SentenceTransformer("BAAI/bge-base-en-v1.5")
    CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


rag_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy==1.26.4",
        "torch==2.4.1",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .pip_install(
        "sentence-transformers==3.1.1",
        "transformers==4.44.2",
        "pymongo==4.7.0",
        "pyyaml==6.0.1",
        "fastapi==0.115.0",
    )
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .add_local_dir("config", remote_path="/root/config", copy=True)
    .run_function(_preload_retrieval_models)
)

ollama_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("curl", "zstd")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .pip_install("requests==2.32.3", "fastapi==0.115.0")
    .add_local_dir("src", remote_path="/root/src", copy=True)
    .env(
        {
            "OLLAMA_HOST": "0.0.0.0:11434",
            "OLLAMA_MODELS": "/root/.ollama/models",
            "OLLAMA_CONTEXT_LENGTH": "4096",
        }
    )
)

# Adds the local data files on top of rag_image, only used by build_vector_store
build_image = rag_image.add_local_dir(
    "data/preprocessed", remote_path="/root/data/preprocessed", copy=True
)


# ── Retrieval service (CPU): embed → MongoDB vector search → rerank ─────────

@app.cls(
    image=rag_image,
    cpu=2,
    memory=4096,
    secrets=[mongo_secret],
    scaledown_window=300,
)
class RAGService:
    @modal.enter()
    def load(self):
        import sys

        import yaml

        sys.path.insert(0, "/root")

        # The committed config.yaml points at local MongoDB — swap in the
        # Atlas URI from the Modal secret at container start.
        with open("/root/config/config.yaml") as f:
            cfg = yaml.safe_load(f)
        cfg["mongodb"]["uri"] = os.environ["MONGO_URI"]

        config_path = "/tmp/config.yaml"
        with open(config_path, "w") as f:
            yaml.safe_dump(cfg, f)

        from src.vector_store.search import VectorSearcher

        self.searcher = VectorSearcher(config_path)

    @modal.method()
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
    ) -> dict:
        results, context = self.searcher.search_and_format(
            query, top_k=top_k, filter_category=filter_category
        )
        return {"results": results, "context": context}


# ── Generation service (GPU): Mistral 7B via Ollama ──────────────────────────

@app.cls(
    image=ollama_image,
    gpu=GPU_TYPE,
    volumes={"/root/.ollama": ollama_volume},
    scaledown_window=300,
    timeout=400,
    startup_timeout=600,
)
class OllamaMistral:
    @modal.enter()
    def start(self):
        import subprocess
        import sys
        import time

        import requests

        sys.path.insert(0, "/root")

        subprocess.Popen(["ollama", "serve"])

        for _ in range(60):
            try:
                requests.get("http://localhost:11434/api/tags", timeout=5)
                break
            except requests.exceptions.RequestException:
                time.sleep(1)
        else:
            raise RuntimeError("Ollama server did not start in time")

        tags = requests.get("http://localhost:11434/api/tags", timeout=10).json()
        models = [m["name"] for m in tags.get("models", [])]
        if not any(OLLAMA_MODEL in m for m in models):
            # First cold start only — cached on the ollama-models volume
            # for every container after this one.
            subprocess.run(["ollama", "pull", OLLAMA_MODEL], check=True)

        from src.rag.generator import Generator

        self.generator = Generator(model=OLLAMA_MODEL)

    @modal.method()
    def generate(self, query: str, context: str) -> str:
        return self.generator.generate(query, context)


# ── Web endpoints ─────────────────────────────────────────────────────────────

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]==0.115.0"
)


def _check_api_key(x_api_key: Optional[str]):
    expected = os.environ.get("API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


@app.function(image=web_image)
@modal.fastapi_endpoint(method="GET")
def health():
    return {"status": "ok", "service": APP_NAME}


@app.function(image=web_image, secrets=[api_key_secret])
@modal.fastapi_endpoint(method="POST")
def search(item: dict, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)

    query = item.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    return RAGService().search.remote(
        query, top_k=item.get("top_k"), filter_category=item.get("category")
    )


@app.function(image=web_image, secrets=[api_key_secret], timeout=900)
@modal.fastapi_endpoint(method="POST")
def chat(item: dict, x_api_key: Optional[str] = Header(default=None)):
    _check_api_key(x_api_key)

    query = item.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Missing 'query' field")

    retrieval = RAGService().search.remote(
        query, top_k=item.get("top_k"), filter_category=item.get("category")
    )
    answer = OllamaMistral().generate.remote(query, retrieval["context"])

    return {
        "query": query,
        "answer": answer,
        "context": retrieval["context"],
        "retrieved_chunks": retrieval["results"],
        "top_intents": [c["intent"] for c in retrieval["results"]],
    }


# ── One-time vector store build (run with `modal run`, not deployed) ────────

@app.function(image=build_image, cpu=8, memory=16384, secrets=[mongo_secret], timeout=3600)
def build_vector_store(drop: bool = False):
    import sys

    import yaml

    sys.path.insert(0, "/root")
    os.chdir("/root")

    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg["mongodb"]["uri"] = os.environ["MONGO_URI"]

    config_path = "/tmp/config.yaml"
    with open(config_path, "w") as f:
        yaml.safe_dump(cfg, f)

    from src.vector_store.build_store import build, smoke_test

    build(config_path=config_path, drop_existing=drop)
    smoke_test(config_path=config_path)


# ── Local smoke test: `modal run deploy/modal/app.py --query "..."` ─────────

@app.local_entrypoint()
def main(query: str = "How do I cancel my order?"):
    retrieval = RAGService().search.remote(query)
    answer = OllamaMistral().generate.remote(query, retrieval["context"])

    print(f"Q: {query}\n")
    print(f"A: {answer}\n")
    print("Context used:")
    print(retrieval["context"])
