"""
routes.py
─────────
HTTP endpoints for the chatbot.

Routes
──────
  GET  /health         — liveness + dependency check (unauthenticated)
  POST /search         — vector search only (no LLM call)
  POST /chat           — full RAG: retrieve + generate

All write/expensive endpoints require the X-API-Key header.
"""

from __future__ import annotations

import logging
import time

import requests
from fastapi import APIRouter, Depends, HTTPException, Request, status

from .auth import verify_api_key
from .dependencies import get_pipeline
from .schemas import (
    ChatRequest,
    ChatResponse,
    Chunk,
    HealthResponse,
    SearchRequest,
    SearchResponse,
)

logger = logging.getLogger("api.routes")

router = APIRouter()


# ── Health ────────────────────────────────────────────────────────────────────

@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness + dependency probe",
    tags=["meta"],
)
def health(request: Request) -> HealthResponse:
    """
    Reports app + dependency status. Used by Azure Container Apps
    health probes and by the support portal to surface incidents.
    """
    pipeline = getattr(request.app.state, "pipeline", None)

    mongo_state, ollama_state = "down", "down"
    n_docs = 0
    emb_model, gen_model = "?", "?"

    if pipeline is not None:
        try:
            n_docs = pipeline.searcher.store.count()
            mongo_state = "ok"
        except Exception as e:
            logger.warning(f"Mongo probe failed: {e}")
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=2)
            r.raise_for_status()
            ollama_state = "ok"
        except Exception:
            pass

        emb_model = pipeline.searcher.embedder.model_name
        gen_model = pipeline.generator.model

    overall = "ok" if mongo_state == "ok" and ollama_state == "ok" else "degraded"
    return HealthResponse(
        status=overall,
        mongo=mongo_state,
        ollama=ollama_state,
        vector_store_docs=n_docs,
        embedding_model=emb_model,
        generation_model=gen_model,
    )


# ── Search (retrieval only) ───────────────────────────────────────────────────

@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Vector search over the support knowledge base",
    tags=["rag"],
    dependencies=[Depends(verify_api_key)],
)
def search(body: SearchRequest, request: Request) -> SearchResponse:
    pipeline = get_pipeline(request)
    started = time.perf_counter()
    try:
        results = pipeline.searcher.search(
            query=body.query,
            top_k=body.top_k,
            filter_category=body.filter_category,
        )
    except Exception as e:
        logger.exception("search failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {e}",
        )
    latency_ms = int((time.perf_counter() - started) * 1000)
    return SearchResponse(
        query=body.query,
        results=[_to_chunk(r) for r in results],
        latency_ms=latency_ms,
    )


# ── Chat (full RAG) ───────────────────────────────────────────────────────────

@router.post(
    "/chat",
    response_model=ChatResponse,
    summary="Retrieve + generate a grounded support answer",
    tags=["rag"],
    dependencies=[Depends(verify_api_key)],
)
def chat(body: ChatRequest, request: Request) -> ChatResponse:
    pipeline = get_pipeline(request)
    started = time.perf_counter()

    k = body.top_k or pipeline.top_k
    try:
        # Apply the optional category filter for this request without
        # mutating pipeline state.
        if body.filter_category:
            original_filter = pipeline.filter_category
            pipeline.filter_category = body.filter_category
            try:
                result = pipeline.run(body.query, top_k=k)
            finally:
                pipeline.filter_category = original_filter
        else:
            result = pipeline.run(body.query, top_k=k)
    except Exception as e:
        logger.exception("chat failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {e}",
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    chunks = (
        [_to_chunk(c) for c in result.retrieved_chunks]
        if body.include_chunks
        else None
    )
    return ChatResponse(
        query=body.query,
        answer=result.generated_answer,
        top_intents=result.top_intents,
        chunks=chunks,
        latency_ms=latency_ms,
    )


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_chunk(r: dict) -> Chunk:
    return Chunk(
        chunk_id=r.get("chunk_id", ""),
        score=float(r.get("score", 0.0)),
        source=r.get("source", ""),
        category=r.get("category", ""),
        intent=r.get("intent", ""),
        instruction=r.get("instruction", ""),
        response=r.get("response", ""),
    )
