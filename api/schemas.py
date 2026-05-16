"""
schemas.py
──────────
Pydantic request/response models for the REST API.

Kept deliberately flat so portal integrations (web widget, ticket triage,
Slack bot, etc.) can map fields one-to-one without nested unpacking.
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── Requests ──────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The customer's question.",
        examples=["How do I cancel my order?"],
    )
    top_k: Optional[int] = Field(
        None,
        ge=1,
        le=20,
        description="Override default top_k from config (1–20).",
    )
    filter_category: Optional[str] = Field(
        None,
        max_length=64,
        description="Optional pre-filter on chunk category (e.g. 'ORDER').",
    )
    include_chunks: bool = Field(
        False,
        description="If true, the response includes the retrieved chunks.",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = Field(None, ge=1, le=20)
    filter_category: Optional[str] = Field(None, max_length=64)


# ── Responses ─────────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    chunk_id: str
    score: float
    source: str
    category: str
    intent: str
    instruction: str
    response: str


class ChatResponse(BaseModel):
    query: str
    answer: str
    top_intents: list[str]
    chunks: Optional[list[Chunk]] = None
    latency_ms: int


class SearchResponse(BaseModel):
    query: str
    results: list[Chunk]
    latency_ms: int


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["ok", "degraded"])
    mongo: str
    ollama: str
    vector_store_docs: int
    embedding_model: str
    generation_model: str


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
