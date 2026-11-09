"""Pydantic models for the embedding pipeline.

Input side mirrors the normalized historical-budget JSON (a budget with a list
of components). Output side carries chunks ready to embed and, once embedded,
the vectors plus aggregate stats.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Closed universe of client sectors present in the sample dataset. Kept as a
# Literal so a typo or an unexpected sector fails validation loudly instead of
# silently leaking into the metadata.
Sector = Literal["finance", "ecommerce", "healthcare", "industrial"]
Complexity = Literal["low", "medium", "high"]


class ClientMetadata(BaseModel):
    """Who the budget belongs to. Travels as filterable context, not embedded."""

    name: str = Field(description="Client company name.")
    sector: Sector = Field(description="Client business sector.")
    country: str = Field(description="ISO-ish country code, e.g. 'ES'.")


class BudgetComponent(BaseModel):
    """A single line item of a historical budget."""

    component_id: str = Field(description="Stable id within the budget, e.g. 'AUTH-001'.")
    name: str = Field(description="Short human-readable component name.")
    description: str = Field(description="Detailed description of the work.")
    module: str | None = Field(
        default=None,
        description="Functional block this component/task belongs to (e.g. 'Payments').",
    )
    tech_stack: list[str] = Field(
        default_factory=list, description="Technologies involved in this component."
    )
    estimated_hours: int = Field(ge=0, description="Hours estimated for this component.")
    complexity: Complexity = Field(description="Coarse complexity bucket.")
    dependencies: list[str] = Field(
        default_factory=list, description="component_ids this one depends on."
    )


class Budget(BaseModel):
    """A complete historical budget with its components."""

    budget_id: str = Field(description="Stable budget id, e.g. 'BUD-2024-014'.")
    client_metadata: ClientMetadata
    project_summary: str = Field(description="One-line summary of the project.")
    main_technology: str = Field(description="Primary technology / stack of the project.")
    year: int = Field(ge=2000, le=2100, description="Year the budget was produced.")
    total_estimated_hours: int = Field(ge=0, description="Sum of component hours, as recorded.")
    components: list[BudgetComponent] = Field(min_length=1, description="Budget line items.")


class Chunk(BaseModel):
    """A fragment ready to be embedded.

    ``text`` is what gets sent to the embeddings API; ``metadata`` carries
    filterable fields that travel alongside the chunk but are NOT embedded.
    """

    chunk_id: str = Field(description="Traceable id, format '{budget_id}::{component_id}'.")
    text: str = Field(description="Embeddable text: parent context + component detail.")
    metadata: dict = Field(default_factory=dict, description="Filterable, non-embedded fields.")
    token_count: int = Field(ge=0, description="Token count of ``text`` (tiktoken).")


class EmbeddedChunk(Chunk):
    """A :class:`Chunk` with its embedding vector attached."""

    embedding: list[float] = Field(
        description="Dense embedding vector (1536 dims for text-embedding-3-small)."
    )


class IngestRequest(BaseModel):
    """Payload for ``POST /embeddings/ingest`` (Session 8: persisting contract).

    One request = one document. ``content`` is the full budget JSON, validated
    against :class:`Budget` so a malformed corpus fails with a 422 before
    touching the database or the embeddings API.
    """

    source_path: str = Field(
        min_length=1, description="Provenance of the document, unique per ingest."
    )
    document_type: str = Field(
        min_length=1, max_length=50, description="Document family, e.g. 'historical_budget'."
    )
    content: Budget = Field(description="Full budget JSON, as produced upstream.")
    chunk_type: str = Field(
        default="budget_component",
        max_length=50,
        description="chunk_type stamped on every chunk (filterable). Defaults keep S08 behaviour; "
        "the task corpus uses 'historical_task'.",
    )


class IngestResponse(BaseModel):
    """Response for ``POST /embeddings/ingest``: identifiers + ingest metrics.

    Vectors no longer travel over HTTP — they are persisted in pgvector.
    """

    document_id: int = Field(description="Primary key of the persisted document.")
    chunks_created: int = Field(ge=0, description="Chunks persisted for this document.")
    embedding_dimension: int = Field(description="Dimensionality of the stored vectors.")
    ingestion_time_ms: int = Field(ge=0, description="Wall-clock ingest time.")


class SearchRequest(BaseModel):
    """Payload for ``POST /search``."""

    query: str = Field(min_length=1, description="Free-text semantic query.")
    k: int = Field(default=5, ge=1, le=50, description="Number of nearest chunks to return.")


class SearchHit(BaseModel):
    """One ranked chunk. ``chunk_id`` is the DB primary key; the traceable
    corpus id ('BUD-X::COMP-Y' parts) travels inside ``metadata``."""

    chunk_id: int
    document_id: int
    chunk_type: str
    content: str
    distance: float = Field(description="Cosine distance (lower = more similar).")
    metadata: dict


class SearchResponse(BaseModel):
    """Response for ``POST /search``."""

    query: str
    k: int
    search_time_ms: int = Field(ge=0)
    results: list[SearchHit]


# ---------------------------------------------------------------------------
# Session 9 — RAG estimation pipeline (query understanding → generation).
#
# These types implement the locked contract from the Session 9 articles. They
# live alongside the Session 8 search types above; nothing here replaces them.
# ---------------------------------------------------------------------------

Scale = Literal["small", "medium", "large", "unknown"]
Confidence = Literal["high", "medium", "low", "insufficient"]
Relevance = Literal["primary", "supporting", "tangential"]
Impact = Literal["high", "medium", "low"]


class EstimationQuery(BaseModel):
    """Structured brief distilled from a raw meeting transcript.

    This is the output of the query-understanding stage: a transcript is full of
    digressions, so we extract only what drives retrieval (what to build, with
    which tech, under which constraints) instead of embedding the raw text.
    """

    function: str = Field(description="Functional summary of the project.")
    technologies: list[str] = Field(default_factory=list)
    sector: str | None = None
    scale: Scale = "unknown"
    country: str | None = None
    regulations: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """One chunk returned by the metadata-filtered retriever.

    ``id`` is the chunk's DB primary key (cited as a ``source id`` downstream).
    ``sector``/``project_year`` are flattened from the chunk's JSONB metadata
    (``client_sector``/``year``) so the generator and the citation validator see
    a stable, typed shape.
    """

    id: int
    content: str
    sector: str
    project_year: int
    chunk_type: str
    distance: float = Field(description="Cosine distance (lower = more similar).")
    budget_id: str | None = Field(
        default=None,
        description="Traceable corpus id of the parent budget (from JSONB metadata). "
        "Used by the Session 10 retrieval evaluation to grade precision per budget.",
    )
    # --- Session 10 debug fields (optional, so the S09 contract is unchanged) ---
    document_id: int | None = Field(
        default=None, description="DB id of the parent document (budget)."
    )
    lexical_score: float | None = Field(
        default=None,
        description="ts_rank_cd score from the full-text branch (higher = better). "
        "None when the chunk was not surfaced by the lexical branch. Never mixed "
        "with ``distance``.",
    )
    rrf_score: float | None = Field(
        default=None,
        description="Reciprocal Rank Fusion score in hybrid mode (higher = better). "
        "None in vector mode.",
    )
    rerank_score: float | None = Field(
        default=None,
        description="Cross-encoder relevance score (higher = better). None when "
        "reranking is off.",
    )


class RetrievalResult(BaseModel):
    """Outcome of the retrieval stage."""

    chunks: list[RetrievedChunk]
    low_confidence: bool = Field(
        description="True when no chunk crossed the distance threshold (soft-fail)."
    )
    candidates_evaluated: int = Field(
        ge=0, description="Total chunks scored before applying the threshold/limit."
    )


class SourceCitation(BaseModel):
    """A reference from the estimate back to a retrieved chunk."""

    source_id: int = Field(description="DB id of the cited chunk (a RetrievedChunk.id).")
    relevance: Relevance
    used_for: str = Field(description="What this source contributed to the estimate.")


class Assumption(BaseModel):
    """An estimate component NOT backed by any retrieved source."""

    description: str
    impact: Impact
    rationale: str


class TaskItem(BaseModel):
    """One concrete engineering task inside a functional module, in engineer-days.

    ``sources`` cite the historical chunk(s) the task was derived from; a task
    with no historical analog is left uncited and should surface as an
    :class:`Assumption` instead.
    """

    name: str
    description: str | None = Field(
        default=None, description="One-line scope of the task."
    )
    engineer_days: int = Field(ge=0)
    sources: list[int] = Field(
        default_factory=list, description="Chunk ids that back this task."
    )


class WorkModule(BaseModel):
    """A functional block (e.g. Auth, Payments, Data, Frontend, Infra, QA, PM)
    grouping the concrete tasks needed to deliver it."""

    name: str
    description: str | None = Field(
        default=None, description="What this functional block covers."
    )
    tasks: list[TaskItem] = Field(default_factory=list)


class Estimate(BaseModel):
    """Grounded estimate produced from retrieved historical budgets.

    Hours-based (engineer-days) with mandatory citations — distinct from the
    Session 4 ``EstimationResult`` (euros/weeks/phases). The breakdown is
    organised into functional modules, each decomposed into concrete tasks.
    When the retrieved context is insufficient, ``confidence='insufficient'``
    and the numeric totals stay ``None`` with ``modules`` empty (enforced by
    :func:`validation.check_coherence`).
    """

    total_engineer_days: int | None = None
    modules: list[WorkModule] = Field(default_factory=list)
    duration_weeks: int | None = None
    sources: list[SourceCitation] = Field(default_factory=list)
    assumptions: list[Assumption] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = Field(description="How the estimate was derived from the sources.")
    insufficient_context_explanation: str | None = None


# ---- HTTP request models for the Session 9 routers ------------------------
# Named ``RetrievalRequest``/``EstimateRequest`` (not ``SearchRequest``) to
# avoid colliding with the Session 8 ``SearchRequest`` above.


class RetrievalRequest(BaseModel):
    """Payload for ``POST /v1/retrieval/search`` (threshold + structural filters)."""

    query_text: str = Field(min_length=10, max_length=2000)
    top_k: int = Field(default=10, ge=1, le=30)
    distance_threshold: float = Field(default=0.6, ge=0.0, le=2.0)
    sectors: list[str] | None = None
    project_year_min: int | None = Field(default=None, ge=2010, le=2100)
    project_year_max: int | None = Field(default=None, ge=2010, le=2100)
    chunk_types: list[str] | None = None
    # Session 10 overrides (None = fall back to runtime/settings default). These
    # make the four measurement configurations invocable per request.
    search_mode: Literal["vector", "hybrid"] | None = None
    rerank: bool | None = None


class EstimateRequest(BaseModel):
    """Payload for ``POST /v1/estimate/from-transcript``."""

    transcript: str = Field(min_length=100, max_length=50_000)
    idempotency_key: str | None = Field(default=None, max_length=128)


# ---- Per-stage request/response models for the wizard (S09 teaching aid) ---
# The full pipeline (``estimate_from_transcript``) hides its intermediate
# artifacts; these stateless stage endpoints expose each step so a UI can run
# the pipeline one stage at a time. They REUSE the pure functions in this
# package — they do not re-implement any pipeline logic. Retrieval reuses
# ``RetrievalRequest``/``RetrievalResult`` above (zero new schema).


class ReformulateRequest(BaseModel):
    """Payload for ``POST /v1/estimate/stages/reformulate``."""

    transcript: str = Field(min_length=100, max_length=50_000)


class ReformulationResult(BaseModel):
    """Output of the query-understanding stage: the structured brief plus the
    canonical search text that gets embedded for retrieval."""

    query: EstimationQuery
    search_text: str = Field(description="Corpus-aligned text fed to the embedder.")


class AssembleRequest(BaseModel):
    """Payload for ``POST /v1/estimate/stages/assemble``.

    ``max_context_tokens`` defaults (server-side) to ``MAX_CONTEXT_TOKENS`` when
    omitted; a small value lets a demo show whole-chunk truncation."""

    chunks: list[RetrievedChunk]
    max_context_tokens: int | None = Field(default=None, ge=256, le=64_000)


class AssembleResult(BaseModel):
    """Output of the augmentation stage: the assembled ``<source>`` block plus
    what survived the token budget."""

    context_block: str
    kept_chunks: list[RetrievedChunk]
    dropped_count: int = Field(ge=0, description="Chunks dropped by the token budget.")
    token_count: int = Field(ge=0, description="Tokens in the assembled context block.")


class GenerateRequest(BaseModel):
    """Payload for ``POST /v1/estimate/stages/generate``.

    ``kept_chunks`` are the chunks the context block was built from; they are
    used to validate citations (no fabricated source ids) after generation."""

    context_block: str = Field(min_length=1)
    query: EstimationQuery
    kept_chunks: list[RetrievedChunk] = Field(default_factory=list)


class GenerateResult(BaseModel):
    """Output of the generation stage: the estimate plus the grounding signals
    the wizard surfaces (instead of auto-retrying like the full pipeline)."""

    estimate: Estimate
    fabricated_source_ids: list[int] = Field(
        default_factory=list,
        description="Cited source ids not present in kept_chunks (empty = clean).",
    )
    coherent: bool = Field(
        description="False when an insufficient estimate still carries numbers."
    )
