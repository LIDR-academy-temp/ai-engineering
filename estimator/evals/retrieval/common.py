"""Shared helpers for the Session 10 retrieval evaluation scripts.

Both scripts call the retrieval pipeline DIRECTLY (no HTTP API, no LLM, no
estimate generation): only retrieval is measured.

Conventions (documented in the report):

* Relevance is graded per **budget** (document). ``relevant_document_ids`` holds
  the traceable corpus ``budget_id`` (e.g. ``BUD-2024-003``), not the DB primary
  key, which changes on every re-ingest.
* Only the base corpus is graded: ``chunk_types=["budget_component"]`` (the 17
  real historical budgets), so the score does not depend on whether the
  synthetic task corpus is ingested.
* A permissive distance threshold is used so the top-5 is never truncated by the
  Session 9 relevance floor: we compare RANKING quality, not the soft-fail gate.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path

from app.config import get_settings
from app.dependencies import get_embedder
from app.generation.rag.retrieval.pipeline import SearchMode, retrieve
from app.generation.rag.schemas import RetrievalResult

GOLDEN_SET_PATH = Path(__file__).resolve().parent / "golden_set.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

K = 5
CHUNK_TYPES = ["budget_component"]
# Cosine distance is in [0, 2]: 2.0 disables the relevance floor.
NO_FLOOR_THRESHOLD = 2.0


@dataclass(frozen=True)
class Config:
    """One of the four measured configurations."""

    id: str
    search_mode: SearchMode
    rerank: bool

    @property
    def label(self) -> str:
        return f"{self.id} ({self.search_mode}{' + rerank' if self.rerank else ''})"


CONFIGS: tuple[Config, ...] = (
    Config("A", "vector", False),
    Config("B", "hybrid", False),
    Config("C", "vector", True),
    Config("D", "hybrid", True),
)


@dataclass(frozen=True)
class GoldenQuery:
    id: str
    query: str
    relevant_document_ids: frozenset[str]
    notes: str


def load_golden_set(path: Path = GOLDEN_SET_PATH) -> list[GoldenQuery]:
    """Load the golden set in file order (deterministic)."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [
        GoldenQuery(
            id=item["id"],
            query=item["query"],
            relevant_document_ids=frozenset(item.get("relevant_document_ids", [])),
            notes=item.get("notes", ""),
        )
        for item in raw
    ]


async def embed_queries(queries: list[GoldenQuery]) -> dict[str, list[float]]:
    """Embed every query ONCE (an OpenAI round-trip shared by all configs, so it
    is excluded from the retrieval latency)."""
    embedder = get_embedder()
    if embedder is None:
        raise SystemExit("OPENAI_API_KEY is required to embed the golden-set queries.")
    vectors = {}
    for query in queries:
        vectors[query.id] = await asyncio.to_thread(embedder.embed_one, query.query)
    return vectors


async def run_config(
    config: Config, query: GoldenQuery, embedding: list[float], *, k: int = K
) -> RetrievalResult:
    """Run the retrieval pipeline exactly as the API would, for one config."""
    settings = get_settings()
    return await retrieve(
        query_embedding=embedding,
        query_text=query.query,
        search_mode=config.search_mode,
        rerank=config.rerank,
        top_k=k,
        recall_k=settings.RETRIEVAL_RECALL_TOP_K,
        rerank_top_n=k,
        distance_threshold=NO_FLOOR_THRESHOLD,
        rrf_k=settings.RRF_K,
        vector_candidate_limit=settings.VECTOR_CANDIDATE_LIMIT,
        lexical_candidate_limit=settings.LEXICAL_CANDIDATE_LIMIT,
        chunk_types=CHUNK_TYPES,
    )
