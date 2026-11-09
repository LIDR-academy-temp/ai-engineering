"""Input validation of ``POST /v1/retrieval/search`` for the Session 10 knobs.

The pipeline is stubbed: we only assert the 422 boundary of ``RetrievalRequest``
(bounded ``top_k``, ``search_mode`` as a closed Literal) and that a valid mode /
rerank flag reaches ``retrieve`` unchanged. Auth stays on (valid key sent).
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import app.api.routers.retrieval as retrieval_router
import app.api.security as security
from app.generation.rag.schemas import RetrievalResult
from app.main import app

RET_KEY = "retrieval-secret"
HEADERS = {"X-API-Key": RET_KEY}
QUERY = "ecommerce storefront with card checkout"


@pytest.fixture
def calls(monkeypatch):
    received: list[dict] = []
    monkeypatch.setattr(
        security,
        "get_settings",
        lambda: type("S", (), {"RETRIEVAL_API_KEY": RET_KEY, "ESTIMATE_API_KEY": None})(),
    )

    async def fake_retrieve(**kwargs):
        received.append(kwargs)
        return RetrievalResult(chunks=[], low_confidence=True, candidates_evaluated=0)

    fake_runtime = type(
        "RT",
        (),
        {"effective_search_mode": lambda self: "vector", "effective_rerank": lambda self: False},
    )()
    monkeypatch.setattr(
        retrieval_router,
        "get_embedder",
        lambda: type("E", (), {"embed_one": staticmethod(lambda t: [0.0] * 1536)})(),
    )
    monkeypatch.setattr(retrieval_router, "get_runtime_retrieval_config", lambda: fake_runtime)
    monkeypatch.setattr(retrieval_router, "retrieve", fake_retrieve)
    return received


@pytest.fixture
def client():
    return TestClient(app)


@pytest.mark.parametrize("mode", ["bm25", "HYBRID", "", "vector; DROP TABLE chunks"])
def test_invalid_search_mode_is_422(client, calls, mode):
    r = client.post(
        "/v1/retrieval/search", json={"query_text": QUERY, "search_mode": mode}, headers=HEADERS
    )
    assert r.status_code == 422
    assert calls == []


@pytest.mark.parametrize("top_k", [0, -1, 31, 10_000])
def test_out_of_range_top_k_is_422(client, calls, top_k):
    r = client.post(
        "/v1/retrieval/search", json={"query_text": QUERY, "top_k": top_k}, headers=HEADERS
    )
    assert r.status_code == 422
    assert calls == []


def test_non_boolean_rerank_is_422(client, calls):
    r = client.post(
        "/v1/retrieval/search", json={"query_text": QUERY, "rerank": "maybe"}, headers=HEADERS
    )
    assert r.status_code == 422


@pytest.mark.parametrize(("mode", "rerank"), [("vector", False), ("hybrid", True)])
def test_valid_mode_and_rerank_reach_the_pipeline(client, calls, mode, rerank):
    r = client.post(
        "/v1/retrieval/search",
        json={"query_text": QUERY, "search_mode": mode, "rerank": rerank},
        headers=HEADERS,
    )
    assert r.status_code == 200
    assert calls[0]["search_mode"] == mode
    assert calls[0]["rerank"] is rerank


def test_omitted_knobs_fall_back_to_runtime_defaults(client, calls):
    r = client.post("/v1/retrieval/search", json={"query_text": QUERY}, headers=HEADERS)
    assert r.status_code == 200
    assert calls[0]["search_mode"] == "vector"
    assert calls[0]["rerank"] is False
