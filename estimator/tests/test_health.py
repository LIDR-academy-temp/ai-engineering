from fastapi.testclient import TestClient


def test_health_returns_200(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_health_reports_reranker_readiness_without_loading_it(client: TestClient) -> None:
    data = client.get("/health").json()
    assert data["reranker"]["model"]
    # RERANKER_PRELOAD=false in the suite (conftest): the model must not load.
    assert data["reranker"]["loaded"] is False
