"""Reproducible retrieval evaluation of the four Session 10 configurations.

    docker compose exec estimator python -m evals.retrieval.run_retrieval_eval \n        [--runs N] [--golden-set PATH]

A = vector, B = hybrid, C = vector + rerank, D = hybrid + rerank. The script calls
the retrieval pipeline directly (no HTTP API, no LLM, no estimate generation) over
the golden set (``golden_set.json``) and reports, per configuration:

* **precision@5 (chunk level)** = chunks in the top-5 whose parent budget is
  relevant / 5. Several chunks of the same budget can fill the top-5, so we also
  report the **distinct relevant budgets in the top-5** and the per-query
  **ceiling** (max achievable precision@5 = min(5, chunks of the relevant
  budgets) / 5), without which the absolute numbers are misleading.
* **latency** of retrieval only (search + fusion + rerank) with
  ``time.perf_counter``: the reranker is loaded and warmed up BEFORE measuring,
  the query embeddings are computed once upfront (shared OpenAI round-trip,
  excluded), one warm-up pass per configuration, then ``N`` runs per query
  (default 5). Reported as p50 / p95 in milliseconds.

Configurations run in a fixed order (A, B, C, D) on the same environment. The
ranking is deterministic (no sampling, fixed tie-breaks); only latencies vary —
the script verifies that every repeated run returned the same ranking.

Outputs ``results/retrieval_eval_<timestamp>.json`` (raw per-query/per-config
results) and ``results/retrieval_eval_<timestamp>.md`` (comparison tables).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

from app.config import get_settings
from app.dependencies import get_async_session_factory, get_reranker
from app.generation.rag.store.models import TEXT_SEARCH_CONFIG, ChunkRow
from evals.retrieval.common import (
    CHUNK_TYPES,
    CONFIGS,
    GOLDEN_SET_PATH,
    K,
    RESULTS_DIR,
    Config,
    GoldenQuery,
    embed_queries,
    load_golden_set,
    run_config,
)

DEFAULT_RUNS = 5
# p95/p50 above this ratio is flagged as a high-variance latency.
HIGH_VARIANCE_RATIO = 2.0


def percentile(samples: list[float], pct: float) -> float:
    """Linear-interpolated percentile (inclusive method), deterministic."""
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * pct / 100
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


async def chunks_per_budget() -> dict[str, int]:
    """How many graded chunks each budget has (for the precision ceiling)."""
    budget = ChunkRow.metadata_["budget_id"].astext
    async with get_async_session_factory()() as session:
        rows = await session.execute(
            select(budget, func.count())
            .where(ChunkRow.chunk_type.in_(CHUNK_TYPES))
            .group_by(budget)
        )
        return {budget_id: count for budget_id, count in rows if budget_id}


def grade(budget_ids: list[str | None], relevant: frozenset[str], k: int) -> dict[str, Any]:
    hits = sum(1 for b in budget_ids[:k] if b in relevant)
    return {
        "precision_at_k": hits / k,
        "relevant_chunks": hits,
        "distinct_relevant_docs": len({b for b in budget_ids[:k] if b in relevant}),
    }


def environment(settings) -> dict[str, Any]:
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_version = torch.__version__
        threads = torch.get_num_threads()
    except Exception:  # noqa: BLE001 — torch is optional for the report
        device, torch_version, threads = "unknown", "unknown", None
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch_version,
        "torch_threads": threads,
        "reranker_device": device,
        "reranker_model": settings.RERANKER_MODEL,
        "embedding_model": settings.EMBEDDING_MODEL,
        "text_search_config": TEXT_SEARCH_CONFIG,
        "recall_pool": settings.RETRIEVAL_RECALL_TOP_K,
        "vector_candidate_limit": settings.VECTOR_CANDIDATE_LIMIT,
        "lexical_candidate_limit": settings.LEXICAL_CANDIDATE_LIMIT,
        "rrf_k": settings.RRF_K,
        "k": K,
        "chunk_types": CHUNK_TYPES,
    }


async def evaluate(runs: int, golden_set: Path = GOLDEN_SET_PATH) -> dict[str, Any]:
    settings = get_settings()
    queries = load_golden_set(golden_set)
    unlabeled = [q.id for q in queries if not q.relevant_document_ids]
    if unlabeled:
        raise SystemExit(f"Golden set not annotated yet (empty relevant ids): {unlabeled}")

    # Model load and query embedding are excluded from every measurement.
    reranker = get_reranker()
    await asyncio.to_thread(reranker.warmup)
    embeddings = await embed_queries(queries)
    budget_sizes = await chunks_per_budget()

    per_query: dict[str, dict[str, Any]] = {q.id: {"query": q.query} for q in queries}
    per_config: dict[str, dict[str, Any]] = {}
    for query in queries:
        relevant_chunks = sum(budget_sizes.get(b, 0) for b in query.relevant_document_ids)
        per_query[query.id]["relevant_document_ids"] = sorted(query.relevant_document_ids)
        per_query[query.id]["ceiling_precision_at_k"] = min(K, relevant_chunks) / K
        per_query[query.id]["configs"] = {}

    for config in CONFIGS:
        await _warmup(config, queries, embeddings)
        latencies: list[float] = []
        for query in queries:
            samples, rankings = [], []
            for _ in range(runs):
                started = time.perf_counter()
                result = await run_config(config, query, embeddings[query.id])
                samples.append((time.perf_counter() - started) * 1000)
                rankings.append([chunk.id for chunk in result.chunks])
            latencies += samples
            budget_ids = [chunk.budget_id for chunk in result.chunks]
            per_query[query.id]["configs"][config.id] = {
                **grade(budget_ids, query.relevant_document_ids, K),
                "top_k_budget_ids": budget_ids,
                "top_k_chunk_ids": rankings[-1],
                "deterministic": all(r == rankings[0] for r in rankings),
                "latency_ms": [round(s, 2) for s in samples],
                "latency_p50_ms": round(percentile(samples, 50), 2),
            }
        rows = [per_query[q.id]["configs"][config.id] for q in queries]
        per_config[config.id] = {
            "search_mode": config.search_mode,
            "rerank": config.rerank,
            "precision_at_k_mean": statistics.mean(r["precision_at_k"] for r in rows),
            "distinct_relevant_docs_mean": statistics.mean(
                r["distinct_relevant_docs"] for r in rows
            ),
            "latency_p50_ms": round(percentile(latencies, 50), 2),
            "latency_p95_ms": round(percentile(latencies, 95), 2),
            "latency_samples": len(latencies),
        }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "golden_set": golden_set.name,
        "runs_per_query": runs,
        "environment": environment(settings),
        "ceiling_precision_at_k_mean": statistics.mean(
            per_query[q.id]["ceiling_precision_at_k"] for q in queries
        ),
        "configs": per_config,
        "queries": per_query,
        "anomalies": _anomalies(queries, per_query, per_config),
    }


async def _warmup(config: Config, queries: list[GoldenQuery], embeddings) -> None:
    """One unmeasured pass so connection-pool and model caches are hot."""
    for query in queries:
        await run_config(config, query, embeddings[query.id])


def _anomalies(queries, per_query, per_config) -> list[str]:
    notes = []
    for query in queries:
        cfg = per_query[query.id]["configs"]
        for base, reranked in (("A", "C"), ("B", "D")):
            if cfg[reranked]["precision_at_k"] < cfg[base]["precision_at_k"]:
                notes.append(
                    f"{query.id}: reranking worsened precision@{K} "
                    f"({base} {cfg[base]['precision_at_k']:.2f} → "
                    f"{reranked} {cfg[reranked]['precision_at_k']:.2f})."
                )
        if not all(c["deterministic"] for c in cfg.values()):
            notes.append(f"{query.id}: ranking changed between repeated runs.")
    if all(
        per_query[q.id]["configs"]["A"]["top_k_chunk_ids"]
        == per_query[q.id]["configs"]["B"]["top_k_chunk_ids"]
        for q in queries
    ):
        notes.append("Hybrid (B) returned exactly the vector (A) ranking for every query.")
    if all(
        q["configs"][c]["precision_at_k"] == q["ceiling_precision_at_k"]
        for q in per_query.values()
        for c in per_config
    ):
        notes.append(
            f"Metric saturated: every config reaches the precision@{K} ceiling on every "
            "query, so the golden set cannot discriminate between A/B/C/D (queries too "
            "easy / one dominant budget each). Compare the non-relevant tail instead."
        )
    for config_id, stats in per_config.items():
        if stats["latency_p50_ms"] and (
            stats["latency_p95_ms"] / stats["latency_p50_ms"] > HIGH_VARIANCE_RATIO
        ):
            notes.append(
                f"{config_id}: high latency variance (p95/p50 = "
                f"{stats['latency_p95_ms'] / stats['latency_p50_ms']:.1f})."
            )
    return notes


def to_markdown(report: dict[str, Any]) -> str:
    env = report["environment"]
    configs = report["configs"]
    queries = report["queries"]
    lines = [
        f"# Retrieval evaluation — {report['generated_at']}",
        "",
        f"Golden set: `{report['golden_set']}` ({len(queries)} queries, base corpus "
        f"`{', '.join(env['chunk_types'])}`). Runs per query: {report['runs_per_query']} "
        "(after 1 warm-up pass per config).",
        "",
        f"| Config | Search | Reranking | Precision@{K} (mean) | "
        f"Distinct relevant docs in top-{K} (mean) | Latency p50 (ms) | Latency p95 (ms) |",
        "|---|---|---|---|---|---|---|",
    ]
    for config_id, s in configs.items():
        lines.append(
            f"| {config_id} | {s['search_mode']} | {'yes' if s['rerank'] else 'no'} | "
            f"{s['precision_at_k_mean']:.2f} | {s['distinct_relevant_docs_mean']:.2f} | "
            f"{s['latency_p50_ms']:.1f} | {s['latency_p95_ms']:.1f} |"
        )
    lines += [
        "",
        f"Mean precision@{K} ceiling (max achievable, see method): "
        f"{report['ceiling_precision_at_k_mean']:.2f}.",
        "",
        f"## Precision@{K} per query",
        "",
        "| Query | Ceiling | A | B | C | D |",
        "|---|---|---|---|---|---|",
    ]
    for query_id, q in queries.items():
        cells = " | ".join(f"{q['configs'][c]['precision_at_k']:.2f}" for c in configs)
        lines.append(f"| {query_id} | {q['ceiling_precision_at_k']:.2f} | {cells} |")
    lines += [
        "",
        f"## Top-{K} budgets per query (relevant in **bold**)",
        "",
        "| Query | Config | Top-5 budget_id |",
        "|---|---|---|",
    ]
    for query_id, q in queries.items():
        relevant = set(q["relevant_document_ids"])
        for config_id in configs:
            ids = [
                f"**{b}**" if b in relevant else str(b)
                for b in q["configs"][config_id]["top_k_budget_ids"]
            ]
            lines.append(f"| {query_id} | {config_id} | {', '.join(ids)} |")
    lines += ["", "## Anomalies", ""]
    lines += [f"- {note}" for note in report["anomalies"]] or ["- None detected."]
    lines += [
        "",
        "## Method",
        "",
        f"- precision@{K} = top-{K} chunks whose parent budget is relevant / {K}; "
        f"relevance is per budget (`budget_id`). Because one budget has 2–6 chunks, "
        f"we also report distinct relevant budgets in the top-{K} and a per-query "
        f"ceiling = min({K}, chunks of the relevant budgets) / {K}.",
        "- Latency = retrieval only (vector/lexical search + RRF + rerank), "
        "`time.perf_counter`; query embedding and model load excluded; reranker "
        "warmed up; 1 warm-up pass per config; p50/p95 over all queries × runs.",
        "- No relevance floor (distance threshold 2.0) so the top-5 is always full.",
        "",
        "## Environment",
        "",
    ]
    lines += [f"- {key}: `{value}`" for key, value in env.items()]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="Measured runs per query.")
    parser.add_argument(
        "--golden-set", type=Path, default=GOLDEN_SET_PATH, help="Golden-set JSON to evaluate."
    )
    args = parser.parse_args()
    if args.runs < 1:
        raise SystemExit("--runs must be >= 1")

    report = asyncio.run(evaluate(args.runs, args.golden_set))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = RESULTS_DIR / f"retrieval_eval_{stamp}.json"
    md_path = RESULTS_DIR / f"retrieval_eval_{stamp}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = to_markdown(report)
    md_path.write_text(markdown, encoding="utf-8")
    print(markdown)
    print(f"Saved {json_path} and {md_path}")


if __name__ == "__main__":
    main()
