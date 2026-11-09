# Session 10 pre-work: hybrid search (RRF) and cross-encoder reranking

Adds hybrid retrieval (vector + English full-text, fused with Reciprocal Rank Fusion), integrates
the existing cross-encoder reranker with a recall-then-rerank pipeline (50 → 5), and measures the
four A/B/C/D configurations against a golden set for precision@5 and latency.

**No public contract changes.** The estimate endpoint is untouched. `/v1/retrieval/search` keeps its
optional `search_mode` / `rerank` fields; `RetrievedChunk` only gains optional debug fields
(`document_id`, `lexical_score`, `rrf_score`, `rerank_score`). Auth (`X-API-Key`) and rate limits
are unchanged.

## Changes by step

1. **FTS column + GIN index.** The existing migration `0003_session10_fts` (generated, stored
   `content_tsv` + `ix_chunks_content_tsv`) was verified with `upgrade → downgrade -1 → upgrade`
   (282/282 chunks populated, data intact). We kept the **`english`** text-search config instead
   of `spanish` because the shipped corpus is in English. The config is now a single shared
   constant (`TEXT_SEARCH_CONFIG`).
2. **Lexical branch + RRF + hybrid search.**
   - `retrieval/lexical_search.py`: parametrised `websearch_to_tsquery` (never `to_tsquery`),
     ranked by `ts_rank_cd`. The query is rewritten to an **OR of word tokens**: with the default
     AND semantics a full project description matched no chunk, so hybrid silently equalled
     vector. Only `\w+` tokens survive the rewrite, so no tsquery operator can be injected.
   - `retrieval/fusion.py`: pure RRF, generic over items (`key=`), 1-based ranks, deterministic
     tie-break (best individual rank, then id), `limit=`, validated `k > 0`.
   - `retrieval/vector_search.py` and `retrieval/hybrid_search.py`: both branches run
     concurrently (`asyncio.gather`, one `AsyncSession` each). A failing branch is logged and
     re-raised, never silently dropped.
   - `retrieval/pipeline.py` only composes the stages.
   - Per-stage structured logs (`vector_search`, `lexical_search`, `fusion`, `rerank`) carry sizes
     and durations, never the query text.
3. **Reranker integration.** The cross-encoder is loaded and warmed up **once in the lifespan**
   (`RERANKER_PRELOAD`). `/health` reports `reranker.loaded`. Inference runs in
   `asyncio.to_thread`, scores the same text the search used (the reformulated query in the
   estimate flow) and exposes `rerank_score`.
4. **Tests.**
   - RRF: k, ties, limit, duplicates, empty and single rankings.
   - Lexical: injection-safe rewrite; opt-in DB integration covering stemming, stop-words and
     hostile input.
   - Pipeline: rerank off/on, pool size, both branches, failure propagation.
   - API: 422 on an invalid `search_mode`, an out-of-range `top_k` or a non-boolean `rerank`.
5. **Evaluation.** `evals/retrieval/`: golden set, `list_candidates.py` (for manual annotation)
   and `run_retrieval_eval.py` (calls the pipeline directly; no API, no LLM).

## How to run

```bash
# Migration (applied automatically on container start)
docker compose exec estimator alembic upgrade head

# Reranker pre-flight
docker compose exec estimator python -m app.generation.rag.retrieval.verify_reranker

# Tests + lint (host; on OneDrive set UV_LINK_MODE=copy)
cd estimator && uv run pytest && uv run ruff check .
# Optional DB integration tests for the lexical branch
TEST_DATABASE_URL=postgresql+asyncpg://<user>:<password>@localhost:5433/<db> \
  uv run pytest tests/generation/rag/test_lexical_search_integration.py

# Evaluation (cleanest latency: one-off container without the uvicorn reloader)
docker compose run --rm --no-deps estimator python -m evals.retrieval.run_retrieval_eval [--runs 5] [--golden-set PATH]
docker compose exec estimator python -m evals.retrieval.list_candidates
```

## New configuration (defaults keep the previous behaviour)

| Variable | Default | Purpose |
|---|---|---|
| `RETRIEVAL_SEARCH_MODE` | `vector` | `vector` or `hybrid` (existing; also per request / at runtime) |
| `RERANKER_ENABLED` | `false` | Rerank by default (existing; also per request / at runtime) |
| `RERANKER_PRELOAD` | `true` | Load + warm up the cross-encoder once at startup |
| `RETRIEVAL_RECALL_TOP_K` | `50` (1–200) | Candidate pool handed to the reranker / cap on the fused ranking |
| `RERANK_TOP_N` | `5` (1–50) | Results kept after reranking |
| `RRF_K` | `60` (>0) | RRF smoothing constant |
| `VECTOR_CANDIDATE_LIMIT` | `50` (1–200) | Vector candidates before fusion |
| `LEXICAL_CANDIDATE_LIMIT` | `50` (1–200) | Lexical candidates before fusion |

## Results (A/B/C/D)

Hardened golden set (`evals/retrieval/golden_set.json`): 5 paraphrased project descriptions with
2 relevant budgets each. The base corpus has 17 budgets / 60 chunks. Measured N=5 runs per query
after warm-up, on CPU (WSL2, 14 logical CPUs, no GPU).

| Config | Search | Reranking | Precision@5 (mean) | Distinct relevant docs in top-5 (mean) | Latency p50 (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|
| A | vector | no | 0.92 | 2.00 | 5.4 | 6.1 |
| B | hybrid | no | 0.92 | 2.00 | 7.3 | 8.2 |
| C | vector | yes | 0.68 | 2.00 | 6456.7 | 7975.8 |
| D | hybrid | yes | 0.68 | 2.00 | 6511.9 | 7443.6 |

- Hybrid lifts the exact-term query (COBOL / strangler / OAuth) from 0.60 to 1.00.
- The reranker drops two queries from 1.00 to 0.40 by promoting shared-vocabulary traps.
- C and D are identical because the 50-candidate pool is almost the whole 60-chunk corpus.
- A first, easier golden set saturated the metric (every config hit the 0.68 ceiling). It is
  kept as `golden_set_v1_easy.json`.

Full reports are in `evals/retrieval/results/`. Relevance labels were proposed by Claude and are
pending human review.

## Conclusions

See [`evals/retrieval/CONCLUSIONS.md`](CONCLUSIONS.md) (draft, in Spanish). In short: **use B
(hybrid, no reranking)**. Reranking does not pay off with this model and corpus.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
