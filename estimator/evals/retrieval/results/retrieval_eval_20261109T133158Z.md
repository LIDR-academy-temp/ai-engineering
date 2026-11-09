# Retrieval evaluation — 2026-11-09T13:31:58.072369+00:00

Golden set: `golden_set.json` (5 queries, base corpus `budget_component`). Runs per query: 5 (after 1 warm-up pass per config).

| Config | Search | Reranking | Precision@5 (mean) | Distinct relevant docs in top-5 (mean) | Latency p50 (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|
| A | vector | no | 0.68 | 1.00 | 6.3 | 7.6 |
| B | hybrid | no | 0.68 | 1.00 | 7.5 | 8.7 |
| C | vector | yes | 0.68 | 1.00 | 6760.7 | 7439.3 |
| D | hybrid | yes | 0.68 | 1.00 | 5843.1 | 6807.6 |

Mean precision@5 ceiling (max achievable, see method): 0.68.

## Precision@5 per query

| Query | Ceiling | A | B | C | D |
|---|---|---|---|---|---|
| q1 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 |
| q2 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 |
| q3 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |
| q4 | 0.80 | 0.80 | 0.80 | 0.80 | 0.80 |
| q5 | 0.60 | 0.60 | 0.60 | 0.60 | 0.60 |

## Top-5 budgets per query (relevant in **bold**)

| Query | Config | Top-5 budget_id |
|---|---|---|
| q1 | A | **BUD-2024-012**, **BUD-2024-012**, **BUD-2024-012**, BUD-2024-009, BUD-2024-002 |
| q1 | B | **BUD-2024-012**, **BUD-2024-012**, **BUD-2024-012**, BUD-2024-009, BUD-2024-009 |
| q1 | C | **BUD-2024-012**, **BUD-2024-012**, **BUD-2024-012**, BUD-2024-009, BUD-2024-009 |
| q1 | D | **BUD-2024-012**, **BUD-2024-012**, **BUD-2024-012**, BUD-2024-009, BUD-2024-009 |
| q2 | A | **BUD-2024-015**, **BUD-2024-015**, **BUD-2024-015**, BUD-2024-013, BUD-2024-013 |
| q2 | B | **BUD-2024-015**, **BUD-2024-015**, **BUD-2024-015**, BUD-2024-013, BUD-2024-013 |
| q2 | C | **BUD-2024-015**, **BUD-2024-015**, BUD-2024-013, BUD-2024-013, **BUD-2024-015** |
| q2 | D | **BUD-2024-015**, **BUD-2024-015**, BUD-2024-013, BUD-2024-013, **BUD-2024-015** |
| q3 | A | **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, BUD-2024-003 |
| q3 | B | **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, BUD-2024-003 |
| q3 | C | **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, BUD-2024-003 |
| q3 | D | **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, **BUD-2024-006**, BUD-2024-003 |
| q4 | A | **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, BUD-2024-006 |
| q4 | B | **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, BUD-2024-006 |
| q4 | C | **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, BUD-2024-006 |
| q4 | D | **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, BUD-2024-006 |
| q5 | A | **BUD-2024-007**, **BUD-2024-007**, **BUD-2024-007**, BUD-2024-014, BUD-2024-005 |
| q5 | B | **BUD-2024-007**, **BUD-2024-007**, **BUD-2024-007**, BUD-2024-005, BUD-2024-014 |
| q5 | C | **BUD-2024-007**, **BUD-2024-007**, **BUD-2024-007**, BUD-2024-016, BUD-2024-004 |
| q5 | D | **BUD-2024-007**, **BUD-2024-007**, **BUD-2024-007**, BUD-2024-016, BUD-2024-004 |

## Anomalies

- Metric saturated: every config reaches the precision@5 ceiling on every query, so the golden set cannot discriminate between A/B/C/D (queries too easy / one dominant budget each). Compare the non-relevant tail instead.

## Method

- precision@5 = top-5 chunks whose parent budget is relevant / 5; relevance is per budget (`budget_id`). Because one budget has 2–6 chunks, we also report distinct relevant budgets in the top-5 and a per-query ceiling = min(5, chunks of the relevant budgets) / 5.
- Latency = retrieval only (vector/lexical search + RRF + rerank), `time.perf_counter`; query embedding and model load excluded; reranker warmed up; 1 warm-up pass per config; p50/p95 over all queries × runs.
- No relevance floor (distance threshold 2.0) so the top-5 is always full.

## Environment

- platform: `Linux-6.18.33.2-microsoft-standard-WSL2-x86_64-with-glibc2.41`
- processor: `x86_64`
- cpu_count: `14`
- python: `3.11.18`
- torch: `2.10.0+cu128`
- torch_threads: `7`
- reranker_device: `cpu`
- reranker_model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
- embedding_model: `text-embedding-3-small`
- text_search_config: `english`
- recall_pool: `50`
- vector_candidate_limit: `50`
- lexical_candidate_limit: `50`
- rrf_k: `60`
- k: `5`
- chunk_types: `['budget_component']`
