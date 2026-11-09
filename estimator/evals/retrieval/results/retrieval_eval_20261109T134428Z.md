# Retrieval evaluation — 2026-11-09T13:44:28.805706+00:00

Golden set: `golden_set.json` (5 queries, base corpus `budget_component`). Runs per query: 5 (after 1 warm-up pass per config).

| Config | Search | Reranking | Precision@5 (mean) | Distinct relevant docs in top-5 (mean) | Latency p50 (ms) | Latency p95 (ms) |
|---|---|---|---|---|---|---|
| A | vector | no | 0.92 | 2.00 | 5.4 | 6.1 |
| B | hybrid | no | 0.92 | 2.00 | 7.3 | 8.2 |
| C | vector | yes | 0.68 | 2.00 | 6456.7 | 7975.8 |
| D | hybrid | yes | 0.68 | 2.00 | 6511.9 | 7443.6 |

Mean precision@5 ceiling (max achievable, see method): 1.00.

## Precision@5 per query

| Query | Ceiling | A | B | C | D |
|---|---|---|---|---|---|
| h1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| h2 | 1.00 | 1.00 | 0.80 | 0.40 | 0.40 |
| h3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| h4 | 1.00 | 0.60 | 1.00 | 0.60 | 0.60 |
| h5 | 1.00 | 1.00 | 0.80 | 0.40 | 0.40 |

## Top-5 budgets per query (relevant in **bold**)

| Query | Config | Top-5 budget_id |
|---|---|---|
| h1 | A | **BUD-2024-010**, **BUD-2024-010**, **BUD-2024-010**, **BUD-2024-009**, **BUD-2024-009** |
| h1 | B | **BUD-2024-010**, **BUD-2024-010**, **BUD-2024-009**, **BUD-2024-009**, **BUD-2024-010** |
| h1 | C | **BUD-2024-010**, **BUD-2024-010**, **BUD-2024-009**, **BUD-2024-009**, **BUD-2024-009** |
| h1 | D | **BUD-2024-010**, **BUD-2024-010**, **BUD-2024-009**, **BUD-2024-009**, **BUD-2024-009** |
| h2 | A | **BUD-2024-001**, **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-002**, **BUD-2024-001** |
| h2 | B | **BUD-2024-001**, **BUD-2024-002**, BUD-2024-016, **BUD-2024-001**, **BUD-2024-002** |
| h2 | C | BUD-2024-006, BUD-2024-004, **BUD-2024-001**, **BUD-2024-002**, BUD-2024-017 |
| h2 | D | BUD-2024-006, BUD-2024-004, **BUD-2024-001**, **BUD-2024-002**, BUD-2024-017 |
| h3 | A | **BUD-2024-008**, **BUD-2024-008**, **BUD-2024-008**, **BUD-2024-005**, **BUD-2024-005** |
| h3 | B | **BUD-2024-005**, **BUD-2024-005**, **BUD-2024-008**, **BUD-2024-005**, **BUD-2024-008** |
| h3 | C | **BUD-2024-005**, **BUD-2024-005**, **BUD-2024-005**, **BUD-2024-005**, **BUD-2024-008** |
| h3 | D | **BUD-2024-005**, **BUD-2024-005**, **BUD-2024-005**, **BUD-2024-005**, **BUD-2024-008** |
| h4 | A | **BUD-2024-016**, **BUD-2024-016**, **BUD-2024-001**, BUD-2024-003, BUD-2024-003 |
| h4 | B | **BUD-2024-016**, **BUD-2024-001**, **BUD-2024-016**, **BUD-2024-001**, **BUD-2024-001** |
| h4 | C | **BUD-2024-016**, **BUD-2024-016**, **BUD-2024-001**, BUD-2024-003, BUD-2024-009 |
| h4 | D | **BUD-2024-016**, **BUD-2024-016**, **BUD-2024-001**, BUD-2024-003, BUD-2024-009 |
| h5 | A | **BUD-2024-014**, **BUD-2024-014**, **BUD-2024-014**, **BUD-2024-014**, **BUD-2024-013** |
| h5 | B | **BUD-2024-014**, **BUD-2024-013**, **BUD-2024-013**, **BUD-2024-013**, BUD-2024-015 |
| h5 | C | **BUD-2024-013**, **BUD-2024-014**, BUD-2024-015, BUD-2024-016, BUD-2024-004 |
| h5 | D | **BUD-2024-013**, **BUD-2024-014**, BUD-2024-015, BUD-2024-016, BUD-2024-004 |

## Anomalies

- h2: reranking worsened precision@5 (A 1.00 → C 0.40).
- h2: reranking worsened precision@5 (B 0.80 → D 0.40).
- h4: reranking worsened precision@5 (B 1.00 → D 0.60).
- h5: reranking worsened precision@5 (A 1.00 → C 0.40).
- h5: reranking worsened precision@5 (B 0.80 → D 0.40).

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
