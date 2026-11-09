"""List candidate budgets per golden-set query, for MANUAL relevance annotation.

For each query it runs the four configurations (A/B/C/D), takes the union of
their top-5 chunks, groups them by budget and prints, per budget: its id,
sector, project title, which configs retrieved it (and at which rank) and a
short excerpt of the best matching component. It also lists the base budgets
that NO configuration retrieved, so the annotator can spot a relevant budget
that all four missed. It never writes relevance labels — a person does::

    docker compose exec estimator python -m evals.retrieval.list_candidates

Output: stdout + ``evals/retrieval/results/candidates_<timestamp>.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select

from app.dependencies import get_async_session_factory, get_reranker
from app.generation.rag.store.models import ChunkRow
from evals.retrieval.common import (
    CHUNK_TYPES,
    CONFIGS,
    GOLDEN_SET_PATH,
    K,
    RESULTS_DIR,
    embed_queries,
    load_golden_set,
    run_config,
)

_HEADER = re.compile(r"^\[Project: (?P<title>[^\]]*)\]", re.MULTILINE)
_COMPONENT = re.compile(r"^Component: (?P<name>.*)$", re.MULTILINE)
_DESCRIPTION = re.compile(r"^Description: (?P<text>.*)$", re.MULTILINE)
EXCERPT_CHARS = 140


@dataclass
class Candidate:
    budget_id: str
    sector: str
    title: str
    hits: dict[str, list[int]] = field(default_factory=dict)  # config id -> ranks
    excerpt: str = ""


def _describe(content: str) -> tuple[str, str]:
    title = _HEADER.search(content)
    component = _COMPONENT.search(content)
    description = _DESCRIPTION.search(content)
    excerpt = " — ".join(
        part
        for part in (
            component.group("name") if component else "",
            description.group("text") if description else "",
        )
        if part
    )
    return (title.group("title") if title else "?"), excerpt[:EXCERPT_CHARS]


async def _all_base_budgets() -> dict[str, tuple[str, str]]:
    """budget_id -> (sector, project title) for every ingested base budget."""
    async with get_async_session_factory()() as session:
        rows = await session.execute(
            select(ChunkRow.metadata_, ChunkRow.content)
            .where(ChunkRow.chunk_type.in_(CHUNK_TYPES))
            .order_by(ChunkRow.id)
        )
        budgets: dict[str, tuple[str, str]] = {}
        for metadata, content in rows:
            budget_id = metadata.get("budget_id")
            if budget_id and budget_id not in budgets:
                budgets[budget_id] = (
                    str(metadata.get("client_sector", "?")),
                    _describe(content)[0],
                )
        return budgets


async def main(golden_set: Path = GOLDEN_SET_PATH) -> None:
    queries = load_golden_set(golden_set)
    get_reranker().warmup()
    embeddings = await embed_queries(queries)
    all_budgets = await _all_base_budgets()

    lines: list[str] = [f"# Golden-set candidates (union of top-{K} over A/B/C/D)", ""]
    for query in queries:
        candidates: dict[str, Candidate] = {}
        for config in CONFIGS:
            result = await run_config(config, query, embeddings[query.id])
            for rank, chunk in enumerate(result.chunks, start=1):
                budget_id = chunk.budget_id or f"doc-{chunk.document_id}"
                title, excerpt = _describe(chunk.content)
                candidate = candidates.setdefault(
                    budget_id, Candidate(budget_id, chunk.sector, title, excerpt=excerpt)
                )
                candidate.hits.setdefault(config.id, []).append(rank)

        lines += [f"## {query.id}", "", f"> {query.query}", "", f"_Notes: {query.notes}_", ""]
        lines += ["| budget_id | sector | project | retrieved by (config: ranks) | excerpt |"]
        lines += ["|---|---|---|---|---|"]
        ordered = sorted(
            candidates.values(),
            key=lambda c: (-sum(len(r) for r in c.hits.values()), c.budget_id),
        )
        for c in ordered:
            hits = ", ".join(f"{cid}:{','.join(map(str, c.hits[cid]))}" for cid in sorted(c.hits))
            lines.append(f"| {c.budget_id} | {c.sector} | {c.title} | {hits} | {c.excerpt} |")
        missing = sorted(set(all_budgets) - set(candidates))
        if missing:
            lines += ["", "Not retrieved by any config:", ""]
            lines += [f"- {b} ({all_budgets[b][0]}) {all_budgets[b][1]}" for b in missing]
        lines.append("")

    report = "\n".join(lines)
    print(report)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RESULTS_DIR / f"candidates_{stamp}.md"
    path.write_text(report, encoding="utf-8")
    print(f"\nSaved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="List golden-set candidates.")
    parser.add_argument("--golden-set", type=Path, default=GOLDEN_SET_PATH)
    asyncio.run(main(parser.parse_args().golden_set))
