"""Reciprocal Rank Fusion (RRF) — combine independent rankings into one.

RRF fuses rankings by POSITION, not by score, which is exactly what we want when
the two branches produce incomparable scores: cosine distance (lower = better)
from the vector branch and ``ts_rank_cd`` (higher = better) from the lexical one.
Normalising those onto a shared scale is fiddly and corpus-dependent; RRF sidesteps
it by only trusting the order each branch puts things in.

The score of an item is ``sum(1 / (k + rank))`` over every ranking it appears in,
with ``rank`` starting at 1 (Cormack et al., 2009). The smoothing constant ``k``
damps the contribution of top positions: a large ``k`` flattens the curve so an
item must rank well in *several* rankings to win, while a small ``k`` lets a
single #1 dominate. 60 is the value from the original paper and a sane default.

Pure and synchronous on purpose — trivially unit-testable, no I/O.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Sequence
from typing import Any, TypeVar

DEFAULT_RRF_K = 60

# Scores are rounded before sorting so that mathematically equal sums that differ
# only by float error still tie, and the deterministic tie-break decides.
_SCORE_PRECISION = 12

T = TypeVar("T")


def _identity(item: Any) -> Any:
    return item


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[T]],
    *,
    k: int = DEFAULT_RRF_K,
    limit: int | None = None,
    key: Callable[[T], Hashable] = _identity,
) -> list[tuple[T, float]]:
    """Fuse several rankings into one, best first.

    Parameters
    ----------
    rankings:
        One sequence per branch, each already ordered best→worst. An empty
        ranking simply adds no score. Duplicates within a single ranking count
        only once, at their first (best) position.
    k:
        RRF smoothing constant (see module docstring). Must be positive.
    limit:
        Keep at most this many fused items (``None`` = all). Must be ``>= 0``.
    key:
        Extracts the identity of an item (default: the item itself, e.g. an int
        chunk id). Items with the same key are the same document across rankings;
        the object kept is the one from the FIRST ranking that contains it.

    Returns
    -------
    list[tuple[T, float]]
        ``(item, rrf_score)`` sorted by score descending. Ties break
        deterministically by best individual rank (ascending), then by key
        (ascending), so the output never depends on dict/set ordering.
    """
    if k <= 0:
        raise ValueError("RRF smoothing constant k must be positive")
    if limit is not None and limit < 0:
        raise ValueError("limit must be >= 0")

    scores: dict[Hashable, float] = {}
    best_rank: dict[Hashable, int] = {}
    items: dict[Hashable, T] = {}
    for ranking in rankings:
        seen: set[Hashable] = set()
        for rank, item in enumerate(ranking, start=1):
            item_key = key(item)
            if item_key in seen:
                continue
            seen.add(item_key)
            items.setdefault(item_key, item)
            scores[item_key] = scores.get(item_key, 0.0) + 1.0 / (k + rank)
            best_rank[item_key] = min(best_rank.get(item_key, rank), rank)

    ordered = sorted(
        scores,
        key=lambda item_key: (
            -round(scores[item_key], _SCORE_PRECISION),
            best_rank[item_key],
            item_key,
        ),
    )
    if limit is not None:
        ordered = ordered[:limit]
    return [(items[item_key], scores[item_key]) for item_key in ordered]
