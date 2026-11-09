"""Unit tests for Reciprocal Rank Fusion (pure, no I/O)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion


def test_document_in_both_rankings_outranks_single_branch_docs():
    # id 2 appears in BOTH branches (near the top of each); ids 1 and 3 each
    # appear in only one. Two contributions beat one — the core RRF behaviour.
    vector = [1, 2]
    lexical = [2, 3]
    fused = reciprocal_rank_fusion([vector, lexical], k=60)
    ids = [cid for cid, _ in fused]
    assert ids[0] == 2


def test_score_is_sum_of_reciprocal_ranks():
    # Ranks start at 1 (Cormack et al.): with k=60, id 5 at rank 1 → 1/61, id 6
    # at rank 2 → 1/62. An id in two rankings sums both contributions.
    fused = dict(reciprocal_rank_fusion([[5, 6]], k=60))
    assert fused[5] == pytest.approx(1 / 61)
    assert fused[6] == pytest.approx(1 / 62)
    both = dict(reciprocal_rank_fusion([[5, 6], [6]], k=60))
    assert both[6] == pytest.approx(1 / 62 + 1 / 61)


def test_smoothing_constant_changes_ordering_sensitivity():
    # A small k lets a single #1 dominate; a large k rewards appearing in both.
    vector = [1, 2]
    lexical = [2, 1]
    small_k = dict(reciprocal_rank_fusion([vector, lexical], k=1))
    # With k=1: id1 = 1/2 + 1/3; id2 = 1/3 + 1/2 → exact tie.
    assert small_k[1] == pytest.approx(small_k[2])
    # A single #1 vs. two mid ranks: small k rewards the #1, large k the overlap.
    rankings = [[1, 3, 2], [4, 5, 2]]
    assert reciprocal_rank_fusion(rankings, k=1)[0][0] == 1
    assert reciprocal_rank_fusion(rankings, k=60)[0][0] == 2


def test_ties_break_by_ascending_id_for_determinism():
    fused = reciprocal_rank_fusion([[10, 20], [20, 10]], k=60)
    # Symmetric input → equal scores and equal best rank → ascending-id order.
    assert [cid for cid, _ in fused] == [10, 20]


def test_ties_break_by_best_individual_rank_before_id():
    # id 9: ranks 2 and 3; id 3: ranks 1 and ... absent → compare equal-score
    # pairs built so the sums tie but the best single position differs.
    # k=1: id 50 → 1/2 (rank 1 once) ; id 7 → 1/4 + 1/4 (rank 3 twice) = 1/2.
    fused = reciprocal_rank_fusion([[50, 1, 7], [2, 3, 7]], k=1)
    scores = dict(fused)
    assert scores[50] == pytest.approx(scores[7])
    ids = [cid for cid, _ in fused]
    assert ids.index(50) < ids.index(7)  # best rank 1 beats best rank 3 despite id


def test_duplicate_ids_within_a_ranking_count_once_at_best_position():
    # The repeated id 7 must not be double-credited; only its first (best) slot.
    once = dict(reciprocal_rank_fusion([[7, 8]], k=60))
    twice = dict(reciprocal_rank_fusion([[7, 8, 7]], k=60))
    assert once[7] == pytest.approx(twice[7])


def test_empty_branch_contributes_nothing():
    fused = reciprocal_rank_fusion([[1, 2], []], k=60)
    assert {cid for cid, _ in fused} == {1, 2}


def test_no_rankings_or_all_empty_returns_empty():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_single_ranking_preserves_its_order():
    assert [cid for cid, _ in reciprocal_rank_fusion([[3, 1, 2]], k=60)] == [3, 1, 2]


def test_limit_caps_the_fused_output():
    fused = reciprocal_rank_fusion([[1, 2, 3], [4, 5, 6]], k=60, limit=2)
    assert len(fused) == 2
    assert reciprocal_rank_fusion([[1, 2]], k=60, limit=0) == []
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], k=60, limit=-1)


def test_key_fuses_objects_by_id_and_keeps_first_rankings_object():
    vector = [SimpleNamespace(id=1, src="vector"), SimpleNamespace(id=2, src="vector")]
    lexical = [SimpleNamespace(id=2, src="lexical"), SimpleNamespace(id=3, src="lexical")]
    fused = reciprocal_rank_fusion([vector, lexical], k=60, key=lambda c: c.id)
    assert [c.id for c, _ in fused] == [2, 1, 3]
    assert fused[0][0].src == "vector"  # object from the first ranking wins


def test_non_positive_k_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([[1]], k=0)
