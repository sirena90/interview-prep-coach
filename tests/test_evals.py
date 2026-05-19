"""Tests for the evals/ module — agreement metric, leaderboard, graders.

These cover the pure logic of the evaluation pipeline (no LLM, no API):
the quadratic-weighted Cohen's kappa, the comparison leaderboard, and the
deterministic planner grader.
"""
import pytest

from evals.baskets import PLANNER_BASKET
from evals.graders import grade_planner
from evals.leaderboard import (
    LEADERBOARD,
    add_to_leaderboard,
    format_leaderboard,
    reset_leaderboard,
)
from evals.metrics import kappa_label, quadratic_weighted_kappa


class TestQuadraticWeightedKappa:
    def test_perfect_agreement_is_one(self):
        assert quadratic_weighted_kappa([1, 3, 5, 2], [1, 3, 5, 2]) == 1.0

    def test_constant_grader_scores_zero(self):
        # A lazy grader that always says "3" has no discriminative power.
        kappa = quadratic_weighted_kappa([1, 2, 3, 4, 5], [3, 3, 3, 3, 3])
        assert kappa == pytest.approx(0.0)

    def test_off_by_one_is_still_substantial(self):
        # Quadratic weighting forgives small errors.
        kappa = quadratic_weighted_kappa([5, 3, 1, 5, 3, 1], [4, 2, 2, 4, 2, 2])
        assert 0.5 < kappa < 1.0

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            quadratic_weighted_kappa([1, 2], [1])

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            quadratic_weighted_kappa([], [])


class TestKappaLabel:
    @pytest.mark.parametrize("kappa,expected", [
        (0.90, "near-human"),
        (0.70, "substantial"),
        (0.50, "moderate"),
        (0.30, "fair"),
        (0.10, "poor"),
    ])
    def test_bands(self, kappa, expected):
        assert kappa_label(kappa) == expected


class TestLeaderboard:
    def setup_method(self):
        reset_leaderboard()  # the leaderboard is a module global

    def test_adds_a_new_version_row(self):
        add_to_leaderboard("v1", {"acc": 0.8})
        assert LEADERBOARD == [{"version": "v1", "acc": 0.8}]

    def test_merges_scores_into_an_existing_version(self):
        add_to_leaderboard("v1", {"acc": 0.8})
        add_to_leaderboard("v1", {"kappa": 0.6})
        assert len(LEADERBOARD) == 1
        assert LEADERBOARD[0] == {"version": "v1", "acc": 0.8, "kappa": 0.6}

    def test_separate_versions_are_separate_rows(self):
        add_to_leaderboard("v1", {"acc": 0.8})
        add_to_leaderboard("v2", {"acc": 0.9})
        assert len(LEADERBOARD) == 2

    def test_format_reports_when_empty(self):
        assert "empty" in format_leaderboard()

    def test_format_shows_version_and_metric(self):
        add_to_leaderboard("v1", {"acc": 0.8})
        out = format_leaderboard()
        assert "v1" in out and "acc" in out


class TestGradePlanner:
    def test_planner_basket_all_pass(self, fake_kb):
        result = grade_planner(fake_kb, PLANNER_BASKET)
        assert result["planner_acc"] == 1.0
        assert all(row["pass"] for row in result["rows"])
