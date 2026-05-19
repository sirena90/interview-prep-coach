"""Layer 1 — deterministic unit tests for the Pydantic models.

Exercises pure Python logic (no LLM, no I/O): the rolling-score EWMA,
score normalisation, schema validation bounds, and session completion.
"""
import pytest
from pydantic import ValidationError

from core.models import Difficulty, DimensionScore, RollingScore, SlotType, Topic


class TestRollingScore:
    def test_defaults_to_neutral_prior(self):
        rs = RollingScore()
        assert rs.score == 0.5
        assert rs.count == 0

    def test_single_update_is_ewma(self):
        rs = RollingScore()
        rs.update(1.0)  # 0.5*1.0 + 0.5*0.5
        assert rs.score == pytest.approx(0.75)
        assert rs.count == 1

    def test_repeated_updates_decay_old_values(self):
        rs = RollingScore()
        rs.update(1.0)  # -> 0.75
        rs.update(1.0)  # 0.5*1.0 + 0.5*0.75
        assert rs.score == pytest.approx(0.875)
        assert rs.count == 2

    def test_alpha_controls_recency_weight(self):
        rs = RollingScore()
        rs.update(1.0, alpha=0.1)  # 0.1*1.0 + 0.9*0.5
        assert rs.score == pytest.approx(0.55)


class TestScoreReportNormalized:
    @pytest.mark.parametrize("overall,expected", [(1, 0.0), (3, 0.5), (5, 1.0)])
    def test_maps_1_5_to_0_1(self, make_score_report, overall, expected):
        assert make_score_report(overall=overall).normalized() == pytest.approx(expected)


class TestDimensionScoreValidation:
    @pytest.mark.parametrize("score", [1, 3, 5])
    def test_accepts_scores_in_range(self, score):
        DimensionScore(score=score, comment="ok")

    @pytest.mark.parametrize("score", [0, 6, -1])
    def test_rejects_scores_out_of_range(self, score):
        with pytest.raises(ValidationError):
            DimensionScore(score=score, comment="bad")


class TestScoreReportFeedbackBounds:
    def test_rejects_empty_feedback(self, make_score_report):
        with pytest.raises(ValidationError):
            make_score_report(feedback=[])

    def test_rejects_more_than_four_feedback_items(self, make_score_report):
        with pytest.raises(ValidationError):
            make_score_report(feedback=["a", "b", "c", "d", "e"])

    def test_accepts_one_to_four_feedback_items(self, make_score_report):
        make_score_report(feedback=["a"])
        make_score_report(feedback=["a", "b", "c", "d"])


class TestSessionStateCompletion:
    def test_turn_count_reflects_recorded_turns(self, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID)]
        assert make_session(turns=turns).turn_count() == 1

    def test_complete_when_target_turns_reached(self, make_session, make_turn):
        turns = [make_turn(i, SlotType.COVER, Topic.SQL, Difficulty.MID) for i in range(1, 6)]
        assert make_session(turns=turns).is_complete()

    def test_not_complete_below_target(self, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID)]
        assert not make_session(turns=turns).is_complete()
