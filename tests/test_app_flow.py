"""Layer 3 — orchestration / wiring tests for the per-answer flow.

These tests target `app.evaluate_and_decide` — the pure function pulled out
of `_handle_user_answer` so the seam between the UI and the agents is
testable without Streamlit. They are the regression net for the class of
bug we previously hit, where a follow-up answer was graded against the
*original* question's rubric.

What each test asserts about the wiring:
  - the active question used for grading flips to the follow-up when one
    is set (this is the original bug);
  - a clean MOVE_ON clears any active follow-up;
  - a non-MOVE_ON with a valid rubric installs a synthetic follow-up
    Question that downstream code will grade against;
  - the self-consistency check degrades a non-MOVE_ON whose generated
    rubric grades its own reference at <4;
  - the 1-round loop guard suppresses further follow-ups even if the
    Director would otherwise pick one.
"""
import pytest

from app import evaluate_and_decide
from core.models import (
    DirectorAction,
    DirectorChoice,
    Rubric,
)


# ============================================================================
# Recording fakes — agents that record what they were called with
# ============================================================================

class RecordingEvaluator:
    """Fake evaluator that records every (question, answer) pair.

    `report_queue` is consumed in order; tests use it to script Evaluator
    verdicts (e.g. low score on the self-consistency check to force a
    degrade).
    """

    def __init__(self, report_queue):
        self.calls = []
        self._queue = list(report_queue)

    def evaluate(self, *, question, user_answer):
        self.calls.append({"question": question, "user_answer": user_answer})
        if not self._queue:
            raise AssertionError(
                f"RecordingEvaluator queue empty on call #{len(self.calls)}"
            )
        return self._queue.pop(0)


class ScriptedDirector:
    """Fake director that returns a pre-built DirectorChoice and records
    which question + answer it was called with."""

    def __init__(self, choice):
        self.calls = []
        self._choice = choice

    def decide_next_action(self, *, question, user_answer, score_report,
                          turn_history_for_question):
        self.calls.append({
            "question": question,
            "user_answer": user_answer,
            "score_report": score_report,
            "turn_history_for_question": turn_history_for_question,
        })
        # Pydantic objects are mutable; clone via model_copy so the loop
        # guard's mutation in evaluate_and_decide doesn't leak into the next
        # test that reuses the same fixture choice.
        return self._choice.model_copy(deep=True)


# ============================================================================
# Helpers
# ============================================================================

def _good_followup_choice():
    """A DirectorChoice the self-consistency check should accept."""
    return DirectorChoice(
        action=DirectorAction.DIG_DEEPER,
        text="How would you prioritise bugs you found during testing?",
        follow_up_rubric=Rubric(
            content=["names a prioritisation axis (impact/frequency/severity)",
                    "ties priority to user or business outcome"],
            clarity=["uses concrete examples rather than abstractions"],
            structure=["leads with the axis, then the rationale"],
        ),
        follow_up_reference=(
            "Prioritise by user impact and frequency: blockers first, then "
            "high-impact regressions, then minor UI bugs. Cross-check with "
            "release risk to escalate near-deadline issues."
        ),
    )


# ============================================================================
# Tests
# ============================================================================

class TestEvaluateAndDecide:
    """The pure orchestrator that decides which question is graded and
    whether a follow-up is installed for the next turn."""

    def test_follow_up_answer_is_graded_against_follow_up_question(
        self, make_question, make_score_report,
    ):
        # The bug we previously shipped: a candidate's reply to a dig_deeper
        # question was graded against the *original* slot rubric. This test
        # locks the fixed behaviour in: when followup_question is set, that
        # is the Question passed to the Evaluator.
        slot_q = make_question(qid="qa-002")
        followup_q = make_question(qid="qa-002-followup")
        evaluator = RecordingEvaluator([make_score_report(overall=3)])
        director = ScriptedDirector(
            DirectorChoice(action=DirectorAction.MOVE_ON, text="")
        )

        result = evaluate_and_decide(
            slot_question=slot_q,
            followup_question=followup_q,
            answer="prioritise by user impact and frequency",
            director_rounds=1,
            agents={"evaluator": evaluator, "director": director},
        )

        # Both the result and the Evaluator's call log must agree.
        assert result["active_question"] is followup_q
        assert evaluator.calls[0]["question"] is followup_q
        # And the Director sees the same active question on the follow-up turn.
        assert director.calls[0]["question"] is followup_q

    def test_no_follow_up_grades_against_slot_question(
        self, make_question, make_score_report,
    ):
        slot_q = make_question(qid="da-001")
        evaluator = RecordingEvaluator([make_score_report(overall=4)])
        director = ScriptedDirector(
            DirectorChoice(action=DirectorAction.MOVE_ON, text="")
        )

        result = evaluate_and_decide(
            slot_question=slot_q,
            followup_question=None,
            answer="INNER vs LEFT explanation",
            director_rounds=0,
            agents={"evaluator": evaluator, "director": director},
        )

        assert result["active_question"] is slot_q
        assert evaluator.calls[0]["question"] is slot_q

    def test_move_on_returns_no_next_follow_up(
        self, make_question, make_score_report,
    ):
        # MOVE_ON must clear any active follow-up — the UI layer reads
        # next_followup_question == None as "wipe st.session_state.followup_question".
        evaluator = RecordingEvaluator([make_score_report(overall=4)])
        director = ScriptedDirector(
            DirectorChoice(action=DirectorAction.MOVE_ON, text="")
        )

        result = evaluate_and_decide(
            slot_question=make_question(),
            followup_question=None,
            answer="answer",
            director_rounds=0,
            agents={"evaluator": evaluator, "director": director},
        )

        assert result["choice"].action is DirectorAction.MOVE_ON
        assert result["next_followup_question"] is None

    def test_non_move_on_with_valid_rubric_installs_follow_up_question(
        self, make_question, make_score_report,
    ):
        # The Director picks dig_deeper with a valid rubric + reference; the
        # self-consistency check passes (score 5/5). The orchestrator must
        # build a synthetic follow-up Question with the generated rubric so
        # the next user reply gets graded correctly.
        slot_q = make_question(qid="qa-002")
        # First Evaluator call grades the user's answer; second is the
        # self-consistency check inside validate_followup_choice.
        evaluator = RecordingEvaluator([
            make_score_report(overall=4),  # user answer
            make_score_report(overall=5),  # self-consistency on Director's reference
        ])
        director = ScriptedDirector(_good_followup_choice())

        result = evaluate_and_decide(
            slot_question=slot_q,
            followup_question=None,
            answer="A good bug report has a title, repro, env...",
            director_rounds=0,
            agents={"evaluator": evaluator, "director": director},
        )

        choice = result["choice"]
        followup_q = result["next_followup_question"]
        assert choice.action is DirectorAction.DIG_DEEPER
        assert followup_q is not None
        # The synthetic question carries the Director's rubric and reference
        # — and that is what the next user answer will be graded against.
        assert followup_q.rubric is choice.follow_up_rubric
        assert followup_q.reference_answer == choice.follow_up_reference
        assert followup_q.question == choice.text
        # Slot classification (topic / subtopic / difficulty) is preserved.
        assert followup_q.topic is slot_q.topic
        assert followup_q.difficulty is slot_q.difficulty

    def test_self_consistency_failure_degrades_non_move_on_to_move_on(
        self, make_question, make_score_report,
    ):
        # If the Director's reference answer scores <4 against its own rubric
        # the rubric is broken and we MUST not grade a real candidate
        # against it. Defence #3 catches this at runtime.
        evaluator = RecordingEvaluator([
            make_score_report(overall=2),  # user answer
            make_score_report(overall=2),  # self-consistency check fails
        ])
        director = ScriptedDirector(_good_followup_choice())

        result = evaluate_and_decide(
            slot_question=make_question(),
            followup_question=None,
            answer="weak answer",
            director_rounds=0,
            agents={"evaluator": evaluator, "director": director},
        )

        assert result["choice"].action is DirectorAction.MOVE_ON
        assert result["next_followup_question"] is None

    def test_loop_guard_forces_move_on_after_one_round(
        self, make_question, make_score_report,
    ):
        # After 1 follow-up round, the next Director output must be
        # downgraded to MOVE_ON regardless of what the LLM returns. This
        # keeps sessions short and avoids grilling the candidate.
        evaluator = RecordingEvaluator([make_score_report(overall=4)])
        director = ScriptedDirector(_good_followup_choice())

        result = evaluate_and_decide(
            slot_question=make_question(),
            followup_question=make_question(qid="q-followup"),
            answer="follow-up answer",
            director_rounds=1,  # already used one round
            agents={"evaluator": evaluator, "director": director},
        )

        assert result["choice"].action is DirectorAction.MOVE_ON
        assert result["choice"].follow_up_rubric is None
        assert result["next_followup_question"] is None
        # Critically: only ONE evaluator call — the self-consistency check
        # must not run when the choice was already downgraded.
        assert len(evaluator.calls) == 1
