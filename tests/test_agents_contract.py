"""Layers 2 & 4 — contract tests for the four LLM agents.

These run with FakeLLM (no API calls). They check the deterministic glue
around each agent: prompt assembly, defensive fallbacks, and how v2's
per-criterion scores are combined in code.
"""
import pytest

from core.agents import (
    ConversationDirectorAgent,
    CoachingSummariserAgent,
    EvaluatorAgent,
    InterviewerAgent,
)
from core.models import (
    CriterionJudgement,
    DirectorAction,
    DirectorChoice,
    Difficulty,
    InterviewerChoice,
    ScoreReport,
    SessionSummary,
    SlotType,
    Topic,
)


class TestInterviewerAgent:
    def test_single_candidate_skips_the_llm(self, fake_llm, make_question):
        q = make_question(qid="only-1")
        choice = InterviewerAgent().ask(
            candidates=[q], topic=Topic.SQL, difficulty=Difficulty.MID
        )
        assert choice.id == "only-1"
        assert fake_llm.call_count == 0  # fast path: no LLM call

    def test_invented_id_falls_back_to_first_candidate(self, fake_llm, make_question):
        c1, c2 = make_question(qid="real-1"), make_question(qid="real-2")
        fake_llm.queue(InterviewerChoice,
                       InterviewerChoice(id="hallucinated", phrased="made up"))
        choice = InterviewerAgent().ask(
            candidates=[c1, c2], topic=Topic.SQL, difficulty=Difficulty.MID
        )
        assert choice.id == "real-1"

    def test_valid_choice_passes_through(self, fake_llm, make_question):
        c1, c2 = make_question(qid="real-1"), make_question(qid="real-2")
        fake_llm.queue(InterviewerChoice,
                       InterviewerChoice(id="real-2", phrased="Tell me about joins."))
        choice = InterviewerAgent().ask(
            candidates=[c1, c2], topic=Topic.SQL, difficulty=Difficulty.MID
        )
        assert choice.id == "real-2"
        assert choice.phrased == "Tell me about joins."

    def test_cv_skills_appear_in_the_prompt(self, fake_llm, make_question, make_cv):
        c1, c2 = make_question(qid="real-1"), make_question(qid="real-2")
        fake_llm.queue(InterviewerChoice,
                       InterviewerChoice(id="real-1", phrased="Tell me about joins."))
        InterviewerAgent().ask(
            candidates=[c1, c2], topic=Topic.SQL, difficulty=Difficulty.MID,
            cv_profile=make_cv(skills=["PostgreSQL"]),
        )
        assert "PostgreSQL" in fake_llm.calls[0].user

    def test_raises_when_no_candidates(self, fake_llm):
        with pytest.raises(ValueError):
            InterviewerAgent().ask(candidates=[], topic=Topic.SQL,
                                   difficulty=Difficulty.MID)


class TestEvaluatorV1:
    def test_makes_one_call_and_returns_the_report(
        self, fake_llm, make_question, make_score_report
    ):
        report = make_score_report(content=4, clarity=3, structure=3, overall=4)
        fake_llm.queue(ScoreReport, report)
        result = EvaluatorAgent(version="v1").evaluate(
            question=make_question(), user_answer="my answer"
        )
        assert result is report
        assert fake_llm.call_count == 1
        assert fake_llm.calls[0].schema == "ScoreReport"


class TestEvaluatorV2:
    """v2 splits grading into one judge per criterion and combines in code."""

    @staticmethod
    def _judgement(score, improvement="do X"):
        return CriterionJudgement(comment="reasoning", score=score,
                                  improvement=improvement)

    def test_makes_three_criterion_calls(self, fake_llm, make_question):
        fake_llm.queue(CriterionJudgement,
                       self._judgement(4), self._judgement(3), self._judgement(5))
        EvaluatorAgent(version="v2").evaluate(
            question=make_question(), user_answer="answer"
        )
        assert fake_llm.call_count == 3
        assert {c.schema for c in fake_llm.calls} == {"CriterionJudgement"}

    def test_overall_is_rounded_mean(self, fake_llm, make_question):
        # 4, 4, 5 -> mean 4.33 -> rounds to 4
        fake_llm.queue(CriterionJudgement,
                       self._judgement(4), self._judgement(4), self._judgement(5))
        result = EvaluatorAgent(version="v2").evaluate(
            question=make_question(), user_answer="a"
        )
        assert result.overall == 4

    def test_overall_rounds_up_above_half(self, fake_llm, make_question):
        # 5, 5, 4 -> mean 4.67 -> rounds to 5
        fake_llm.queue(CriterionJudgement,
                       self._judgement(5), self._judgement(5), self._judgement(4))
        result = EvaluatorAgent(version="v2").evaluate(
            question=make_question(), user_answer="a"
        )
        assert result.overall == 5

    def test_maps_each_score_to_its_dimension(self, fake_llm, make_question):
        # call order is content, clarity, structure
        fake_llm.queue(CriterionJudgement,
                       self._judgement(2, "fix content"),
                       self._judgement(3, "fix clarity"),
                       self._judgement(4, "fix structure"))
        result = EvaluatorAgent(version="v2").evaluate(
            question=make_question(), user_answer="a"
        )
        assert (result.content.score, result.clarity.score,
                result.structure.score) == (2, 3, 4)

    def test_feedback_collects_the_three_improvements(self, fake_llm, make_question):
        fake_llm.queue(CriterionJudgement,
                       self._judgement(3, "fix content"),
                       self._judgement(3, "fix clarity"),
                       self._judgement(3, "fix structure"))
        result = EvaluatorAgent(version="v2").evaluate(
            question=make_question(), user_answer="a"
        )
        assert result.actionable_feedback == [
            "fix content", "fix clarity", "fix structure"
        ]


class TestConversationDirector:
    def test_passes_through_the_llm_choice(
        self, fake_llm, make_question, make_score_report
    ):
        fake_llm.queue(DirectorChoice,
                       DirectorChoice(action=DirectorAction.MOVE_ON, text=""))
        result = ConversationDirectorAgent().decide_next_action(
            question=make_question(), user_answer="answer",
            score_report=make_score_report(),
        )
        assert result.action is DirectorAction.MOVE_ON

    def test_prompt_includes_the_answer_and_overall_score(
        self, fake_llm, make_question, make_score_report
    ):
        fake_llm.queue(DirectorChoice,
                       DirectorChoice(action=DirectorAction.CLARIFY, text="more?"))
        ConversationDirectorAgent().decide_next_action(
            question=make_question(), user_answer="my unique answer text",
            score_report=make_score_report(overall=2),
        )
        prompt = fake_llm.calls[0].user
        assert "my unique answer text" in prompt
        assert "2/5" in prompt


class TestCoachingSummariser:
    def test_returns_summary_and_includes_cv_in_prompt(
        self, fake_llm, make_session, make_turn, make_cv
    ):
        summary = SessionSummary(
            total_turns=1, overall_score=0.6, per_topic={},
            strengths=["s"], gaps=["g"], study_suggestions=["x"],
            coaching_letter="letter",
        )
        fake_llm.queue(SessionSummary, summary)
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID)]
        session = make_session(turns=turns, cv_profile=make_cv(skills=["Tableau"]))

        result = CoachingSummariserAgent().summarise(session)

        assert result is summary
        assert "Tableau" in fake_llm.calls[0].user


class TestCoachingSummariserKbPath:
    """With a kb, the summariser takes the tool-use path (call_llm_with_tools)."""

    def test_with_kb_routes_through_tool_use_and_executor_works(
        self, monkeypatch, make_session, make_turn, make_cv, make_question
    ):
        import core.agents as agents_mod

        reference_q = make_question(qid="da-001")

        class FakeKB:
            def get(self, qid):
                if qid == "da-001":
                    return reference_q
                raise KeyError(qid)

        captured = {}

        def fake_call_llm_with_tools(**kwargs):
            captured["tools"] = kwargs["tools"]
            captured["tool_executor"] = kwargs["tool_executor"]
            return SessionSummary(
                total_turns=1, overall_score=0.5, per_topic={},
                strengths=["s"], gaps=["g"], study_suggestions=["x"],
                coaching_letter="letter",
            )

        monkeypatch.setattr(agents_mod, "call_llm_with_tools",
                            fake_call_llm_with_tools)

        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID)]
        session = make_session(turns=turns, cv_profile=make_cv())
        result = CoachingSummariserAgent(kb=FakeKB()).summarise(session)

        # routed through the tool path, carrying the coach's tools
        assert isinstance(result, SessionSummary)
        assert captured["tools"][0]["name"] == "lookup_reference_answer"

        # the lookup_reference_answer executor behaves correctly
        executor = captured["tool_executor"]
        assert reference_q.reference_answer in executor(
            "lookup_reference_answer", {"question_id": "da-001"}
        )
        assert "No question found" in executor(
            "lookup_reference_answer", {"question_id": "missing"}
        )
        assert "required" in executor(
            "lookup_reference_answer", {"question_id": ""}
        )
        assert "Unknown tool" in executor("some_other_tool", {})
