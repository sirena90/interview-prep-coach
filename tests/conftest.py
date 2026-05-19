"""Shared fixtures for the test suite.

Factories are exposed as fixtures that RETURN a builder function, so each
test constructs exactly the objects it needs without repeating the Pydantic
boilerplate. FakeLLM lets agent tests run with zero API calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pytest

from core.models import (
    CVProfile,
    Difficulty,
    DimensionScore,
    Question,
    Role,
    Rubric,
    ScoreReport,
    SessionState,
    SlotType,
    Topic,
    TurnRecord,
)


# ============================================================================
# FakeLLM — drop-in replacement for core.llm.call_llm
# ============================================================================

@dataclass
class RecordedCall:
    """One captured call to the fake LLM — lets tests assert on prompts."""
    system: str
    user: str
    schema: str
    kwargs: dict


class FakeLLM:
    """Stand-in for call_llm: returns canned objects, records prompts.

    Usage:
        fake.queue(ScoreReport, some_report)
        result = agent.evaluate(...)            # pops the queued report
        assert "PostgreSQL" in fake.calls[0].user
    """

    def __init__(self) -> None:
        self._responses: dict[str, list[Any]] = {}
        self.calls: list[RecordedCall] = []

    def queue(self, schema: type, *objects: Any) -> "FakeLLM":
        """Queue one or more responses for a schema, returned in call order."""
        self._responses.setdefault(schema.__name__, []).extend(objects)
        return self

    def __call__(self, *, system: str, user: str, schema: type, **kwargs: Any):
        self.calls.append(
            RecordedCall(system=system, user=user, schema=schema.__name__, kwargs=kwargs)
        )
        queue = self._responses.get(schema.__name__, [])
        if not queue:
            raise AssertionError(
                f"FakeLLM: no canned {schema.__name__} response left "
                f"(this was call #{len(self.calls)})"
            )
        return queue.pop(0)

    @property
    def call_count(self) -> int:
        return len(self.calls)


@pytest.fixture
def fake_llm(monkeypatch) -> FakeLLM:
    """Patch call_llm in every module that imports it, so no test hits the API."""
    import core.agents as agents_mod
    import core.cv_parser as cv_mod

    fake = FakeLLM()
    monkeypatch.setattr(agents_mod, "call_llm", fake)
    monkeypatch.setattr(cv_mod, "call_llm", fake)
    return fake


# ============================================================================
# Fake knowledge base
# ============================================================================

# Topics each role is allowed to be asked about. Mirrors the real KB's
# topics_for_role() so the fake KB below behaves like the real one.
_ROLE_TOPICS: dict[Role, set[Topic]] = {
    Role.DATA_ANALYST: {
        Topic.SQL, Topic.DATA_VISUALIZATION, Topic.BUSINESS_METRICS,
        Topic.EXPERIMENTATION, Topic.STATISTICS,
    },
    Role.QA_ENGINEER: {
        Topic.TEST_DESIGN, Topic.TEST_AUTOMATION, Topic.API_TESTING,
        Topic.BUG_LIFECYCLE, Topic.TEST_STRATEGY,
    },
    Role.DATA_ENGINEER: {
        Topic.DATA_PIPELINES, Topic.DATA_WAREHOUSING, Topic.DATA_MODELING,
        Topic.DISTRIBUTED_SYSTEMS,
    },
    Role.FRONTEND_DEVELOPER: {
        Topic.FRONTEND_CORE, Topic.FRONTEND_FRAMEWORK, Topic.FRONTEND_PERFORMANCE,
        Topic.FRONTEND_ACCESSIBILITY, Topic.FRONTEND_TESTING, Topic.FRONTEND_SECURITY,
    },
}


class FakeKB:
    """Minimal stand-in for KnowledgeBase.

    SimplePlanner only ever calls topics_for_role(), so that is all this
    fake implements — no Chroma, no JSONL, no embedding model.
    """

    def topics_for_role(self, role: Role) -> set[Topic]:
        return set(_ROLE_TOPICS[role])


@pytest.fixture
def fake_kb() -> FakeKB:
    return FakeKB()


# ============================================================================
# Model builders
# ============================================================================

@pytest.fixture
def make_score_report():
    """Build a ScoreReport. `feedback` left unset -> a valid one-item list;
    pass an explicit list (incl. []) to exercise validation bounds."""
    sentinel = object()

    def _make(content=3, clarity=3, structure=3, overall=3, feedback=sentinel):
        items = ["improve X"] if feedback is sentinel else feedback
        return ScoreReport(
            content=DimensionScore(score=content, comment="c"),
            clarity=DimensionScore(score=clarity, comment="c"),
            structure=DimensionScore(score=structure, comment="c"),
            actionable_feedback=items,
            overall=overall,
        )

    return _make


@pytest.fixture
def make_question():
    def _make(qid="q-001", topic=Topic.SQL, difficulty=Difficulty.MID,
              rubric=None, skill_tags=None):
        return Question(
            id=qid,
            topic=topic,
            subtopic="joins",
            difficulty=difficulty,
            question="What is the difference between an INNER JOIN and a LEFT JOIN?",
            reference_answer="INNER JOIN returns only matching rows; LEFT JOIN "
                             "returns all left-table rows plus matches.",
            rubric=rubric or Rubric(
                content=["defines INNER as matching", "defines LEFT correctly"],
                clarity=["uses correct SQL terminology"],
                structure=["definition first, then contrast"],
            ),
            skill_tags=skill_tags or ["sql", "joins"],
        )

    return _make


@pytest.fixture
def make_cv():
    def _make(skills=None, projects=None, seniority="mid"):
        return CVProfile(
            skills=skills or [],
            projects=projects or [],
            seniority=seniority,
        )

    return _make


@pytest.fixture
def make_turn(make_score_report):
    def _make(turn_id, slot_type, topic, difficulty, overall=3, scored=True):
        return TurnRecord(
            turn_id=turn_id,
            slot_type=slot_type,
            question_id=f"q-{turn_id}",
            question_text=f"Question {turn_id}",
            topic=topic,
            difficulty=difficulty,
            user_answer="some answer",
            score_report=make_score_report(overall=overall) if scored else None,
        )

    return _make


@pytest.fixture
def make_session():
    def _make(role=Role.DATA_ANALYST, difficulty=Difficulty.MID, turns=None,
              cv_profile=None):
        return SessionState(
            session_id="test-session",
            role=role,
            difficulty=difficulty,
            turns=turns or [],
            cv_profile=cv_profile,
            started_at=datetime(2026, 5, 17, 12, 0, 0),
        )

    return _make
