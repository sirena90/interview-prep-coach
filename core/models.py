"""Pydantic schemas — the contracts between every component in the system.

This module is intentionally dependency-free beyond Pydantic so it can be
imported from any other module without pulling in heavy deps (chromadb,
anthropic, streamlit).

Models grouped by purpose:
  Enums                — Topic, Difficulty, Role, SlotType, DirectorAction
  Knowledge base       — Rubric, Question
  Agent outputs        — DimensionScore, ScoreReport, InterviewerChoice,
                         DirectorChoice, CVProfile
  Session state        — RollingScore, TurnRecord, SessionState
  Final report         — SessionSummary
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, conint


# ============================================================================
# Enums
# ============================================================================

class Topic(str, Enum):
    """All topics covered in the KB across the four roles plus behavioural."""
    # Data Analyst
    SQL = "sql"
    DATA_VISUALIZATION = "data_visualization"
    BUSINESS_METRICS = "business_metrics"
    EXPERIMENTATION = "experimentation"
    STATISTICS = "statistics"
    # QA Engineer
    TEST_DESIGN = "test_design"
    TEST_AUTOMATION = "test_automation"
    API_TESTING = "api_testing"
    BUG_LIFECYCLE = "bug_lifecycle"
    TEST_STRATEGY = "test_strategy"
    # Data Engineer
    DATA_PIPELINES = "data_pipelines"
    DATA_WAREHOUSING = "data_warehousing"
    DATA_MODELING = "data_modeling"
    DISTRIBUTED_SYSTEMS = "distributed_systems"
    # Frontend Developer
    FRONTEND_CORE = "frontend_core"
    FRONTEND_FRAMEWORK = "frontend_framework"
    FRONTEND_PERFORMANCE = "frontend_performance"
    FRONTEND_ACCESSIBILITY = "frontend_accessibility"
    FRONTEND_TESTING = "frontend_testing"
    FRONTEND_SECURITY = "frontend_security"
    # Behavioural (role-agnostic, evaluated via STAR)
    BEHAVIOURAL = "behavioural"


class Difficulty(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"


class Role(str, Enum):
    DATA_ANALYST = "da"
    QA_ENGINEER = "qa"
    DATA_ENGINEER = "de"
    FRONTEND_DEVELOPER = "fe"


class SlotType(str, Enum):
    """Which slot in the 5-question session this turn fills.

    Session pattern: Q1=cover, Q2=reinforce, Q3=behavioral, Q4=cover, Q5=reinforce
    """
    COVER = "cover"
    REINFORCE = "reinforce"
    BEHAVIORAL = "behavioral"


class DirectorAction(str, Enum):
    """The four actions the Conversation Director can take after each answer.

    clarify    — answer was ambiguous, ask user to expand on same question
    followup   — answer was acceptable, ask a deeper follow-up on same topic
    dig_deeper — answer was strong, escalate to a harder variant
    move_on    — done with this question, hand back to Planner for next slot
    """
    CLARIFY = "clarify"
    FOLLOWUP = "followup"
    DIG_DEEPER = "dig_deeper"
    MOVE_ON = "move_on"


# ============================================================================
# Knowledge base items
# ============================================================================

class Rubric(BaseModel):
    """Anchored criteria per evaluation dimension.

    Each string is one observable criterion the evaluator checks for.
    """
    content: list[str] = Field(min_length=1)
    clarity: list[str] = Field(default_factory=list)
    structure: list[str] = Field(default_factory=list)


class Question(BaseModel):
    """One curated question loaded from the JSONL knowledge base."""
    id: str
    topic: Topic
    subtopic: str
    difficulty: Difficulty
    question: str
    reference_answer: str
    rubric: Rubric
    tags: list[str] = Field(default_factory=list)
    skill_tags: list[str] = Field(default_factory=list)
    # ^ concrete tool/concept tags for CV-aware retrieval routing


# ============================================================================
# Agent outputs
# ============================================================================

class DimensionScore(BaseModel):
    """One scored dimension (content / clarity / structure)."""
    score: conint(ge=1, le=5)
    comment: str


class ScoreReport(BaseModel):
    """Evaluator output for one user answer.

    Three scored dimensions + actionable feedback bullets + an overall score.
    """
    content: DimensionScore
    clarity: DimensionScore
    structure: DimensionScore
    actionable_feedback: list[str] = Field(min_length=1, max_length=4)
    overall: conint(ge=1, le=5)

    def normalized(self) -> float:
        """Map overall 1..5 -> 0..1 for the Planner's rolling topic scores."""
        return (self.overall - 1) / 4.0


class CriterionJudgement(BaseModel):
    """One per-criterion judge's output in the v2 (split) Evaluator.

    The v2 Evaluator grades content / clarity / structure with three separate
    focused prompts ("one criterion per judge") and combines them in code.
    `comment` is listed before `score` on purpose: it asks the model to reason
    before committing to a number, which makes the verdict auditable.
    """
    comment: str
    score: conint(ge=1, le=5)
    improvement: str


class InterviewerChoice(BaseModel):
    """Interviewer agent output: which candidate was picked + phrasing."""
    id: str
    phrased: str = Field(min_length=3)


class DirectorChoice(BaseModel):
    """Conversation Director output: what to do after each answer.

    If action == MOVE_ON, `text` is empty; planner picks the next question.
    Otherwise `text` is the next message shown to the user, and the same
    question stays active for the next round of evaluation.
    """
    action: DirectorAction
    text: str = ""


class CVProfile(BaseModel):
    """Extracted profile from the user's CV.

    Produced once by the CV Parser when the user uploads their CV. Stored
    in SessionState and consumed by (a) the Retriever for CV-aware question
    routing and (b) the Coaching Summariser for personalised final feedback.
    """
    skills: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    seniority: str = ""  # rough estimate: "entry" / "mid" / "senior"
    claimed_strengths: list[str] = Field(default_factory=list)
    likely_gaps: list[str] = Field(default_factory=list)


# ============================================================================
# Session state (held in st.session_state in Streamlit)
# ============================================================================

class RollingScore(BaseModel):
    """EWMA-style per-topic score in [0, 1]. Starts at 0.5 (neutral prior).

    alpha = 0.5 means the most recent answer carries 50 percent weight,
    the previous one 25 percent, then 12.5 percent, etc.
    """
    score: float = 0.5
    count: int = 0

    def update(self, new_score: float, alpha: float = 0.5) -> None:
        self.score = alpha * new_score + (1 - alpha) * self.score
        self.count += 1


class TurnRecord(BaseModel):
    """One completed turn in the session."""
    turn_id: int  # 1..target_turns
    slot_type: SlotType
    question_id: str
    question_text: str
    topic: Topic
    difficulty: Difficulty
    user_answer: str
    score_report: Optional[ScoreReport] = None


class SessionState(BaseModel):
    """Full in-memory state of one interview session.

    Lives in st.session_state and is mutated as the session progresses.
    No persistence in the simplified version; state is lost on page refresh.
    """
    session_id: str
    role: Role
    difficulty: Difficulty
    target_turns: int = 5
    turns: list[TurnRecord] = Field(default_factory=list)
    topic_scores: dict[Topic, RollingScore] = Field(default_factory=dict)
    asked_ids: set[str] = Field(default_factory=set)
    cv_profile: Optional[CVProfile] = None
    current_question_id: Optional[str] = None
    started_at: datetime
    ended_at: Optional[datetime] = None

    def turn_count(self) -> int:
        return len(self.turns)

    def is_complete(self) -> bool:
        return self.ended_at is not None or self.turn_count() >= self.target_turns


# ============================================================================
# End-of-session output
# ============================================================================

class SessionSummary(BaseModel):
    """Coaching Summariser output: personalised final report for the user."""
    total_turns: int
    overall_score: float = Field(ge=0, le=1)
    per_topic: dict[Topic, float]
    strengths: list[str]
    gaps: list[str]
    study_suggestions: list[str]
    coaching_letter: str  # free-form personalised text that references the CV
