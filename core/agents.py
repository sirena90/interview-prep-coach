"""Four LLM agents that drive the interview coaching session.

Each agent is a small class with:
  - A system prompt (instructions for the model)
  - A user prompt template (filled in with per-call data)
  - One public method that calls llm.call_llm() and returns a validated
    Pydantic object

Agents:
  InterviewerAgent         — picks one question from candidates & phrases it
  EvaluatorAgent           — grades the user answer against the rubric
  ConversationDirectorAgent — decides next action (clarify/followup/dig_deeper/move_on)
  CoachingSummariserAgent   — writes the final personalised coaching report
"""
from __future__ import annotations

import json
from typing import Optional

from core.llm import call_llm
from core.models import (
    CVProfile,
    DirectorChoice,
    Difficulty,
    InterviewerChoice,
    Question,
    Role,
    ScoreReport,
    SessionState,
    SessionSummary,
    Topic,
    TurnRecord,
)


# ============================================================================
# Interviewer Agent
# ============================================================================

INTERVIEWER_SYSTEM = """You are an interview coach choosing the next question to ask the candidate.

You will receive a list of candidate questions from the knowledge base. Your job:
1. Pick the most appropriate question for the candidate (semantic fit, variety).
2. Paraphrase it naturally so it sounds like a real interviewer talking.
3. If the candidate's CV mentions a project or skill that naturally connects to the question,
   reference it in your phrasing. Do NOT force a connection where none exists.

You MUST return JSON matching this exact schema (no prose, no code fences):
{schema}
"""

INTERVIEWER_USER = """Topic: {topic}
Difficulty: {difficulty}

Candidate questions to pick from:
{candidates_json}

User's CV signals (use only where natural):
{cv_signals}

Pick the ID of the question you choose, and write your phrased version.
Phrasing should be one sentence, sound like a real interviewer, and reference CV only if it
genuinely fits."""


class InterviewerAgent:
    """Picks one question from retrieved candidates and phrases it naturally."""

    def ask(
        self,
        *,
        candidates: list[Question],
        topic: Topic,
        difficulty: Difficulty,
        cv_profile: Optional[CVProfile] = None,
    ) -> InterviewerChoice:
        if not candidates:
            raise ValueError("InterviewerAgent received no candidates")

        # Fast path: only one candidate → no need for an LLM call.
        if len(candidates) == 1:
            q = candidates[0]
            return InterviewerChoice(id=q.id, phrased=q.question)

        candidates_repr = [
            {"id": q.id, "question": q.question, "subtopic": q.subtopic}
            for q in candidates
        ]
        cv_text = _format_cv_for_prompt(cv_profile)

        system = INTERVIEWER_SYSTEM.format(
            schema=json.dumps(InterviewerChoice.model_json_schema(), indent=2)
        )
        user = INTERVIEWER_USER.format(
            topic=topic.value,
            difficulty=difficulty.value,
            candidates_json=json.dumps(candidates_repr, indent=2),
            cv_signals=cv_text,
        )

        choice = call_llm(system=system, user=user, schema=InterviewerChoice)

        # Defensive: if the model invented an id not in candidates, fall back.
        valid_ids = {q.id for q in candidates}
        if choice.id not in valid_ids:
            return InterviewerChoice(id=candidates[0].id, phrased=candidates[0].question)
        return choice


# ============================================================================
# Evaluator Agent
# ============================================================================

EVALUATOR_SYSTEM = """You are a strict but fair interview evaluator.

Your job: read the candidate's answer to one interview question and grade it against the rubric.

Scoring:
- content   (1-5): is the answer factually correct and complete?
- clarity   (1-5): is it explained in a clear, well-articulated way?
- structure (1-5): is it organised logically (definition then example, etc.)?
- overall   (1-5): your overall judgement, not just the average.

Also write 2-3 actionable feedback bullets — concrete things the candidate could do to improve.

You MUST return JSON matching this exact schema (no prose, no code fences):
{schema}
"""

EVALUATOR_USER = """Question: {question_text}

Rubric (criteria the evaluator looks for):
content criteria:
{content_criteria}
clarity criteria:
{clarity_criteria}
structure criteria:
{structure_criteria}

Reference answer (what a strong answer looks like):
{reference_answer}

Candidate's answer:
{user_answer}

Grade strictly. Cite specific gaps in your comments. Write the JSON now."""


class EvaluatorAgent:
    """Grades a user answer against a question's rubric. Returns a ScoreReport."""

    def evaluate(
        self,
        *,
        question: Question,
        user_answer: str,
    ) -> ScoreReport:
        system = EVALUATOR_SYSTEM.format(
            schema=json.dumps(ScoreReport.model_json_schema(), indent=2)
        )
        user = EVALUATOR_USER.format(
            question_text=question.question,
            content_criteria=_bullets(question.rubric.content),
            clarity_criteria=_bullets(question.rubric.clarity) or "(none)",
            structure_criteria=_bullets(question.rubric.structure) or "(none)",
            reference_answer=question.reference_answer,
            user_answer=user_answer.strip() or "(empty answer)",
        )
        return call_llm(system=system, user=user, schema=ScoreReport)


# ============================================================================
# Conversation Director Agent — the agentic core
# ============================================================================

DIRECTOR_SYSTEM = """You are the conductor of an interview-prep conversation.

After each answer, you choose ONE action from this fixed set:

  clarify    — the answer is ambiguous or incomplete in a way that's worth asking back.
               Write a short clarifying question that points to the gap without giving the answer.

  followup   — the answer is acceptable but shallow. Ask a deeper question ON THE SAME topic,
               extending the conversation rather than starting fresh.

  dig_deeper — the answer is strong (overall >= 4). Escalate to a harder variant on the same topic.

  move_on    — the question is exhausted. Hand back to the planner; it picks the next slot.

Loop guards:
- If the candidate has already had 2 clarify/followup/dig_deeper turns on the SAME question, choose move_on.
- Default to move_on unless one of the other actions clearly applies.

You MUST return JSON matching this exact schema (no prose, no code fences):
{schema}
"""

DIRECTOR_USER = """Question just asked: {question_text}

Candidate's answer: {user_answer}

Score report:
- content:   {content_score}/5 — {content_comment}
- clarity:   {clarity_score}/5 — {clarity_comment}
- structure: {structure_score}/5 — {structure_comment}
- overall:   {overall}/5

Conversation history on this question (so far):
{turn_history}

Decide ONE action and (if not move_on) write the next message to the candidate.
If move_on, leave `text` empty."""


class ConversationDirectorAgent:
    """The agentic loop: picks the next action from a fixed action set.

    This is what makes the system formally agentic — it observes (score report),
    chooses an action (from { clarify, followup, dig_deeper, move_on }), and
    feeds the choice back into the loop.
    """

    def decide_next_action(
        self,
        *,
        question: Question,
        user_answer: str,
        score_report: ScoreReport,
        turn_history_for_question: str = "(this is the first turn on this question)",
    ) -> DirectorChoice:
        system = DIRECTOR_SYSTEM.format(
            schema=json.dumps(DirectorChoice.model_json_schema(), indent=2)
        )
        user = DIRECTOR_USER.format(
            question_text=question.question,
            user_answer=user_answer.strip() or "(empty answer)",
            content_score=score_report.content.score,
            content_comment=score_report.content.comment,
            clarity_score=score_report.clarity.score,
            clarity_comment=score_report.clarity.comment,
            structure_score=score_report.structure.score,
            structure_comment=score_report.structure.comment,
            overall=score_report.overall,
            turn_history=turn_history_for_question,
        )
        return call_llm(system=system, user=user, schema=DirectorChoice)


# ============================================================================
# Coaching Summariser Agent — generative personalisation
# ============================================================================

SUMMARISER_SYSTEM = """You are a personal interview coach writing the final report for one practice session.

Use the candidate's CV to personalise the report. Where a CV project naturally connects
to a topic from the session, reference it concretely. Where no natural connection exists,
give clean generic advice — do NOT force a CV reference.

Cite specific turns by their number when discussing strengths/gaps.

You MUST return JSON matching this exact schema (no prose, no code fences):
{schema}
"""

SUMMARISER_USER = """Role: {role}
Difficulty: {difficulty}
Total turns completed: {total_turns}

Per-turn breakdown:
{turns_summary}

Average score per topic:
{topic_scores}

Candidate's CV:
{cv_summary}

Write the SessionSummary:
- overall_score: average of all turn overall scores, normalised to 0-1
- per_topic: dict of topic -> normalised score
- strengths: 2-3 specific strengths (cite turn numbers)
- gaps: 2-3 specific gaps (cite turn numbers)
- study_suggestions: 3-4 concrete steps tailored to CV where possible
- coaching_letter: 3-5 sentences, personalised, references CV experience"""


class CoachingSummariserAgent:
    """Writes a personalised final report. Generative agent #2.

    This is a second agentic pattern alongside Director: instead of action selection
    in a loop, this agent synthesises long-form personalised content from rich context.
    """

    def summarise(self, session_state: SessionState) -> SessionSummary:
        turns_summary = _format_turns(session_state.turns)
        topic_scores = _format_topic_scores(session_state)
        cv_summary = _format_cv_for_prompt(session_state.cv_profile)

        system = SUMMARISER_SYSTEM.format(
            schema=json.dumps(SessionSummary.model_json_schema(), indent=2)
        )
        user = SUMMARISER_USER.format(
            role=session_state.role.value,
            difficulty=session_state.difficulty.value,
            total_turns=session_state.turn_count(),
            turns_summary=turns_summary,
            topic_scores=topic_scores,
            cv_summary=cv_summary,
        )
        # Higher max_tokens for the coaching letter; slightly warmer for natural prose.
        return call_llm(
            system=system,
            user=user,
            schema=SessionSummary,
            max_tokens=2048,
            temperature=0.3,
        )


# ============================================================================
# Formatting helpers
# ============================================================================

def _bullets(items: list[str]) -> str:
    """Render a list of strings as a bullet block for prompts."""
    if not items:
        return ""
    return "\n".join(f"  - {it}" for it in items)


def _format_cv_for_prompt(cv: Optional[CVProfile]) -> str:
    """Render the CV profile in a compact form for prompts."""
    if cv is None:
        return "(no CV uploaded)"
    parts = []
    if cv.seniority:
        parts.append(f"Estimated seniority: {cv.seniority}")
    if cv.skills:
        parts.append(f"Skills mentioned: {', '.join(cv.skills)}")
    if cv.projects:
        parts.append("Projects:")
        for p in cv.projects[:5]:  # cap at 5 for prompt brevity
            parts.append(f"  - {p}")
    if cv.claimed_strengths:
        parts.append(f"Claimed strengths: {', '.join(cv.claimed_strengths)}")
    if cv.likely_gaps:
        parts.append(f"Likely gaps for role: {', '.join(cv.likely_gaps)}")
    return "\n".join(parts) if parts else "(empty profile)"


def _format_turns(turns: list[TurnRecord]) -> str:
    """Compact per-turn breakdown for the summariser prompt."""
    if not turns:
        return "(no turns recorded)"
    lines = []
    for t in turns:
        score = t.score_report.overall if t.score_report else "—"
        lines.append(
            f"Turn {t.turn_id} [{t.slot_type.value}] {t.topic.value} ({t.difficulty.value}): "
            f"overall {score}/5"
        )
        if t.score_report:
            lines.append(f"  Q: {t.question_text}")
            lines.append(f"  A: {t.user_answer[:200]}{'...' if len(t.user_answer) > 200 else ''}")
            lines.append(
                f"  Scores: content {t.score_report.content.score}, "
                f"clarity {t.score_report.clarity.score}, "
                f"structure {t.score_report.structure.score}"
            )
    return "\n".join(lines)


def _format_topic_scores(state: SessionState) -> str:
    """Render rolling topic scores for the summariser prompt."""
    if not state.topic_scores:
        return "(no topic scores yet)"
    lines = []
    for topic, rs in state.topic_scores.items():
        if rs.count > 0:
            lines.append(f"  {topic.value}: {rs.score:.2f} (over {rs.count} turn(s))")
    return "\n".join(lines) or "(no scored turns yet)"
