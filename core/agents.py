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

from core.llm import call_llm, call_llm_with_tools, traceable
from core.models import (
    CriterionJudgement,
    CVProfile,
    DimensionScore,
    DirectorAction,
    DirectorChoice,
    Difficulty,
    InterviewerChoice,
    Question,
    Role,
    Rubric,
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
3. Anchor the phrasing in the candidate's CV whenever there is *any* reasonable
   overlap between the question's topic/skills and a CV skill, project, or
   claimed strength. This is the default behaviour, not the exception.
   - Examples of reasonable overlap: question is about SQL joins and the CV
     lists PostgreSQL or MySQL; question is about dashboards and the CV
     mentions Tableau or Power BI; question is about testing strategy and the
     CV lists pytest or Cypress.
   - Lead with the CV reference, e.g. "In your Tableau dashboard project, ...",
     "Given your PostgreSQL experience at <company>, ...", "You listed pytest
     as a strength — ...".
4. Only fall back to a plain, generic phrasing when the CV is genuinely empty
   or has zero connection to the topic (e.g. behavioural question with a CV
   that lists only unrelated technical skills). Do not invent CV facts that
   aren't there.

Return ONLY a JSON object filled in with real values that satisfy the schema
below. Do NOT return the schema definition itself — return actual data.
No prose, no code fences.

JSON schema the object must satisfy:
{schema}
"""

INTERVIEWER_USER = """Topic: {topic}
Difficulty: {difficulty}

Candidate questions to pick from:
{candidates_json}

User's CV signals:
{cv_signals}

Pick the ID of the question you choose, and write your phrased version.
Phrasing should be one sentence and sound like a real interviewer. If any CV
skill, project, or strength is related to the topic above, your phrasing MUST
open by referencing it concretely. Only skip the CV reference if there is no
plausible connection at all."""


class InterviewerAgent:
    """Picks one question from retrieved candidates and phrases it naturally."""

    @traceable(name="InterviewerAgent.ask")
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
            return InterviewerChoice(id=candidates[0].id, phrased=candidates[0].question, cv_signal_used=None)
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

Return ONLY a JSON object filled in with real values that satisfy the schema
below. Do NOT return the schema definition itself — return actual data.
No prose, no code fences.

JSON schema the object must satisfy:
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


# --- v2 prompts: one judge per criterion ------------------------------------
# v1 grades content/clarity/structure in a single prompt. v2 splits them into
# three focused prompts and combines the scores in code — less halo effect,
# easier to calibrate. Both versions are kept so the eval leaderboard can
# compare them side by side.

_CRITERION_DEFS = {
    "content": "Content = is the answer factually correct and complete?",
    "clarity": "Clarity = is it explained in a clear, well-articulated way?",
    "structure": "Structure = is it organised logically (e.g. definition, then example)?",
}

_SCORE_ANCHORS = """- 1: fails this dimension entirely
- 2: weak — major gaps
- 3: acceptable — meets the basics
- 4: strong — only minor gaps
- 5: excellent — fully satisfies the rubric"""

CRITERION_JUDGE_SYSTEM = """You are an interview evaluator grading ONE specific dimension of a candidate's answer.

You are grading: {criterion_name}
{criterion_definition}

Grade ONLY this dimension. Do not let other qualities of the answer raise or lower this score.

Score scale (1-5):
{score_anchors}

Return ONLY a JSON object filled in with real values that satisfy the schema
below. Do NOT return the schema definition itself — return actual data.
No prose, no code fences.

JSON schema the object must satisfy:
{schema}
"""

CRITERION_JUDGE_USER = """Question: {question_text}

Rubric criteria for {criterion_name}:
{rubric_criteria}

Reference answer (what a strong answer looks like):
{reference_answer}

Candidate's answer:
{user_answer}

Grade the {criterion_name} dimension only. Reason first in `comment`, then give the `score`,
then one concrete `improvement` suggestion for this dimension."""


class EvaluatorAgent:
    """Grades a user answer against a question's rubric. Returns a ScoreReport.

    Two grading strategies, selected at construction:
      - "v1": one prompt grades all dimensions at once (original behaviour).
      - "v2": one focused prompt per dimension, scores combined in code.
    Keeping both lets the eval suite measure them side by side.
    """

    def __init__(self, version: str = "v1") -> None:
        if version not in ("v1", "v2"):
            raise ValueError(
                f"EvaluatorAgent version must be 'v1' or 'v2', got {version!r}"
            )
        self.version = version

    @traceable(name="EvaluatorAgent.evaluate")
    def evaluate(
        self,
        *,
        question: Question,
        user_answer: str,
    ) -> ScoreReport:
        if self.version == "v2":
            return self._evaluate_v2(question, user_answer)
        return self._evaluate_v1(question, user_answer)

    # ---- v1: single combined prompt ---------------------------------------

    def _evaluate_v1(self, question: Question, user_answer: str) -> ScoreReport:
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

    # ---- v2: one judge per criterion --------------------------------------

    def _evaluate_v2(self, question: Question, user_answer: str) -> ScoreReport:
        answer = user_answer.strip() or "(empty answer)"
        rubric_by_criterion = {
            "content": question.rubric.content,
            "clarity": question.rubric.clarity,
            "structure": question.rubric.structure,
        }
        judgements: dict[str, CriterionJudgement] = {}
        for criterion in ("content", "clarity", "structure"):
            judgements[criterion] = self._judge_one(
                criterion=criterion,
                question=question,
                rubric_criteria=rubric_by_criterion[criterion],
                answer=answer,
            )

        # Combine the three criterion scores in code: overall = rounded mean.
        scores = [judgements[c].score for c in ("content", "clarity", "structure")]
        overall = round(sum(scores) / len(scores))

        return ScoreReport(
            content=DimensionScore(score=judgements["content"].score,
                                   comment=judgements["content"].comment),
            clarity=DimensionScore(score=judgements["clarity"].score,
                                   comment=judgements["clarity"].comment),
            structure=DimensionScore(score=judgements["structure"].score,
                                     comment=judgements["structure"].comment),
            actionable_feedback=[judgements[c].improvement
                                 for c in ("content", "clarity", "structure")],
            overall=overall,
        )

    @traceable(name="EvaluatorAgent.judge_criterion")
    def _judge_one(
        self,
        *,
        criterion: str,
        question: Question,
        rubric_criteria: list[str],
        answer: str,
    ) -> CriterionJudgement:
        system = CRITERION_JUDGE_SYSTEM.format(
            criterion_name=criterion,
            criterion_definition=_CRITERION_DEFS[criterion],
            score_anchors=_SCORE_ANCHORS,
            schema=json.dumps(CriterionJudgement.model_json_schema(), indent=2),
        )
        user = CRITERION_JUDGE_USER.format(
            question_text=question.question,
            criterion_name=criterion,
            rubric_criteria=_bullets(rubric_criteria) or "(none)",
            reference_answer=question.reference_answer,
            user_answer=answer,
        )
        return call_llm(system=system, user=user, schema=CriterionJudgement)


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
- If the candidate has already had 1 clarify/followup/dig_deeper turn on the SAME question, choose move_on.
- Default to move_on unless one of the other actions clearly applies.

WHEN THE ACTION IS NOT move_on, YOU MUST ALSO EMIT:
- `follow_up_rubric`  — a Rubric used to grade the candidate's NEXT answer to the
                        follow-up question you just wrote. Criteria must be:
                          • observable in a written answer (not "shows insight"),
                          • specific to the follow-up question you actually asked
                            (not a copy of the original rubric),
                          • independently evaluable (a grader can mark each one true/false).
                        Provide 2-4 `content` criteria; 1-2 `clarity` and `structure`
                        criteria are optional but encouraged.
- `follow_up_reference` — a model answer (3-6 sentences) that a strong candidate would
                          give to your follow-up question. It must satisfy every
                          criterion in `follow_up_rubric`; the system will check this
                          and discard your follow-up if it doesn't.

When the action IS move_on, leave `follow_up_rubric` and `follow_up_reference` null.

Return ONLY a JSON object filled in with real values that satisfy the schema
below. Do NOT return the schema definition itself — return actual data.
No prose, no code fences.

JSON schema the object must satisfy:
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

    @traceable(name="ConversationDirectorAgent.decide_next_action")
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


# --- Follow-up question construction + self-consistency check (defence #3) --

# Minimum overall score we require the Evaluator to give the Director's own
# `follow_up_reference` answer when graded against `follow_up_rubric`. If a
# *strong* model answer doesn't satisfy its own rubric, the rubric is broken.
FOLLOWUP_SELF_CONSISTENCY_FLOOR = 4


def build_followup_question(*, choice: DirectorChoice, slot_question: Question) -> Question:
    """Synthesise a Question from a Director follow-up choice.

    `slot_question` is the original KB question for this slot — its topic /
    subtopic / difficulty are reused so downstream code (TurnRecord, topic
    scoring) sees a coherent classification.

    Caller must ensure choice.action != MOVE_ON and the follow-up fields are
    populated; in practice this goes through `validate_followup_choice` first.
    """
    if choice.follow_up_rubric is None or not choice.follow_up_reference:
        raise ValueError(
            "build_followup_question requires follow_up_rubric and "
            "follow_up_reference — call validate_followup_choice first."
        )
    return Question(
        id=f"{slot_question.id}-followup",
        topic=slot_question.topic,
        subtopic=slot_question.subtopic,
        difficulty=slot_question.difficulty,
        question=choice.text,
        reference_answer=choice.follow_up_reference,
        rubric=choice.follow_up_rubric,
    )


def validate_followup_choice(
    *,
    choice: DirectorChoice,
    slot_question: Question,
    evaluator: EvaluatorAgent,
    floor: int = FOLLOWUP_SELF_CONSISTENCY_FLOOR,
) -> DirectorChoice:
    """Runtime defence: only let a non-MOVE_ON choice through if its
    self-generated rubric grades its self-generated reference answer at
    or above `floor`. Otherwise degrade to MOVE_ON.

    Why: the Director's follow-up rubric is LLM-written, so quality varies.
    A model reference answer that fails its own rubric is the clearest
    signal the rubric is unusable. Catching this at runtime means we never
    grade a real candidate against a broken rubric — we just stop the
    follow-up branch and move on.
    """
    if choice.action == DirectorAction.MOVE_ON:
        return choice
    if choice.follow_up_rubric is None or not choice.follow_up_reference:
        return _move_on()
    synth = build_followup_question(choice=choice, slot_question=slot_question)
    report = evaluator.evaluate(question=synth, user_answer=choice.follow_up_reference)
    if report.overall < floor:
        return _move_on()
    return choice


def _move_on() -> DirectorChoice:
    """Build a clean MOVE_ON choice — used when validation fails."""
    return DirectorChoice(action=DirectorAction.MOVE_ON, text="")


# ============================================================================
# Coaching Summariser Agent — generative personalisation
# ============================================================================

SUMMARISER_SYSTEM = """You are a personal interview coach writing the final report for one practice session.

Use the candidate's CV to personalise the report. Where a CV project naturally connects
to a topic from the session, reference it concretely. Where no natural connection exists,
give clean generic advice — do NOT force a CV reference.

Cite specific turns by their number when discussing strengths/gaps.

TOOL USE:
You have access to a `lookup_reference_answer` tool. Use it sparingly and intentionally:
- Call it for 1-3 questions where the candidate scored low (overall <= 3) and a precise
  study suggestion would benefit from the gold reference.
- Do NOT look up every question — only the ones where the reference will sharpen your
  advice. If the candidate scored high on a question, skip it.
- After looking up, base your study suggestion on what's actually in the reference, not
  on guesses.

When you are done using tools, emit your final JSON answer.

Return ONLY a JSON object filled in with real values that satisfy the schema
below. Do NOT return the schema definition itself — return actual data.
No prose, no code fences.

JSON schema the object must satisfy:
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


# Tools available to the Coaching Summariser agent
COACH_TOOLS = [
    {
        "name": "lookup_reference_answer",
        "description": (
            "Look up the gold/reference answer for a question by ID. Use this when "
            "you want to give specific, accurate study advice for a question the "
            "candidate missed or scored low on. The question IDs are shown in the "
            "per-turn breakdown (e.g. 'da-001', 'qa-003', 'behav-002')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question_id": {
                    "type": "string",
                    "description": "The question ID, e.g. 'da-001'.",
                }
            },
            "required": ["question_id"],
        },
    }
]


class CoachingSummariserAgent:
    """Writes a personalised final report. Tool-using agent #2.

    Two agentic patterns demonstrated by the system:
      - Conversation Director  — action selection in a closed loop
      - Coaching Summariser   — tool use to fetch gold-standard answers for
        questions the candidate scored low on, so study suggestions are grounded
        in the actual KB content rather than the model's parametric memory.
    """

    def __init__(self, kb=None) -> None:
        """Optional KB reference; needed for the lookup_reference_answer tool.

        If kb is None, the agent falls back to a plain (non-tool) call.
        """
        self._kb = kb

    @traceable(name="CoachingSummariserAgent.summarise")
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

        # No KB available -> plain call (backward-compatible fallback)
        if self._kb is None:
            return call_llm(
                system=system,
                user=user,
                schema=SessionSummary,
                max_tokens=2048,
                temperature=0.3,
            )

        # Tool-use loop: the agent decides whether to look up reference answers
        kb = self._kb

        def tool_executor(tool_name: str, tool_input: dict) -> str:
            if tool_name == "lookup_reference_answer":
                qid = tool_input.get("question_id", "").strip()
                if not qid:
                    return "Error: question_id is required."
                try:
                    q = kb.get(qid)
                except KeyError:
                    return f"No question found with id '{qid}'."
                return (
                    f"Question (id={q.id}, topic={q.topic.value}, "
                    f"difficulty={q.difficulty.value}):\n"
                    f"{q.question}\n\n"
                    f"Reference answer:\n{q.reference_answer}"
                )
            return f"Unknown tool: {tool_name}"

        return call_llm_with_tools(
            system=system,
            user=user,
            schema=SessionSummary,
            tools=COACH_TOOLS,
            tool_executor=tool_executor,
            max_tokens=2048,
            temperature=0.3,
            max_iterations=12,
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
