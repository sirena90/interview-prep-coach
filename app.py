"""Streamlit UI for Interview Prep Coach.

One-file Streamlit app that wires together:
  - core/kb.py        (RAG retrieval)
  - core/planner.py   (slot scheduling)
  - core/agents.py    (Interviewer, Evaluator, Director, Coach)
  - core/cv_parser.py (CV upload)

Run with:
    streamlit run app.py

State lives in st.session_state["app"] as a Python dict. The Pydantic
SessionState is kept inside that dict under the key "session_state".
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone

import streamlit as st
from dotenv import load_dotenv

from core.agents import (
    ConversationDirectorAgent,
    CoachingSummariserAgent,
    EvaluatorAgent,
    InterviewerAgent,
    build_followup_question,
    validate_followup_choice,
)
from core.cv_parser import extract_cv_text, parse_cv
from core.kb import KnowledgeBase
from core.llm import set_retry_notifier
from core.models import (
    CVProfile,
    DimensionScore,
    DirectorAction,
    Difficulty,
    RollingScore,
    Role,
    ScoreReport,
    SessionState,
    SlotType,
    Topic,
    TurnRecord,
)
from core.planner import SimplePlanner, TurnPlan


load_dotenv()

# Show a toast in the UI whenever an LLM call is being retried after a rate
# limit, so a free-tier limit looks like a brief wait, not a frozen app.
set_retry_notifier(lambda message: st.toast(message))


# ============================================================================
# Streamlit setup + singletons
# ============================================================================

st.set_page_config(
    page_title="Interview Prep Coach",
    page_icon="🎤",
    layout="centered",
)


@st.cache_resource
def get_kb() -> KnowledgeBase:
    """KnowledgeBase is heavy (loads + indexes 129 questions). Cache it once."""
    return KnowledgeBase()


@st.cache_resource
def get_agents() -> dict:
    """Instantiate agents once per session.

    Coach receives the KB so it can use the lookup_reference_answer tool
    to fetch gold-standard answers for questions the candidate missed.
    """
    kb = get_kb()
    return {
        "interviewer": InterviewerAgent(),
        "evaluator": EvaluatorAgent(),
        "director": ConversationDirectorAgent(),
        "coach": CoachingSummariserAgent(kb=kb),
    }


@st.cache_resource
def get_planner() -> SimplePlanner:
    return SimplePlanner(kb=get_kb())


# ============================================================================
# Session state initialisation
# ============================================================================

def init_app_state() -> None:
    """Initialise the per-user state on first run."""
    if "phase" not in st.session_state:
        st.session_state.phase = "setup"           # setup / interview / done
        st.session_state.session = None            # SessionState (Pydantic)
        st.session_state.current_question = None   # the slot's KB Question
        st.session_state.followup_question = None  # synthesised Question for an active follow-up
        st.session_state.current_slot = None       # the active TurnPlan
        st.session_state.chat_messages = []        # list of {role, content}
        st.session_state.director_rounds = 0       # follow-up rounds within current question
        st.session_state.final_summary = None      # SessionSummary at end
        st.session_state.final_summary_error = None  # last Coach failure, if any
        st.session_state.fatal_error = None        # last unhandled exception in a phase
        st.session_state.fatal_error_type = None
        st.session_state.cv_profile = None         # CVProfile from upload


# ============================================================================
# Setup phase — role / difficulty / CV upload
# ============================================================================

ROLE_LABELS = {
    Role.DATA_ANALYST: "Data Analyst",
    Role.QA_ENGINEER: "QA Engineer",
    Role.DATA_ENGINEER: "Data Engineer",
    Role.FRONTEND_DEVELOPER: "Frontend Developer",
}

DIFFICULTY_LABELS = {
    Difficulty.ENTRY: "Entry",
    Difficulty.MID: "Mid",
    Difficulty.SENIOR: "Senior",
}


def render_setup() -> None:
    st.title("🎤 Interview Prep Coach")
    st.markdown(
        "Multi-turn coaching for technical and behavioural interview prep. "
        "Pick a role, upload your CV, and run a 5-question practice session."
    )

    with st.form("setup_form"):
        role_label = st.selectbox(
            "Role you're preparing for:",
            options=list(ROLE_LABELS.values()),
        )
        diff_label = st.radio(
            "Difficulty level:",
            options=list(DIFFICULTY_LABELS.values()),
            horizontal=True,
            index=1,  # default to "Mid"
        )
        uploaded = st.file_uploader(
            "Upload your CV (PDF or .txt):",
            type=["pdf", "txt"],
            help="Your CV is used to personalise question phrasing and the final report.",
        )
        submitted = st.form_submit_button("Start interview", type="primary")

    if submitted:
        if uploaded is None:
            st.error("Please upload your CV first.")
            return

        role = _label_to_role(role_label)
        difficulty = _label_to_difficulty(diff_label)

        with st.spinner("Reading your CV..."):
            try:
                cv_text = extract_cv_text(uploaded)
            except ValueError as e:
                st.error(f"Could not read CV: {e}")
                return

        with st.spinner("Extracting profile from CV..."):
            try:
                cv_profile = parse_cv(cv_text, role=role)
            except Exception as e:
                st.warning(f"CV parsing failed; continuing without it. ({e})")
                cv_profile = CVProfile()

        st.session_state.cv_profile = cv_profile
        st.session_state.session = SessionState(
            session_id=str(uuid.uuid4()),
            role=role,
            difficulty=difficulty,
            target_turns=5,
            cv_profile=cv_profile,
            started_at=datetime.now(timezone.utc),
        )
        st.session_state.phase = "interview"
        _advance_to_next_question()
        st.rerun()


def _label_to_role(label: str) -> Role:
    for r, lbl in ROLE_LABELS.items():
        if lbl == label:
            return r
    raise ValueError(f"Unknown role label: {label}")


def _label_to_difficulty(label: str) -> Difficulty:
    for d, lbl in DIFFICULTY_LABELS.items():
        if lbl == label:
            return d
    raise ValueError(f"Unknown difficulty label: {label}")


# ============================================================================
# Interview phase — main loop
# ============================================================================

def render_interview() -> None:
    state: SessionState = st.session_state.session
    cv: CVProfile = st.session_state.cv_profile

    # Header: progress and CV chip
    st.markdown(
        f"**{ROLE_LABELS[state.role]}** · {DIFFICULTY_LABELS[state.difficulty]} · "
        f"Question **{state.turn_count() + 1}** of {state.target_turns}"
    )
    if cv and cv.skills:
        st.caption("CV skills: " + ", ".join(cv.skills[:6]) + ("..." if len(cv.skills) > 6 else ""))

    # Replay chat so far
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Skip button — for the candidate who genuinely doesn't know. Commits a
    # placeholder low-score turn (no LLM call) so the topic still registers
    # as a weak area in the final report, then advances.
    if st.button("⏭️ I don't know — skip this question",
                 help="Move on without answering. Counts as a weak attempt "
                      "for this topic, no feedback is generated."):
        _skip_current_question()
        st.rerun()

    # Get user answer
    user_input = st.chat_input("Type your answer...")
    if user_input:
        _handle_user_answer(user_input)
        st.rerun()


def _advance_to_next_question() -> None:
    """Plan + retrieve + ask the next question. Appends to chat history."""
    state: SessionState = st.session_state.session
    cv: CVProfile = st.session_state.cv_profile
    kb = get_kb()
    agents = get_agents()
    planner = get_planner()

    plan: TurnPlan = planner.plan_next_turn(state)
    st.session_state.current_slot = plan
    st.session_state.director_rounds = 0
    st.session_state.followup_question = None

    # Retrieve candidate questions
    if plan.slot_type == SlotType.BEHAVIORAL:
        candidates = kb.retrieve_behavioural(
            difficulty=plan.difficulty,
            excluded_ids=state.asked_ids,
            k=3,
        )
    else:
        candidates = kb.retrieve(
            topic=plan.topic,
            difficulty=plan.difficulty,
            excluded_ids=state.asked_ids,
            cv_skills=cv.skills if cv else None,
            k=5,
        )

    if not candidates:
        # Defensive: skip this slot if KB exhausted
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": "_(no questions available for this slot; advancing)_",
        })
        return

    # Interviewer picks + paraphrases
    choice = agents["interviewer"].ask(
        candidates=candidates,
        topic=plan.topic,
        difficulty=plan.difficulty,
        cv_profile=cv,
    )
    question = kb.get(choice.id)
    st.session_state.current_question = question
    state.current_question_id = question.id

    # Header for this question
    slot_label = {
        SlotType.COVER: "Topic question",
        SlotType.REINFORCE: "Follow-up on same topic",
        SlotType.BEHAVIORAL: "Behavioural question",
    }[plan.slot_type]

    q_num = state.turn_count() + 1
    cv_match_line = _cv_match_caption(question, cv)
    header = (
        f"### Q{q_num} — {slot_label}\n"
        f"*Topic: {plan.topic.value} · Difficulty: {plan.difficulty.value}*\n"
        f"{cv_match_line}\n"
        f"{choice.phrased}"
    )
    st.session_state.chat_messages.append({"role": "assistant", "content": header})


def _cv_match_caption(question, cv: CVProfile | None) -> str:
    """Return a one-line caption showing which CV skills drove this choice.

    Empty string when there's no overlap or no CV — keeps the header clean.
    Surfaces the otherwise-invisible CV-aware reranking so the user can see
    that uploading their CV actually changed what they were asked.
    """
    if cv is None or not cv.skills or not question.skill_tags:
        return ""
    tag_set = {t.lower().strip() for t in question.skill_tags}
    # Preserve the user's original CV casing in the displayed match.
    matched = [s for s in cv.skills if s.lower().strip() in tag_set]
    if not matched:
        return ""
    rendered = ", ".join(f"**{s}**" for s in matched[:4])
    return f"\n> 📄 Picked from your CV: {rendered}\n"


def evaluate_and_decide(
    *,
    slot_question,
    followup_question,
    answer: str,
    director_rounds: int,
    agents: dict,
    turn_history_summary: str | None = None,
) -> dict:
    """Pure orchestrator for one user-answer turn.

    Stateless on purpose: takes the inputs the Streamlit layer would normally
    read from `st.session_state`, returns the outputs the Streamlit layer
    would normally write back. Pulled out of `_handle_user_answer` so the
    end-to-end behaviour (which question is graded, whether a follow-up is
    installed, whether the loop guard fires) can be tested directly without
    mocking Streamlit.

    Returns a dict with:
      active_question      — Question actually used for grading this answer
      score_report         — Evaluator's verdict
      choice               — possibly-degraded DirectorChoice
      next_followup_question — synthetic Question to install if action != MOVE_ON,
                               else None (signal to clear any active follow-up)
    """
    active_question = followup_question or slot_question

    score_report = agents["evaluator"].evaluate(
        question=active_question, user_answer=answer,
    )

    history = turn_history_summary or _default_history_summary(director_rounds)
    choice = agents["director"].decide_next_action(
        question=active_question,
        user_answer=answer,
        score_report=score_report,
        turn_history_for_question=history,
    )

    # Loop guard: cap follow-ups at 1 to keep sessions short and avoid grilling.
    if director_rounds >= 1 and choice.action != DirectorAction.MOVE_ON:
        choice.action = DirectorAction.MOVE_ON
        choice.text = ""
        choice.follow_up_rubric = None
        choice.follow_up_reference = None

    # Defence #3: self-consistency check on the generated follow-up rubric.
    if choice.action != DirectorAction.MOVE_ON:
        choice = validate_followup_choice(
            choice=choice,
            slot_question=slot_question,
            evaluator=agents["evaluator"],
        )

    next_followup = None
    if choice.action != DirectorAction.MOVE_ON:
        next_followup = build_followup_question(
            choice=choice, slot_question=slot_question,
        )

    return {
        "active_question": active_question,
        "score_report": score_report,
        "choice": choice,
        "next_followup_question": next_followup,
    }


def _default_history_summary(director_rounds: int) -> str:
    if director_rounds == 0:
        return "(this is the first answer to this question)"
    return (f"(the candidate has already had {director_rounds} "
            f"follow-up round(s) on this question)")


def _skip_current_question() -> None:
    """Commit a placeholder turn for a skipped question and advance.

    No LLM call. The TurnRecord stores a 1/5 ScoreReport with a `(skipped)`
    feedback note, so the per-topic rolling score and the final coaching
    report still treat the topic as weak — which is the truthful signal
    when the candidate said "I don't know".
    """
    state: SessionState = st.session_state.session
    slot_question = st.session_state.current_question
    plan = st.session_state.current_slot
    if slot_question is None or plan is None or state is None:
        st.error("No active question to skip.")
        return

    st.session_state.chat_messages.append({
        "role": "user", "content": "_(skipped)_",
    })
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": "_Question skipped — moving on. This topic counts as a "
                   "weak area in your final report._",
    })

    skip_report = ScoreReport(
        content=DimensionScore(score=1, comment="skipped"),
        clarity=DimensionScore(score=1, comment="skipped"),
        structure=DimensionScore(score=1, comment="skipped"),
        actionable_feedback=["candidate skipped — practice this topic"],
        overall=1,
    )

    # If the active question was a follow-up, skipping closes both rounds
    # of the slot. Clear the follow-up and commit against the slot question.
    st.session_state.followup_question = None
    _commit_turn("(skipped)", skip_report)

    if state.turn_count() >= state.target_turns:
        _finish_session()
    else:
        _advance_to_next_question()


def _handle_user_answer(answer: str) -> None:
    """Apply `evaluate_and_decide` to st.session_state and the UI."""
    state: SessionState = st.session_state.session
    agents = get_agents()
    slot_question = st.session_state.current_question
    plan: TurnPlan = st.session_state.current_slot

    if slot_question is None or plan is None:
        st.error("No active question. Refresh and start over.")
        return

    # Show user answer
    st.session_state.chat_messages.append({"role": "user", "content": answer})

    with st.spinner("Evaluating your answer..."):
        result = evaluate_and_decide(
            slot_question=slot_question,
            followup_question=st.session_state.followup_question,
            answer=answer,
            director_rounds=st.session_state.director_rounds,
            agents=agents,
        )
    score_report: ScoreReport = result["score_report"]
    choice = result["choice"]

    # Render feedback
    feedback_md = _format_feedback(score_report)
    st.session_state.chat_messages.append({"role": "assistant", "content": feedback_md})

    # If staying on this slot, install the synthesised follow-up Question and
    # wait for the next user reply.
    if choice.action != DirectorAction.MOVE_ON:
        st.session_state.followup_question = result["next_followup_question"]
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"_({choice.action.value})_\n\n{choice.text}",
        })
        st.session_state.director_rounds += 1
        return

    # MOVE_ON: persist this turn and advance. Clear any active follow-up.
    st.session_state.followup_question = None
    _commit_turn(answer, score_report)

    if state.turn_count() >= state.target_turns:
        _finish_session()
    else:
        _advance_to_next_question()


def _commit_turn(user_answer: str, score_report: ScoreReport) -> None:
    state: SessionState = st.session_state.session
    plan: TurnPlan = st.session_state.current_slot
    question = st.session_state.current_question

    turn = TurnRecord(
        turn_id=state.turn_count() + 1,
        slot_type=plan.slot_type,
        question_id=question.id,
        question_text=question.question,
        topic=question.topic,
        difficulty=question.difficulty,
        user_answer=user_answer,
        score_report=score_report,
    )
    state.turns.append(turn)
    state.asked_ids.add(question.id)
    state.topic_scores.setdefault(question.topic, RollingScore()).update(
        score_report.normalized()
    )
    state.current_question_id = None


def _format_feedback(report: ScoreReport) -> str:
    """Compact one-glance feedback: overall + per-dim inline, brief comments, tips joined."""
    tips = "; ".join(report.actionable_feedback)
    return (
        f"**{report.overall}/5 overall** — content {report.content.score}, "
        f"clarity {report.clarity.score}, structure {report.structure.score}\n\n"
        f"- Content: {report.content.comment}\n"
        f"- Clarity: {report.clarity.comment}\n"
        f"- Structure: {report.structure.comment}\n\n"
        f"**Improve:** {tips}"
    )


# ============================================================================
# Final report phase
# ============================================================================

def _finish_session() -> None:
    """Generate the coaching report. On Coach failure, transition to `done`
    anyway with `final_summary=None` and a stored error message; render_final
    will surface a Retry button. Session state is fully preserved so retry
    is a single re-call against the same SessionState.
    """
    state: SessionState = st.session_state.session
    state.ended_at = datetime.now(timezone.utc)
    agents = get_agents()
    try:
        with st.spinner("Writing your personalised coaching report..."):
            st.session_state.final_summary = agents["coach"].summarise(state)
        st.session_state.final_summary_error = None
    except Exception as exc:  # noqa: BLE001 — Coach can raise many things
        st.session_state.final_summary = None
        st.session_state.final_summary_error = f"{type(exc).__name__}: {exc}"
    st.session_state.phase = "done"


def render_final() -> None:
    state: SessionState = st.session_state.session
    summary = st.session_state.final_summary
    err = st.session_state.get("final_summary_error")

    # Coach failed (e.g. JSON decode error, exhausted rate-limit retries).
    # Don't lose the session — show what we have, explain, offer a retry.
    if err and not summary:
        st.title("🏁 Session complete — final report unavailable")
        st.markdown(
            f"**{ROLE_LABELS[state.role]}** · {DIFFICULTY_LABELS[state.difficulty]} · "
            f"{state.turn_count()} questions answered"
        )
        st.warning(
            f"Could not generate the coaching report: {err}\n\n"
            "Your answers are still saved. This is usually transient — "
            "try again."
        )
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("🔄 Retry final report", type="primary"):
                _finish_session()
                st.rerun()
        with col_b:
            if st.button("Start a new session"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        return

    st.title("🏁 Session complete")
    st.markdown(
        f"**{ROLE_LABELS[state.role]}** · {DIFFICULTY_LABELS[state.difficulty]} · "
        f"{summary.total_turns} questions answered"
    )

    # Overall + per-topic
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Overall score", f"{int(round(summary.overall_score * 100))}%")
    with col2:
        st.metric("Questions answered", summary.total_turns)

    st.subheader("Per-topic scores")
    for topic, score in sorted(summary.per_topic.items(), key=lambda kv: -kv[1]):
        st.markdown(f"- **{topic.value}**: {int(round(score * 100))}%")

    st.subheader("Strengths")
    for s in summary.strengths:
        st.markdown(f"- {s}")

    st.subheader("Gaps")
    for g in summary.gaps:
        st.markdown(f"- {g}")

    st.subheader("Study suggestions")
    for s in summary.study_suggestions:
        st.markdown(f"- {s}")

    st.subheader("Your personal coaching letter")
    st.info(summary.coaching_letter)

    if st.button("Start a new session", type="primary"):
        # Reset everything
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ============================================================================
# Main router
# ============================================================================

def main() -> None:
    init_app_state()

    # If the previous run raised, surface the error and offer a retry
    # *before* trying to render the failed phase again. Session state is
    # preserved so retry resumes from where the user was.
    if st.session_state.get("fatal_error"):
        _render_fatal_error_banner()
        return

    phase = st.session_state.phase

    try:
        if phase == "setup":
            render_setup()
        elif phase == "interview":
            render_interview()
        elif phase == "done":
            render_final()
        else:
            st.error(f"Unknown phase: {phase}")
    except Exception as exc:  # noqa: BLE001 — any agent / network / parser error
        st.session_state.fatal_error = str(exc) or type(exc).__name__
        st.session_state.fatal_error_type = type(exc).__name__
        st.rerun()


def _render_fatal_error_banner() -> None:
    """Generic 'session paused' UI for any unhandled exception in a phase.

    The user's progress is intact (chat history, session, CV); only the
    *current* operation failed. Retry simply clears the error flag and
    reruns the same phase. Common triggers: transient LLM JSON decode
    failures, exhausted rate-limit retries, network blips.
    """
    err = st.session_state.get("fatal_error", "unknown error")
    err_type = st.session_state.get("fatal_error_type", "Error")
    st.error(
        f"⚠️ **Session paused — {err_type}**\n\n"
        f"{err}\n\n"
        "Your progress is saved. This is usually transient (API rate limit "
        "or a model output that failed to parse). Click Retry to resume."
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🔄 Retry", type="primary"):
            st.session_state.fatal_error = None
            st.session_state.fatal_error_type = None
            st.rerun()
    with col_b:
        if st.button("Start over"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


if __name__ == "__main__":
    main()
