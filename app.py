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
)
from core.cv_parser import extract_cv_text, parse_cv
from core.kb import KnowledgeBase
from core.llm import set_retry_notifier
from core.models import (
    CVProfile,
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
        st.session_state.phase = "setup"          # setup / interview / done
        st.session_state.session = None            # SessionState (Pydantic)
        st.session_state.current_question = None   # the Question being asked
        st.session_state.current_slot = None       # the active TurnPlan
        st.session_state.chat_messages = []        # list of {role, content}
        st.session_state.director_rounds = 0       # follow-up rounds within current question
        st.session_state.slot_first_answer = None  # first answer to the current slot
        st.session_state.slot_score_report = None  # score of the first answer to the current slot
        st.session_state.slot_followup_answers = []  # any clarifications after the first answer
        st.session_state.final_summary = None      # SessionSummary at end
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
    # Reset per-slot answer/score caches — only the first answer is evaluated
    # per slot; follow-up clarifications reuse the first score and are
    # appended to the answer text at commit time.
    st.session_state.slot_first_answer = None
    st.session_state.slot_score_report = None
    st.session_state.slot_followup_answers = []

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
    header = (
        f"### Q{q_num} — {slot_label}\n"
        f"*Topic: {plan.topic.value} · Difficulty: {plan.difficulty.value}*\n\n"
        f"{choice.phrased}"
    )
    st.session_state.chat_messages.append({"role": "assistant", "content": header})


def _handle_user_answer(answer: str) -> None:
    """Evaluate, run Director, decide whether to stay on Q or advance."""
    state: SessionState = st.session_state.session
    agents = get_agents()
    question = st.session_state.current_question
    plan: TurnPlan = st.session_state.current_slot

    if question is None or plan is None:
        st.error("No active question. Refresh and start over.")
        return

    # Show user answer
    st.session_state.chat_messages.append({"role": "user", "content": answer})

    is_first_turn_in_slot = st.session_state.slot_score_report is None

    if is_first_turn_in_slot:
        # Evaluate the first answer to this slot — the score sticks for the
        # whole slot, even if the Director asks a follow-up afterwards.
        with st.spinner("Evaluating your answer..."):
            score_report: ScoreReport = agents["evaluator"].evaluate(
                question=question,
                user_answer=answer,
            )
        st.session_state.slot_first_answer = answer
        st.session_state.slot_score_report = score_report

        # Render feedback once, after the first answer.
        feedback_md = _format_feedback(score_report)
        st.session_state.chat_messages.append({"role": "assistant", "content": feedback_md})
    else:
        # Follow-up answer — reuse the original score (clarifications are not
        # rubric-graded on their own; we'd unfairly penalise a one-sentence
        # clarification against e.g. a full STAR rubric).
        score_report = st.session_state.slot_score_report
        st.session_state.slot_followup_answers.append(answer)

    # Director — only after the first round on this Q (subsequent rounds also call it)
    history_summary = _summarise_director_history()
    with st.spinner("Deciding next move..."):
        choice = agents["director"].decide_next_action(
            question=question,
            user_answer=answer,
            score_report=score_report,
            turn_history_for_question=history_summary,
        )

    # Loop guard: cap follow-ups at 1 to keep sessions short and avoid grilling
    if st.session_state.director_rounds >= 1 and choice.action != DirectorAction.MOVE_ON:
        choice.action = DirectorAction.MOVE_ON
        choice.text = ""

    # If staying on this question, just send Director's text and wait
    if choice.action != DirectorAction.MOVE_ON:
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": f"_({choice.action.value})_\n\n{choice.text}",
        })
        st.session_state.director_rounds += 1
        return

    # MOVE_ON: persist this turn and advance.
    # The committed answer is the first answer plus any clarifications, so
    # the Coach sees the full picture in the final report. The score remains
    # the one computed on the first answer.
    combined_answer = st.session_state.slot_first_answer or answer
    for extra in st.session_state.slot_followup_answers:
        combined_answer += f"\n\n[Clarification] {extra}"
    _commit_turn(combined_answer, score_report)

    if state.turn_count() >= state.target_turns:
        _finish_session()
    else:
        _advance_to_next_question()


def _summarise_director_history() -> str:
    rounds = st.session_state.director_rounds
    if rounds == 0:
        return "(this is the first answer to this question)"
    return f"(the candidate has already had {rounds} follow-up round(s) on this question)"


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
    state: SessionState = st.session_state.session
    state.ended_at = datetime.now(timezone.utc)
    agents = get_agents()
    with st.spinner("Writing your personalised coaching report..."):
        st.session_state.final_summary = agents["coach"].summarise(state)
    st.session_state.phase = "done"


def render_final() -> None:
    state: SessionState = st.session_state.session
    summary = st.session_state.final_summary

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
    except RuntimeError as exc:
        # Raised when the LLM provider keeps rate-limiting after all retries,
        # or when a provider is misconfigured. Show the message, not a crash.
        st.error(str(exc))


if __name__ == "__main__":
    main()
