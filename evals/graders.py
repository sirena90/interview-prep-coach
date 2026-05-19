"""Graders — turn a basket run into a leaderboard number.

Grader types:
  - deterministic   : grade_planner   (exact match vs expected slot/topic/difficulty)
  - deterministic   : grade_retriever (rule check over the retrieved set)
  - decision-match  : grade_director  (LLM's chosen action must be in the accepted set)

Each grader returns {"<metric>": float, "rows": [...]} — the metric goes on
the leaderboard, the rows let you inspect individual passes/failures.
"""
from __future__ import annotations

from datetime import datetime

from core.agents import ConversationDirectorAgent, EvaluatorAgent
from core.models import (
    Difficulty,
    DimensionScore,
    Question,
    Rubric,
    ScoreReport,
    SessionState,
    Topic,
    TurnRecord,
)
from core.planner import SimplePlanner
from evals.metrics import quadratic_weighted_kappa


# --- helpers ----------------------------------------------------------------

def _score_report(overall: int) -> ScoreReport:
    """A flat ScoreReport with every dimension at `overall` — enough for the
    planner (which only reads `overall`) and as a Director fixture."""
    dim = DimensionScore(score=overall, comment="-")
    return ScoreReport(
        content=dim, clarity=dim, structure=dim,
        actionable_feedback=["-"], overall=overall,
    )


def _session_from_case(case: dict) -> SessionState:
    """Replay a planner case's prior_turns into a SessionState."""
    turns = []
    for i, t in enumerate(case["prior_turns"], start=1):
        report = _score_report(t["overall"]) if t.get("overall") is not None else None
        turns.append(TurnRecord(
            turn_id=i, slot_type=t["slot"], question_id=f"q-{i}",
            question_text=f"Q{i}", topic=t["topic"], difficulty=t["difficulty"],
            user_answer="answer", score_report=report,
        ))
    return SessionState(
        session_id="eval", role=case["role"], difficulty=case["difficulty"],
        turns=turns, started_at=datetime(2026, 5, 17),
    )


# A generic question used as the fixed context for every Director case — the
# Director's decision should depend on the score/history, not the question.
_EVAL_QUESTION = Question(
    id="eval-q", topic=Topic.SQL, subtopic="joins", difficulty=Difficulty.MID,
    question="What is the difference between an INNER JOIN and a LEFT JOIN?",
    reference_answer="INNER JOIN keeps only matching rows; LEFT JOIN keeps all "
                     "left-table rows, NULL-filling unmatched right-side columns.",
    rubric=Rubric(content=["defines both join types"], clarity=[], structure=[]),
)


# --- graders ----------------------------------------------------------------

def grade_planner(kb, basket: list[dict]) -> dict:
    """Deterministic: planned slot (and topic/difficulty when specified) must
    match the expected values exactly."""
    planner = SimplePlanner(kb, rng_seed=0)
    rows, passed = [], 0
    for case in basket:
        plan = planner.plan_next_turn(_session_from_case(case))
        ok = plan.slot_type is case["expected_slot"]
        if "expected_topic" in case:
            ok = ok and plan.topic is case["expected_topic"]
        if "expected_difficulty" in case:
            ok = ok and plan.difficulty is case["expected_difficulty"]
        passed += ok
        rows.append({"id": case["id"], "pass": ok,
                     "got": f"{plan.slot_type.value}/{plan.topic.value}/"
                            f"{plan.difficulty.value}"})
    return {"planner_acc": passed / len(basket), "rows": rows}


def grade_retriever(kb, basket: list[dict]) -> dict:
    """Deterministic: each case applies a rule to the retrieved set."""
    rows, passed = [], 0
    for case in basket:
        results = kb.retrieve(
            topic=case["topic"], difficulty=case["difficulty"],
            cv_skills=case.get("cv_skills"), k=5,
        )
        if case["check"] == "topic_match":
            ok = bool(results) and all(q.topic is case["topic"] for q in results)
        elif case["check"] == "cv_rerank":
            skill = case["cv_skills"][0].lower()
            top_tags = [t.lower() for t in results[0].skill_tags] if results else []
            ok = skill in top_tags
        else:
            ok = False
        passed += ok
        rows.append({"id": case["id"], "pass": ok, "n_results": len(results)})
    return {"retriever_pass": passed / len(basket), "rows": rows}


def grade_director(basket: list[dict]) -> dict:
    """Decision-match: the Director's chosen action must be in the accepted set.

    Runs the real Director agent — this makes one Anthropic call per case.
    """
    director = ConversationDirectorAgent()
    rows, passed = [], 0
    for case in basket:
        s = case["scores"]
        report = ScoreReport(
            content=DimensionScore(score=s["content"], comment="-"),
            clarity=DimensionScore(score=s["clarity"], comment="-"),
            structure=DimensionScore(score=s["structure"], comment="-"),
            actionable_feedback=["-"],
            overall=s["overall"],
        )
        choice = director.decide_next_action(
            question=_EVAL_QUESTION,
            user_answer=case["user_answer"],
            score_report=report,
            turn_history_for_question=case["turn_history"],
        )
        ok = choice.action in case["acceptable_actions"]
        passed += ok
        rows.append({"id": case["id"], "pass": ok,
                     "predicted": choice.action.value,
                     "accepted": sorted(a.value for a in case["acceptable_actions"])})
    return {"director_acc": passed / len(basket), "rows": rows}


def grade_evaluator(kb, golden: list[dict], version: str) -> dict:
    """LLM-as-judge calibration: run the Evaluator over the golden set and
    measure agreement with the human `overall` labels.

    Returns quadratic-weighted Cohen's kappa plus exact / within-1 rates.
    Makes real Anthropic calls — one per case for v1, three per case for v2.
    """
    agent = EvaluatorAgent(version=version)
    human, model, rows = [], [], []
    for case in golden:
        question = kb.get(case["question_id"])
        report = agent.evaluate(question=question, user_answer=case["answer"])
        h, m = case["human_overall"], report.overall
        human.append(h)
        model.append(m)
        rows.append({
            "id": case["id"], "human": h, "model": m,
            "pass": h == m, "within1": abs(h - m) <= 1,
        })
    return {
        "kappa": quadratic_weighted_kappa(human, model),
        "exact": sum(r["pass"] for r in rows) / len(rows),
        "within1": sum(r["within1"] for r in rows) / len(rows),
        "rows": rows,
    }
