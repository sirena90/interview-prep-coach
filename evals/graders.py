"""Graders — turn a basket run into a leaderboard number.

Grader types:
  - deterministic   : grade_planner    (exact match vs expected slot/topic/difficulty)
  - deterministic   : grade_retriever  (rule check over the retrieved set)
  - decision-match  : grade_director   (LLM's chosen action must be in the accepted set)
  - mixed           : grade_interviewer (selection + CV anchor [deterministic] + fidelity [LLM judge])
  - perturbation    : grade_bias       (delta between baseline and perturbed inputs)

Each grader returns {"<metric>": float, "rows": [...]} — the metric goes on
the leaderboard, the rows let you inspect individual passes/failures.
"""
from __future__ import annotations

import json
from datetime import datetime

from pydantic import BaseModel, Field

from core.agents import ConversationDirectorAgent, EvaluatorAgent, InterviewerAgent
from core.llm import call_llm
from core.models import (
    CVProfile,
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


# ---- Interviewer grader ----------------------------------------------------

class FidelityVerdict(BaseModel):
    """Output schema for the reformulation-fidelity LLM judge.

    `same_intent` and `leaks_answer` are the load-bearing booleans; the
    comment lets a human auditor see why the judge decided what it did.
    """
    same_intent: bool
    leaks_answer: bool
    comment: str = Field(min_length=1)


FIDELITY_JUDGE_SYSTEM = """You are an independent reviewer checking whether a paraphrased
interview question is faithful to the original source question.

Decide two things:
  same_intent   — does the paraphrase ask the candidate to answer the same underlying
                  question? (Allowed: rewording, CV anchoring like "given your X experience,"
                  changes in tone. Not allowed: changing the topic, narrowing/widening the scope,
                  swapping concepts.)
  leaks_answer  — does the paraphrase reveal part of the answer that the original kept hidden?
                  (e.g., the original asks "what is X?" and the paraphrase says "X works by ..." —
                  that leaks the answer.)

Return ONLY a JSON object filled in with real values that satisfy the schema below.
No prose, no code fences.

JSON schema the object must satisfy:
{schema}
"""

FIDELITY_JUDGE_USER = """Original source question:
{source_question}

Reference answer (so you know what the candidate is *supposed* to produce):
{reference_answer}

Paraphrased version shown to the candidate:
{phrased}

Now judge. Be strict about leakage but tolerant of paraphrasing and CV anchoring."""


def _judge_fidelity(*, source_question: str, reference_answer: str,
                    phrased: str) -> FidelityVerdict:
    system = FIDELITY_JUDGE_SYSTEM.format(
        schema=json.dumps(FidelityVerdict.model_json_schema(), indent=2)
    )
    user = FIDELITY_JUDGE_USER.format(
        source_question=source_question,
        reference_answer=reference_answer,
        phrased=phrased,
    )
    return call_llm(system=system, user=user, schema=FidelityVerdict)


def grade_interviewer(kb, basket: list[dict]) -> dict:
    """Run the InterviewerAgent on each case and grade three dimensions:

      - selection : when `expected_skill_in_choice` is set, the chosen
                    question's skill_tags must contain that skill.
      - anchor    : when `anchor_must_appear` is set, that substring must
                    appear in the phrasing (case-insensitive).
      - fidelity  : when `check_fidelity` is True, an LLM judge confirms the
                    paraphrase asks the same thing and doesn't leak the answer.

    The combined score is the mean across all enabled checks. Each dimension
    is also reported separately so a regression can be located.
    """
    agent = InterviewerAgent()
    rows = []
    sel_total = sel_pass = 0
    anc_total = anc_pass = 0
    fid_total = fid_pass = 0
    for case in basket:
        candidates = [kb.get(qid) for qid in case["candidate_ids"]]
        cv_skills = case.get("cv_skills")
        cv_profile = CVProfile(skills=cv_skills) if cv_skills else None
        choice = agent.ask(
            candidates=candidates,
            topic=case["topic"],
            difficulty=case["difficulty"],
            cv_profile=cv_profile,
        )
        chosen = kb.get(choice.id)

        row = {"id": case["id"], "chosen_id": choice.id,
               "phrased_excerpt": choice.phrased[:120]}

        if "expected_skill_in_choice" in case:
            sel_total += 1
            want = case["expected_skill_in_choice"].lower()
            tags = {t.lower() for t in chosen.skill_tags}
            ok = want in tags
            sel_pass += ok
            row["selection_pass"] = ok

        if "anchor_must_appear" in case:
            anc_total += 1
            term = case["anchor_must_appear"].lower()
            ok = term in choice.phrased.lower()
            anc_pass += ok
            row["anchor_pass"] = ok

        if case.get("check_fidelity"):
            fid_total += 1
            verdict = _judge_fidelity(
                source_question=chosen.question,
                reference_answer=chosen.reference_answer,
                phrased=choice.phrased,
            )
            ok = verdict.same_intent and not verdict.leaks_answer
            fid_pass += ok
            row["fidelity_pass"] = ok
            row["fidelity_comment"] = verdict.comment[:160]

        # Combined per-case pass = all enabled checks must pass.
        checks = [v for k, v in row.items() if k.endswith("_pass")]
        row["pass"] = bool(checks) and all(checks)
        rows.append(row)

    def _ratio(p, t):
        return (p / t) if t else 1.0

    sel_acc = _ratio(sel_pass, sel_total)
    anc_acc = _ratio(anc_pass, anc_total)
    fid_acc = _ratio(fid_pass, fid_total)
    # Weighted combined score: equal weight per dimension that ran.
    parts = []
    if sel_total: parts.append(sel_acc)
    if anc_total: parts.append(anc_acc)
    if fid_total: parts.append(fid_acc)
    combined = sum(parts) / len(parts) if parts else 1.0

    return {
        "interviewer_score": combined,
        "interviewer_selection": sel_acc,
        "interviewer_anchor": anc_acc,
        "interviewer_fidelity": fid_acc,
        "rows": rows,
    }


# ---- Bias grader -----------------------------------------------------------
# Each case is a perturbation pair. We run the agent on both the baseline and
# the perturbed input, then check whether the delta is within tolerance.
# A bias is "controlled" when small perturbations of the irrelevant feature
# produce small deltas. A large delta means the model is being swayed by
# something it shouldn't care about.


def _evaluate(agent: EvaluatorAgent, question: Question, answer: str) -> ScoreReport:
    return agent.evaluate(question=question, user_answer=answer)


def grade_bias(kb, basket: list[dict]) -> dict:
    """Run each bias case and return per-bias pass rates + a combined score.

    The handled biases:
      halo            — appending an unrelated confidence boast should not
                        move clarity / structure (the dimensions that don't
                        depend on the boast's content).
      length          — padding a mediocre answer with filler should not
                        raise the overall score by more than 1.
      lexical_mirror  — an answer that parrots rubric phrases without
                        substance should not earn a content score above the
                        case's threshold (default 3).
      position        — shuffling the Interviewer's candidate order should
                        leave the chosen id unchanged.

    Returns one row per case with the raw scores so a regression is
    diagnosable, not just a thumbs-up/-down.
    """
    evaluator = EvaluatorAgent()
    interviewer = InterviewerAgent()
    rows = []
    by_kind_pass: dict[str, int] = {}
    by_kind_total: dict[str, int] = {}

    for case in basket:
        kind = case["kind"]
        by_kind_total[kind] = by_kind_total.get(kind, 0) + 1
        row = {"id": case["id"], "kind": kind}
        ok = False

        if kind == "halo":
            q = kb.get(case["question_id"])
            base = _evaluate(evaluator, q, case["baseline_answer"])
            pert = _evaluate(evaluator, q, case["perturbed_answer"])
            tol = case.get("dimension_max_delta", 1)
            d_clarity = pert.clarity.score - base.clarity.score
            d_structure = pert.structure.score - base.structure.score
            ok = abs(d_clarity) <= tol and abs(d_structure) <= tol
            row.update({
                "base_overall": base.overall, "pert_overall": pert.overall,
                "d_clarity": d_clarity, "d_structure": d_structure,
                "tolerance": tol,
            })

        elif kind == "length":
            q = kb.get(case["question_id"])
            base = _evaluate(evaluator, q, case["baseline_answer"])
            pert = _evaluate(evaluator, q, case["perturbed_answer"])
            tol = case.get("overall_max_delta", 1)
            d_overall = pert.overall - base.overall
            # The bias is "padded answer rated higher". Negative deltas are fine.
            ok = d_overall <= tol
            row.update({
                "base_overall": base.overall, "pert_overall": pert.overall,
                "d_overall": d_overall, "tolerance": tol,
            })

        elif kind == "lexical_mirror":
            q = kb.get(case["question_id"])
            base = _evaluate(evaluator, q, case["baseline_answer"])
            pert = _evaluate(evaluator, q, case["perturbed_answer"])
            ceiling = case.get("content_max_score_for_perturbed", 3)
            ok = pert.content.score <= ceiling
            row.update({
                "base_content": base.content.score,
                "pert_content": pert.content.score,
                "content_ceiling": ceiling,
            })

        elif kind == "position":
            candidates = [kb.get(qid) for qid in case["candidate_ids"]]
            cv = CVProfile(skills=case["cv_skills"]) if case.get("cv_skills") else None
            choice_a = interviewer.ask(
                candidates=candidates,
                topic=case["topic"], difficulty=case["difficulty"],
                cv_profile=cv,
            )
            choice_b = interviewer.ask(
                candidates=list(reversed(candidates)),
                topic=case["topic"], difficulty=case["difficulty"],
                cv_profile=cv,
            )
            ok = choice_a.id == choice_b.id
            row.update({"order_a": choice_a.id, "order_b": choice_b.id})

        else:
            row["error"] = f"unknown bias kind: {kind}"

        row["pass"] = ok
        by_kind_pass[kind] = by_kind_pass.get(kind, 0) + (1 if ok else 0)
        rows.append(row)

    total = len(basket)
    overall = sum(1 for r in rows if r["pass"]) / total if total else 1.0
    out = {"bias_pass": overall, "rows": rows}
    for kind, total_k in by_kind_total.items():
        out[f"bias_{kind}"] = by_kind_pass.get(kind, 0) / total_k
    return out


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
