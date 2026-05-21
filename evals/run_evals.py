"""Run the eval baskets and print the leaderboard.

    python -m evals.run_evals                       # free angles only (planner, retriever)
    python -m evals.run_evals --director            # also run the Director (real API calls)
    python -m evals.run_evals --interviewer         # also run the Interviewer (real API calls)
    python -m evals.run_evals --bias                # also run the bias suite (real API calls)
    python -m evals.run_evals --all                 # everything

Each basket -> grader -> append to the leaderboard. The leaderboard frames
evaluation as comparison; rerun it after any change to the system and
compare the rows.
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from core.kb import KnowledgeBase
from evals.baskets import (
    BIAS_BASKET,
    DIRECTOR_BASKET,
    INTERVIEWER_BASKET,
    PLANNER_BASKET,
    RETRIEVER_BASKET,
)
from evals.graders import (
    grade_bias,
    grade_director,
    grade_interviewer,
    grade_planner,
    grade_retriever,
)
from evals.leaderboard import add_to_leaderboard, show_leaderboard

VERSION = "baseline"


def _print_rows(rows: list[dict]) -> None:
    for r in rows:
        mark = "PASS" if r["pass"] else "FAIL"
        extra = {k: v for k, v in r.items() if k not in ("id", "pass")}
        print(f"  [{mark}] {r['id']}  {extra}")


def main() -> None:
    load_dotenv()  # pick up ANTHROPIC_API_KEY / LANGSMITH_* from .env
    parser = argparse.ArgumentParser(description="Run the Interview Prep Coach evals.")
    parser.add_argument(
        "--director", action="store_true",
        help="also run the Director basket (makes real LLM API calls)",
    )
    parser.add_argument(
        "--interviewer", action="store_true",
        help="also run the Interviewer basket (selection / anchor / fidelity; real LLM calls)",
    )
    parser.add_argument(
        "--bias", action="store_true",
        help="also run the bias basket (halo / length / lexical / position; real LLM calls)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="run every basket (planner, retriever, director, interviewer, bias)",
    )
    args = parser.parse_args()
    run_director = args.director or args.all
    run_interviewer = args.interviewer or args.all
    run_bias = args.bias or args.all

    print("Building the knowledge base (Chroma index)...")
    kb = KnowledgeBase()

    # --- Planner (deterministic, free) -------------------------------------
    print(f"\nPlanner basket - {len(PLANNER_BASKET)} cases")
    planner_result = grade_planner(kb, PLANNER_BASKET)
    _print_rows(planner_result["rows"])
    add_to_leaderboard(VERSION, {"planner_acc": planner_result["planner_acc"]})

    # --- Retriever (deterministic, free) -----------------------------------
    print(f"\nRetriever basket - {len(RETRIEVER_BASKET)} cases")
    retriever_result = grade_retriever(kb, RETRIEVER_BASKET)
    _print_rows(retriever_result["rows"])
    add_to_leaderboard(VERSION, {"retriever_pass": retriever_result["retriever_pass"]})

    # --- Director (LLM decision — costs API credits) -----------------------
    if run_director:
        print(f"\nDirector basket - {len(DIRECTOR_BASKET)} cases (real API calls)")
        director_result = grade_director(DIRECTOR_BASKET)
        _print_rows(director_result["rows"])
        add_to_leaderboard(VERSION, {"director_acc": director_result["director_acc"]})
    else:
        print("\nDirector basket skipped - pass --director to run it "
              "(it makes real LLM calls).")

    # --- Interviewer (selection + anchor + fidelity — costs API credits) ---
    if run_interviewer:
        print(f"\nInterviewer basket - {len(INTERVIEWER_BASKET)} cases (real API calls)")
        interviewer_result = grade_interviewer(kb, INTERVIEWER_BASKET)
        _print_rows(interviewer_result["rows"])
        add_to_leaderboard(VERSION, {
            "interviewer_score":     interviewer_result["interviewer_score"],
            "interviewer_selection": interviewer_result["interviewer_selection"],
            "interviewer_anchor":    interviewer_result["interviewer_anchor"],
            "interviewer_fidelity":  interviewer_result["interviewer_fidelity"],
        })
    else:
        print("\nInterviewer basket skipped - pass --interviewer to run it "
              "(it makes real LLM calls).")

    # --- Bias suite (perturbation pairs — costs API credits) ---------------
    if run_bias:
        print(f"\nBias basket - {len(BIAS_BASKET)} cases (real API calls)")
        bias_result = grade_bias(kb, BIAS_BASKET)
        _print_rows(bias_result["rows"])
        bias_metrics = {k: v for k, v in bias_result.items() if k != "rows"}
        add_to_leaderboard(VERSION, bias_metrics)
    else:
        print("\nBias basket skipped - pass --bias to run it "
              "(it makes real LLM calls).")

    print("\n=== LEADERBOARD ===")
    show_leaderboard()


if __name__ == "__main__":
    main()
