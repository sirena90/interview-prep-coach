"""Run the eval baskets and print the leaderboard.

    python -m evals.run_evals               # free angles only (planner, retriever)
    python -m evals.run_evals --director    # also run the Director (real API calls)

Each basket -> grader -> append to the leaderboard. The leaderboard frames
evaluation as comparison; rerun it after any change to the system and
compare the rows.
"""
from __future__ import annotations

import argparse

from dotenv import load_dotenv

from core.kb import KnowledgeBase
from evals.baskets import DIRECTOR_BASKET, PLANNER_BASKET, RETRIEVER_BASKET
from evals.graders import grade_director, grade_planner, grade_retriever
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
        help="also run the Director basket (makes real Anthropic API calls)",
    )
    args = parser.parse_args()

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
    if args.director:
        print(f"\nDirector basket - {len(DIRECTOR_BASKET)} cases (real API calls)")
        director_result = grade_director(DIRECTOR_BASKET)
        _print_rows(director_result["rows"])
        add_to_leaderboard(VERSION, {"director_acc": director_result["director_acc"]})
    else:
        print("\nDirector basket skipped - pass --director to run it "
              "(it makes real Anthropic calls).")

    print("\n=== LEADERBOARD ===")
    show_leaderboard()


if __name__ == "__main__":
    main()
