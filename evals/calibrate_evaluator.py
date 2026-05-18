"""Calibrate the Evaluator (LLM-as-judge) against human labels.

    python -m evals.calibrate_evaluator

Runs the golden set through Evaluator v1 (single combined prompt) and v2
(one judge per criterion), computes quadratic-weighted Cohen's kappa against
the human `overall` labels, and prints a leaderboard comparing the two.

Makes real API calls: ~24 for v1, ~72 for v2.
"""
from __future__ import annotations

import json
from pathlib import Path

from dotenv import load_dotenv

from core.kb import KnowledgeBase
from evals.graders import grade_evaluator
from evals.leaderboard import add_to_leaderboard, show_leaderboard
from evals.metrics import kappa_label

GOLDEN_PATH = Path(__file__).parent / "golden" / "evaluator_golden.jsonl"


def load_golden(path: Path = GOLDEN_PATH) -> list[dict]:
    """Read the golden JSONL, skipping blank lines and # comments."""
    cases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(json.loads(line))
    return cases


def _report(version: str, result: dict) -> None:
    print(f"\n--- Evaluator {version} vs human labels ---")
    print(f"{'id':<10}{'human':>7}{'model':>7}   match")
    for r in result["rows"]:
        flag = "exact" if r["pass"] else ("within1" if r["within1"] else "OFF")
        print(f"{r['id']:<10}{r['human']:>7}{r['model']:>7}   {flag}")
    print(f"exact={result['exact']:.2f}  within1={result['within1']:.2f}  "
          f"kappa={result['kappa']:.3f} ({kappa_label(result['kappa'])})")


def main() -> None:
    load_dotenv()  # pick up ANTHROPIC_API_KEY / LANGSMITH_* from .env
    golden = load_golden()
    print(f"Loaded {len(golden)} golden examples.")
    print("Building the knowledge base...")
    kb = KnowledgeBase()

    for version in ("v1", "v2"):
        print(f"\nRunning Evaluator {version} over the golden set "
              f"(real API calls)...")
        result = grade_evaluator(kb, golden, version=version)
        _report(version, result)
        add_to_leaderboard(f"evaluator-{version}", {
            "kappa": result["kappa"],
            "exact": result["exact"],
            "within1": result["within1"],
        })

    print("\n=== CALIBRATION LEADERBOARD ===")
    show_leaderboard()
    print("\nNote: the human labels in evaluator_golden.jsonl are DRAFT. "
          "Review them before trusting these kappa values.")


if __name__ == "__main__":
    main()
