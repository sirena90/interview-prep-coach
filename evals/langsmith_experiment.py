"""Run the Evaluator calibration as a LangSmith experiment.

    python -m evals.langsmith_experiment

Same calibration as evals/calibrate_evaluator.py, but pushed to LangSmith:
the golden set becomes a versioned Dataset, and v1 / v2 each run as an
Experiment with persistent traces and a side-by-side comparison view.

Requires LANGSMITH_API_KEY (free Developer tier — see smith.langchain.com).
Without it the script prints setup steps and exits; nothing else in the
project depends on LangSmith.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv

from core.agents import EvaluatorAgent
from core.kb import KnowledgeBase
from evals.calibrate_evaluator import load_golden

DATASET_NAME = "interview-coach-evaluator-golden"


def _require_langsmith() -> bool:
    if os.getenv("LANGSMITH_API_KEY"):
        return True
    print(
        "LANGSMITH_API_KEY is not set - skipping the experiment.\n"
        "  1. Sign up (free) at https://smith.langchain.com\n"
        "  2. Settings -> API Keys -> create a key\n"
        "  3. Set these environment variables, then re-run:\n"
        "       LANGSMITH_API_KEY=<your key>\n"
        "       LANGSMITH_TRACING=true\n"
    )
    return False


def push_dataset(client, golden: list[dict]) -> None:
    """Create the LangSmith dataset from the golden set (once)."""
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Dataset '{DATASET_NAME}' already exists - reusing it.")
        return
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Interview Prep Coach - Evaluator golden set "
                    "(question_id + answer + human overall label).",
    )
    client.create_examples(
        dataset_id=dataset.id,
        inputs=[{"question_id": c["question_id"], "answer": c["answer"]}
                for c in golden],
        outputs=[{"human_overall": c["human_overall"]} for c in golden],
    )
    print(f"Created dataset '{DATASET_NAME}' with {len(golden)} examples.")


# --- LangSmith evaluators: compare the run's output to the golden label -----

def exact_match_evaluator(run, example):
    predicted = run.outputs["overall"]
    expected = example.outputs["human_overall"]
    return {"key": "exact_match", "score": float(predicted == expected)}


def within_one_evaluator(run, example):
    predicted = run.outputs["overall"]
    expected = example.outputs["human_overall"]
    return {"key": "within_1", "score": float(abs(predicted - expected) <= 1)}


def main() -> None:
    load_dotenv()  # pick up ANTHROPIC_API_KEY / LANGSMITH_* from .env
    if not _require_langsmith():
        return

    from langsmith import Client
    from langsmith.evaluation import evaluate as ls_evaluate

    golden = load_golden()
    client = Client()
    push_dataset(client, golden)

    print("Building the knowledge base...")
    kb = KnowledgeBase()

    for version in ("v1", "v2"):
        agent = EvaluatorAgent(version=version)

        # _agent bound as a default arg so each closure keeps its own version.
        def predict(inputs: dict, _agent=agent) -> dict:
            question = kb.get(inputs["question_id"])
            report = _agent.evaluate(question=question, user_answer=inputs["answer"])
            return {"overall": report.overall}

        print(f"\nRunning experiment for Evaluator {version} (real API calls)...")
        ls_evaluate(
            predict,
            data=DATASET_NAME,
            evaluators=[exact_match_evaluator, within_one_evaluator],
            experiment_prefix=f"evaluator-{version}",
        )

    print("\nDone. Open https://smith.langchain.com to compare the two "
          "experiments side by side, with full traces attached.")


if __name__ == "__main__":
    main()
