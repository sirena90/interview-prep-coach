# Testing Guide — Interview Prep Coach

A walkthrough of how testing is organised in this project, how to run it, and
what every part contains. Written for someone who did **not** write the tests
and just needs to understand and use them.

---

## 1. The big picture

The project is tested in **two separate suites**, because they answer two
different questions:

| Suite | Folder | Answers | Cost | Run it... |
|---|---|---|---|---|
| **Test suite** | `tests/` | "Is the code still correct?" | Free, ~1 sec | Every time you change code |
| **Eval suite** | `evals/` | "How *good* is the system, and is version B better than A?" | Some free, some paid | Before a release / for the report |

The guiding rule: **cheap checks first.**
Plain code tests catch most bugs for free and instantly; the expensive
checks — ones that call a real LLM — are kept separate and run on demand.

- `tests/` uses **pytest**, the standard Python test runner. These tests
  never call a real LLM (they use a fake — see §6), so they cost nothing.
- `evals/` is a small home-grown evaluation pipeline: it runs the system on
  curated examples and scores it. Some of it calls the real LLM API.

---

## 2. Quick start

### One-time setup

```bash
# Install the test tools (pytest etc.) — separate from the app's own deps.
pip install -r requirements-dev.txt
```

That's all you need for the **free** tests. For the extras:
- The **integration tests** need `chromadb` (already in `requirements.txt`).
  Their first run also downloads a ~80 MB embedding model, once.
- The **eval suite's paid parts** need a working `.env` with an LLM API key.

### The commands you'll actually use

```bash
pytest                       # the fast test suite — 73 tests, ~1 sec, free
pytest -v                    # same, but prints every test name
pytest -m integration        # the 7 knowledge-base tests (builds the index)
pytest -m "not llm"          # everything safe to run — fast + integration (80)
pytest tests/test_planner.py # run just one file
pytest -k difficulty         # run tests whose name contains "difficulty"

python -m evals.run_evals                       # eval: planner + retriever (free)
python -m evals.run_evals --director            # eval: + the Director agent (real API calls)
python -m evals.run_evals --interviewer         # eval: + the Interviewer agent (real API calls)
python -m evals.run_evals --bias                # eval: + the bias suite (real API calls)
python -m evals.run_evals --followup-rubric     # eval: + the Director's follow-up rubric (real API calls)
python -m evals.run_evals --all                 # eval: every basket above
python -m evals.calibrate_evaluator             # eval: Evaluator v1 vs v2 (real API calls)
```

A plain `pytest` run will **never** cost money or need the network — that is
by design (see §5).
python -m evals.run_evals
---

## 3. The test suite — `tests/`

Eight files. Together: **~114 fast tests + 7 integration tests.**

| File | What it checks | Needs |
|---|---|---|
| `test_models.py` | Data objects: score maths, validation rules | nothing |
| `test_planner.py` | The question-scheduling logic | nothing |
| `test_llm_wrapper.py` | The LLM wrapper + provider detection | nothing |
| `test_agents_contract.py` | The 4 LLM agents + follow-up validator (with a fake LLM) | nothing |
| `test_app_flow.py` | The per-answer orchestrator wiring (which question gets graded, when a follow-up is installed, loop guard) | nothing |
| `test_cv_parser.py` | CV text extraction + parsing | nothing |
| `test_kb_retrieval.py` | The knowledge base / search | `chromadb` |
| `conftest.py` | Shared test setup (not tests itself) | — |

### File-by-file

**`test_models.py`** — The project's data structures (`core/models.py`) are
Pydantic objects with built-in rules. These tests confirm those rules hold:
the rolling-score average maths is correct, a 1–5 score normalises to 0–1
properly, an out-of-range score (0 or 6) is rejected, a session is marked
"complete" only after 5 turns. Pure maths and validation — no surprises.

**`test_planner.py`** — `core/planner.py` decides what question comes next.
It is plain Python with fixed rules, so it can be checked exactly: question 1
is always a "cover" slot, question 3 is always behavioural, a weak answer
makes the next question easier, a strong answer makes it harder, difficulty
never drops below "entry" or above "senior". A seeded random generator makes
it reproducible.

**`test_llm_wrapper.py`** — `core/llm.py` is the single point every LLM call
goes through. These tests cover the parts that don't need a real API call:
**provider auto-detection** (the system picks Anthropic / OpenAI / Mistral
based on which API key is in `.env`), stripping stray ```` ``` ```` fences
from model output, normalising token counts, and the observability log
(see §8) — including that a broken log path can never crash the app.

**`test_agents_contract.py`** — The four LLM agents (Interviewer, Evaluator,
Director, Coaching Summariser). The agents normally call an LLM, which is slow
and costs money — so these tests swap in a **fake LLM** (see §6) that returns
canned answers. That lets them check the *code around* the model for free:
does the Evaluator's "v2" mode correctly combine three separate scores, does
the Interviewer fall back safely if the model returns a bad ID, does the
prompt actually include the candidate's CV. Also covers the runtime defences
that gate the Director's LLM-generated follow-up rubric: `validate_followup_choice`
must degrade to MOVE_ON when the reference answer fails self-consistency or
when the follow-up fields are missing.

**`test_app_flow.py`** — The orchestration seam between the Streamlit UI and
the agents. `app.evaluate_and_decide` is a pure function (no `st.session_state`)
that decides which question gets graded on each turn and whether a follow-up
is installed for the next one. These tests lock in the wiring invariant that
the previous version of the code got wrong: *if a follow-up question is
active, the candidate's next answer is graded against the follow-up's rubric,
not the original slot's rubric.* They are the regression net for that whole
class of bug — exactly the kind of issue that earlier tests missed because
each agent was tested in isolation but the wiring between them was not.

**`test_cv_parser.py`** — `core/cv_parser.py` reads an uploaded CV. Tests
cover the parts that don't need an LLM: extracting text from a file, cutting
off over-long CVs, rejecting unsupported input — plus a fake-LLM check that an
empty CV is handled without even calling the model.

**`test_kb_retrieval.py`** — `core/kb.py` is the searchable question bank
(129 questions in a Chroma vector database). These tests build the **real**
index, so they are slower (~10 sec) and marked `integration` — a plain
`pytest` skips them. They check that all 129 questions load, that a search
returns questions of the right topic, and that CV skills correctly re-rank
results.

**`conftest.py`** — Not tests. This is shared **setup** that pytest loads
automatically. It defines the **fake LLM** and small "builder" helpers
(fixtures) that other test files reuse to construct sample objects without
boilerplate.

---

## 4. The eval suite — `evals/`

Where `tests/` asks "is it correct?", `evals/` asks "is it *good*?". It runs
the system on curated example sets and produces scores on a **leaderboard**.

| File | What it is |
|---|---|
| `baskets.py` | The example sets ("task baskets") — planner, retriever, director, interviewer, bias, follow-up rubric |
| `graders.py` | The scoring functions — one per eval angle |
| `metrics.py` | Cohen's kappa (agreement metric, see §6) |
| `leaderboard.py` | Collects scores into a comparison table |
| `golden/evaluator_golden.jsonl` | 24 hand-labelled answers — the "correct" grades |
| `run_evals.py` | Runs the planner / retriever / director / interviewer / bias / follow-up rubric evals |
| `calibrate_evaluator.py` | Compares Evaluator v1 vs v2 |
| `langsmith_experiment.py` | Same comparison, uploaded to the LangSmith dashboard |

### What each eval measures

- **Planner eval** (free) — feeds the scheduler past situations and checks it
  picks the expected next question. Deterministic.
- **Retriever eval** (free) — checks the question search returns on-topic
  results and that CV-skill re-ranking works.
- **Director eval** (paid — real API) — gives the Director agent scripted
  situations and checks it picks a reasonable next action (move on, dig
  deeper, clarify…). Because an LLM's answer varies, each case allows a *set*
  of acceptable actions.
- **Interviewer eval** (paid — real API) — runs the Interviewer agent on
  hand-built candidate lists and grades three things:
  *selection* (when one candidate carries a CV-overlapping skill_tag, does the
  agent pick it?), *CV anchoring* (does the phrasing actually mention the CV
  term?), and *reformulation fidelity* (an LLM judge confirms the paraphrase
  asks the same thing and does not leak the answer).
- **Bias eval** (paid — real API) — probes four biases by running the same
  agent on a baseline input and a deliberately perturbed twin and checking
  the score delta stays small:
  - *halo effect* — append an unrelated confidence boast to a strong answer;
    clarity and structure should not jump.
  - *length bias* — pad a mediocre answer with filler; overall should not rise.
  - *lexical mirroring* — an answer that parrots rubric phrases without
    substance; content should still be modest (≤3).
  - *position bias* — shuffle the Interviewer's candidate order; the chosen
    id should be stable.
- **Follow-up rubric eval** (paid — real API) — when the Director picks
  `dig_deeper` / `followup` / `clarify`, it emits a rubric and a reference
  answer for grading the candidate's next reply (Option B, see
  [`ARCHITECTURE.md`](ARCHITECTURE.md) §4.7). That rubric is LLM-generated
  and so needs quality measurement. The eval runs the Director on a few
  curated cases and checks the rubric is:
  - *observable* — ≥2 content criteria that are concrete enough to evaluate;
  - *consistent* — the Director's own reference answer scores ≥4 against
    its own rubric (same self-consistency check defence #3 enforces at
    runtime);
  - *distinct* — the generated criteria don't just duplicate the slot
    question's curated rubric.
- **Evaluator calibration** (paid — real API) — the big one. The Evaluator
  agent grades candidate answers; is it any good? `calibrate_evaluator.py`
  runs it over the 24 hand-labelled answers in `golden/` and measures how well
  it agrees with the human grades, for two versions of the Evaluator (v1 and
  v2). See §6 for "kappa".

### Running it

```bash
python -m evals.run_evals                       # planner + retriever, free
python -m evals.run_evals --director            # + Director eval, costs API credits
python -m evals.run_evals --interviewer         # + Interviewer eval, costs API credits
python -m evals.run_evals --bias                # + bias eval, costs API credits
python -m evals.run_evals --followup-rubric     # + follow-up rubric eval, costs API credits
python -m evals.run_evals --all                 # every basket above
python -m evals.calibrate_evaluator             # Evaluator v1 vs v2, costs API credits
```

Each run prints a **leaderboard** — the point of an eval is comparison, so
you rerun it after a change and compare rows.

> The 24 grades in `golden/evaluator_golden.jsonl` are a **draft**. They are
> the "ground truth" the Evaluator is measured against, so they should be
> reviewed by a human before the calibration numbers are trusted.

---

## 5. Test markers — why `pytest` skips some tests

A "marker" is a label on a test. This project uses two, both configured in
`pytest.ini`:

- `integration` — the test builds the real Chroma index (slower).
- `llm` — reserved for tests that would call the real LLM API (none exist
  yet; the real-API work lives in `evals/`, run separately).

By default `pytest` runs **neither** — `pytest.ini` deselects them. So:

| Command | Runs |
|---|---|
| `pytest` | 121 fast tests only |
| `pytest -m integration` | the 7 integration tests only |
| `pytest -m "not llm"` | everything except paid — all 128 |

This is the safety net: someone can clone the repo and run `pytest` with no
API key, no network, and no cost.

---

## 6. Concepts you need to know

**Fixture** — a reusable piece of test setup provided by pytest. In this
project they live in `conftest.py` and act as "builders": e.g. `make_question`
hands a test a ready-made sample question. Tests receive them by naming them
as arguments.

**FakeLLM** — a stand-in for the real LLM, defined in `conftest.py`. You tell
it what to return; it records the prompts it was given. This is how the agent
tests run for free: the agent thinks it called a model, but the "model" was a
fake returning a canned answer. The test can then inspect both the result and
the prompt that was built.

**Golden set** — a file of examples with known-correct answers
(`evals/golden/evaluator_golden.jsonl`). It is the human "ground truth" used
to judge the Evaluator.

**Cohen's kappa** — a number from roughly 0 to 1 measuring how much two
graders *agree*, corrected for lucky guesses. Used to ask "does the LLM
Evaluator grade answers the way a human would?" Plain accuracy is misleading
here (a lazy grader that always says "3" can look 60% accurate); kappa is
not fooled. Rough scale: 0.6 = solid, 0.8 = near-human.

**Leaderboard** — a small table in `evals/` that holds one row per system
version. Evaluation is about *comparison*: you change something, rerun the
evals, and compare the new row to the old one.

**v1 vs v2 Evaluator** — the Evaluator agent has two modes. v1 grades an
answer with one prompt; v2 uses three focused prompts (one per criterion) and
combines the scores. The calibration eval exists to measure which is better.

---

## 7. What costs money, what doesn't

| Activity | Cost |
|---|---|
| `pytest` (any marker) | **Free** — never calls a real LLM |
| `python -m evals.run_evals` (no flag) | **Free** — planner + retriever only |
| `python -m evals.run_evals --director` | **Paid** — real API calls |
| `python -m evals.run_evals --interviewer` | **Paid** — real API calls |
| `python -m evals.run_evals --bias` | **Paid** — real API calls |
| `python -m evals.run_evals --followup-rubric` | **Paid** — real API calls |
| `python -m evals.run_evals --all` | **Paid** — every basket above |
| `python -m evals.calibrate_evaluator` | **Paid** — ~100 API calls |
| `python -m evals.langsmith_experiment` | **Paid** + needs a LangSmith key |

If in doubt: pytest is always free; an `evals` command is paid only if it
says so above.

---

## 8. Observability — the call log

Separate from testing, but related. Every real LLM call (in normal app use or
in the paid evals) is recorded two ways, both wired into `core/llm.py`:

- **`logs/llm_calls.jsonl`** — a local file, one line per call: which
  provider, model, how long it took, token counts, success/failure. Always
  on, needs nothing.
- **LangSmith** — if a `LANGSMITH_API_KEY` is set in `.env`, the same calls
  also appear in the LangSmith web dashboard. If no key is set, this silently
  does nothing — the app is unaffected.

---

## 9. How to add a new test

1. Find the right file in `tests/` (e.g. planner logic → `test_planner.py`).
2. Add a function named `test_something`. Use a `conftest.py` builder if you
   need a sample object.
3. For agent code, take the `fake_llm` fixture as an argument and queue a
   canned response — never call the real API in `tests/`.
4. Run `pytest` and confirm it passes.

When a bug is found in production, the habit is: reproduce it as a failing
test first, then fix the code. That test then stays forever and stops the bug
coming back.

---

## 10. Reference — full file map

```
pytest.ini                  Test configuration (markers, default options)
requirements-dev.txt        Test-only dependencies (pytest)

tests/
  conftest.py               Shared setup: FakeLLM + object builders
  test_models.py            Data objects: score maths, validation
  test_planner.py           Question-scheduling rules
  test_llm_wrapper.py       LLM wrapper + provider auto-detection
  test_agents_contract.py   The 4 LLM agents + follow-up validator (fake LLM)
  test_app_flow.py          Per-answer orchestrator wiring (fake LLM)
  test_cv_parser.py         CV extraction + parsing
  test_kb_retrieval.py      Knowledge-base search (integration)

evals/
  baskets.py                Curated example sets
  graders.py                Scoring functions
  metrics.py                Cohen's kappa
  leaderboard.py            The comparison table
  run_evals.py              Run planner / retriever / director / interviewer / bias evals
  calibrate_evaluator.py    Evaluator v1 vs v2 calibration
  langsmith_experiment.py   Calibration as a LangSmith experiment
  golden/
    evaluator_golden.jsonl  24 hand-labelled answers (ground truth)
```
