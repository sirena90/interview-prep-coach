# Interview Prep Coach

A multi-turn coaching chatbot for technical and behavioural interview prep.
Powered by RAG over a curated question bank and four LLM agents. Works with
Anthropic Claude, OpenAI, or Mistral — the provider is auto-detected from
whichever API key you put in `.env`.

## What it does

The user picks a **role** (Data Analyst, QA Engineer, Data Engineer, Frontend
Developer) and a **difficulty level** (entry, mid, senior), uploads a CV, and
goes through a session of **5 questions** with structured feedback after each
answer. At the end they receive a personalised coaching report that uses the
CV for concrete recommendations.

## Session structure

```
Q1 — COVER       (a fresh topic for the role)
Q2 — REINFORCE   (same topic as Q1, difficulty adapts to score)
Q3 — BEHAVIORAL  (STAR question)
Q4 — COVER       (a different fresh topic)
Q5 — REINFORCE   (same topic as Q4, difficulty adapts to score)
```

After each answer the **Conversation Director agent** decides whether to
stay on the current question (`clarify` / `followup` / `dig_deeper`) or
to move on to the next planned slot (`move_on`). The Director is capped at
**1 follow-up round per slot** so a single slot can't trap the session.

For reinforce slots the difficulty adapts to the previous score: weak
(< 0.4) → step down, strong (> 0.85) → step up, otherwise stay. Floored at
ENTRY, capped at SENIOR.

## Quick start

### 1. Dependencies

```bash
# Create a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate    # Mac / Linux
# .venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt

# (Optional) tests + eval pipeline — adds pytest and coverage
pip install -r requirements-dev.txt
```

> Two requirements files on purpose: `requirements.txt` is what you need
> **to run the app**; `requirements-dev.txt` adds **test/eval** tooling
> (pytest, pytest-cov). If you're just running the coach, the first file is
> enough.

### 2. API key

The app supports **Anthropic Claude, OpenAI, and Mistral** — the active
provider is auto-detected from whichever `*_API_KEY` is set in your `.env`:

- `ANTHROPIC_API_KEY` → Anthropic Claude
- `OPENAI_API_KEY` → OpenAI
- `MISTRAL_API_KEY` → Mistral

Get a key from your chosen provider, then:

```bash
# Copy the template
cp .env.example .env

# Edit .env and fill in ONE of the keys. If several are set, priority is
# Anthropic > OpenAI > Mistral.
```

The model name per provider is also overridable in `.env` — useful if your
account doesn't have access to the default model, or you want to use a
different one without touching the code:

```bash
# ANTHROPIC_DEFAULT_MODEL=claude-sonnet-4-5-20250929
# OPENAI_DEFAULT_MODEL=gpt-4o
# MISTRAL_DEFAULT_MODEL=mistral-large-latest
```

Optional: set `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` in `.env` to
trace every LLM call in the LangSmith dashboard. Without it, calls still log
locally to `logs/llm_calls.jsonl`.

### 3. Run

```bash
streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

## How to use (for the candidate)

### Before you start

- Have a **CV** ready as a PDF or .txt file. Easiest path: **export your
  LinkedIn profile as PDF** (Profile → More → Save to PDF).
- No LinkedIn? Write a short .txt with your skills, experience and projects
  (200–300 words is enough).
- You can pick **any role** — the CV is used to personalise the questions
  *within* that role, not to choose the role. Practising a role outside
  your background is a valid use case.

### Walkthrough

You'll see three screens in sequence.

**1. Setup** — pick your **role** (Data Analyst, QA, Data Engineer, Frontend
Developer), pick a **difficulty** (entry / mid / senior), upload your **CV**,
then click *Start interview*. The app parses your CV once (you'll see a brief
spinner) and starts the session.

**2. Interview** — a chat screen for the 5-question session.
- The header shows your role, difficulty, and progress (`Question 2 of 5`).
- Each question carries a small caption — *Topic: …, Difficulty: …* — and,
  when your CV matches the question, a *📄 Picked from your CV: …* badge so
  you can see which of your skills drove the choice.
- Type your answer in the chat box and press Enter. If you genuinely don't
  know the question, click **⏭️ I don't know — skip this question** — the
  app records a low score for the topic (no feedback) and moves on.
- After each answer you get a **structured score**: overall 1–5, plus per-
  dimension scores (content / clarity / structure), short comments per
  dimension, and 2–4 concrete tips under *Improve:*.
- Sometimes the coach will follow up on the same question (a `dig_deeper`,
  `followup`, or `clarify` round) before moving on — you'll see a small
  label like `(dig_deeper)` above the new question. Each slot is capped at
  one follow-up to keep the session short.

**3. Final report** — once the 5 questions are done, you get:
- an **overall score** and a per-topic breakdown,
- **strengths** and **gaps** with the turn number each one came from,
- **study suggestions** the coach grounds in real reference answers from
  the knowledge base,
- and a short **personal coaching letter** that references your CV
  experience.

Click *Start a new session* to wipe the state and run another session.

## Architecture in short

```
app.py                       Streamlit UI (3 screens: setup, interview, final)
├── core/config.py           Multi-provider settings (auto-detect from .env)
├── core/models.py           Pydantic schemas (16 models, dependency-free)
├── core/llm.py              Provider-agnostic LLM wrapper (Anthropic / OpenAI /
│                            Mistral) with JSON validation, one-shot repair,
│                            rate-limit retry with Retry-After honouring,
│                            LangSmith tracing + offline logs/llm_calls.jsonl,
│                            and a provider-agnostic tool-use loop
├── core/kb.py               KB loader + Chroma retriever (129 questions,
│                            metadata filter + semantic search + CV-aware
│                            rerank + difficulty fallback)
├── core/cv_parser.py        PDF / text → CVProfile (one LLM call)
├── core/planner.py          Deterministic 5-slot scheduler (no LLM)
└── core/agents.py           Four LLM agents:
                               - Interviewer (picks + paraphrases the question)
                               - Evaluator (grades against the rubric; v1/v2)
                               - ConversationDirector (chooses the next action) ★
                               - CoachingSummariser (final report, uses tools) ★
```

★ = formally agentic component (Director = action selection in a closed
loop; Coach = tool use against the KB).

See `ARCHITECTURE.md` for a detailed description of each component and the
data flow.

## Knowledge base

**129 curated questions** in 3 JSONL files under `data/`:

| File                              | # questions | Content                            |
|-----------------------------------|-------------|------------------------------------|
| `questions_da_qa.jsonl`           | 60          | Data Analyst + QA Engineer         |
| `questions_de_fe.jsonl`           | 60          | Data Engineer + Frontend Developer |
| `questions_behavioural.jsonl`     |  9          | STAR questions, role-agnostic      |

Every question has a topic, subtopic, difficulty, reference answer, a rubric
(3 dimensions), tags, and `skill_tags` for CV-aware retrieval.

## Troubleshooting

**`No LLM API key set in .env`** — make sure `.env` is in the project root
and contains at least one of `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or
`MISTRAL_API_KEY`.

**Chroma error on startup** — on first run, Chroma indexes all 129
questions, which can take 10–20 seconds. The second run is instant (cache
is at `.chroma/`).

**`pypdf` error on CV upload** — the CV must be a PDF with a text layer
(exported from Word, LinkedIn, Google Docs). Scanned PDFs do not work.

**Session disappears on refresh** — this is by design; the simplified
version keeps state in memory, no SQLite. A refresh starts a new session.

## Technologies

- **Streamlit** for the UI (single-file Python web app)
- **Anthropic Claude / OpenAI / Mistral** — multi-provider, auto-detected
  from the API key in `.env`
- **pydantic-settings** for typed config (`core/config.py`)
- **ChromaDB** for the vector store and semantic retrieval
- **Pydantic** for type-safe data contracts
- **pypdf** for PDF text extraction
- **LangSmith** (optional) for tracing every LLM call, with a local
  `logs/llm_calls.jsonl` fallback when no key is set
- **No LangChain** — direct provider SDKs (Anthropic + OpenAI; Mistral uses
  the OpenAI-compatible API)

## Development

Tests live in `tests/` (pytest) and `evals/` (eval pipeline). See
[`TESTING.md`](TESTING.md) for the full guide.

**Before running any test or eval command**, install the dev dependencies:

```bash
pip install -r requirements-dev.txt      # adds pytest, pytest-cov — required for tests and evals
```

Then:

```bash
pytest                # 121 fast tests, no API calls, no cost
pytest -m integration # + 7 KB tests (builds the Chroma index)
```

> **Important:** eval scripts must be run as Python *modules* (with `-m`), not
> as plain scripts. Use `python -m evals.run_evals`, **not**
> `python evals/run_evals.py` — the latter causes an `ImportError` because the
> package imports won't resolve.

```bash
python -m evals.run_evals                       # planner + retriever evals (free)
python -m evals.run_evals --director            # + Director agent (real API calls)
python -m evals.run_evals --interviewer         # + Interviewer agent: selection / anchor / fidelity (paid)
python -m evals.run_evals --bias                # + bias suite: halo / length / lexical / position (paid)
python -m evals.run_evals --followup-rubric     # + Director follow-up rubric quality (paid)
python -m evals.run_evals --all                 # every basket above
python -m evals.calibrate_evaluator             # Evaluator v1 vs v2 (paid, ~96 calls)
python -m evals.langsmith_experiment            # same calibration, pushed to LangSmith
```

Each eval grader appends its score to a shared **leaderboard** so you can
re-run after a change and compare versions side by side.

## License

University course project — Interview Prep Coach, 2025.
