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

After each answer the **Conversation Director agent** decides whether to stay
on the current question (`clarify` / `followup` / `dig_deeper`) or to move on
to the next planned slot (`move_on`).

## Quick start

### 1. Dependencies

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies (creates .venv automatically)
uv sync --group dev

# Activate virtualenv
source .venv/bin/activate    # Mac / Linux
# .venv\Scripts\activate     # Windows
```

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

Optional: set `LANGSMITH_API_KEY` + `LANGSMITH_TRACING=true` in `.env` to
trace every LLM call in the LangSmith dashboard. Without it, calls still log
locally to `logs/llm_calls.jsonl`.

### 3. Run

```bash
uv run streamlit run app.py
```

The browser opens automatically at `http://localhost:8501`.

## What to prepare

- **A CV** as a PDF or .txt file. Easiest path: **export your LinkedIn
  profile as PDF** (Profile → More → Save to PDF).
- If you don't have a LinkedIn profile, write a small .txt with your skills,
  experience and projects (200–300 words is enough).

## Architecture in short

```
app.py                       Streamlit UI (3 screens: setup, interview, final)
├── core/config.py           Multi-provider settings (auto-detect from .env)
├── core/models.py           Pydantic schemas
├── core/llm.py              Provider-agnostic LLM wrapper (Anthropic / OpenAI /
│                            Mistral) with JSON validation, rate-limit retry,
│                            LangSmith tracing, and a provider-agnostic
│                            tool-use loop
├── core/kb.py               KB loader + Chroma retriever (129 questions)
├── core/cv_parser.py        PDF / text → CVProfile
├── core/planner.py          Deterministic 5-slot scheduler
└── core/agents.py           Four LLM agents:
                               - Interviewer (picks + paraphrases the question)
                               - Evaluator (grades against the rubric; v1/v2)
                               - ConversationDirector (chooses the next action) ★
                               - CoachingSummariser (final report, uses tools) ★
```

★ = formally agentic component (action selection in a loop + generative
personalisation).

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

```bash
# Lint and format
uv run ruff check .
uv run ruff format .

# Install all dependencies (creates .venv automatically)
uv sync --group dev

uv run pytest                # 108 fast tests, no API calls, no cost
uv run pytest -m integration # + 7 KB tests (builds the Chroma index)

uv run python -m evals.run_evals             # planner + retriever evals (free)
uv run python -m evals.calibrate_evaluator   # Evaluator v1 vs v2 (paid)

```

## License

University course project — Interview Prep Coach, 2025.
