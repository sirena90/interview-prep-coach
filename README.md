# Interview Prep Coach

A multi-turn coaching chatbot for technical and behavioural interview prep.
Powered by RAG over a curated question bank and four LLM agents (Anthropic Claude).

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
cd interview-prep-coach/simple

# Create a virtualenv (recommended)
python -m venv venv
source venv/bin/activate    # Mac / Linux
# venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt
```

### 2. API key

You need an **Anthropic API key**. Sign up at
[console.anthropic.com](https://console.anthropic.com/settings/keys),
click "Create Key", and copy the key (it starts with `sk-ant-...`).

```bash
# Copy the template
cp .env.example .env

# Edit .env and paste your real key in place of the placeholder
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Run

```bash
streamlit run app.py
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
├── core/models.py           Pydantic schemas (16 models)
├── core/llm.py              Thin Anthropic wrapper with JSON validation
├── core/kb.py               KB loader + Chroma retriever (129 questions)
├── core/cv_parser.py        PDF / text → CVProfile
├── core/planner.py          Deterministic 5-slot scheduler
└── core/agents.py           Four LLM agents:
                               - Interviewer (picks + paraphrases the question)
                               - Evaluator (grades against the rubric)
                               - ConversationDirector (chooses the next action) ★
                               - CoachingSummariser (final report) ★
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

**`ANTHROPIC_API_KEY not set`** — make sure `.env` is in the `simple/`
folder and contains your real key.

**Chroma error on startup** — on first run, Chroma indexes all 129
questions, which can take 10–20 seconds. The second run is instant (cache
is at `simple/.chroma/`).

**`pypdf` error on CV upload** — the CV must be a PDF with a text layer
(exported from Word, LinkedIn, Google Docs). Scanned PDFs do not work.

**Session disappears on refresh** — this is by design; the simplified
version keeps state in memory, no SQLite. A refresh starts a new session.

## Technologies

- **Streamlit** for the UI (single-file Python web app)
- **Anthropic Claude** (`claude-sonnet-4-5`) for every LLM call
- **ChromaDB** for the vector store and semantic retrieval
- **Pydantic** for type-safe data contracts
- **pypdf** for PDF text extraction
- **No LangChain** — direct Anthropic SDK for simplicity and transparency

## Development

Run the tests (once they exist):

```bash
pytest tests/
```

## License

University course project — Interview Prep Coach, 2025.
