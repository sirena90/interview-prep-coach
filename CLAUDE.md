# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
cp .env.example .env          # add ANTHROPIC_API_KEY
uv sync --group dev           # installs runtime + ruff

# Run
uv run streamlit run app.py   # opens at http://localhost:8501
# Note: `VIRTUAL_ENV=/usr does not match...` warning from uv run is harmless — ignored automatically

# Lint
uv run ruff check .
uv run ruff format .

# Tests (none yet — directory doesn't exist)
uv run pytest tests/
```

## Architecture

Single-file Streamlit UI (`app.py`) drives a 3-screen flow (setup → interview loop → final report). All session state lives in `st.session_state` as a `SessionState` Pydantic object — **no persistence**, cleared on browser refresh.

```
app.py
├── core/models.py      All Pydantic schemas (source of truth for inter-module contracts)
├── core/llm.py         call_llm(system, user, schema) → validated Pydantic object
├── core/kb.py          ChromaDB over 129 JSONL questions; CV-aware reranking via skill_tags
├── core/cv_parser.py   PDF/txt → CVProfile (one LLM call)
├── core/planner.py     Deterministic 5-slot scheduler — no LLM, pure Python
└── core/agents.py      Four LLM agents (see below)
```

### The 5-slot session pattern

```
Q1=COVER → Q2=REINFORCE(Q1 topic) → Q3=BEHAVIORAL → Q4=COVER → Q5=REINFORCE(Q4 topic)
```

REINFORCE difficulty adapts: score < 0.4 → step down; score > 0.85 → step up; else stay.

### Four agents in `core/agents.py`

| Agent | Input → Output | Notes |
|---|---|---|
| `InterviewerAgent` | candidates + CV → `InterviewerChoice` | Skips LLM if only 1 candidate |
| `EvaluatorAgent` | question + answer → `ScoreReport` | Grades content/clarity/structure 1–5 |
| `ConversationDirectorAgent` ★ | question + answer + score → `DirectorChoice` | Picks: clarify / followup / dig_deeper / move_on. Stays on slot until move_on |
| `CoachingSummariserAgent` ★ | full `SessionState` → `SessionSummary` | Writes personalised coaching letter; uses higher max_tokens + temperature=0.3 |

★ = formally agentic (Director = action-selection loop; Summariser = generative personalisation).

### LLM wrapper pattern

Every agent is one `call_llm()` call. The wrapper strips JSON fences, validates against the passed Pydantic `schema`, and retries once with a repair prompt on parse failure. Default: `claude-sonnet-4-5-20250929`, `temperature=0`, `max_tokens=1024`.

### Adding a new agent

1. Define its output schema in `core/models.py`
2. Write system + user prompt templates in `core/agents.py`
3. Call `call_llm(system=..., user=..., schema=YourSchema)`
4. Wire into `app.py`

### KB structure

Questions in `data/*.jsonl` — `skill_tags` field drives CV-aware retrieval. `kb.retrieve()` falls back through difficulty levels if no candidates exist at the requested difficulty.
