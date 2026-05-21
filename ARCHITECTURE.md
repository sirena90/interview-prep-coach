# Interview Prep Coach — Architecture (Simplified Streamlit Version)

> Reference document for the project. Describes each component, the data
> flow, and the agentic patterns. Serves both as a presentation aid and as
> a team-facing readme.

---

## 1. What the project is

**Interview Prep Coach** is a multi-turn coaching chatbot that helps
candidates prepare for technical and behavioural interviews for four roles:

- Data Analyst
- QA Engineer
- Data Engineer
- Frontend Developer

The user picks a role and a difficulty level, uploads a CV, and goes through
a 5-question session. After every answer they receive structured feedback.
At the end they receive a personalised coaching report that uses the CV for
concrete recommendations.

---

## 2. Architecture at a glance

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI (app.py)                          │
│        Setup → Interview loop (Q1..Q5) → Final report                   │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ session_state (Pydantic SessionState)
                                 ↓
┌──────────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│  Planner             │  │  KB / RAG       │  │  5 LLM components       │
│  (Python, no LLM)    │  │  (Chroma)       │  │                         │
│                      │  │                 │  │  • CV Parser  (setup)   │
│  5-slot pattern,     │  │  129 questions  │  │  • Interviewer          │
│  difficulty          │  │  in 3 JSONL,    │  │  • Evaluator (v1 / v2)  │
│  adaptation          │  │  CV-aware       │  │  • Director  ★          │
│                      │  │  rerank         │  │  • Coach     ★ (tools)  │
└──────────────────────┘  └─────────────────┘  └─────────────┬───────────┘
                                                             │ every LLM call
                                                             ↓
┌──────────────────────────────────────────────────────────────────────────────┐
│                       core/llm.py   —   LLM wrapper                          │
│                                                                              │
│   call_llm()              — plain JSON-validated calls                       │
│   call_llm_with_tools()   — provider-agnostic tool-use loop                  │
│                                                                              │
│   • per-provider routing   • rate-limit retry   • repair retry               │
└─────┬──────────────────────────────────────────────────────────────────┬─────┘
      │ provider                                                         │ trace
      ↓                                                                  ↓
┌──────────────────────────────┐                  ┌────────────────────────────┐
│ Settings (core/config.py)    │                  │ Observability              │
│                              │                  │                            │
│ Provider auto-detected from  │                  │ • LangSmith (when set)     │
│ the API key in .env:         │                  │ • logs/llm_calls.jsonl     │
│  ANTHROPIC / OPENAI /        │                  │   (always on)              │
│  MISTRAL                     │                  │                            │
└──────────────────────────────┘                  └────────────────────────────┘
```

**★** = formally agentic component (action selection in a loop + tool-using
personalisation). Everything else is deterministic Python or single-shot
LLM calls.

---

## 3. Session — the 5-question pattern

A session has a **fixed slot pattern**:

| Slot | Type        | Logic                                                                  |
|------|-------------|------------------------------------------------------------------------|
| Q1   | COVER       | Random uncovered topic for the role, base difficulty                   |
| Q2   | REINFORCE   | Same topic as Q1, difficulty adjusted by Q1's score                    |
| Q3   | BEHAVIORAL  | STAR question, base difficulty                                         |
| Q4   | COVER       | A different uncovered topic, base difficulty                           |
| Q5   | REINFORCE   | Same topic as Q4, difficulty adjusted by Q4's score                    |

### Adaptive difficulty rules (for reinforce slots)

```
prev_score < 0.4 (1-2/5)      → STEP DOWN  (e.g. mid → entry)
prev_score > 0.85 (5/5)       → STEP UP    (e.g. mid → senior)
otherwise (3-4/5)             → STAY at base difficulty
```

Difficulty is floored at ENTRY (easiest) and capped at SENIOR (hardest).

### The Director Agent can loop inside a slot

After the Evaluator, the **Director Agent** decides what comes next. If it
picks `clarify`, `followup`, or `dig_deeper`, the system **stays on the
current question** and waits for another answer. Only when the Director
picks `move_on` does the Planner choose the next slot.

That means a single slot can have multiple rounds of conversation before it
closes.

---

## 4. Components — what each one does

### 4.1 Streamlit UI (`app.py`)

Everything the user sees. Drives the flow:

1. **Setup screen** — pick role (dropdown), difficulty (radio), upload CV (file_uploader)
2. **Loop screen** — for each turn:
   - Display the question (Q1…Q5)
   - Receive the answer via `st.chat_input`
   - Show feedback (the ScoreReport)
   - If the Director says "stay" — continue the conversation
   - If the Director says "move_on" — Planner picks the next slot
3. **Final report screen** — display the SessionSummary including the coaching letter

All session state lives in `st.session_state` (in-memory, cleared on refresh).

### 4.2 Pydantic schemas (`core/models.py`)

All data structures. 16 models in 5 groups:

- **Enums**: `Topic`, `Difficulty`, `Role`, `SlotType`, `DirectorAction`
- **KB**: `Question`, `Rubric`
- **Agent outputs**: `InterviewerChoice`, `ScoreReport`, `DimensionScore`, `DirectorChoice`, `CVProfile`
- **Session state**: `SessionState`, `TurnRecord`, `RollingScore`
- **Final**: `SessionSummary`

Pydantic forces every agent to return valid JSON that maps onto type-safe
Python objects.

### 4.3 Settings (`core/config.py`)

`pydantic-settings`-based config. The active LLM provider is **auto-
detected** from whichever API key is set in `.env`:

- `ANTHROPIC_API_KEY` → Anthropic
- `OPENAI_API_KEY` → OpenAI
- `MISTRAL_API_KEY` → Mistral

Priority order when several keys are set: Anthropic > OpenAI > Mistral.
Other settings (model overrides, generation defaults, data paths) also live
here. Exposed via `from core.config import settings`.

### 4.4 LLM wrapper (`core/llm.py`)

Provider-agnostic wrapper. Every agent calls one function:

```python
result = call_llm(system=SYSTEM_PROMPT, user=user_msg, schema=ScoreReport)
```

What it does:
- Routes the call to the active provider's SDK (Anthropic Messages API, or
  the OpenAI-compatible chat API for OpenAI + Mistral).
- Strips ```` ```json ```` fences if the model adds them, parses JSON, and
  validates against the Pydantic schema.
- Repairs once with a clearer prompt if the first response is malformed.
- Retries on rate-limit errors with exponential backoff (honouring
  `Retry-After`) and surfaces a toast in the UI on each retry.
- Traces every call to LangSmith when `LANGSMITH_API_KEY` is configured;
  also appends one JSON line per call to `logs/llm_calls.jsonl` regardless.

It also exposes `call_llm_with_tools()` — a **provider-agnostic tool-use
loop**. Tools are declared once in a neutral `{name, description, parameters}`
format and translated to each provider's tool spec. The Coaching Summariser
uses this to call `lookup_reference_answer` against the KB.

### 4.5 Knowledge Base + Chroma (`core/kb.py`)

At startup:
- Loads the 3 JSONL files (questions_da_qa, questions_de_fe, questions_behavioural)
- Indexes them in a Chroma vector store
- Document = the question text, Metadata = topic, difficulty, skill_tags

Main methods:
- `retrieve(topic, difficulty, excluded_ids, cv_skills)` → candidates for a technical slot
- `retrieve_behavioural(difficulty, excluded_ids)` → candidates for a behavioural slot
- `topics_for_role(role)` → which topics are valid for the selected role

**CV-aware reranking:** if `cv_skills` is passed, questions whose `skill_tags`
overlap with the CV signals (e.g. the user mentions PostgreSQL → questions
tagged `postgresql` get priority).

**Fallback logic:**
- If there are no candidates at the requested difficulty → drops to an easier one
- If still nothing → any difficulty within the topic

### 4.6 Planner (`core/planner.py`)

Deterministic Python (NOT an LLM agent — deliberately, for predictability
and explainability).

Main method:

```python
plan = planner.plan_next_turn(session_state)
# Returns TurnPlan(slot_type, topic, difficulty)
```

Rules:
- Q1, Q4: COVER → pick an uncovered topic for the role
- Q2, Q5: REINFORCE → reuse the previous COVER's topic, adjust difficulty
- Q3: BEHAVIORAL → always the behavioural topic

### 4.7 Agents (`core/agents.py`)

Four LLM agents, each with one job:

#### InterviewerAgent
- **Input:** list of candidate questions from the Retriever + CV profile
- **Output:** `InterviewerChoice(id, phrased)`
- **Job:** pick one candidate and paraphrase it naturally, anchoring the phrasing
  in the CV whenever there is any reasonable overlap (CV anchoring is the
  default behaviour, not the exception)
- **UI surfacing:** when the chosen question's `skill_tags` overlap the CV
  skills, the UI shows a one-line caption ("📄 Picked from your CV: ...") so
  the user can see that the CV actually affected what was asked
- **Defensive:** if the LLM invents an id not in the candidates → falls back to the first

#### EvaluatorAgent
- **Input:** Question (with rubric, reference_answer) + user_answer
- **Output:** `ScoreReport` (content / clarity / structure 1–5 + feedback bullets + overall)
- **Job:** structured grading against the rubric
- **Two modes** (selected at construction — `EvaluatorAgent(version="v1"|"v2")`):
  - **v1** — one prompt grades all three dimensions at once (original).
  - **v2** — one focused prompt per criterion, scores combined in code.
    Reduces "halo effect" but costs ~3× the API calls. The `evals/`
    pipeline measures which is better against a human-labelled golden set
    using Cohen's kappa.

#### ConversationDirectorAgent ★ AGENT #1
- **Input:** Question + user_answer + ScoreReport + history
- **Output:** `DirectorChoice(action, text)`
- **Actions:** clarify, followup, dig_deeper, move_on
- **Why this is "the real agent":** it picks an action from a fixed set based on observation, in a closed loop. That matches the formal definition of an agent.

#### CoachingSummariserAgent ★ AGENT #2
- **Input:** the full SessionState + CVProfile + the KB (for tool calls)
- **Output:** `SessionSummary` including a coaching_letter
- **Why this is agentic:** **tool use.** The Coach is given a
  `lookup_reference_answer(question_id)` tool and decides — autonomously —
  for which questions to call it (typically the ones the candidate scored
  low on) so that study suggestions are grounded in the actual gold
  reference, not the model's parametric memory. Built on a provider-
  agnostic tool-use loop (`call_llm_with_tools`), so it works with whichever
  provider `.env` selects.

### 4.8 CV Parser (`core/cv_parser.py`)

Two functions:
- `extract_cv_text(file)` — pulls text from a PDF (via `pypdf`) or a .txt file
- `parse_cv(text, role)` — sends the text to the LLM, returns a structured `CVProfile`

CVProfile contains: skills, projects, seniority estimate, claimed_strengths, likely_gaps.

The CV is used in two places:
1. **Retriever** — boosts questions whose skill_tags overlap with the CV skills
2. **Coaching Summariser** — references CV projects inside the final report

---

## 5. Two agentic patterns

The project demonstrates **two distinct agentic patterns**, which matters for
the course rubric:

### Pattern 1: Action Selection (Director)

```
Observation → Decision (from fixed action set) → Action → Loop
```

The Director observes the ScoreReport, picks one of 4 actions, the action
changes system state, and the loop repeats. Classic RL-style agent.

### Pattern 2: Tool Use (Coaching Summariser)

```
Rich context → Decide which tools to call → Execute → Synthesise → Output
```

The Coach takes the full session state and the CV, then decides — on its
own — which questions need a `lookup_reference_answer` call against the
KB before writing the report. Study suggestions are grounded in the actual
gold answer, not the model's parametric memory. The tool-use loop is
provider-agnostic, so this works on Anthropic, OpenAI, or Mistral.

**Plus 2 supporting LLM components:**
- Interviewer (LLM-as-selector)
- Evaluator (LLM-as-judge)
- CV Parser (LLM-as-extractor)

Total **~20 LLM calls per session** (1 CV parse + 5 × Evaluator + 5 × Director + 5 × Interviewer + 1 Coach + a small number of Coach tool-call rounds). The Evaluator's v2 mode adds 2 extra calls per turn.

---

## 6. Data flow — complete walkthrough

The user is a Data Analyst, mid level, with a CV mentioning PostgreSQL,
Tableau, Python, A/B testing.

### Before the session

1. UI: the user picks "Data Analyst" + "mid"
2. UI: uploads CV.pdf (`st.file_uploader`)
3. `extract_cv_text(file)` extracts text from the PDF
4. `parse_cv(text, role=DA)` → LLM call → `CVProfile(skills=["PostgreSQL", "Tableau", ...], projects=[...])`
5. UI: creates `SessionState(role=DA, difficulty=mid, target_turns=5, cv_profile=...)`
6. Click "Start"

### Q1 — COVER, SQL

1. `planner.plan_next_turn(state)` → `TurnPlan(COVER, SQL, mid)`
2. `kb.retrieve(SQL, mid, excluded={}, cv_skills=["PostgreSQL","Tableau","Python"])` → 5 candidates, postgresql-tagged ones first
3. `interviewer.ask(candidates, SQL, mid, cv_profile)` → `InterviewerChoice(id="da-001", phrased="Since you've worked with PostgreSQL at...")`
4. UI shows the question
5. The user types an answer
6. `evaluator.evaluate(question, answer)` → `ScoreReport(content=3, clarity=4, structure=3, overall=3)`
7. UI displays feedback
8. `director.decide_next_action(question, answer, score)` → `DirectorChoice(action=MOVE_ON, text="")`
9. Update SessionState: append TurnRecord, update topic_scores[SQL]
10. Director said MOVE_ON → go to the next slot

### Q2 — REINFORCE, SQL (stays at mid because Q1 score = 3, in the stay zone)

`planner.plan_next_turn(state)` → `TurnPlan(REINFORCE, SQL, mid)` ← same topic, same difficulty.

The rest is the same as Q1.

### Q3 — BEHAVIORAL

1. Planner → `TurnPlan(BEHAVIORAL, BEHAVIOURAL, mid)`
2. `kb.retrieve_behavioural(mid, excluded={})` → STAR questions
3. Interviewer picks one (e.g. conflict resolution)
4. The user answers → Evaluator → Director → MOVE_ON

### Q4 — COVER, Data Visualization

1. Planner → `TurnPlan(COVER, DATA_VISUALIZATION, mid)` (a different topic from Q1 because Q1 already covered SQL)
2. The CV has Tableau → Retriever prioritises questions tagged `tableau`
3. Same as before

### Q5 — REINFORCE, Data Visualization

1. Planner looks at the Q4 score, adjusts difficulty
2. Slot type is REINFORCE, same topic as Q4

### Final

1. After Q5 the session ends (turn_count = target_turns)
2. `coach.summarise(session_state)` → `SessionSummary(...)`
   - Uses every TurnRecord
   - Uses CVProfile for personalisation
   - Returns a coaching_letter that references the PostgreSQL/Tableau experience
3. UI shows the final report to the user

---

## 7. Knowledge base structure

**129 questions** in 3 active JSONL files under `data/`:

| File                              | # questions | Content                                              |
|-----------------------------------|-------------|------------------------------------------------------|
| `questions_da_qa.jsonl`           | 60          | 30 Data Analyst + 30 QA Engineer, 10 per (role × difficulty) |
| `questions_de_fe.jsonl`           | 60          | 30 Data Engineer + 30 Frontend Developer, 10 per (role × difficulty) |
| `questions_behavioural.jsonl`     |  9          | STAR questions, 3 per difficulty, role-agnostic       |

Every question has:

```json
{
  "id": "da-001",
  "topic": "sql",
  "subtopic": "joins",
  "difficulty": "entry",
  "question": "What is the difference between an INNER JOIN and a LEFT JOIN?",
  "reference_answer": "INNER JOIN returns only rows where...",
  "rubric": {
    "content": ["Defines INNER as matching", "Defines LEFT as ..."],
    "clarity": ["Uses correct SQL terminology"],
    "structure": ["Definition first, then contrast"]
  },
  "tags": ["sql", "joins", "data_analyst"],
  "skill_tags": ["postgresql", "mysql", "joins", "left_join", "data_analyst"]
}
```

**`skill_tags`** is the key field for CV-aware reranking — concrete tools /
concepts that can appear in a CV.

---

## 8. Running locally

```bash
# 1. Create a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate    # Mac / Linux
# .venv\Scripts\activate     # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up .env with ONE LLM API key (any of the three)
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY, OPENAI_API_KEY, or MISTRAL_API_KEY.
# optional: LANGSMITH_API_KEY + LANGSMITH_TRACING=true to trace to LangSmith.

# 4. Run Streamlit
streamlit run app.py
```

The bot opens the browser at `http://localhost:8501`.

---

## 9. Why this architecture

**Why no LangChain?** Fewer dependencies, no magic, easier to explain.
Multi-provider routing is ~30 lines in `core/llm.py` (`_DISPATCH`); the
tool-use loop is another ~60 lines (`_TOOL_LOOPS`). No framework needed.

**Why no chunking?** Each Question is an atomic unit, short enough to fit
into one embedding.

**Why no Telegram bot?** Streamlit is simpler, no ConversationHandler state
machine, easier demo (`streamlit run app.py`).

**Why no SQLite persistence?** Streamlit `session_state` is enough for the
demo. Not a production app.

**Why a deterministic planner instead of an LLM Planner?** Predictable,
easier to explain, cheaper. The adaptivity comes from the Reinforce slots
plus the Director agent.

**Why 4 separate agents instead of one "mega-agent"?** Each has a clear
responsibility, easier to test, easier to explain to the committee. Single
responsibility principle for agents.

---

## 10. Testing, evaluation, and observability

- **`tests/`** — pytest suite (108 fast tests + 7 integration). Uses a
  FakeLLM so no real API calls are made; covers models, planner, the LLM
  wrapper (rate-limit retry, repair, tool use), the four agents, the CV
  parser, the evals pipeline's pure logic, and KB retrieval. Run with
  `pytest`.
- **`evals/`** — task baskets + graders + a leaderboard. Five baskets:
  - **Planner** — deterministic slot/topic/difficulty scheduling (free).
  - **Retriever** — topic filter + CV-aware rerank (free).
  - **Director** — action-selection LLM-as-judge against an accepted-action
    set (paid).
  - **Interviewer** — selection (CV-overlap → chose a CV-tagged question),
    CV anchoring (phrasing mentions the CV term), and reformulation fidelity
    (an LLM judge confirms the paraphrase asks the same thing and does not
    leak the answer) (paid).
  - **Bias** — perturbation pairs for halo effect, length bias, lexical
    mirroring of rubric phrasing, and position bias in the Interviewer (paid).
  - Plus an Evaluator v1-vs-v2 calibration against a 24-example human-
    labelled golden set (Cohen's kappa).

  Run with `python -m evals.run_evals` (free), add `--director`,
  `--interviewer`, `--bias`, or `--all` for the paid angles, or use
  `python -m evals.calibrate_evaluator` for the v1/v2 comparison.
- **Observability** — every LLM call is traced to LangSmith when
  `LANGSMITH_API_KEY` is configured, and also appended to
  `logs/llm_calls.jsonl` regardless. See [`TESTING.md`](TESTING.md) for
  the full guide.
