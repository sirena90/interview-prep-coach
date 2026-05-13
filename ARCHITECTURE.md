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
│  Setup → Q1 → Q2 → Q3 → Q4 → Q5 → Final Report                          │
└───────┬─────────────────────────────────────────────────────────────────┘
        │
        │ session_state (Pydantic SessionState held in memory)
        │
        ↓
┌──────────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│  Planner             │  │  KB / RAG       │  │  4 LLM Agents           │
│  (Python, no LLM)    │  │  (Chroma)       │  │                         │
│                      │  │                 │  │  - Interviewer          │
│  5-slot pattern:     │  │  129 questions  │  │  - Evaluator            │
│  cover · reinforce · │  │  3 JSONL files  │  │  - Director ★           │
│  behavioral · cover ·│  │  Vector search  │  │  - Coaching Summariser ★│
│  reinforce           │  │  + metadata     │  │                         │
└──────────────────────┘  └─────────────────┘  └─────────────────────────┘
                                │
                                ↓
                        ┌──────────────┐
                        │  CV Parser   │
                        │  (LLM)       │
                        │              │
                        │  CV → profile│
                        └──────────────┘
                                │
                                ↓
                        ┌──────────────┐
                        │  LLM wrapper │
                        │  (Anthropic) │
                        └──────────────┘
```

**★** = formally agentic component (action selection in a loop + generative
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

### 4.3 LLM wrapper (`core/llm.py`)

A thin wrapper around the Anthropic API. One function:

```python
result = call_llm(system=SYSTEM_PROMPT, user=user_msg, schema=ScoreReport)
```

What it does:
- Calls `claude-sonnet-4-5-20250929`
- Strips ```json fences if the model adds them
- Parses JSON
- Validates against the Pydantic schema
- Retries once with a repair prompt if the first response doesn't parse

Without this wrapper every agent would have to parse JSON and run retries
manually — with the wrapper, each agent becomes a one-liner.

### 4.4 Knowledge Base + Chroma (`core/kb.py`)

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

### 4.5 Planner (`core/planner.py`)

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

### 4.6 Agents (`core/agents.py`)

Four LLM agents, each with one job:

#### InterviewerAgent
- **Input:** list of candidate questions from the Retriever + CV profile
- **Output:** `InterviewerChoice(id, phrased)`
- **Job:** pick one candidate and paraphrase it naturally, optionally referencing the CV
- **Fast path:** if there is only one candidate, no LLM call
- **Defensive:** if the LLM invents an id not in the candidates → falls back to the first

#### EvaluatorAgent
- **Input:** Question (with rubric, reference_answer) + user_answer
- **Output:** `ScoreReport` (content / clarity / structure 1–5 + feedback bullets + overall)
- **Job:** structured grading against the rubric

#### ConversationDirectorAgent ★ AGENT #1
- **Input:** Question + user_answer + ScoreReport + history
- **Output:** `DirectorChoice(action, text)`
- **Actions:** clarify, followup, dig_deeper, move_on
- **Why this is "the real agent":** it picks an action from a fixed set based on observation, in a closed loop. That matches the formal definition of an agent.

#### CoachingSummariserAgent ★ AGENT #2
- **Input:** the full SessionState + CVProfile
- **Output:** `SessionSummary` including a coaching_letter
- **Why this is also agentic:** generative personalisation — it synthesises long-form content from rich context, using the CV for concrete references.

### 4.7 CV Parser (`core/cv_parser.py`)

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

### Pattern 2: Generative Personalisation (Coaching Summariser)

```
Rich context → Synthesis → Long-form personalised output
```

The Coach takes the full session state plus the CV and synthesises coherent
personalised text. This is an "agent" in the sense of autonomous creative
synthesis.

**Plus 2 supporting LLM components:**
- Interviewer (LLM-as-selector)
- Evaluator (LLM-as-judge)
- CV Parser (LLM-as-extractor)

Total **5 LLM calls per session minimum** (1 CV parse + 5 × Evaluator + 5 × Director + 5 × Interviewer when there are multiple candidates + 1 Coach ≈ ~20 calls per session).

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
# 1. Clone the repo, cd into simple/
cd interview-prep-coach/simple

# 2. Create a virtualenv (recommended)
python -m venv venv
source venv/bin/activate    # Mac / Linux
# venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up .env with the Anthropic API key
cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY

# 5. Run Streamlit
streamlit run app.py
```

The bot opens the browser at `http://localhost:8501`.

---

## 9. Why this architecture

**Why no LangChain?** Fewer dependencies (~5MB vs ~50MB), no magic, easier to
explain to the committee. Our use case does not need multi-provider routing.

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
