# Interview Prep Coach — Architecture (Simplified Streamlit Version)

> Reference dokument za projekat. Detaljno objašnjava svaku komponentu,
> data flow, i agentske patterne. Pisano da služi i kao priprema za
> prezentaciju projekta i kao readme za tim.

---

## 1. O čemu je projekat

**Interview Prep Coach** je multi-turn coaching chatbot koji pomaže kandidatima
da se pripreme za tehničke i behavioralne intervjue za 4 role:

- Data Analyst
- QA Engineer
- Data Engineer
- Frontend Developer

Korisnik bira rolu i nivo težine, uploaduje CV, i prolazi kroz sesiju od 5
pitanja. Posle svakog odgovora dobija strukturisan feedback. Na kraju dobija
personalizovan coaching report koji koristi CV za konkretne preporuke.

---

## 2. Sažeta arhitektura

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          STREAMLIT UI (app.py)                          │
│  Setup → Q1 → Q2 → Q3 → Q4 → Q5 → Final Report                          │
└───────┬─────────────────────────────────────────────────────────────────┘
        │
        │ session_state (Pydantic SessionState in memory)
        │
        ↓
┌──────────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐
│  Planner             │  │  KB / RAG       │  │  4 LLM Agents           │
│  (Python, no LLM)    │  │  (Chroma)       │  │                         │
│                      │  │                 │  │  - Interviewer          │
│  5-slot pattern:     │  │  129 questions  │  │  - Evaluator            │
│  cover · reinforce · │  │  4 JSONL files  │  │  - Director ★           │
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

**★** = formalni agentski deo sistema (action selection u petlji + generative
personalization). Sve ostalo su deterministički ili LLM komponente.

---

## 3. Sesija — 5-pitanja pattern

Sesija ima **fiksan raspored slotova**:

| Slot | Tip | Logika |
|------|-----|--------|
| Q1   | COVER       | Random uncovered topic for the role, base difficulty |
| Q2   | REINFORCE   | Same topic as Q1, difficulty adjusted by Q1's score  |
| Q3   | BEHAVIORAL  | STAR question, base difficulty                       |
| Q4   | COVER       | Different uncovered topic, base difficulty           |
| Q5   | REINFORCE   | Same topic as Q4, difficulty adjusted by Q4's score  |

### Adaptive difficulty pravila (za reinforce slotove)

```
prev_score < 0.4 (1-2/5)      → STEP DOWN  (e.g. mid → entry)
prev_score > 0.85 (5/5)       → STEP UP    (e.g. mid → senior)
otherwise (3-4/5)             → STAY at base difficulty
```

Difficulty se klipuje na ENTRY (najlakše) i SENIOR (najteže).

### Director Agent može da unutra petlja

Posle Evaluator-a, **Director Agent** odlučuje šta dalje. Ako bira
`clarify`, `followup`, ili `dig_deeper` — sistem ostaje na **istom pitanju**
i traži dodatni odgovor. Tek kad Director kaže `move_on`, Planner bira sledeći slot.

To znači da jedan slot može imati više "round-ova" konverzacije pre nego što
se zatvori.

---

## 4. Komponente — šta svaka radi

### 4.1 Streamlit UI (`app.py`)

Sve što korisnik vidi. Drži flow:

1. **Setup screen** — bira rolu (dropdown), težinu (radio), uploaduje CV (file_uploader)
2. **Loop screen** — za svaki turn:
   - Prikazuje pitanje (Q1...Q5)
   - Prima odgovor preko `st.chat_input`
   - Pokazuje feedback (ScoreReport)
   - Ako Director kaže "stay" — nastavi konverzaciju
   - Ako Director kaže "move_on" — Planner bira sledeći slot
3. **Final report screen** — prikazuje SessionSummary sa coaching letter-om

Sve stanje sesije živi u `st.session_state` (in-memory, briše se na refresh).

### 4.2 Pydantic šeme (`core/models.py`)

Sve strukture podataka. 16 modela u 5 grupa:

- **Enumi**: `Topic`, `Difficulty`, `Role`, `SlotType`, `DirectorAction`
- **KB**: `Question`, `Rubric`
- **Agent outputs**: `InterviewerChoice`, `ScoreReport`, `DimensionScore`, `DirectorChoice`, `CVProfile`
- **Session state**: `SessionState`, `TurnRecord`, `RollingScore`
- **Final**: `SessionSummary`

Pydantic obavezuje da svi agenti vrate validan JSON koji se mapira u tip-safe Python objekte.

### 4.3 LLM wrapper (`core/llm.py`)

Tanak omotač oko Anthropic API-ja. Jedna funkcija:

```python
result = call_llm(system=SYSTEM_PROMPT, user=user_msg, schema=ScoreReport)
```

Šta radi:
- Poziva `claude-sonnet-4-5-20250929`
- Strip-uje ```json fence-ove ako ih model doda
- Parse-uje JSON
- Validira kroz Pydantic
- Retry jednom sa repair prompt-om ako prvi response nije validan

Bez ovog wrapper-a, svaki agent bi morao ručno da parsuje JSON i radi retry — sa wrapper-om agenti su jednolinijaši.

### 4.4 Knowledge Base + Chroma (`core/kb.py`)

Pri startu:
- Učita 3 JSONL fajla (questions_da_qa, questions_de_fe, questions_behavioural)
- Indeksira ih u Chroma vector store
- Document = tekst pitanja, Metadata = topic, difficulty, skill_tags

Glavne metode:
- `retrieve(topic, difficulty, excluded_ids, cv_skills)` → kandidati za tehnički slot
- `retrieve_behavioural(difficulty, excluded_ids)` → kandidati za behavioral slot
- `topics_for_role(role)` → koje teme su validne za izabranu rolu

**CV-aware reranking:** ako `cv_skills` prosleđeno, pitanja čiji se `skill_tags` preklapaju sa CV-jem (npr. korisnik pominje PostgreSQL → pitanja sa `postgresql` tag-om) dobijaju prioritet u rezultatu.

**Fallback logika:**
- Ako nema kandidata na traženoj težini → spušta na lakšu
- Ako ni tu nema → bilo koja težina unutar topic-a

### 4.5 Planner (`core/planner.py`)

Deterministički Python (NIJE LLM agent — to je svesna odluka radi predvidivosti i explainability-ja).

Glavna metoda:

```python
plan = planner.plan_next_turn(session_state)
# Returns TurnPlan(slot_type, topic, difficulty)
```

Pravila:
- Q1, Q4: COVER → bira uncovered topic za rolu
- Q2, Q5: REINFORCE → koristi topic prethodnog COVER turn-a, adjustuje difficulty
- Q3: BEHAVIORAL → uvek behavioural topic

### 4.6 Agenti (`core/agents.py`)

Četiri LLM agenta, svaki sa svojom svrhom:

#### InterviewerAgent
- **Input:** lista kandidata iz Retriever-a + CV profile
- **Output:** `InterviewerChoice(id, phrased)`
- **Zadatak:** izabere jedan kandidat i parafrazira ga prirodno, eventualno referencirajući CV
- **Fast path:** ako ima samo 1 kandidat, ne zove LLM
- **Defensive:** ako LLM izmisli ID koji nije u kandidatima → fallback na prvi

#### EvaluatorAgent
- **Input:** Question (sa rubric, reference_answer) + user_answer
- **Output:** `ScoreReport` (content/clarity/structure 1-5 + feedback bullets + overall)
- **Zadatak:** strukturirana ocena po rubricu

#### ConversationDirectorAgent ★ AGENT #1
- **Input:** Question + user_answer + ScoreReport + history
- **Output:** `DirectorChoice(action, text)`
- **Akcije:** clarify, followup, dig_deeper, move_on
- **Zašto je ovo "pravi agent":** bira akciju iz fiksnog skupa na osnovu posmatranja, u zatvorenoj petlji. To je formalna definicija agenta.

#### CoachingSummariserAgent ★ AGENT #2
- **Input:** ceo SessionState + CVProfile
- **Output:** `SessionSummary` sa coaching_letter
- **Zašto je ovo agentski:** generativna personalizacija — sintetiše long-form content iz bogatog konteksta, koristi CV za konkretne reference

### 4.7 CV Parser (`core/cv_parser.py`)

Dve funkcije:
- `extract_cv_text(file)` — pulls text iz PDF-a (`pypdf`) ili .txt fajla
- `parse_cv(text, role)` — šalje tekst LLM-u, vraća strukturirani `CVProfile`

CVProfile sadrži: skills, projects, seniority estimate, claimed_strengths, likely_gaps.

CV se koristi na dva mesta:
1. **Retriever** — boost-uje pitanja čiji se skill_tags preklapaju sa CV skills
2. **Coaching Summariser** — referencira projekte iz CV-ja u final report-u

---

## 5. Dva agentska patterna

Naš projekat demonstrira **dva različita agentska patterna**, što je važno za rubriku kursa:

### Pattern 1: Action Selection (Director)

```
Observation → Decision (from fixed action set) → Action → Loop
```

Director posmatra ScoreReport, bira jednu od 4 akcije, akcija menja stanje sistema, pa se loop ponavlja. To je klasični RL-style agent.

### Pattern 2: Generative Personalization (Coaching Summariser)

```
Rich context → Synthesis → Long-form personalized output
```

Coach prima ceo session state plus CV, sintetiše coherent personalized text. To je "agent" u smislu autonomne kreativne sinteze.

**Plus 2 supporting LLM komponente:**
- Interviewer (LLM-as-selector)
- Evaluator (LLM-as-judge)
- CV Parser (LLM-as-extractor)

Ukupno **5 LLM poziva po sesiji minimum** (1 CV parse + 5×Evaluator + 5×Director + 5×Interviewer ako ima više kandidata + 1 Coach = ~20 poziva po sesiji).

---

## 6. Data flow — kompletan walkthrough

Korisnik je Data Analyst, mid level, ima CV koji pominje PostgreSQL, Tableau, Python, A/B testing.

### Pre sesije

1. UI: korisnik bira "Data Analyst" + "mid"
2. UI: uploaduje CV.pdf (`st.file_uploader`)
3. `extract_cv_text(file)` izvuče text iz PDF-a
4. `parse_cv(text, role=DA)` → LLM call → `CVProfile(skills=["PostgreSQL", "Tableau", ...], projects=[...])`
5. UI: kreira `SessionState(role=DA, difficulty=mid, target_turns=5, cv_profile=...)`
6. Klik "Start"

### Q1 — COVER, SQL

1. `planner.plan_next_turn(state)` → `TurnPlan(COVER, SQL, mid)`
2. `kb.retrieve(SQL, mid, excluded={}, cv_skills=["PostgreSQL","Tableau","Python"])` → 5 kandidata, postgresql-tagged prvi
3. `interviewer.ask(candidates, SQL, mid, cv_profile)` → `InterviewerChoice(id="da-001", phrased="Since you've worked with PostgreSQL at...")`
4. UI prikazuje pitanje
5. Korisnik kuca odgovor
6. `evaluator.evaluate(question, answer)` → `ScoreReport(content=3, clarity=4, structure=3, overall=3)`
7. UI prikazuje feedback
8. `director.decide_next_action(question, answer, score)` → `DirectorChoice(action=MOVE_ON, text="")`
9. Update SessionState: append TurnRecord, update topic_scores[SQL]
10. Director rekao MOVE_ON → idi na sledeći slot

### Q2 — REINFORCE, SQL (entry, stepped down zbog Q1 score=3)

Wait — Q1 score je 3, što je medium. Pravilo: < 0.4 step down, > 0.85 step up, inače stay. 3/5 normalized je 0.5, što je u "stay" zoni → ostaje mid.

Idi na Q2 sa SQL mid.

Q2: `planner.plan_next_turn(state)` → `TurnPlan(REINFORCE, SQL, mid)` ← same topic, same difficulty

Ostatak isto kao Q1.

### Q3 — BEHAVIORAL

1. Planner → `TurnPlan(BEHAVIORAL, BEHAVIOURAL, mid)`
2. `kb.retrieve_behavioural(mid, excluded={})` → STAR pitanja
3. Interviewer bira jedno (npr. conflict resolution)
4. Korisnik odgovara → Evaluator → Director → MOVE_ON

### Q4 — COVER, Data Visualization

1. Planner → `TurnPlan(COVER, DATA_VISUALIZATION, mid)` (drugačiji od Q1 jer Q1 je već pokrio SQL)
2. CV ima Tableau → Retriever prioritizuje pitanja sa `tableau` tag-om
3. Ostatak isto

### Q5 — REINFORCE, Data Visualization

1. Planner gleda Q4 score, adjustuje difficulty
2. Slot type je REINFORCE, isti topic kao Q4

### Final

1. Posle Q5, sesija je gotova (turn_count = target_turns)
2. `coach.summarise(session_state)` → `SessionSummary(...)`
   - Koristi sve TurnRecord-e
   - Koristi CVProfile za personalizaciju
   - Vrati coaching_letter koji referenciše PostgreSQL/Tableau iskustvo
3. UI prikazuje final report

---

## 7. Knowledge base struktura

**129 pitanja** u 3 aktivna JSONL fajla:

| Fajl | # pitanja | Sadržaj |
|------|-----------|---------|
| `questions_da_qa.jsonl` | 60 | 30 Data Analyst + 30 QA Engineer, 10 po (rola × težina) |
| `questions_de_fe.jsonl` | 60 | 30 Data Engineer + 30 Frontend, 10 po (rola × težina) |
| `questions_behavioural.jsonl` | 9 | STAR pitanja, 3 po težini, role-agnostic |

Svako pitanje ima:

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

**`skill_tags`** je ključno polje za CV-aware rerangiranje — to su konkretni alati / koncepti koji se mogu pojaviti u CV-ju.

---

## 8. Kako se pokreće lokalno

```bash
# 1. Kloniraj repo, uđi u simple/ folder
cd interview-prep-coach/simple

# 2. Napravi virtualenv (preporučeno)
python -m venv venv
source venv/bin/activate    # Mac/Linux
# venv\Scripts\activate     # Windows

# 3. Instaliraj zavisnosti
pip install -r requirements.txt

# 4. Setup .env sa Anthropic API ključem
cp .env.example .env
# uredi .env i upiši svoj ANTHROPIC_API_KEY

# 5. Pokreni Streamlit
streamlit run app.py
```

Bot otvara browser na `http://localhost:8501`.

---

## 9. Trenutni status implementacije

| Korak | Fajl | Linije | Status |
|-------|------|--------|--------|
| 0 | Skill tags na 149 pitanja | — | ✓ |
| 1 | `core/models.py` | 210 | ✓ |
| 2 | `core/llm.py` | 140 | ✓ |
| 3 | `core/kb.py` | 293 | ✓ |
| 4 | `core/agents.py` | 381 | ✓ |
| 5 | `core/cv_parser.py` | 175 | ✓ |
| 6 | `core/planner.py` | 174 | ✓ |
| 7 | `app.py` | ~300 | sledeće |
| 8 | `requirements.txt`, `.env.example`, README | — | sledeće |

Ukupno ~1370 linija koda do sada, plus ~300 za UI = **~1700 linija ukupno**.

---

## 10. Šta dolazi sledeće (Korak 7)

Streamlit UI koji povezuje sve komponente. Tri ekrana:

1. **Setup screen** — role/difficulty/CV upload + "Start" dugme
2. **Interview screen** — chat history + chat input, jedno pitanje po pitanje
3. **Final report screen** — overall score + per-topic breakdown + coaching letter

Sve stanje u `st.session_state["interview_state"]` (Pydantic SessionState).

---

## Appendix: zašto baš ova arhitektura

**Zašto bez LangChain-a?** Manje deps-a (~5MB vs ~50MB), bez magije, lakše objasniti komisiji. Naš use case ne traži multi-provider routing.

**Zašto bez chunking-a?** Svaka Question je atomska jedinica, dovoljno kratka da stane u jedan embedding.

**Zašto bez Telegram bot-a?** Streamlit je jednostavniji, ne traži ConversationHandler state machine, demo je lakši (`streamlit run app.py`).

**Zašto bez SQLite persistencije?** Streamlit `session_state` je dovoljan za demo. Nije production app.

**Zašto deterministički Planner umesto LLM Planner-a?** Predvidiv, lakši za objašnjenje, jeftiniji. Adaptivnost dolazi kroz Reinforce slotove + Director agent.

**Zašto 4 odvojena agenta umesto jednog "mega-agenta"?** Svaki ima jasnu odgovornost, lakši test, jasno objasniti pred komisijom. Single responsibility principle za agente.
