# Interview Prep Coach

A multi-turn coaching chatbot for technical and behavioural interview prep.
Powered by RAG over a curated question bank and four LLM agents (Anthropic Claude).

## Što radi

Korisnik bira **rolu** (Data Analyst, QA Engineer, Data Engineer, Frontend
Developer) i **nivo težine** (entry, mid, senior), uploaduje CV, i prolazi
kroz sesiju od **5 pitanja** sa strukturisanim feedback-om posle svakog
odgovora. Na kraju dobija personalizovan coaching report koji koristi CV za
konkretne preporuke.

## Struktura sesije

```
Q1 — COVER       (fresh topic for the role)
Q2 — REINFORCE   (same topic as Q1, difficulty adapts to score)
Q3 — BEHAVIORAL  (STAR question)
Q4 — COVER       (different fresh topic)
Q5 — REINFORCE   (same topic as Q4, difficulty adapts to score)
```

Posle svakog odgovora **Conversation Director agent** odlučuje da li ostaje
na istom pitanju (clarify / followup / dig_deeper) ili prelazi na sledeći
slot (move_on).

## Brz start

### 1. Zavisnosti

```bash
cd interview-prep-coach/simple

# Make virtualenv (recommended)
python -m venv venv
source venv/bin/activate    # Mac / Linux
# venv\Scripts\activate     # Windows

# Install
pip install -r requirements.txt
```

### 2. API ključ

Treba ti **Anthropic API ključ**. Registruj se na
[console.anthropic.com](https://console.anthropic.com/settings/keys), klikni
"Create Key", i kopiraj ključ (počinje sa `sk-ant-...`).

```bash
# Copy the template
cp .env.example .env

# Edit .env and paste your key
# ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Pokreni

```bash
streamlit run app.py
```

Otvara se browser na `http://localhost:8501`.

## Šta da pripremiš

- **CV** kao PDF ili .txt fajl. Najjednostavniji način: **eksportuj svoj
  LinkedIn profil kao PDF** (Profile → More → Save to PDF).
  - Ako nemaš LinkedIn profil, napiši mali .txt sa skills, experience, i
    projects (oko 200-300 reči je dovoljno).

## Arhitektura ukratko

```
app.py                       Streamlit UI (3 ekrana: setup, interview, final)
├── core/models.py           Pydantic schemas (16 modela)
├── core/llm.py              Thin Anthropic wrapper sa JSON validacijom
├── core/kb.py               KB loader + Chroma retriever (129 pitanja)
├── core/cv_parser.py        PDF/text → CVProfile
├── core/planner.py          Deterministic 5-slot scheduler
└── core/agents.py           4 LLM agenta:
                               - Interviewer (picks + paraphrases Q)
                               - Evaluator (grades against rubric)
                               - ConversationDirector (chooses next action) ★
                               - CoachingSummariser (final report) ★
```

★ = formalni agentski deo (action selection u petlji + generative personalization).

Vidi `ARCHITECTURE.md` za detaljan opis svake komponente i data flow-a.

## Knowledge base

**129 kuriranih pitanja** u 3 JSONL fajla pod `../data/`:

| Fajl | # pitanja | Sadržaj |
|------|-----------|---------|
| `questions_da_qa.jsonl` | 60 | Data Analyst + QA Engineer |
| `questions_de_fe.jsonl` | 60 | Data Engineer + Frontend Developer |
| `questions_behavioural.jsonl` | 9 | STAR questions, role-agnostic |

Svako pitanje ima topic, subtopic, difficulty, reference_answer, rubric (3
dimenzije), tags, i skill_tags za CV-aware retrieval.

## Troubleshooting

**`ANTHROPIC_API_KEY not set`** — proveri da li je `.env` u `simple/`
folderu i da li si upisala ključ.

**Chroma greška pri startu** — pri prvom run-u, Chroma indeksira sva 129
pitanja, što može da potraje 10-20 sekundi. Drugi put će biti instant
(cache je na `simple/.chroma/`).

**`pypdf` greška na CV uploadu** — CV mora da bude PDF sa text layer-om
(eksportovan iz Word-a, LinkedIn-a, Google Docs-a). Skenirani PDF ne radi.

**Sesija nestane na refresh** — to je očekivano; ova jednostavna verzija
drži state u memoriji bez SQLite-a. Refresh = nova sesija.

## Tehnologije

- **Streamlit** za UI (single-file Python web app)
- **Anthropic Claude** (`claude-sonnet-4-5`) za sve LLM pozive
- **ChromaDB** za vector store i semantic retrieval
- **Pydantic** za type-safe data contracts
- **pypdf** za PDF text extraction
- **Bez LangChain-a** — direktan Anthropic SDK za jednostavnost i
  transparentnost

## Razvoj

Pokreni testove (kad postanu):

```bash
pytest tests/
```

## Licenca

University course project — Interview Prep Coach, 2025.
