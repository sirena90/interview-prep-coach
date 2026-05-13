"""Knowledge base loader and Chroma retriever.

At startup, loads questions from the 3 active JSONL files into Chroma:
  - data/questions_da_qa.jsonl        (60 questions: Data Analyst + QA)
  - data/questions_de_fe.jsonl        (60 questions: Data Engineer + Frontend)
  - data/questions_behavioural.jsonl  (9 STAR questions)

Provides two retrieval functions:
  - retrieve()              for technical slots (cover / reinforce)
  - retrieve_behavioural()  for behavioral slots

Both filter by metadata (topic + difficulty) and exclude already-asked IDs.
retrieve() also re-ranks results by CV skill overlap so questions whose
skill_tags match the user's claimed skills surface first.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from core.models import Difficulty, Question, Role, Topic


# ---- Constants -------------------------------------------------------------

# Project layout: simple/core/kb.py is two levels deep from project root.
_PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = _PROJECT_ROOT / "data"
CHROMA_PATH = _PROJECT_ROOT / ".chroma"
COLLECTION_NAME = "interview_questions"

KB_FILES = [
    "questions_da_qa.jsonl",
    "questions_de_fe.jsonl",
    "questions_behavioural.jsonl",
]


# ---- Role → topics mapping -------------------------------------------------
# Which topics are eligible for which role. Planner uses this to decide which
# topics to cycle through during the COVER phase of a session.
ROLE_TOPICS: dict[Role, set[Topic]] = {
    Role.DATA_ANALYST: {
        Topic.SQL,
        Topic.DATA_VISUALIZATION,
        Topic.BUSINESS_METRICS,
        Topic.EXPERIMENTATION,
        Topic.STATISTICS,
    },
    Role.QA_ENGINEER: {
        Topic.TEST_DESIGN,
        Topic.TEST_AUTOMATION,
        Topic.API_TESTING,
        Topic.BUG_LIFECYCLE,
        Topic.TEST_STRATEGY,
    },
    Role.DATA_ENGINEER: {
        Topic.DATA_PIPELINES,
        Topic.DATA_WAREHOUSING,
        Topic.DATA_MODELING,
        Topic.DISTRIBUTED_SYSTEMS,
    },
    Role.FRONTEND_DEVELOPER: {
        Topic.FRONTEND_CORE,
        Topic.FRONTEND_FRAMEWORK,
        Topic.FRONTEND_PERFORMANCE,
        Topic.FRONTEND_ACCESSIBILITY,
        Topic.FRONTEND_TESTING,
        Topic.FRONTEND_SECURITY,
    },
}


# ---- KnowledgeBase ---------------------------------------------------------

class KnowledgeBase:
    """Loads questions from JSONL, indexes them in Chroma, serves retrieval.

    Usage:
        kb = KnowledgeBase()                # loads + indexes once at startup
        results = kb.retrieve(
            topic=Topic.SQL,
            difficulty=Difficulty.MID,
            excluded_ids=session_state.asked_ids,
            cv_skills=["postgresql", "tableau"],
            k=5,
        )
    """

    def __init__(
        self,
        data_dir: Path = DATA_DIR,
        chroma_path: Path = CHROMA_PATH,
    ) -> None:
        # In-memory dict for fast lookup by id (after Chroma returns ids).
        self._questions: dict[str, Question] = {}
        self._load_questions(data_dir)

        chroma_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(COLLECTION_NAME)
        self._ingest_if_needed()

    def _load_questions(self, data_dir: Path) -> None:
        for fname in KB_FILES:
            path = data_dir / fname
            if not path.exists():
                continue  # skip missing files (e.g., the deprecated questions.jsonl)
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    q = Question.model_validate_json(line)
                    self._questions[q.id] = q

    def _ingest_if_needed(self) -> None:
        """Rebuild the Chroma index if it doesn't match the loaded KB count.

        Chroma's PersistentClient persists across runs; we only rebuild when
        the count differs (e.g., a JSONL file was edited between runs).
        """
        if self._collection.count() == len(self._questions):
            return  # already in sync

        # Wipe and rebuild to stay consistent with the source JSONL.
        if self._collection.count() > 0:
            self._client.delete_collection(COLLECTION_NAME)
            self._collection = self._client.get_or_create_collection(COLLECTION_NAME)

        ids, documents, metadatas = [], [], []
        for q in self._questions.values():
            ids.append(q.id)
            documents.append(q.question)  # the question text is what we embed
            metadatas.append({
                "topic": q.topic.value,
                "subtopic": q.subtopic,
                "difficulty": q.difficulty.value,
                # Chroma metadata can't be a list; join skill_tags as csv.
                "skill_tags": ",".join(q.skill_tags),
            })
        self._collection.add(ids=ids, documents=documents, metadatas=metadatas)

    # ---- Public retrieval API ----------------------------------------------

    def retrieve(
        self,
        *,
        topic: Topic,
        difficulty: Difficulty,
        query_text: Optional[str] = None,
        excluded_ids: Optional[set[str]] = None,
        cv_skills: Optional[list[str]] = None,
        k: int = 5,
    ) -> list[Question]:
        """Return up to k Question objects for a (topic, difficulty).

        - Filters by topic + difficulty metadata.
        - Excludes any ids in excluded_ids (already asked this session).
        - If cv_skills provided, reranks so questions whose skill_tags overlap
          with the user's CV skills appear first.
        - Falls back to one easier difficulty if no matches at target level.
        """
        excluded = excluded_ids or set()
        query = query_text or _topic_description(topic)

        # Step 1: Chroma metadata-filtered semantic search.
        candidate_ids = self._chroma_query(topic, difficulty, query, excluded, k)

        # Step 2: Fallback to easier difficulty if empty.
        if not candidate_ids and difficulty != Difficulty.ENTRY:
            fallback = (
                Difficulty.ENTRY if difficulty == Difficulty.MID else Difficulty.MID
            )
            candidate_ids = self._chroma_query(topic, fallback, query, excluded, k)

        # Step 3: Final fallback - any difficulty within the topic.
        if not candidate_ids:
            results = self._collection.query(
                query_texts=[query],
                n_results=max(k * 3, 10),
                where={"topic": topic.value},
            )
            candidate_ids = [
                qid for qid in (results.get("ids", [[]])[0] or [])
                if qid not in excluded
            ]

        # Step 4: CV-aware reranking.
        if cv_skills:
            cv_set = {s.lower().strip() for s in cv_skills}
            candidate_ids = sorted(
                candidate_ids,
                key=lambda qid: -self._skill_overlap(qid, cv_set),
            )

        return [self._questions[qid] for qid in candidate_ids[:k] if qid in self._questions]

    def retrieve_behavioural(
        self,
        *,
        difficulty: Difficulty,
        excluded_ids: Optional[set[str]] = None,
        k: int = 3,
    ) -> list[Question]:
        """Return up to k behavioural (STAR) questions at the requested difficulty."""
        excluded = excluded_ids or set()
        results = self._collection.query(
            query_texts=["behavioural STAR interview question"],
            n_results=max(k * 2, 5),
            where={"$and": [
                {"topic": Topic.BEHAVIOURAL.value},
                {"difficulty": difficulty.value},
            ]},
        )
        ids = [
            qid for qid in (results.get("ids", [[]])[0] or [])
            if qid not in excluded
        ]
        return [self._questions[qid] for qid in ids[:k]]

    def get(self, question_id: str) -> Question:
        """Look up a Question by id (after Chroma returns ids)."""
        return self._questions[question_id]

    def topics_for_role(self, role: Role) -> set[Topic]:
        """Eligible topics for a role (used by Planner for COVER selection)."""
        return ROLE_TOPICS.get(role, set())

    # ---- Internals ---------------------------------------------------------

    def _chroma_query(
        self,
        topic: Topic,
        difficulty: Difficulty,
        query: str,
        excluded: set[str],
        k: int,
    ) -> list[str]:
        where = {"$and": [
            {"topic": topic.value},
            {"difficulty": difficulty.value},
        ]}
        results = self._collection.query(
            query_texts=[query],
            n_results=max(k * 3, 10),  # overfetch for filtering + reranking
            where=where,
        )
        ids = results.get("ids", [[]])[0] or []
        return [qid for qid in ids if qid not in excluded]

    def _skill_overlap(self, question_id: str, cv_skills: set[str]) -> int:
        q = self._questions[question_id]
        tags = {t.lower().strip() for t in q.skill_tags}
        return len(tags & cv_skills)


# ---- Topic descriptions for semantic query --------------------------------

_TOPIC_DESCRIPTIONS: dict[Topic, str] = {
    Topic.SQL: "SQL: joins, aggregation, window functions, performance",
    Topic.DATA_VISUALIZATION: "Data visualization: charts, dashboards, design",
    Topic.BUSINESS_METRICS: "Business metrics: KPIs, funnels, retention, cohorts",
    Topic.EXPERIMENTATION: "Experimentation: A/B testing, hypothesis, power",
    Topic.STATISTICS: "Statistics: hypothesis testing, confidence intervals",
    Topic.TEST_DESIGN: "Test design: pyramid, equivalence partitioning, types",
    Topic.TEST_AUTOMATION: "Test automation: frameworks, CI/CD, page object model",
    Topic.API_TESTING: "API testing: REST endpoints, contract, edge cases",
    Topic.BUG_LIFECYCLE: "Bug lifecycle: reports, severity, priority, tracking",
    Topic.TEST_STRATEGY: "Test strategy: real-time, microservices, performance, security",
    Topic.DATA_PIPELINES: "Data pipelines: ETL, ELT, orchestration, idempotency",
    Topic.DATA_WAREHOUSING: "Data warehousing: partitioning, OLTP vs OLAP, lake vs warehouse",
    Topic.DATA_MODELING: "Data modeling: star schema, SCD, normalization",
    Topic.DISTRIBUTED_SYSTEMS: "Distributed systems: Spark, streaming, exactly-once",
    Topic.FRONTEND_CORE: "Frontend core: JavaScript, CSS, DOM, HTML",
    Topic.FRONTEND_FRAMEWORK: "Frontend frameworks: React, hooks, components, SSR",
    Topic.FRONTEND_PERFORMANCE: "Frontend performance: web vitals, bundle, caching",
    Topic.FRONTEND_ACCESSIBILITY: "Frontend accessibility: ARIA, WCAG, screen readers",
    Topic.FRONTEND_TESTING: "Frontend testing: Jest, RTL, Cypress, Playwright",
    Topic.FRONTEND_SECURITY: "Frontend security: XSS, CSRF, CSP",
    Topic.BEHAVIOURAL: "Behavioural STAR question: situation, task, action, result",
}


def _topic_description(topic: Topic) -> str:
    """Free-text description used for semantic search when no query is provided."""
    return _TOPIC_DESCRIPTIONS.get(topic, topic.value)
