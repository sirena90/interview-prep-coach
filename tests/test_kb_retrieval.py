"""Integration tests for the KB retriever (the RAG layer).

These build a REAL Chroma index over the 129 questions, so they are marked
`integration` and skipped by a plain `pytest` run. Run them with:

    pytest -m integration

The first run downloads Chroma's default embedding model (~80 MB), once.
"""
import pytest

from core.kb import DATA_DIR, KnowledgeBase
from core.models import Difficulty, Question, Role, Topic

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    """Build the KB once per module against a throwaway Chroma path so the
    real .chroma cache is never touched."""
    chroma_dir = tmp_path_factory.mktemp("chroma")
    return KnowledgeBase(data_dir=DATA_DIR, chroma_path=chroma_dir)


class TestKnowledgeBaseLoading:
    def test_loads_all_129_questions(self, kb):
        assert len(kb._questions) == 129

    def test_topics_for_role_returns_role_topics(self, kb):
        assert Topic.SQL in kb.topics_for_role(Role.DATA_ANALYST)
        assert Topic.SQL not in kb.topics_for_role(Role.FRONTEND_DEVELOPER)


class TestRetrieve:
    def test_returns_questions_matching_topic_and_difficulty(self, kb):
        results = kb.retrieve(topic=Topic.SQL, difficulty=Difficulty.ENTRY)
        assert results
        assert all(isinstance(q, Question) for q in results)
        assert all(q.topic is Topic.SQL for q in results)
        assert all(q.difficulty is Difficulty.ENTRY for q in results)

    def test_respects_k_limit(self, kb):
        results = kb.retrieve(topic=Topic.SQL, difficulty=Difficulty.ENTRY, k=2)
        assert len(results) <= 2

    def test_excludes_already_asked_ids(self, kb):
        first = kb.retrieve(topic=Topic.SQL, difficulty=Difficulty.ENTRY, k=5)
        assert first
        excluded = {first[0].id}
        again = kb.retrieve(
            topic=Topic.SQL, difficulty=Difficulty.ENTRY,
            excluded_ids=excluded, k=5,
        )
        assert first[0].id not in {q.id for q in again}


class TestCvAwareReranking:
    """CV-skill overlap should rank a matching question to the top."""

    def test_cv_skill_overlap_ranks_a_matching_question_first(self, kb):
        plain = kb.retrieve(topic=Topic.SQL, difficulty=Difficulty.ENTRY, k=5)
        has_pg = [q for q in plain if "postgresql" in q.skill_tags]
        if not has_pg or len(plain) < 2:
            pytest.skip("not enough SQL/entry questions to exercise reranking")

        reranked = kb.retrieve(
            topic=Topic.SQL, difficulty=Difficulty.ENTRY,
            cv_skills=["postgresql"], k=5,
        )
        assert "postgresql" in reranked[0].skill_tags


class TestRetrieveBehavioural:
    def test_returns_behavioural_questions(self, kb):
        results = kb.retrieve_behavioural(difficulty=Difficulty.ENTRY)
        assert results
        assert all(q.topic is Topic.BEHAVIOURAL for q in results)
