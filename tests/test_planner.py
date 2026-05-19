"""Layer 1 — deterministic unit tests for SimplePlanner.

The planner is pure Python with a seedable RNG, so every rule can be checked
exactly. No LLM, no Chroma — the fake_kb fixture supplies role topics.
"""
import pytest

from core.models import Difficulty, SlotType, Topic
from core.planner import SESSION_PATTERN, SimplePlanner


@pytest.fixture
def planner(fake_kb):
    # Fixed seed -> reproducible topic picks across runs.
    return SimplePlanner(fake_kb, rng_seed=42)


class TestSlotPattern:
    def test_q1_is_cover(self, planner, make_session):
        assert planner.plan_next_turn(make_session()).slot_type is SlotType.COVER

    def test_q3_is_behavioral(self, planner, make_session, make_turn):
        turns = [
            make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID),
            make_turn(2, SlotType.REINFORCE, Topic.SQL, Difficulty.MID),
        ]
        plan = planner.plan_next_turn(make_session(turns=turns))
        assert plan.slot_type is SlotType.BEHAVIORAL
        assert plan.topic is Topic.BEHAVIOURAL

    def test_full_pattern_order(self, planner, make_session, make_turn):
        # Walk all 5 slots, feeding back the planned turn each time.
        observed, turns = [], []
        for i in range(5):
            plan = planner.plan_next_turn(make_session(turns=list(turns)))
            observed.append(plan.slot_type)
            turns.append(make_turn(i + 1, plan.slot_type, plan.topic, plan.difficulty))
        assert observed == SESSION_PATTERN

    def test_raises_when_called_past_pattern(self, planner, make_session, make_turn):
        turns = []
        for i, slot in enumerate(SESSION_PATTERN):
            topic = Topic.BEHAVIOURAL if slot is SlotType.BEHAVIORAL else Topic.SQL
            turns.append(make_turn(i + 1, slot, topic, Difficulty.MID))
        with pytest.raises(RuntimeError):
            planner.plan_next_turn(make_session(turns=turns))


class TestReinforceTopic:
    def test_q2_reuses_q1_topic(self, planner, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.STATISTICS, Difficulty.MID)]
        plan = planner.plan_next_turn(make_session(turns=turns))
        assert plan.slot_type is SlotType.REINFORCE
        assert plan.topic is Topic.STATISTICS

    def test_q4_picks_a_topic_not_yet_covered(self, planner, make_session, make_turn):
        turns = [
            make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID),
            make_turn(2, SlotType.REINFORCE, Topic.SQL, Difficulty.MID),
            make_turn(3, SlotType.BEHAVIORAL, Topic.BEHAVIOURAL, Difficulty.MID),
        ]
        plan = planner.plan_next_turn(make_session(turns=turns))
        assert plan.slot_type is SlotType.COVER
        assert plan.topic is not Topic.SQL


class TestDifficultyAdjustment:
    """Reinforce difficulty adjusts on the previous cover's normalized score:
    < 0.4 steps down, > 0.85 steps up, otherwise stays. Floored/capped."""

    def test_low_score_steps_down(self, planner, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID, overall=1)]
        plan = planner.plan_next_turn(make_session(difficulty=Difficulty.MID, turns=turns))
        assert plan.difficulty is Difficulty.ENTRY

    def test_high_score_steps_up(self, planner, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID, overall=5)]
        plan = planner.plan_next_turn(make_session(difficulty=Difficulty.MID, turns=turns))
        assert plan.difficulty is Difficulty.SENIOR

    def test_mid_score_stays(self, planner, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID, overall=3)]
        plan = planner.plan_next_turn(make_session(difficulty=Difficulty.MID, turns=turns))
        assert plan.difficulty is Difficulty.MID

    def test_floored_at_entry(self, planner, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.ENTRY, overall=1)]
        plan = planner.plan_next_turn(make_session(difficulty=Difficulty.ENTRY, turns=turns))
        assert plan.difficulty is Difficulty.ENTRY

    def test_capped_at_senior(self, planner, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.SENIOR, overall=5)]
        plan = planner.plan_next_turn(make_session(difficulty=Difficulty.SENIOR, turns=turns))
        assert plan.difficulty is Difficulty.SENIOR

    def test_unscored_cover_keeps_base_difficulty(self, planner, make_session, make_turn):
        turns = [make_turn(1, SlotType.COVER, Topic.SQL, Difficulty.MID, scored=False)]
        plan = planner.plan_next_turn(make_session(difficulty=Difficulty.MID, turns=turns))
        assert plan.difficulty is Difficulty.MID


class TestReproducibility:
    def test_same_seed_produces_same_topic(self, fake_kb, make_session):
        p1 = SimplePlanner(fake_kb, rng_seed=7)
        p2 = SimplePlanner(fake_kb, rng_seed=7)
        assert p1.plan_next_turn(make_session()).topic == p2.plan_next_turn(make_session()).topic
