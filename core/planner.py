"""Two-phase planner that schedules every turn in a 5-question session.

The session pattern is fixed:

    Q1 — COVER       fresh topic
    Q2 — REINFORCE   same topic as Q1, difficulty adjusted by Q1's score
    Q3 — BEHAVIORAL  STAR question
    Q4 — COVER       a different fresh topic
    Q5 — REINFORCE   same topic as Q4, difficulty adjusted by Q4's score

The planner is deterministic Python (not an LLM agent). That's deliberate:
  - reproducible across runs (test-friendly)
  - easy to explain to the evaluation committee ("here's the rule")
  - cheap (no LLM call per turn for slot picking)

Public API:
    planner = SimplePlanner(kb)
    plan = planner.plan_next_turn(session_state)
    # plan.slot_type, plan.topic, plan.difficulty
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from core.kb import KnowledgeBase
from core.models import (
    Difficulty,
    SessionState,
    SlotType,
    Topic,
    TurnRecord,
)


# Fixed 5-slot pattern. Index 0 = Q1, index 4 = Q5.
SESSION_PATTERN: list[SlotType] = [
    SlotType.COVER,        # Q1
    SlotType.REINFORCE,    # Q2  (reinforces Q1's topic)
    SlotType.BEHAVIORAL,   # Q3
    SlotType.COVER,        # Q4
    SlotType.REINFORCE,    # Q5  (reinforces Q4's topic)
]


@dataclass
class TurnPlan:
    """What the planner decided for the next turn.

    The orchestrator passes (topic, difficulty) to KnowledgeBase.retrieve()
    to get candidate questions, then to InterviewerAgent.ask() for the final
    pick + phrasing.
    """
    slot_type: SlotType
    topic: Topic
    difficulty: Difficulty


class SimplePlanner:
    """Decides slot, topic, and difficulty for each turn. Stateless.

    Reads from SessionState (turns so far, configured difficulty, role) and
    the KnowledgeBase (which topics are eligible per role). Returns one
    TurnPlan per call.
    """

    def __init__(self, kb: KnowledgeBase, rng_seed: Optional[int] = None) -> None:
        self._kb = kb
        # Seeded RNG for reproducible topic order in tests; default is non-seeded.
        self._rng = random.Random(rng_seed)

    # ---- Public API -------------------------------------------------------

    def plan_next_turn(self, state: SessionState) -> TurnPlan:
        """Look at how many turns have happened and decide what comes next.

        Raises if called past the configured target_turns (defensive).
        """
        turn_idx = state.turn_count()  # 0-indexed; next turn is turn_idx + 1
        if turn_idx >= len(SESSION_PATTERN):
            raise RuntimeError(
                f"Planner called for turn {turn_idx + 1} but the pattern only "
                f"has {len(SESSION_PATTERN)} slots."
            )

        slot_type = SESSION_PATTERN[turn_idx]

        if slot_type is SlotType.BEHAVIORAL:
            return TurnPlan(
                slot_type=slot_type,
                topic=Topic.BEHAVIOURAL,
                difficulty=state.difficulty,
            )

        if slot_type is SlotType.COVER:
            return self._plan_cover(state)

        # SlotType.REINFORCE
        return self._plan_reinforce(state)

    # ---- Internals --------------------------------------------------------

    def _plan_cover(self, state: SessionState) -> TurnPlan:
        """Pick an uncovered topic for the role for a COVER slot."""
        role_topics = self._kb.topics_for_role(state.role)
        if not role_topics:
            raise RuntimeError(f"No topics configured for role {state.role}")

        covered = {
            t.topic for t in state.turns
            if t.slot_type is not SlotType.BEHAVIORAL
        }
        uncovered = role_topics - covered

        # If everything has been touched (unlikely in a 5-Q session), pick any.
        topic_pool = uncovered if uncovered else role_topics

        # Deterministic order before random pick (test-friendly).
        topic = self._rng.choice(sorted(topic_pool, key=lambda t: t.value))

        return TurnPlan(
            slot_type=SlotType.COVER,
            topic=topic,
            difficulty=state.difficulty,
        )

    def _plan_reinforce(self, state: SessionState) -> TurnPlan:
        """Reinforce reuses the most recent COVER turn's topic with adjusted diff."""
        last_cover = self._most_recent_cover(state)

        if last_cover is None:
            # Safety net: should never happen given the fixed pattern, but if
            # somehow we hit reinforce before any cover, fall back to a cover.
            return self._plan_cover(state)

        difficulty = self._adjust_difficulty(state.difficulty, last_cover)

        return TurnPlan(
            slot_type=SlotType.REINFORCE,
            topic=last_cover.topic,
            difficulty=difficulty,
        )

    @staticmethod
    def _most_recent_cover(state: SessionState) -> Optional[TurnRecord]:
        for turn in reversed(state.turns):
            if turn.slot_type is SlotType.COVER:
                return turn
        return None

    @staticmethod
    def _adjust_difficulty(base: Difficulty, last_cover: TurnRecord) -> Difficulty:
        """Pick the reinforce difficulty based on the previous cover's score.

        Rule:
          - score < 0.4  (i.e., 1-2 out of 5)   -> step down one level
          - score > 0.85 (i.e., 5 out of 5)     -> step up one level
          - otherwise                            -> stay at base difficulty

        Floors at ENTRY, ceils at SENIOR.
        """
        if last_cover.score_report is None:
            return base

        normalized = last_cover.score_report.normalized()  # 0..1
        order = [Difficulty.ENTRY, Difficulty.MID, Difficulty.SENIOR]
        idx = order.index(base)

        if normalized < 0.4 and idx > 0:
            return order[idx - 1]
        if normalized > 0.85 and idx < len(order) - 1:
            return order[idx + 1]
        return base
