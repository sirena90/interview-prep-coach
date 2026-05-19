"""Task baskets — curated test cases for each eval angle.

A basket is a list of cases. Each case carries enough context to be re-run
and re-graded: an input, the expected outcome, and tags. Kept deliberately
small — a handful of hand-written cases is enough to start.

Three baskets, one per angle the system needs measured:
  PLANNER_BASKET   — deterministic slot scheduling
  RETRIEVER_BASKET — RAG retrieval + CV reranking
  DIRECTOR_BASKET  — the agent's action selection (needs the real LLM)
"""
from core.models import Difficulty, DirectorAction, Role, SlotType, Topic


# --- Planner basket ---------------------------------------------------------
# prior_turns: turns already completed, replayed into a SessionState by the
# grader. Each is {slot, topic, difficulty, overall}.

PLANNER_BASKET = [
    {
        "id": "plan-01",
        "description": "fresh session -> Q1 is a COVER slot at base difficulty",
        "role": Role.DATA_ANALYST,
        "difficulty": Difficulty.MID,
        "prior_turns": [],
        "expected_slot": SlotType.COVER,
        "expected_difficulty": Difficulty.MID,
        "tags": ["slot-pattern"],
    },
    {
        "id": "plan-02",
        "description": "after Q1 cover -> Q2 reinforces the same topic",
        "role": Role.DATA_ANALYST,
        "difficulty": Difficulty.MID,
        "prior_turns": [
            {"slot": SlotType.COVER, "topic": Topic.SQL,
             "difficulty": Difficulty.MID, "overall": 3},
        ],
        "expected_slot": SlotType.REINFORCE,
        "expected_topic": Topic.SQL,
        "expected_difficulty": Difficulty.MID,
        "tags": ["slot-pattern", "reinforce"],
    },
    {
        "id": "plan-03",
        "description": "weak Q1 (overall 1) -> reinforce steps difficulty down",
        "role": Role.DATA_ANALYST,
        "difficulty": Difficulty.MID,
        "prior_turns": [
            {"slot": SlotType.COVER, "topic": Topic.SQL,
             "difficulty": Difficulty.MID, "overall": 1},
        ],
        "expected_slot": SlotType.REINFORCE,
        "expected_difficulty": Difficulty.ENTRY,
        "tags": ["difficulty-adapt"],
    },
    {
        "id": "plan-04",
        "description": "strong Q1 (overall 5) -> reinforce steps difficulty up",
        "role": Role.DATA_ANALYST,
        "difficulty": Difficulty.MID,
        "prior_turns": [
            {"slot": SlotType.COVER, "topic": Topic.SQL,
             "difficulty": Difficulty.MID, "overall": 5},
        ],
        "expected_slot": SlotType.REINFORCE,
        "expected_difficulty": Difficulty.SENIOR,
        "tags": ["difficulty-adapt"],
    },
    {
        "id": "plan-05",
        "description": "Q3 is always the behavioural slot",
        "role": Role.QA_ENGINEER,
        "difficulty": Difficulty.MID,
        "prior_turns": [
            {"slot": SlotType.COVER, "topic": Topic.TEST_DESIGN,
             "difficulty": Difficulty.MID, "overall": 3},
            {"slot": SlotType.REINFORCE, "topic": Topic.TEST_DESIGN,
             "difficulty": Difficulty.MID, "overall": 3},
        ],
        "expected_slot": SlotType.BEHAVIORAL,
        "expected_topic": Topic.BEHAVIOURAL,
        "tags": ["slot-pattern", "behavioural"],
    },
    {
        "id": "plan-06",
        "description": "weak entry-level Q1 -> difficulty floored at ENTRY",
        "role": Role.DATA_ANALYST,
        "difficulty": Difficulty.ENTRY,
        "prior_turns": [
            {"slot": SlotType.COVER, "topic": Topic.SQL,
             "difficulty": Difficulty.ENTRY, "overall": 1},
        ],
        "expected_slot": SlotType.REINFORCE,
        "expected_difficulty": Difficulty.ENTRY,
        "tags": ["difficulty-adapt", "boundary"],
    },
]


# --- Retriever basket -------------------------------------------------------
# check: "topic_match" -> every result must be in the requested topic.
#        "cv_rerank"   -> the top result must carry the CV skill tag.

RETRIEVER_BASKET = [
    {
        "id": "ret-01",
        "description": "SQL/entry retrieval returns only SQL questions",
        "topic": Topic.SQL,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": None,
        "check": "topic_match",
        "tags": ["filter"],
    },
    {
        "id": "ret-02",
        "description": "CV skill 'postgresql' reranks a matching question to the top",
        "topic": Topic.SQL,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": ["postgresql"],
        "check": "cv_rerank",
        "tags": ["rerank"],
    },
    {
        "id": "ret-03",
        "description": "QA test-design retrieval returns only test-design questions",
        "topic": Topic.TEST_DESIGN,
        "difficulty": Difficulty.MID,
        "cv_skills": None,
        "check": "topic_match",
        "tags": ["filter"],
    },
    {
        "id": "ret-04",
        "description": "frontend-core retrieval returns only frontend-core questions",
        "topic": Topic.FRONTEND_CORE,
        "difficulty": Difficulty.MID,
        "cv_skills": None,
        "check": "topic_match",
        "tags": ["filter"],
    },
]


# --- Director basket --------------------------------------------------------
# The Director's decision is an LLM call, so its action is graded against a
# SET of acceptable actions rather than one single exact answer.

DIRECTOR_BASKET = [
    {
        "id": "dir-01",
        "description": "strong answer (5/5) -> escalate to a harder variant",
        "scores": {"content": 5, "clarity": 5, "structure": 5, "overall": 5},
        "user_answer": "An INNER JOIN returns only rows matching in both tables; "
                       "a LEFT JOIN keeps all left-table rows, NULL-filling the "
                       "right. I'd use LEFT JOIN to find customers with no orders, "
                       "INNER JOIN for valid order-customer pairs.",
        "turn_history": "(this is the first turn on this question)",
        "acceptable_actions": {DirectorAction.DIG_DEEPER},
        "tags": ["strong"],
    },
    {
        "id": "dir-02",
        "description": "ambiguous, incomplete answer -> clarify or follow up",
        "scores": {"content": 2, "clarity": 2, "structure": 2, "overall": 2},
        "user_answer": "They're both joins. One of them keeps more rows I think.",
        "turn_history": "(this is the first turn on this question)",
        "acceptable_actions": {DirectorAction.CLARIFY, DirectorAction.FOLLOWUP},
        "tags": ["weak"],
    },
    {
        "id": "dir-03",
        "description": "loop guard: after 2 prior rounds -> must move on",
        "scores": {"content": 3, "clarity": 3, "structure": 3, "overall": 3},
        "user_answer": "INNER keeps matching rows, LEFT keeps all left-table rows.",
        "turn_history": "Turn 1: a clarify question was asked. Turn 2: a followup "
                        "was asked. This is the 3rd round on this same question.",
        "acceptable_actions": {DirectorAction.MOVE_ON},
        "tags": ["loop-guard"],
    },
    {
        "id": "dir-04",
        "description": "empty answer -> clarify or move on, never dig deeper",
        "scores": {"content": 1, "clarity": 1, "structure": 1, "overall": 1},
        "user_answer": "(empty answer)",
        "turn_history": "(this is the first turn on this question)",
        "acceptable_actions": {DirectorAction.CLARIFY, DirectorAction.MOVE_ON},
        "tags": ["edge-case"],
    },
    {
        "id": "dir-05",
        "description": "solid but shallow answer -> follow up, dig deeper, or move on",
        "scores": {"content": 4, "clarity": 4, "structure": 3, "overall": 4},
        "user_answer": "INNER JOIN returns rows that match in both tables; "
                       "LEFT JOIN returns all rows from the left table.",
        "turn_history": "(this is the first turn on this question)",
        "acceptable_actions": {DirectorAction.FOLLOWUP, DirectorAction.DIG_DEEPER,
                               DirectorAction.MOVE_ON},
        "tags": ["medium"],
    },
]
