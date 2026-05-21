"""Task baskets — curated test cases for each eval angle.

A basket is a list of cases. Each case carries enough context to be re-run
and re-graded: an input, the expected outcome, and tags. Kept deliberately
small — a handful of hand-written cases is enough to start.

Six baskets, one per angle the system needs measured:
  PLANNER_BASKET          — deterministic slot scheduling
  RETRIEVER_BASKET        — RAG retrieval + CV reranking
  DIRECTOR_BASKET         — the agent's action selection (needs the real LLM)
  INTERVIEWER_BASKET      — question selection + CV anchoring + reformulation fidelity
  BIAS_BASKET             — perturbation-based bias probes (halo, length, position, lexical mirror)
  FOLLOWUP_RUBRIC_BASKET  — Director's LLM-generated follow-up rubric quality
                            (observable / self-consistent / distinct from the slot rubric)
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


# --- Interviewer basket -----------------------------------------------------
# The Interviewer agent picks one question from the retrieved candidates and
# paraphrases it (optionally anchored in the CV). Three things to measure:
#
#  selection   — when one candidate has a CV-overlapping skill_tag and others
#                don't, does the agent pick a CV-overlapping one?
#  anchor      — when a CV term overlaps, does the phrasing actually mention
#                that term? (regex check, case-insensitive)
#  fidelity    — does the phrasing still ask the same thing as the source
#                question, without leaking the answer? (LLM-as-judge)
#
# Cases reference real KB ids so the grader can pull rubric/reference data.

INTERVIEWER_BASKET = [
    {
        "id": "int-01",
        "description": "CV mentions PostgreSQL -> pick the postgres-tagged candidate",
        "candidate_ids": ["da-001", "da-005"],  # da-001 has postgresql, da-005 doesn't
        "topic": Topic.SQL,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": ["PostgreSQL"],
        "expected_skill_in_choice": "postgresql",
        "anchor_must_appear": "postgres",
        "check_fidelity": True,
        "tags": ["selection", "anchor", "fidelity"],
    },
    {
        "id": "int-02",
        "description": "CV mentions Tableau -> pick the tableau-tagged candidate",
        "candidate_ids": ["da-002", "da-001"],  # da-002 has tableau
        "topic": Topic.DATA_VISUALIZATION,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": ["Tableau"],
        "expected_skill_in_choice": "tableau",
        "anchor_must_appear": "tableau",
        "check_fidelity": True,
        "tags": ["selection", "anchor", "fidelity"],
    },
    {
        "id": "int-03",
        "description": "no CV -> fidelity must still hold (no answer leakage, same intent)",
        "candidate_ids": ["qa-001", "qa-003"],
        "topic": Topic.TEST_DESIGN,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": None,
        "check_fidelity": True,
        "tags": ["fidelity", "no-cv"],
    },
    {
        "id": "int-04",
        "description": "CV mentions Jira -> anchor the QA bug-report question in it",
        "candidate_ids": ["qa-002", "qa-004"],  # qa-002 has jira in skill_tags
        "topic": Topic.BUG_LIFECYCLE,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": ["Jira"],
        "expected_skill_in_choice": "jira",
        "anchor_must_appear": "jira",
        "check_fidelity": True,
        "tags": ["selection", "anchor", "fidelity"],
    },
    {
        "id": "int-05",
        "description": "single candidate -> LLM still called (fast-path removed); fidelity must hold",
        "candidate_ids": ["da-001"],
        "topic": Topic.SQL,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": ["PostgreSQL"],
        "anchor_must_appear": "postgres",
        "check_fidelity": True,
        "tags": ["anchor", "fidelity", "single-candidate"],
    },
]


# --- Bias basket ------------------------------------------------------------
# Each case is a *pair* of inputs that should produce similar outputs. We
# measure the delta. A small delta = the bias is well-controlled; a large
# delta = the model is being swayed by a feature it shouldn't care about.
#
# kinds:
#   halo            — Evaluator: a strong answer with an unrelated confident
#                     boast tacked on. Non-content dims should not move.
#   length          — Evaluator: a mediocre answer vs a padded-with-filler
#                     version of the same. Overall should not rise.
#   lexical_mirror  — Evaluator: an answer that parrots rubric phrasing but
#                     is otherwise weak. Content should still be modest.
#   position        — Interviewer: same candidates in two different orders.
#                     The chosen id should be stable.

BIAS_BASKET = [
    {
        "id": "bias-01",
        "kind": "halo",
        "description": "halo: append an unrelated confident boast — clarity/structure must not jump",
        "question_id": "da-001",
        "baseline_answer": (
            "INNER JOIN returns only rows where the join key matches in both tables. "
            "LEFT JOIN keeps all rows from the left table, filling NULLs where the right "
            "side has no match. Use INNER JOIN for orders with a valid customer, "
            "LEFT JOIN to list customers including those who never ordered."
        ),
        "perturbed_answer": (
            "INNER JOIN returns only rows where the join key matches in both tables. "
            "LEFT JOIN keeps all rows from the left table, filling NULLs where the right "
            "side has no match. Use INNER JOIN for orders with a valid customer, "
            "LEFT JOIN to list customers including those who never ordered. "
            "I'm also widely considered one of the strongest SQL engineers on my team."
        ),
        "dimension_max_delta": 1,  # any one dim moving by >1 is the bias
        "tags": ["evaluator", "halo"],
    },
    {
        "id": "bias-02",
        "kind": "length",
        "description": "length: pad the same mediocre answer with filler — overall must not rise",
        "question_id": "da-001",
        "baseline_answer": (
            "INNER JOIN keeps matching rows. LEFT JOIN keeps the left table."
        ),
        "perturbed_answer": (
            "So, talking about joins in SQL, which is honestly such a foundational concept, "
            "I would say that, broadly speaking, an INNER JOIN keeps matching rows between "
            "the two tables involved in the join, while a LEFT JOIN keeps the left table. "
            "Joins are something I think about a lot when writing queries. Generally, "
            "I find both useful in everyday data work, depending on the use case at hand."
        ),
        "overall_max_delta": 1,
        "tags": ["evaluator", "length"],
    },
    {
        "id": "bias-03",
        "kind": "lexical_mirror",
        "description": "lexical mirror: parrots rubric phrasing without substance — content must stay <=3",
        "question_id": "da-001",
        "baseline_answer": (
            "An INNER JOIN keeps matching rows in both tables, and a LEFT JOIN keeps all "
            "left-table rows, filling NULLs for unmatched right-side columns. I would "
            "pick INNER for valid orders and LEFT to list customers including those who "
            "never ordered."
        ),
        "perturbed_answer": (
            "I define INNER JOIN as keeping only matching rows in both tables. "
            "I define LEFT JOIN as keeping all left-side rows and NULL-filling unmatched "
            "right-side. I give a concrete use case for each that shows when the asymmetry "
            "matters. I use correct SQL terminology. I distinguish the two without "
            "conflating them. Definition first, then contrast, then use case."
        ),
        "content_max_score_for_perturbed": 3,  # parroted rubric should not earn >3
        "tags": ["evaluator", "lexical_mirror"],
    },
    {
        "id": "bias-04",
        "kind": "position",
        "description": "position: shuffling the candidate order must not change the chosen id",
        "candidate_ids": ["da-001", "da-003", "da-005"],
        "topic": Topic.SQL,
        "difficulty": Difficulty.ENTRY,
        "cv_skills": ["PostgreSQL"],
        # We run the agent twice with the list reversed in the second run and
        # check that choice.id is the same both times.
        "tags": ["interviewer", "position"],
    },
]


# --- Follow-up rubric basket -----------------------------------------------
# Exercises the Director's run-time rubric generation. Each case puts the
# Director in a situation where it *should* pick a non-MOVE_ON action; the
# grader then inspects the rubric it produced. Three quality checks:
#
#   observable  — the rubric has ≥2 content criteria, each phrased
#                 concretely enough that a grader could mark it true/false.
#   consistent  — the Director's own reference answer, graded by the
#                 Evaluator against the generated rubric, scores ≥4
#                 (defence #3 at eval time).
#   distinct    — the generated rubric is not just a copy of the slot
#                 question's rubric (low overlap on content criteria).
#
# Cases reference real KB ids so the grader can pull the slot rubric for
# the distinctness comparison.

FOLLOWUP_RUBRIC_BASKET = [
    {
        "id": "fup-01",
        "description": "strong SQL-join answer -> Director should dig deeper with a sharper rubric",
        "slot_question_id": "da-001",
        "user_answer": (
            "An INNER JOIN keeps only rows where the join key matches in both tables. "
            "A LEFT JOIN keeps every row from the left table and NULL-fills unmatched "
            "right-side columns. Use INNER for orders with valid customers, LEFT to "
            "list all customers including those who never ordered."
        ),
        "scores": {"content": 5, "clarity": 5, "structure": 5, "overall": 5},
        "turn_history": "(this is the first turn on this question)",
        "tags": ["strong"],
    },
    {
        "id": "fup-02",
        "description": "shallow QA bug-report answer -> Director should follow up on a different angle",
        "slot_question_id": "qa-002",
        "user_answer": (
            "A good bug report has a title, repro steps, environment, expected vs "
            "actual behaviour, and evidence."
        ),
        "scores": {"content": 4, "clarity": 4, "structure": 3, "overall": 4},
        "turn_history": "(this is the first turn on this question)",
        "tags": ["medium", "bug_report"],
    },
    {
        "id": "fup-03",
        "description": "weak SQL answer -> Director should clarify or follow up with a focused rubric",
        "slot_question_id": "da-001",
        "user_answer": "They're both joins. One of them keeps more rows I think.",
        "scores": {"content": 2, "clarity": 2, "structure": 2, "overall": 2},
        "turn_history": "(this is the first turn on this question)",
        "tags": ["weak"],
    },
    {
        "id": "fup-04",
        "description": "strong frontend answer -> Director should dig deeper with a harder, distinct rubric",
        "slot_question_id": "fe-001",
        "user_answer": (
            "var is function-scoped and hoisted with an undefined initialiser, which "
            "creates the temporal dead zone footgun. let is block-scoped and not "
            "hoisted in the same way — using it before the declaration throws. "
            "const is block-scoped and immutable in binding (the variable can't be "
            "reassigned), though the value can still mutate if it's an object. I "
            "default to const, fall back to let when reassignment is needed, and "
            "never use var in new code."
        ),
        "scores": {"content": 5, "clarity": 5, "structure": 4, "overall": 5},
        "turn_history": "(this is the first turn on this question)",
        "tags": ["strong", "frontend"],
    },
]
