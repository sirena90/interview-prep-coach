"""The eval leaderboard — evaluation framed as comparison.

A single number from one run is uninformative; what matters is the delta
between system versions on the *same* task baskets. Every grader appends or
merges its score into a version row.
"""
from __future__ import annotations

LEADERBOARD: list[dict] = []


def add_to_leaderboard(version: str, scores: dict) -> None:
    """Append a new version row, or merge scores into an existing one.

    Merging lets each grader contribute only its own metric without having
    to re-pass the others.
    """
    for i, row in enumerate(LEADERBOARD):
        if row["version"] == version:
            LEADERBOARD[i] = {**row, **scores, "version": version}
            return
    LEADERBOARD.append({"version": version, **scores})


def format_leaderboard() -> str:
    if not LEADERBOARD:
        return "(leaderboard empty — no scores yet)"
    cols = sorted({k for row in LEADERBOARD for k in row if k != "version"})
    header = f"{'version':<22}" + "".join(f"{c:>16}" for c in cols)
    lines = [header, "-" * len(header)]
    for row in LEADERBOARD:
        line = f"{row['version']:<22}"
        for c in cols:
            v = row.get(c, "-")
            line += f"{v:>16.3f}" if isinstance(v, (int, float)) else f"{str(v):>16}"
        lines.append(line)
    return "\n".join(lines)


def show_leaderboard() -> None:
    print(format_leaderboard())


def reset_leaderboard() -> None:
    LEADERBOARD.clear()
