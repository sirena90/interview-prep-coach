"""Agreement metrics for calibrating the Evaluator against human labels.

Cohen's kappa measures rater agreement corrected for chance. Plain accuracy
is misleading for a 1-5 judge — a model that always says "3" can look decent.
We use the QUADRATIC-WEIGHTED variant so being off by 1 is penalised far less
than being off by 4 (the scores are ordinal, not categorical).

Agreement targets: kappa >= 0.6 substantial, >= 0.8 near-human.

Implemented inline (no scikit-learn dependency) — it is ~30 lines.
"""
from __future__ import annotations


def quadratic_weighted_kappa(
    human: list[int],
    model: list[int],
    k_min: int = 1,
    k_max: int = 5,
) -> float:
    """Quadratic-weighted Cohen's kappa between two integer rating lists."""
    if len(human) != len(model):
        raise ValueError("human and model rating lists must be the same length")
    n = len(human)
    if n == 0:
        raise ValueError("cannot compute kappa over zero ratings")

    n_classes = k_max - k_min + 1

    # Observed agreement matrix O[i][j] = #(human=i, model=j).
    O = [[0] * n_classes for _ in range(n_classes)]
    for h, m in zip(human, model):
        O[h - k_min][m - k_min] += 1

    # Quadratic weights: distance^2 normalised to [0, 1].
    denom = (n_classes - 1) ** 2
    W = [[((i - j) ** 2) / denom for j in range(n_classes)]
         for i in range(n_classes)]

    # Marginal histograms -> expected matrix under independence.
    hist_h = [sum(O[i]) for i in range(n_classes)]
    hist_m = [sum(O[i][j] for i in range(n_classes)) for j in range(n_classes)]

    num = den = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            expected = hist_h[i] * hist_m[j] / n
            num += W[i][j] * O[i][j]
            den += W[i][j] * expected

    if den == 0:
        return 1.0  # no expected disagreement -> perfect by convention
    return 1.0 - num / den


def kappa_label(kappa: float) -> str:
    """Human-readable band for a kappa value (Landis & Koch scale)."""
    if kappa >= 0.8:
        return "near-human"
    if kappa >= 0.6:
        return "substantial"
    if kappa >= 0.4:
        return "moderate"
    if kappa >= 0.2:
        return "fair"
    return "poor"
