"""McNemar's test for paired nominal comparisons.

Implements exact binomial test and Edwards continuity-corrected chi-squared test
to compare discordant outcomes between two models or system configurations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class McNemarResult:
    """Outcome of McNemar paired statistical test."""
    contingency_table: dict[str, int]
    discordant_a_passes_b_fails: int  # b
    discordant_a_fails_b_passes: int  # c
    total_discordant: int              # b + c
    test_statistic: float              # chi-squared or exact statistic
    p_value: float
    is_statistically_significant: bool # p < alpha
    odds_ratio: float
    method: str
    alpha: float = 0.05

    def to_dict(self) -> dict[str, Any]:
        return {
            "contingency_table": self.contingency_table,
            "discordant_b": self.discordant_a_passes_b_fails,
            "discordant_c": self.discordant_a_fails_b_passes,
            "total_discordant": self.total_discordant,
            "test_statistic": round(self.test_statistic, 4),
            "p_value": round(self.p_value, 6),
            "is_statistically_significant": self.is_statistically_significant,
            "odds_ratio": round(self.odds_ratio, 4) if not math.isinf(self.odds_ratio) else "inf",
            "method": self.method,
            "alpha": self.alpha,
        }


def calculate_mcnemar_test(
    a_results: list[bool],
    b_results: list[bool],
    alpha: float = 0.05,
) -> McNemarResult:
    """Calculate McNemar's test on two aligned boolean outcome vectors.

    Args:
        a_results: Boolean success vector for System A (e.g. Full Kernel + 8B).
        b_results: Boolean success vector for System B (e.g. Raw Frontier).
        alpha: Significance threshold (default 0.05).

    Returns:
        McNemarResult containing statistic, p-value, odds ratio, and significance flag.
    """
    if len(a_results) != len(b_results):
        raise ValueError(
            f"Vector lengths must match: len(a)={len(a_results)} vs len(b)={len(b_results)}"
        )

    a = 0  # both pass
    b = 0  # A passes, B fails
    c = 0  # A fails, B passes
    d = 0  # both fail

    for res_a, res_b in zip(a_results, b_results):
        if res_a and res_b:
            a += 1
        elif res_a and not res_b:
            b += 1
        elif not res_a and res_b:
            c += 1
        else:
            d += 1

    total_discordant = b + c

    # Odds ratio calculation
    if c == 0:
        odds_ratio = float("inf") if b > 0 else 1.0
    else:
        odds_ratio = b / c

    table = {"both_pass": a, "a_only": b, "b_only": c, "both_fail": d}

    if total_discordant == 0:
        return McNemarResult(
            contingency_table=table,
            discordant_a_passes_b_fails=b,
            discordant_a_fails_b_passes=c,
            total_discordant=0,
            test_statistic=0.0,
            p_value=1.0,
            is_statistically_significant=False,
            odds_ratio=1.0,
            method="Exact (zero discordant pairs)",
            alpha=alpha,
        )

    # If small number of discordant pairs, use exact two-sided binomial test
    if total_discordant < 25:
        k = min(b, c)
        prob_sum = 0.0
        for i in range(k + 1):
            prob_sum += math.comb(total_discordant, i) * (0.5 ** total_discordant)
        p_val = min(1.0, 2.0 * prob_sum)
        stat = float(b)
        method = "Exact Binomial Test (n < 25)"
    else:
        # Edwards continuity-corrected chi-squared test
        diff = abs(b - c)
        chi2 = ((diff - 1.0) ** 2) / total_discordant
        stat = chi2
        # Two-tailed p-value for 1 degree of freedom: erfc(sqrt(chi2 / 2))
        p_val = math.erfc(math.sqrt(chi2) / math.sqrt(2.0))
        method = "Edwards Continuity-Corrected Chi-Squared (n >= 25)"

    p_val = max(0.0, min(1.0, p_val))

    return McNemarResult(
        contingency_table=table,
        discordant_a_passes_b_fails=b,
        discordant_a_fails_b_passes=c,
        total_discordant=total_discordant,
        test_statistic=stat,
        p_value=p_val,
        is_statistically_significant=(p_val < alpha),
        odds_ratio=odds_ratio,
        method=method,
        alpha=alpha,
    )
