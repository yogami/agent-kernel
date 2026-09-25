"""Observational Causal Verifier.

Tests Structural Causal Model DAG assumptions against tabular observational data.
Performs conditional independence testing:
1. Continuous variables: Partial correlation via OLS residuals and Fisher's z-transform.
2. Categorical variables: Stratified conditional G-test (log-likelihood ratio) with Wilson-Hilferty p-values.
"""

from __future__ import annotations

from enum import Enum
import math
from typing import Any
import numpy as np
from pydantic import BaseModel, Field

from core.causal_graph import StructuralCausalModel


class IndependenceTestType(str, Enum):
    PARTIAL_CORRELATION = "partial_correlation"
    CONDITIONAL_G_TEST = "conditional_g_test"


class IndependenceTestResult(BaseModel):
    """Result of a single statistical conditional independence test."""

    var_x: str
    var_y: str
    conditioning_set: list[str] = Field(default_factory=list)
    test_type: IndependenceTestType = IndependenceTestType.PARTIAL_CORRELATION
    statistic: float
    p_value: float
    adjusted_p_value: float | None = None
    degrees_of_freedom: float
    sample_size: int
    predicted_independent: bool
    empirically_independent: bool
    is_consistent: bool
    violation_detail: str | None = None


class CausalVerificationReport(BaseModel):
    """Aggregate report comparing DAG d-separation predictions against observational data."""

    scm_name: str
    is_valid: bool
    alpha: float = 0.05
    fdr_applied: bool = False
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    violations: list[IndependenceTestResult] = Field(default_factory=list)
    all_results: list[IndependenceTestResult] = Field(default_factory=list)
    summary: str = ""


class ObservationalCausalVerifier:
    """Verifies Structural Causal Model DAG assumptions against empirical tabular records."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha

    def _is_numeric_column(self, values: list[Any]) -> bool:
        """Check whether column contains continuous numeric values."""
        valid_nums = 0
        for v in values:
            if v is None:
                continue
            if isinstance(v, bool):
                return False
            if isinstance(v, (int, float)):
                valid_nums += 1
            else:
                return False
        return valid_nums > 0

    def compute_partial_correlation(
        self,
        data: list[dict[str, Any]],
        var_x: str,
        var_y: str,
        conditioning_set: list[str],
    ) -> tuple[float, float, float, int]:
        """Compute sample partial correlation, p-value, degrees of freedom, and sample size.

        Uses OLS residual regression and Fisher's z-transformation.
        """
        all_vars = [var_x, var_y] + conditioning_set
        clean_rows = []
        for row in data:
            if all(row.get(v) is not None for v in all_vars):
                try:
                    vals = [float(row[v]) for v in all_vars]
                    clean_rows.append(vals)
                except (ValueError, TypeError):
                    continue

        n = len(clean_rows)
        k = len(conditioning_set)
        df = max(0, n - k - 3)

        if n < k + 4 or df <= 0:
            return 0.0, 1.0, float(df), n

        mat = np.array(clean_rows, dtype=np.float64)
        x_vec = mat[:, 0]
        y_vec = mat[:, 1]

        # Check for zero variance
        if np.std(x_vec) < 1e-9 or np.std(y_vec) < 1e-9:
            return 0.0, 1.0, float(df), n

        if k == 0:
            corr_mat = np.corrcoef(x_vec, y_vec)
            r = float(corr_mat[0, 1])
        else:
            z_mat = mat[:, 2:]
            a_mat = np.column_stack([np.ones(n), z_mat])
            beta_x, _, _, _ = np.linalg.lstsq(a_mat, x_vec, rcond=None)
            beta_y, _, _, _ = np.linalg.lstsq(a_mat, y_vec, rcond=None)
            res_x = x_vec - a_mat @ beta_x
            res_y = y_vec - a_mat @ beta_y

            std_rx = np.std(res_x)
            std_ry = np.std(res_y)
            if std_rx < 1e-9 or std_ry < 1e-9:
                return 0.0, 1.0, float(df), n

            corr_mat = np.corrcoef(res_x, res_y)
            r = float(corr_mat[0, 1])

        if math.isnan(r):
            return 0.0, 1.0, float(df), n

        r_clipped = max(-0.999999, min(0.999999, r))
        fisher_z = 0.5 * math.log((1.0 + r_clipped) / (1.0 - r_clipped))
        z_stat = fisher_z * math.sqrt(df)
        p_val = float(math.erfc(abs(z_stat) / math.sqrt(2.0)))

        return r, p_val, float(df), n

    def compute_conditional_g_test(
        self,
        data: list[dict[str, Any]],
        var_x: str,
        var_y: str,
        conditioning_set: list[str],
    ) -> tuple[float, float, float, int]:
        """Compute stratified conditional log-likelihood G-statistic and Wilson-Hilferty p-value."""
        all_vars = [var_x, var_y] + conditioning_set
        clean_rows = [row for row in data if all(row.get(v) is not None for v in all_vars)]
        n = len(clean_rows)

        if n < 4:
            return 0.0, 1.0, 0.0, n

        # Group data into strata by conditioning set values
        strata: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        for row in clean_rows:
            key = tuple(str(row[v]) for v in conditioning_set)
            if key not in strata:
                strata[key] = []
            strata[key].append(row)

        total_g2 = 0.0
        total_df = 0

        for key, s_rows in strata.items():
            if len(s_rows) < 2:
                continue

            counts: dict[tuple[str, str], int] = {}
            row_totals: dict[str, int] = {}
            col_totals: dict[str, int] = {}

            for r in s_rows:
                vx = str(r[var_x])
                vy = str(r[var_y])
                counts[(vx, vy)] = counts.get((vx, vy), 0) + 1
                row_totals[vx] = row_totals.get(vx, 0) + 1
                col_totals[vy] = col_totals.get(vy, 0) + 1

            n_stratum = len(s_rows)
            num_r = len(row_totals)
            num_c = len(col_totals)

            if num_r < 2 or num_c < 2:
                continue

            stratum_df = (num_r - 1) * (num_c - 1)
            total_df += stratum_df

            for (vx, vy), o_val in counts.items():
                expected = (row_totals[vx] * col_totals[vy]) / float(n_stratum)
                if expected > 0 and o_val > 0:
                    total_g2 += 2.0 * o_val * math.log(o_val / expected)

        if total_df <= 0 or total_g2 <= 0.0:
            return 0.0, 1.0, float(total_df), n

        # Wilson-Hilferty normal transformation for chi-square survival
        term1 = (total_g2 / float(total_df)) ** (1.0 / 3.0)
        term2 = 1.0 - (2.0 / (9.0 * float(total_df)))
        denom = math.sqrt(2.0 / (9.0 * float(total_df)))
        z_score = (term1 - term2) / denom
        p_val = max(0.0, min(1.0, 0.5 * math.erfc(z_score / math.sqrt(2.0))))

        return float(total_g2), float(p_val), float(total_df), n

    def verify_dataset(
        self,
        scm: StructuralCausalModel,
        data: list[dict[str, Any]],
        custom_tests: list[dict[str, Any]] | None = None,
        check_direct_edges: bool = True,
        apply_fdr: bool = True,
    ) -> CausalVerificationReport:
        """Evaluate observational records against the SCM's d-separation topology."""
        test_specs: list[dict[str, Any]] = []

        if custom_tests is not None:
            for spec in custom_tests:
                test_specs.append({
                    "var_x": spec["x"],
                    "var_y": spec["y"],
                    "conditioning_set": spec.get("z", []),
                    "predicted_independent": scm.is_d_separated(spec["x"], spec["y"], spec.get("z", [])),
                })
        else:
            implied = scm.find_implied_independencies(max_conditioning_size=2)
            for imp in implied:
                test_specs.append({
                    "var_x": imp["var_x"],
                    "var_y": imp["var_y"],
                    "conditioning_set": imp["conditioning_set"],
                    "predicted_independent": True,
                })

            if check_direct_edges:
                checked_pairs = set()
                for edge in scm.all_edges:
                    pair = (edge.source, edge.target)
                    if pair not in checked_pairs:
                        checked_pairs.add(pair)
                        test_specs.append({
                            "var_x": edge.source,
                            "var_y": edge.target,
                            "conditioning_set": [],
                            "predicted_independent": False,
                        })

        raw_test_records: list[dict[str, Any]] = []

        for spec in test_specs:
            vx = spec["var_x"]
            vy = spec["var_y"]
            z_set = spec["conditioning_set"]
            pred_indep = spec["predicted_independent"]

            x_vals = [r.get(vx) for r in data if r.get(vx) is not None]
            y_vals = [r.get(vy) for r in data if r.get(vy) is not None]

            if not x_vals or not y_vals:
                continue

            use_numeric = self._is_numeric_column(x_vals) and self._is_numeric_column(y_vals)

            if use_numeric:
                stat, p_val, df, n = self.compute_partial_correlation(data, vx, vy, z_set)
                test_type = IndependenceTestType.PARTIAL_CORRELATION
            else:
                stat, p_val, df, n = self.compute_conditional_g_test(data, vx, vy, z_set)
                test_type = IndependenceTestType.CONDITIONAL_G_TEST

            raw_test_records.append({
                "vx": vx,
                "vy": vy,
                "z_set": z_set,
                "pred_indep": pred_indep,
                "stat": stat,
                "p_val": p_val,
                "df": df,
                "n": n,
                "test_type": test_type,
            })

        m = len(raw_test_records)
        q_values = [1.0] * m
        if apply_fdr and m > 0:
            # Benjamini-Hochberg procedure: q_(i) = min(q_(i+1), (m / rank) * p_(i))
            sorted_indices = sorted(range(m), key=lambda i: raw_test_records[i]["p_val"])
            cum_min = 1.0
            for rank_idx in range(m - 1, -1, -1):
                orig_idx = sorted_indices[rank_idx]
                rank = rank_idx + 1
                raw_p = raw_test_records[orig_idx]["p_val"]
                val = min(1.0, (m / rank) * raw_p)
                cum_min = min(cum_min, val)
                q_values[orig_idx] = cum_min

        results: list[IndependenceTestResult] = []
        violations: list[IndependenceTestResult] = []

        for idx, rec in enumerate(raw_test_records):
            vx = rec["vx"]
            vy = rec["vy"]
            z_set = rec["z_set"]
            pred_indep = rec["pred_indep"]
            stat = rec["stat"]
            p_val = rec["p_val"]
            df = rec["df"]
            n = rec["n"]
            test_type = rec["test_type"]
            adj_p = q_values[idx] if apply_fdr else None

            effective_p = adj_p if apply_fdr else p_val
            emp_indep = effective_p >= self.alpha
            is_consistent = (pred_indep == emp_indep)

            violation_msg = None
            if not is_consistent:
                if pred_indep and not emp_indep:
                    violation_msg = (
                        f"DAG predicts independence for ({vx} _||_ {vy} | {z_set}), "
                        f"but observational data shows statistically significant dependency "
                        f"(statistic = {stat:.4f}, p = {p_val:.5f}, q_fdr = {effective_p:.5f} < {self.alpha})."
                    )
                else:
                    violation_msg = (
                        f"DAG posits direct mechanism ({vx} -> {vy}), "
                        f"but observational data shows no significant correlation "
                        f"(statistic = {stat:.4f}, p = {p_val:.5f}, q_fdr = {effective_p:.5f} >= {self.alpha})."
                    )

            res = IndependenceTestResult(
                var_x=vx,
                var_y=vy,
                conditioning_set=z_set,
                test_type=test_type,
                statistic=stat,
                p_value=p_val,
                adjusted_p_value=adj_p,
                degrees_of_freedom=df,
                sample_size=n,
                predicted_independent=pred_indep,
                empirically_independent=emp_indep,
                is_consistent=is_consistent,
                violation_detail=violation_msg,
            )
            results.append(res)
            if not is_consistent:
                violations.append(res)

        total = len(results)
        failed = len(violations)
        passed = total - failed
        is_valid = (failed == 0)

        fdr_note = " (FDR corrected)" if apply_fdr else ""
        summary = (
            f"Verified {total} causal assumptions across '{scm.name}' against {len(data)} records{fdr_note}. "
            f"Passed: {passed}, Violations: {failed} (alpha = {self.alpha})."
        )

        return CausalVerificationReport(
            scm_name=scm.name,
            is_valid=is_valid,
            alpha=self.alpha,
            fdr_applied=apply_fdr,
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            violations=violations,
            all_results=results,
            summary=summary,
        )

    verify_dag_against_data = verify_dataset
