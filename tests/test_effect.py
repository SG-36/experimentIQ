"""Unit tests for fixed-horizon treatment-effect estimation."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from data.synthetic_experiment import generate_experiment
from pipeline.effect import estimate_effect
from pipeline.ingest import run_ingest


def _ingested(scenario: str, n_users: int = 2000, seed: int = 42) -> pd.DataFrame:
    df = generate_experiment(n_users=n_users, seed=seed, scenario=scenario)
    clean, _ = run_ingest(df)
    return clean


class TestContinuous:
    def test_matches_scipy_welch(self) -> None:
        clean = _ingested("clean_experiment")
        res = estimate_effect(clean)
        t = clean.loc[clean["treatment"] == 1, "metric"]
        c = clean.loc[clean["treatment"] == 0, "metric"]
        sp = stats.ttest_ind(t, c, equal_var=False)
        assert res["method"] == "welch_t"
        assert res["p_value"] == pytest.approx(float(sp.pvalue), rel=1e-6)
        assert res["test_statistic"] == pytest.approx(float(sp.statistic), rel=1e-6)

    def test_ci_brackets_point_estimate(self) -> None:
        res = estimate_effect(_ingested("clean_experiment"))
        assert res["ci_low"] < res["effect"] < res["ci_high"]

    def test_significant_on_real_effect(self) -> None:
        res = estimate_effect(_ingested("clean_experiment"))
        assert res["significant"] is True
        assert res["ci_low"] > 0  # whole CI above zero

    def test_significance_matches_ci_excluding_zero(self) -> None:
        # The decision must be internally consistent: significant iff the CI
        # excludes zero. (We don't assert a particular outcome on no_effect:
        # a single realization can be a false positive within the 5% rate.)
        res = estimate_effect(_ingested("no_effect"))
        ci_excludes_zero = not (res["ci_low"] <= 0 <= res["ci_high"])
        assert res["significant"] == ci_excludes_zero

    def test_identical_arms_not_significant(self) -> None:
        # Deterministically null: both arms have the same values -> p == 1.
        vals = list(np.linspace(0, 10, 100))
        df = pd.DataFrame(
            {
                "user_id": [str(i) for i in range(200)],
                "treatment": [0] * 100 + [1] * 100,
                "timestamp": pd.date_range("2024-01-01", periods=200, freq="h"),
                "metric": vals + vals,
            }
        )
        res = estimate_effect(df)
        assert res["effect"] == pytest.approx(0.0)
        assert res["significant"] is False
        assert res["ci_low"] < 0 < res["ci_high"]

    def test_relative_lift_computed(self) -> None:
        res = estimate_effect(_ingested("clean_experiment"))
        expected = res["effect"] / res["mean_control"]
        assert res["relative_lift"] == pytest.approx(expected)


class TestBinary:
    def _binary_df(self, p_c: float, p_t: float, n: int = 4000, seed: int = 1) -> pd.DataFrame:
        rng = np.random.default_rng(seed)
        treat = rng.integers(0, 2, size=n)
        metric = np.where(
            treat == 1,
            rng.random(n) < p_t,
            rng.random(n) < p_c,
        ).astype(int)
        return pd.DataFrame(
            {
                "user_id": [str(i) for i in range(n)],
                "treatment": treat,
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
                "metric": metric,
            }
        )

    def test_detects_binary_and_uses_proportion_test(self) -> None:
        res = estimate_effect(self._binary_df(0.10, 0.13))
        assert res["metric_type"] == "binary"
        assert res["method"] == "two_proportion_z"

    def test_matches_manual_pooled_z(self) -> None:
        df = self._binary_df(0.10, 0.14)
        res = estimate_effect(df)
        t = df.loc[df["treatment"] == 1, "metric"]
        c = df.loc[df["treatment"] == 0, "metric"]
        n_t, n_c = len(t), len(c)
        p_t, p_c = t.mean(), c.mean()
        p_pool = (t.sum() + c.sum()) / (n_t + n_c)
        se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_t + 1 / n_c))
        z = (p_t - p_c) / se
        p = 2 * stats.norm.sf(abs(z))
        assert res["p_value"] == pytest.approx(float(p), rel=1e-6)
        assert res["test_statistic"] == pytest.approx(float(z), rel=1e-6)

    def test_detects_real_binary_lift(self) -> None:
        res = estimate_effect(self._binary_df(0.10, 0.16, n=6000))
        assert res["significant"] is True
        assert res["effect"] > 0


class TestCupedColumn:
    def test_runs_on_cuped_column(self) -> None:
        clean = _ingested("clean_experiment").copy()
        clean["metric_cuped"] = clean["metric"]
        res = estimate_effect(clean, metric_col="metric_cuped")
        assert res["status"] == "ok"
        assert res["metric_col"] == "metric_cuped"


class TestSkip:
    def test_insufficient_data_skips(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["a", "b"],
                "treatment": [0, 1],
                "timestamp": pd.date_range("2024-01-01", periods=2),
                "metric": [1.0, 2.0],
            }
        )
        res = estimate_effect(df)
        assert res["status"] == "skip"
