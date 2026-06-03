"""Unit tests for the heterogeneous-treatment-effects stage (CausalForestDML).

A light forest config (few trees) is used to keep the suite fast; the
heterogeneity signal in the custom fixtures is strong enough to be recovered.
"""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_experiment import generate_experiment
from pipeline.hte import plot_cate_distribution, run_hte, select_hte_covariates
from pipeline.ingest import run_ingest

# Light hyperparameters shared across tests for speed.
FAST = dict(n_estimators=60, min_samples_leaf=15, random_state=0)


def _strong_hte_df(n: int = 1500, seed: int = 0) -> pd.DataFrame:
    """A dataset where the treatment effect is driven almost entirely by `driver`."""
    rng = np.random.default_rng(seed)
    driver = rng.normal(0, 1, size=n)
    noise_cov = rng.normal(0, 1, size=n)  # irrelevant covariate
    treatment = rng.integers(0, 2, size=n)
    # Effect is large and positive when driver>0, ~0 otherwise.
    effect = np.where(driver > 0, 2.0, 0.0)
    metric = 0.5 * noise_cov + treatment * effect + rng.normal(0, 0.3, size=n)
    return pd.DataFrame(
        {
            "user_id": [str(i) for i in range(n)],
            "treatment": treatment,
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
            "metric": metric,
            "driver": driver,
            "noise_cov": noise_cov,
        }
    )


class TestCovariateSelection:
    def test_picks_numeric_covariates(self) -> None:
        df = _strong_hte_df(200)
        cols = select_hte_covariates(df)
        assert set(cols) == {"driver", "noise_cov"}


class TestRecovery:
    def test_identifies_true_driver(self) -> None:
        df = _strong_hte_df()
        scored, report = run_hte(df, min_per_arm=100, **FAST)
        assert report["status"] in ("ok", "warn")
        # The dominant heterogeneity driver should be `driver`, not the noise.
        assert report["heterogeneity_drivers"][0]["covariate"] == "driver"

    def test_cate_separates_by_driver_sign(self) -> None:
        df = _strong_hte_df()
        scored, _ = run_hte(df, min_per_arm=100, **FAST)
        hi = scored.loc[scored["driver"] > 0, "cate"].mean()
        lo = scored.loc[scored["driver"] <= 0, "cate"].mean()
        assert hi > lo + 0.5  # clear separation

    def test_cate_column_added(self) -> None:
        df = _strong_hte_df(400)
        scored, _ = run_hte(df, min_per_arm=100, **FAST)
        assert "cate" in scored.columns
        assert scored["cate"].notna().sum() == len(df)


class TestRealScenarios:
    def test_heterogeneous_scenario_directional(self) -> None:
        # In heterogeneous_effects only the young (<30) get the effect.
        clean, _ = run_ingest(
            generate_experiment(n_users=2000, seed=42, scenario="heterogeneous_effects")
        )
        scored, report = run_hte(clean, min_per_arm=100, **FAST)
        young = scored["age_years"] < 30
        assert scored.loc[young, "cate"].mean() > scored.loc[~young, "cate"].mean()

    def test_no_effect_cate_near_zero(self) -> None:
        clean, _ = run_ingest(
            generate_experiment(n_users=2000, seed=42, scenario="no_effect")
        )
        scored, report = run_hte(clean, min_per_arm=100, **FAST)
        assert abs(report["ate"]) < 0.05


class TestReliabilityFlag:
    def test_small_sample_flagged_unreliable(self) -> None:
        df = _strong_hte_df(400)
        _, report = run_hte(df, min_per_arm=500, **FAST)
        # ~200 per arm < 500 -> unreliable warning.
        assert report["reliable"] is False
        assert report["status"] == "warn"

    def test_large_sample_reliable(self) -> None:
        df = _strong_hte_df(1500)
        _, report = run_hte(df, min_per_arm=500, **FAST)
        assert report["reliable"] is True
        assert report["status"] == "ok"


class TestSkipPaths:
    def test_no_covariates_skips(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": [str(i) for i in range(50)],
                "treatment": [0, 1] * 25,
                "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
                "metric": np.random.default_rng(0).normal(size=50),
            }
        )
        scored, report = run_hte(df, **FAST)
        assert report["status"] == "skip"
        assert report["covariates_used"] == []

    def test_too_few_rows_skips(self) -> None:
        df = _strong_hte_df(30)
        _, report = run_hte(df, min_samples_leaf=15, n_estimators=60)
        assert report["status"] == "skip"


class TestPlot:
    def test_plot_returns_figure_with_data(self) -> None:
        df = _strong_hte_df(400)
        scored, report = run_hte(df, min_per_arm=100, **FAST)
        fig = plot_cate_distribution(scored, report)
        assert len(fig.data) >= 1
