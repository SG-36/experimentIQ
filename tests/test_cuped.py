"""Unit tests for the CUPED variance-reduction stage."""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_experiment import generate_experiment
from pipeline.cuped import run_cuped, select_cuped_covariates
from pipeline.ingest import run_ingest


def _ingested(scenario: str, n_users: int = 2000, seed: int = 42) -> pd.DataFrame:
    df = generate_experiment(n_users=n_users, seed=seed, scenario=scenario)
    clean, _ = run_ingest(df)
    return clean


class TestCovariateSelection:
    def test_default_picks_numeric_covariates(self) -> None:
        cols = select_cuped_covariates(_ingested("clean_experiment"))
        assert "prior_sessions" in cols
        assert "account_age_days" in cols
        assert "treatment" not in cols and "metric" not in cols

    def test_constant_column_dropped(self) -> None:
        df = _ingested("clean_experiment").copy()
        df["const_cov"] = 7.0
        cols = select_cuped_covariates(df)
        assert "const_cov" not in cols

    def test_explicit_list_respected(self) -> None:
        cols = select_cuped_covariates(
            _ingested("clean_experiment"), covariates=["prior_sessions"]
        )
        assert cols == ["prior_sessions"]


class TestApplyCuped:
    def test_variance_is_reduced(self) -> None:
        clean = _ingested("clean_experiment")
        adjusted, report = run_cuped(clean)
        assert report["status"] == "applied"
        assert report["variance_cuped"] < report["variance_original"]
        assert 0.0 <= report["variance_reduction"] <= 1.0
        assert report["variance_reduction"] > 0.0

    def test_ate_is_preserved(self) -> None:
        clean = _ingested("clean_experiment")
        _, report = run_cuped(clean)
        # CUPED must not move the treatment effect by more than noise.
        assert abs(report["ate_cuped"] - report["ate_original"]) < 0.02

    def test_mean_preserved(self) -> None:
        clean = _ingested("clean_experiment")
        adjusted, _ = run_cuped(clean)
        assert adjusted["metric_cuped"].mean() == pytest.approx(
            adjusted["metric"].mean(), abs=1e-6
        )

    def test_original_metric_untouched(self) -> None:
        clean = _ingested("clean_experiment")
        before = clean["metric"].copy()
        adjusted, _ = run_cuped(clean)
        assert "metric_cuped" in adjusted.columns
        pd.testing.assert_series_equal(adjusted["metric"], before, check_names=False)

    def test_theta_keys_match_covariates(self) -> None:
        clean = _ingested("clean_experiment")
        _, report = run_cuped(clean)
        assert set(report["theta"].keys()) == set(report["covariates_used"])

    def test_strong_covariate_gives_large_reduction(self) -> None:
        # Construct a covariate that explains most of the metric variance.
        rng = np.random.default_rng(0)
        n = 1000
        x = rng.normal(0, 1, size=n)
        metric = 5.0 * x + rng.normal(0, 0.5, size=n)  # x dominates variance
        df = pd.DataFrame(
            {
                "user_id": [str(i) for i in range(n)],
                "treatment": rng.integers(0, 2, size=n),
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
                "metric": metric,
                "pre_metric": x,
            }
        )
        _, report = run_cuped(df, covariates=["pre_metric"])
        assert report["variance_reduction"] > 0.8


class TestSkipPath:
    def test_no_covariates_skips(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d"],
                "treatment": [0, 1, 0, 1],
                "timestamp": pd.date_range("2024-01-01", periods=4),
                "metric": [1.0, 2.0, 3.0, 4.0],
            }
        )
        adjusted, report = run_cuped(df)
        assert report["status"] == "skipped"
        assert report["covariates_used"] == []
        # Skip path still yields a usable, unchanged metric_cuped column.
        pd.testing.assert_series_equal(
            adjusted["metric_cuped"], adjusted["metric"], check_names=False
        )


class TestMissingCovariates:
    def test_incomplete_rows_fall_back_to_original(self) -> None:
        clean = _ingested("clean_experiment").copy()
        clean.loc[clean.index[:50], "prior_sessions"] = np.nan
        adjusted, report = run_cuped(clean, covariates=["prior_sessions"])
        assert report["status"] == "applied"
        assert report["n_unadjusted"] == 50
        # The 50 incomplete rows keep their original metric value.
        incomplete = adjusted.loc[clean["prior_sessions"].isna()]
        assert (incomplete["metric_cuped"] == incomplete["metric"]).all()
