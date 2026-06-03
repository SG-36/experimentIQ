"""Unit tests for the sequential-testing stage (mSPRT, always-valid p-values)."""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_experiment import generate_experiment
from pipeline.ingest import run_ingest
from pipeline.sequential import run_sequential_test


def _ingested(scenario: str, n_users: int = 2000, seed: int = 42) -> pd.DataFrame:
    df = generate_experiment(n_users=n_users, seed=seed, scenario=scenario)
    clean, _ = run_ingest(df)
    return clean


class TestAlwaysValidProperties:
    def test_p_values_monotone_non_increasing(self) -> None:
        r = run_sequential_test(_ingested("clean_experiment"))
        p = np.array(r["p_values"])
        assert np.all(np.diff(p) <= 1e-12)

    def test_p_values_in_unit_interval(self) -> None:
        r = run_sequential_test(_ingested("clean_experiment"))
        p = np.array(r["p_values"])
        assert p.min() >= 0.0 and p.max() <= 1.0

    def test_series_lengths_match(self) -> None:
        r = run_sequential_test(_ingested("clean_experiment"))
        n = len(r["sample_sizes"])
        assert len(r["effect"]) == n
        assert len(r["log_lambda"]) == n
        assert len(r["p_values"]) == n


class TestDecisions:
    def test_real_effect_crosses_and_rejects(self) -> None:
        r = run_sequential_test(_ingested("clean_experiment"), alpha=0.05)
        assert r["crossed"] is True
        assert r["decision"] == "reject_h0"
        assert r["final_p_value"] < 0.05
        assert 0 < r["crossing_sample_size"] <= r["n_total"]

    def test_no_effect_does_not_cross(self) -> None:
        r = run_sequential_test(_ingested("no_effect"), alpha=0.05)
        assert r["crossed"] is False
        assert r["decision"] == "no_decision"
        assert r["final_p_value"] > 0.05

    def test_strong_effect_stops_early(self) -> None:
        # A large, obvious effect should cross well before the full sample.
        rng = np.random.default_rng(0)
        n = 4000
        treatment = rng.integers(0, 2, size=n)
        metric = treatment * 1.0 + rng.normal(0, 1.0, size=n)  # ~1 sigma effect
        df = pd.DataFrame(
            {
                "user_id": [str(i) for i in range(n)],
                "treatment": treatment,
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="min"),
                "metric": metric,
            }
        )
        r = run_sequential_test(df, alpha=0.05)
        assert r["crossed"] is True
        assert r["crossing_sample_size"] < n  # stopped before exhausting data


class TestParameters:
    def test_target_sample_size_reported(self) -> None:
        r = run_sequential_test(_ingested("clean_experiment"), target_sample_size=1000)
        assert "decision_at_target" in r
        dt = r["decision_at_target"]
        assert dt["sample_size"] >= 1000
        assert isinstance(dt["significant"], bool)

    def test_smaller_alpha_crosses_later_or_equal(self) -> None:
        clean = _ingested("clean_experiment")
        loose = run_sequential_test(clean, alpha=0.10)
        strict = run_sequential_test(clean, alpha=0.01)
        # A stricter boundary can only require at least as much data to cross.
        assert strict["crossing_sample_size"] >= loose["crossing_sample_size"]

    def test_runs_on_cuped_column(self) -> None:
        clean = _ingested("clean_experiment").copy()
        clean["metric_cuped"] = clean["metric"]  # stand-in column
        r = run_sequential_test(clean, metric_col="metric_cuped")
        assert r["status"] == "ok"
        assert r["metric_col"] == "metric_cuped"


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
        r = run_sequential_test(df)
        assert r["status"] == "skip"
        assert r["crossed"] is False
