"""Unit tests for the EDA stage (numeric report; plotting not rendered here)."""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_experiment import generate_experiment
from pipeline.eda import (
    flag_anomalies,
    run_eda,
    summarize_group_sizes,
    summarize_metric,
    summarize_timeline,
)
from pipeline.ingest import run_ingest


def _ingested(scenario: str, n_users: int = 2000, seed: int = 42) -> pd.DataFrame:
    df = generate_experiment(n_users=n_users, seed=seed, scenario=scenario)
    clean, _ = run_ingest(df)
    return clean


class TestGroupSizes:
    def test_counts_sum_to_total(self) -> None:
        df = _ingested("clean_experiment")
        g = summarize_group_sizes(df)
        assert g["n_control"] + g["n_treatment"] == g["n_total"] == len(df)
        assert g["prop_control"] + g["prop_treatment"] == pytest.approx(1.0)

    def test_balanced_clean_experiment(self) -> None:
        g = summarize_group_sizes(_ingested("clean_experiment"))
        assert abs(g["prop_treatment"] - 0.5) < 0.05

    def test_srm_scenario_is_imbalanced(self) -> None:
        g = summarize_group_sizes(_ingested("srm_injected"))
        # srm_injected uses a 0.6 treatment ratio.
        assert g["prop_treatment"] > 0.55


class TestMetricSummary:
    def test_arms_and_overall_present(self) -> None:
        m = summarize_metric(_ingested("clean_experiment"))
        for arm in ("overall", "control", "treatment"):
            for key in ("n", "mean", "std", "min", "median", "max", "skew"):
                assert key in m[arm]

    def test_treatment_lifts_mean(self) -> None:
        # clean_experiment has a positive ATE, so treatment mean should exceed control.
        m = summarize_metric(_ingested("clean_experiment"))
        assert m["treatment"]["mean"] > m["control"]["mean"]

    def test_no_effect_means_close(self) -> None:
        m = summarize_metric(_ingested("no_effect"))
        assert abs(m["treatment"]["mean"] - m["control"]["mean"]) < 0.1


class TestTimeline:
    def test_span_and_active_days(self) -> None:
        t = summarize_timeline(_ingested("clean_experiment"))
        # generator staggers assignments across 21 days.
        assert t["day_span"] == 21
        assert t["n_active_days"] == 21
        assert sum(
            d["control"] + d["treatment"] for d in t["assignments_per_day"].values()
        ) == 2000


class TestAnomalies:
    def test_clean_experiment_has_no_high_severity(self) -> None:
        report = run_eda(_ingested("clean_experiment"))
        highs = [a for a in report["anomalies"] if a["severity"] == "high"]
        assert highs == []

    def test_empty_arm_flagged(self) -> None:
        df = pd.DataFrame({
            "user_id": [str(i) for i in range(50)],
            "treatment": [0] * 50,
            "timestamp": pd.date_range("2024-01-01", periods=50, freq="h"),
            "metric": np.random.default_rng(0).normal(size=50),
        })
        g = summarize_group_sizes(df)
        m = summarize_metric(df)
        t = summarize_timeline(df)
        codes = {a["code"] for a in flag_anomalies(df, g, m, t)}
        assert "empty_arm" in codes

    def test_constant_metric_flagged(self) -> None:
        df = pd.DataFrame({
            "user_id": [str(i) for i in range(40)],
            "treatment": [0, 1] * 20,
            "timestamp": pd.date_range("2024-01-01", periods=40, freq="h"),
            "metric": [3.0] * 40,
        })
        report = run_eda(df)
        codes = {a["code"] for a in report["anomalies"]}
        assert "constant_metric" in codes

    def test_point_mass_flagged(self) -> None:
        metric = [0.0] * 70 + list(np.random.default_rng(1).normal(5, 1, size=30))
        df = pd.DataFrame({
            "user_id": [str(i) for i in range(100)],
            "treatment": [0, 1] * 50,
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="h"),
            "metric": metric,
        })
        report = run_eda(df)
        codes = {a["code"] for a in report["anomalies"]}
        assert "point_mass" in codes


class TestReportShape:
    def test_run_eda_keys(self) -> None:
        report = run_eda(_ingested("clean_experiment"))
        for key in ("group_sizes", "metric", "timeline", "anomalies", "n_anomalies"):
            assert key in report
        assert report["n_anomalies"] == len(report["anomalies"])
