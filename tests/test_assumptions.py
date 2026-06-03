"""Unit tests for the assumption-testing stage (SRM, SUTVA, novelty)."""

import numpy as np
import pandas as pd
import pytest

from data.synthetic_experiment import generate_experiment
from pipeline.assumptions import (
    check_novelty,
    check_srm,
    check_sutva,
    run_assumption_tests,
)
from pipeline.ingest import run_ingest


def _ingested(scenario: str, n_users: int = 2000, seed: int = 42) -> pd.DataFrame:
    df = generate_experiment(n_users=n_users, seed=seed, scenario=scenario)
    clean, _ = run_ingest(df)
    return clean


class TestSRM:
    def test_clean_experiment_passes(self) -> None:
        res = check_srm(_ingested("clean_experiment"))
        assert res["status"] == "pass"
        assert res["p_value"] >= res["alpha"]

    def test_srm_injected_fails(self) -> None:
        # srm_injected assigns to treatment at ratio 0.6 but we expect 0.5.
        res = check_srm(_ingested("srm_injected"), expected_treatment_ratio=0.5)
        assert res["status"] == "fail"
        assert res["p_value"] < res["alpha"]
        assert res["observed_treatment_ratio"] > 0.55

    def test_correct_expected_ratio_passes(self) -> None:
        # If we tell SRM the true 0.6 split, it should no longer fire.
        res = check_srm(_ingested("srm_injected"), expected_treatment_ratio=0.6)
        assert res["status"] == "pass"

    def test_invalid_ratio_skips(self) -> None:
        res = check_srm(_ingested("clean_experiment"), expected_treatment_ratio=0.0)
        assert res["status"] == "skip"


class TestSUTVA:
    def test_clean_data_passes(self) -> None:
        res = check_sutva(_ingested("clean_experiment"))
        assert res["status"] == "pass"
        assert res["n_crossover_users"] == 0

    def test_crossover_user_flagged(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c"],
                "treatment": [0, 1, 0, 1],
                "timestamp": pd.date_range("2024-01-01", periods=4),
                "metric": [1.0, 2.0, 3.0, 4.0],
            }
        )
        res = check_sutva(df)
        assert res["status"] == "fail"
        assert res["n_crossover_users"] == 1
        assert "a" in res["crossover_user_ids"]

    def test_run_uses_raw_df_for_sutva(self) -> None:
        # A crossover user in raw data is removed by ingest; SUTVA must see raw.
        raw = pd.DataFrame(
            {
                "user_id": ["a", "a", "b", "c", "d"],
                "treatment": [0, 1, 0, 1, 0],
                "timestamp": pd.date_range("2024-01-01", periods=5),
                "metric": [1.0, 2.0, 3.0, 4.0, 5.0],
            }
        )
        clean, _ = run_ingest(raw)
        res = run_assumption_tests(clean, raw_df=raw)
        assert res["sutva"]["ran_on_raw_data"] is True
        assert res["sutva"]["status"] == "fail"
        # Without raw_df the deduplicated frame hides the crossover.
        res_no_raw = run_assumption_tests(clean)
        assert res_no_raw["sutva"]["status"] == "pass"
        assert res_no_raw["sutva"]["ran_on_raw_data"] is False


class TestNovelty:
    def test_clean_experiment_passes(self) -> None:
        res = check_novelty(_ingested("clean_experiment"))
        assert res["status"] == "pass"

    def test_no_effect_passes(self) -> None:
        res = check_novelty(_ingested("no_effect"))
        assert res["status"] == "pass"

    def test_novelty_scenario_fails_with_decay(self) -> None:
        res = check_novelty(_ingested("novelty_effect"))
        assert res["status"] == "fail"
        # Novelty effect: early-cohort effect should exceed late-cohort effect.
        assert res["effect_early"] > res["effect_late"]
        assert res["p_value"] < res["alpha"]

    def test_empty_skips(self) -> None:
        df = pd.DataFrame(
            {"user_id": [], "treatment": [], "timestamp": [], "metric": []}
        )
        res = check_novelty(df)
        assert res["status"] == "skip"


class TestRunAssumptionTests:
    def test_clean_overall_pass(self) -> None:
        clean = _ingested("clean_experiment")
        res = run_assumption_tests(clean)
        assert res["overall"] == "pass"
        assert res["n_failed"] == 0
        for key in ("srm", "sutva", "novelty"):
            assert key in res

    def test_srm_scenario_overall_fail(self) -> None:
        clean = _ingested("srm_injected")
        res = run_assumption_tests(clean, expected_treatment_ratio=0.5)
        assert res["overall"] == "fail"
        assert res["srm"]["status"] == "fail"
        assert res["n_failed"] >= 1

    def test_every_check_has_explanation_and_recommendation(self) -> None:
        res = run_assumption_tests(_ingested("clean_experiment"))
        for key in ("srm", "sutva", "novelty"):
            assert res[key]["explanation"]
            assert res[key]["recommendation"]
