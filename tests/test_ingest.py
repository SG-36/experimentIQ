"""Unit tests for data ingestion and validation."""

import pandas as pd
import pytest

from data.synthetic_experiment import generate_experiment
from pipeline.ingest import (
    check_duplicate_users,
    check_invalid_treatment,
    check_nulls,
    encode_treatment,
    run_ingest,
    validate_schema,
)


@pytest.fixture
def fixture_clean() -> pd.DataFrame:
    return generate_experiment(n_users=2000, seed=42, scenario="clean_experiment")


def _base_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "user_id": ["a", "b", "c", "d"],
            "treatment": [0, 1, 0, 1],
            "timestamp": pd.date_range("2024-01-01", periods=4),
            "metric": [1.0, 2.0, 3.0, 4.0],
        }
    )


class TestSchema:
    def test_clean_schema_passes(self, fixture_clean: pd.DataFrame) -> None:
        cleaned, report = run_ingest(fixture_clean)
        assert report["schema"]["status"] == "pass"
        assert report["overall"] in ("pass", "warn")
        assert cleaned["user_id"].is_unique

    def test_missing_required_column_fails(self) -> None:
        df = _base_df().drop(columns=["metric"])
        cleaned, report = run_ingest(df)
        assert report["schema"]["status"] == "fail"
        assert report["overall"] == "fail"
        assert "metric" in report["schema"]["missing_columns"]

    def test_column_mapping_renames(self) -> None:
        df = _base_df().rename(columns={"user_id": "uid", "metric": "y"})
        cleaned, report = run_ingest(df, column_mapping={"user_id": "uid", "metric": "y"})
        assert report["schema"]["status"] == "pass"
        assert "user_id" in cleaned.columns
        assert "metric" in cleaned.columns

    def test_dtypes_coerced(self) -> None:
        df = _base_df()
        df["treatment"] = df["treatment"].astype(str)
        cleaned, _ = run_ingest(df)
        assert cleaned["treatment"].dtype.kind in ("i", "u")
        assert pd.api.types.is_datetime64_any_dtype(cleaned["timestamp"])


class TestNulls:
    def test_null_rows_reported_and_dropped(self) -> None:
        df = _base_df()
        df.loc[1, "metric"] = None
        cleaned, report = run_ingest(df)
        assert report["nulls"]["status"] == "warn"
        assert report["nulls"]["n_rows_with_nulls"] == 1
        assert len(cleaned) == 3
        assert not cleaned["metric"].isna().any()

    def test_no_nulls_passes(self) -> None:
        report = check_nulls(_base_df())
        assert report["status"] == "pass"
        assert report["n_rows_with_nulls"] == 0


class TestInvalidTreatment:
    def test_invalid_treatment_dropped(self) -> None:
        df = _base_df()
        df.loc[2, "treatment"] = 5
        cleaned, report = run_ingest(df)
        assert report["invalid_treatment"]["status"] == "warn"
        assert report["invalid_treatment"]["n_invalid_rows"] == 1
        assert 5 in report["invalid_treatment"]["invalid_values"]
        assert set(cleaned["treatment"].unique()).issubset({0, 1})

    def test_valid_treatment_passes(self) -> None:
        report = check_invalid_treatment(_base_df())
        assert report["status"] == "pass"
        assert report["n_invalid_rows"] == 0


class TestDuplicates:
    def test_duplicate_user_ids_resolved_by_earliest(self) -> None:
        df = _base_df()
        dup = pd.DataFrame(
            {
                "user_id": ["a"],
                "treatment": [1],
                "timestamp": pd.to_datetime(["2024-02-01"]),
                "metric": [99.0],
            }
        )
        df = pd.concat([df, dup], ignore_index=True)
        cleaned, report = run_ingest(df)
        assert report["duplicates"]["status"] == "warn"
        assert report["duplicates"]["n_duplicate_rows"] == 1
        assert cleaned["user_id"].is_unique
        # Earliest row (2024-01-01, metric 1.0) is kept, not the later dup.
        kept = cleaned.loc[cleaned["user_id"] == "a"].iloc[0]
        assert kept["metric"] == 1.0

    def test_no_duplicates_passes(self) -> None:
        report = check_duplicate_users(_base_df())
        assert report["status"] == "pass"
        assert report["n_duplicate_rows"] == 0


class TestReportShape:
    def test_report_keys_present(self, fixture_clean: pd.DataFrame) -> None:
        _, report = run_ingest(fixture_clean)
        for key in ("schema", "nulls", "invalid_treatment", "duplicates", "overall", "rows_in", "rows_out"):
            assert key in report

    def test_covariates_preserved(self, fixture_clean: pd.DataFrame) -> None:
        cleaned, report = run_ingest(fixture_clean)
        for col in ("prior_sessions", "account_age_days", "age_years"):
            assert col in cleaned.columns
            assert col in report["covariate_columns"]

    def test_rows_in_out_consistent(self) -> None:
        df = generate_experiment(n_users=300, seed=3, scenario="clean_experiment")
        cleaned, report = run_ingest(df)
        assert report["rows_in"] == len(df)
        assert report["rows_out"] == len(cleaned)


class TestSchemaHelper:
    def test_validate_schema_missing(self) -> None:
        df = _base_df().drop(columns=["treatment"])
        _, schema = validate_schema(df)
        assert schema["status"] == "fail"
        assert schema["missing_columns"] == ["treatment"]


class TestTreatmentEncoding:
    def test_numeric_0_1_passthrough(self) -> None:
        enc, rep = encode_treatment(pd.Series([0, 1, 1, 0]))
        assert rep["method"] == "numeric_0_1"
        assert enc.tolist() == [0, 1, 1, 0]

    def test_control_treatment_labels(self) -> None:
        df = _base_df()
        df["treatment"] = ["control", "treatment", "control", "treatment"]
        cleaned, report = run_ingest(df)
        assert report["treatment_encoding"]["method"] == "auto_label"
        assert set(cleaned["treatment"].unique()) == {0, 1}
        assert len(cleaned) == 4

    def test_a_b_labels(self) -> None:
        enc, rep = encode_treatment(pd.Series(["A", "B", "B", "A"]))
        assert rep["method"] == "auto_label"
        assert enc.tolist() == [0, 1, 1, 0]

    def test_boolean_treatment(self) -> None:
        enc, rep = encode_treatment(pd.Series([True, False, True, False]))
        assert rep["method"] == "boolean"
        assert enc.tolist() == [1, 0, 1, 0]

    def test_explicit_mapping_takes_precedence(self) -> None:
        df = _base_df()
        df["treatment"] = ["grp_x", "grp_y", "grp_x", "grp_y"]
        cleaned, report = run_ingest(df, treatment_mapping={"grp_x": 0, "grp_y": 1})
        assert report["treatment_encoding"]["method"] == "explicit_mapping"
        assert set(cleaned["treatment"].unique()) == {0, 1}

    def test_unrecognized_labels_fail_with_guidance(self) -> None:
        df = _base_df()
        df["treatment"] = ["red", "blue", "red", "blue"]
        cleaned, report = run_ingest(df)
        enc = report["treatment_encoding"]
        assert enc["status"] == "fail"
        assert report["overall"] == "fail"
        assert "treatment_mapping" in enc["message"]

    def test_more_than_two_arms_flagged(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c"],
                "treatment": ["control", "variant_b", "variant_c"],
                "timestamp": pd.date_range("2024-01-01", periods=3),
                "metric": [1.0, 2.0, 3.0],
            }
        )
        _, report = run_ingest(df)
        enc = report["treatment_encoding"]
        assert enc["status"] == "fail"
        assert enc["n_distinct_arms"] == 3

    def test_select_two_of_three_arms_via_mapping(self) -> None:
        df = pd.DataFrame(
            {
                "user_id": ["a", "b", "c", "d"],
                "treatment": ["control", "variant_b", "variant_c", "control"],
                "timestamp": pd.date_range("2024-01-01", periods=4),
                "metric": [1.0, 2.0, 3.0, 4.0],
            }
        )
        cleaned, report = run_ingest(
            df, treatment_mapping={"control": 0, "variant_b": 1}
        )
        # variant_c is unmapped -> NaN -> dropped as a null row.
        assert set(cleaned["treatment"].unique()) == {0, 1}
        assert len(cleaned) == 3
