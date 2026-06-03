"""
Data ingestion and validation.

Loads experiment data, validates the schema, and checks for nulls, duplicate
user_ids, and invalid treatment values. Returns a cleaned DataFrame alongside
a validation report.

Cleaning decisions (all are reported in the validation report):
  * Rows with nulls in any required column are dropped (unusable).
  * Rows with a treatment value other than 0 or 1 are dropped (unusable).
  * Duplicate user_ids are resolved by keeping the earliest row per user
    (first assignment wins); no aggregation of metric or treatment is done.

No outlier handling, imputation, or variance/assumption testing happens here.
Those belong to later stages of the pipeline.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

REQUIRED_COLUMNS = ["user_id", "treatment", "timestamp", "metric"]
VALID_TREATMENT_VALUES = (0, 1)


def get_covariate_columns(df: pd.DataFrame) -> list[str]:
    """Return the optional covariate columns (everything not required)."""
    return [c for c in df.columns if c not in REQUIRED_COLUMNS]


def validate_schema(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Check that required columns exist and coerce dtypes.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment data.
    column_mapping : dict, optional
        Maps canonical names (user_id, treatment, timestamp, metric) to the
        actual column names in ``df``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        A copy with canonical column names and coerced dtypes, plus the
        schema section of the validation report. When required columns are
        missing the frame is returned unchanged with ``status == "fail"``.
    """
    work = df.copy()

    if column_mapping:
        # Map actual column names -> canonical names.
        inv = {actual: canonical for canonical, actual in column_mapping.items()}
        work = work.rename(columns=inv)

    missing = [c for c in REQUIRED_COLUMNS if c not in work.columns]
    if missing:
        return work, {
            "status": "fail",
            "missing_columns": missing,
            "message": f"Missing required columns: {missing}",
        }

    work["user_id"] = work["user_id"].astype(str)
    work["treatment"] = pd.to_numeric(work["treatment"], errors="coerce")
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
    work["metric"] = pd.to_numeric(work["metric"], errors="coerce")

    return work, {
        "status": "pass",
        "missing_columns": [],
        "message": "All required columns present.",
    }


def check_invalid_treatment(df: pd.DataFrame) -> dict[str, Any]:
    """Report non-null treatment values that are not 0 or 1."""
    treatment = df["treatment"]
    invalid_mask = treatment.notna() & ~treatment.isin(VALID_TREATMENT_VALUES)
    n_invalid = int(invalid_mask.sum())
    invalid_values = sorted(treatment[invalid_mask].unique().tolist())
    return {
        "status": "warn" if n_invalid else "pass",
        "n_invalid_rows": n_invalid,
        "invalid_values": invalid_values,
        "message": (
            f"{n_invalid} row(s) had treatment values outside {{0, 1}}: "
            f"{invalid_values}. These rows are dropped."
            if n_invalid
            else "All treatment values are 0 or 1."
        ),
    }


def check_nulls(df: pd.DataFrame) -> dict[str, Any]:
    """Report nulls in required columns."""
    null_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    n_rows_with_nulls = int(df[REQUIRED_COLUMNS].isna().any(axis=1).sum())
    return {
        "status": "warn" if n_rows_with_nulls else "pass",
        "n_rows_with_nulls": n_rows_with_nulls,
        "null_counts": null_counts,
        "message": (
            f"{n_rows_with_nulls} row(s) had nulls in required columns "
            f"({null_counts}). These rows are dropped."
            if n_rows_with_nulls
            else "No nulls in required columns."
        ),
    }


def check_duplicate_users(df: pd.DataFrame) -> dict[str, Any]:
    """Report how many user_ids appear more than once."""
    n_rows = len(df)
    n_unique = df["user_id"].nunique()
    n_duplicate_rows = n_rows - n_unique
    n_affected_users = int((df["user_id"].value_counts() > 1).sum())
    return {
        "status": "warn" if n_duplicate_rows else "pass",
        "n_duplicate_rows": n_duplicate_rows,
        "n_affected_users": n_affected_users,
        "message": (
            f"{n_affected_users} user_id(s) appeared more than once "
            f"({n_duplicate_rows} extra row(s)). Earliest row per user is kept."
            if n_duplicate_rows
            else "All user_ids are unique."
        ),
    }


def run_ingest(
    df: pd.DataFrame,
    column_mapping: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Validate and clean raw experiment data.

    Runs schema validation, null detection, invalid-treatment detection, and
    duplicate-user detection, then produces a cleaned DataFrame:
      1. drop rows with nulls in required columns,
      2. drop rows with invalid treatment values,
      3. keep the earliest row per duplicated user_id.

    Parameters
    ----------
    df : pd.DataFrame
        Raw experiment data.
    column_mapping : dict, optional
        Maps canonical column names to the actual column names in ``df``.

    Returns
    -------
    tuple[pd.DataFrame, dict]
        The cleaned DataFrame and the validation report. If the schema check
        fails, the (uncleaned) frame is returned with a report whose
        ``overall`` status is ``"fail"``.
    """
    report: dict[str, Any] = {}

    work, schema = validate_schema(df, column_mapping)
    report["schema"] = schema
    if schema["status"] == "fail":
        report["overall"] = "fail"
        report["rows_in"] = len(df)
        report["rows_out"] = 0
        return work, report

    report["nulls"] = check_nulls(work)
    report["invalid_treatment"] = check_invalid_treatment(work)

    rows_in = len(work)

    null_mask = work[REQUIRED_COLUMNS].isna().any(axis=1)
    invalid_mask = work["treatment"].notna() & ~work["treatment"].isin(VALID_TREATMENT_VALUES)
    clean = work.loc[~(null_mask | invalid_mask)].copy()

    report["duplicates"] = check_duplicate_users(clean)
    clean = (
        clean.sort_values("timestamp", kind="stable")
        .drop_duplicates(subset="user_id", keep="first")
        .reset_index(drop=True)
    )

    clean["treatment"] = clean["treatment"].astype(int)

    statuses = [
        report["schema"]["status"],
        report["nulls"]["status"],
        report["invalid_treatment"]["status"],
        report["duplicates"]["status"],
    ]
    if "fail" in statuses or clean.empty:
        report["overall"] = "fail"
    elif "warn" in statuses:
        report["overall"] = "warn"
    else:
        report["overall"] = "pass"

    report["rows_in"] = rows_in
    report["rows_out"] = len(clean)
    report["n_users"] = len(clean)
    report["covariate_columns"] = get_covariate_columns(clean)

    return clean, report
