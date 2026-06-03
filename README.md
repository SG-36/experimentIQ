# ExperimentIQ

Open-source A/B test analysis platform that automates the data scientist workflow—from raw experiment data to assumption-checked causal analysis and a plain-language report.

## Status

**Data ingestion and validation** is implemented with unit tests and fixed synthetic fixtures. The remaining analysis stages are planned next.

## Quick start

```bash
cd experimentiq
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest tests/test_ingest.py -v
```

## Input format

CSV with required columns:

| Column | Description |
|--------|-------------|
| `user_id` | Unique user identifier |
| `treatment` | `0` = control, `1` = treatment |
| `timestamp` | Assignment time |
| `metric` | Outcome (e.g. retention, revenue) |

Optional: any number of pre-experiment covariate columns.

## Synthetic data

```bash
python -m data.synthetic_experiment --scenario clean_experiment --output /tmp/clean.csv
python scripts/generate_fixtures.py  # regenerate tests/fixtures/
```

Scenarios: `clean_experiment`, `srm_injected`, `novelty_effect`, `noncompliance`, `heterogeneous_effects`, `no_effect`.

## Ingestion API

```python
from pipeline.ingest import run_ingest
import pandas as pd

df = pd.read_csv("your_experiment.csv", parse_dates=["timestamp"])
cleaned, report = run_ingest(df)
```

`run_ingest` returns a cleaned DataFrame and a validation `report` dict with
pass/warn/fail status for the schema, null, invalid-treatment, and duplicate
checks plus an `overall` status. Cleaning is intentionally minimal and fully
reported:

- rows with nulls in required columns are dropped,
- rows with treatment values other than `0`/`1` are dropped,
- duplicate `user_id`s are resolved by keeping the earliest row per user.

Optional covariate columns are passed through untouched.

## License

MIT (add license file when ready for public release)
