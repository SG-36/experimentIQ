"""One-off script to generate committed test fixtures."""

from pathlib import Path

from data.synthetic_experiment import generate_experiment

FIXTURES = [
    ("fixture_clean.csv", "clean_experiment", 42, {}),
    ("fixture_srm.csv", "srm_injected", 42, {}),
    ("fixture_novelty.csv", "novelty_effect", 42, {}),
    ("fixture_noncompliance.csv", "noncompliance", 42, {}),
    ("fixture_het.csv", "heterogeneous_effects", 42, {}),
    ("fixture_null.csv", "no_effect", 42, {}),
]

OUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for filename, scenario, seed, kwargs in FIXTURES:
        df = generate_experiment(n_users=2000, seed=seed, scenario=scenario, **kwargs)
        path = OUT_DIR / filename
        df.to_csv(path, index=False)
        print(f"Wrote {path} ({len(df)} rows, scenario={scenario})")


if __name__ == "__main__":
    main()
