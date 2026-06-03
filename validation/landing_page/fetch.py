"""
Download the Udacity "ab_data.csv" landing-page A/B test dataset.

Widely-used public dataset (Udacity Data Analyst Nanodegree): an e-commerce
landing-page experiment with ~294k visits over ~3 weeks of January 2017. Unlike
the Hillstrom file, every row carries a real per-user ``timestamp``, so this
dataset exercises the time-based modules (sequential testing, novelty effect).

Columns: user_id, timestamp, group (control/treatment), landing_page
(old_page/new_page), converted (0/1).

Run:  python validation/landing_page/fetch.py
Writes to validation/data/ab_data.csv (gitignored).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = (
    "https://raw.githubusercontent.com/"
    "jemc36/Udacity-DAND-AB-test-ecommerce/master/ab_data.csv"
)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "ab_data.csv"


def fetch(force: bool = False) -> Path:
    """Download the CSV if not already present. Returns the local path."""
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        print(f"Already present: {DATA_PATH}")
        return DATA_PATH
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, DATA_PATH)
    print(f"Wrote {DATA_PATH} ({DATA_PATH.stat().st_size:,} bytes)")
    return DATA_PATH


if __name__ == "__main__":
    fetch()
