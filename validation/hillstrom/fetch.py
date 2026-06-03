"""
Download the Hillstrom MineThatData e-mail experiment dataset.

Public dataset from Kevin Hillstrom's 2008 "MineThatData E-Mail Analytics and
Data Mining Challenge". 64,000 customers randomized into three arms
(no e-mail / men's e-mail / women's e-mail).

Run:  python validation/hillstrom/fetch.py
Writes to validation/data/hillstrom.csv (gitignored).
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

URL = (
    "http://www.minethatdata.com/"
    "Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "hillstrom.csv"


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
