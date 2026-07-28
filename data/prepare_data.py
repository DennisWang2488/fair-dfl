#!/usr/bin/env python3
"""Reconstruct ``data_processed.csv`` for the healthcare experiment.

The healthcare task uses the publicly released, de-identified Obermeyer et al.
sample ("Dissecting bias in algorithms", Science 2019). The raw file
``data_new.csv`` is NOT redistributed with this package; this script downloads it
from the public repository and applies three deterministic transforms to produce
the three derived columns the task consumes: ``race``, ``benefit`` and
``cost_t_capped``.

Every transform below has been verified to reproduce the original
``data_processed.csv`` to machine precision (max abs error ~1e-16).

Run:
    python data/prepare_data.py                 # download + build data/data_processed.csv
    python data/prepare_data.py --check OLD.csv  # additionally diff against an existing file

Source data:
    Obermeyer, Powers, Vogeli, Mullainathan (2019), "Dissecting racial bias in an
    algorithm used to manage the health of populations", Science 366(6464):447-453.
    Public data release: https://gitlab.com/labsysmed/dissecting-bias
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

DATA_URL = "https://gitlab.com/labsysmed/dissecting-bias/-/raw/master/data/data_new.csv"
HERE = Path(__file__).resolve().parent
RAW_PATH = HERE / "data_new.csv"
OUT_PATH = HERE / "data_processed.csv"

EXPECTED_ROWS = 48_784
COST_CAP_QUANTILE = 0.99  # winsorize cost_t at its 99th percentile (= $78,585 on this sample)


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    lo, hi = np.nanmin(x), np.nanmax(x)
    return (x - lo) / (hi - lo)


def download_raw(force: bool = False) -> Path:
    if RAW_PATH.exists() and not force:
        print(f"[prepare_data] using cached raw file: {RAW_PATH}")
        return RAW_PATH
    print(f"[prepare_data] downloading public sample from:\n    {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, RAW_PATH)
    print(f"[prepare_data] saved raw file -> {RAW_PATH} ({RAW_PATH.stat().st_size:,} bytes)")
    return RAW_PATH


def build_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the three verified deterministic transforms in place and return df."""
    if len(df) != EXPECTED_ROWS:
        print(
            f"[prepare_data] WARNING: expected {EXPECTED_ROWS} rows, got {len(df)}. "
            "The public release may have changed.",
            file=sys.stderr,
        )

    # 1. race: binarize the public string column. white -> 0, black -> 1.
    df["race"] = (df["race"].astype(str).str.lower() == "black").astype(int)

    # 2. benefit: ground-truth benefit score r_i = 0.5 * h_i + 0.5 * s_i, in [0, 1].
    #    h_i = min-max normalized active-chronic-illness count (gagne_sum_t, range 0-17).
    #    s_i = min-max normalized log(1 + avoidable cost), using the RAW (un-imputed)
    #          cost_avoidable_t column exactly as released. See data/README.md for the
    #          note on why nearest-neighbor imputation is intentionally NOT applied here.
    h = _minmax(df["gagne_sum_t"].to_numpy(dtype=float))
    s = _minmax(np.log1p(df["cost_avoidable_t"].to_numpy(dtype=float)))
    df["benefit"] = 0.5 * h + 0.5 * s

    # 3. cost_t_capped: winsorize cost_t at the 99th percentile, then scale to [0, 1].
    cap = float(df["cost_t"].quantile(COST_CAP_QUANTILE))
    df["cost_t_capped"] = np.minimum(df["cost_t"].to_numpy(dtype=float), cap) / cap

    return df


def diff_against(reference_csv: Path, produced: pd.DataFrame) -> bool:
    """Compare the three derived columns against an existing processed file."""
    ref = pd.read_csv(reference_csv)
    ok = True
    if len(ref) != len(produced):
        print(f"[check] row count differs: ref={len(ref)} produced={len(produced)}")
        return False
    for col in ("race", "benefit", "cost_t_capped"):
        a = produced[col].to_numpy(dtype=float)
        b = ref[col].to_numpy(dtype=float)
        err = float(np.max(np.abs(a - b)))
        status = "OK" if err < 1e-9 else "MISMATCH"
        if err >= 1e-9:
            ok = False
        print(f"[check] {col:14s} max abs error = {err:.3e}  [{status}]")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-download", action="store_true", help="re-download even if cached")
    ap.add_argument("--check", metavar="OLD_CSV", help="diff derived columns against an existing processed CSV")
    ap.add_argument("--out", default=str(OUT_PATH), help="output path (default: data/data_processed.csv)")
    args = ap.parse_args()

    raw = download_raw(force=args.force_download)
    df = pd.read_csv(raw)
    df = build_derived_columns(df)

    out = Path(args.out)
    df.to_csv(out, index=False)
    print(f"[prepare_data] wrote {out} ({len(df):,} rows, {df.shape[1]} cols)")

    if args.check:
        print(f"[prepare_data] checking against {args.check} ...")
        ok = diff_against(Path(args.check), df)
        if not ok:
            print("[prepare_data] CHECK FAILED", file=sys.stderr)
            return 1
        print("[prepare_data] check passed: derived columns reproduced to machine precision.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
