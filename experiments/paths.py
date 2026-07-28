"""Central filesystem paths and environment setup for the experiments repo.

Single source of truth for: the repository root, the healthcare data file, solver
license locations, and the result generations the paper reports.

Import this instead of hard-coding relative paths or
``os.path.abspath("data/...")`` -- the latter resolves against the *current
working directory*, not the repo root, so it silently breaks when a script is
launched from anywhere other than the repo root (e.g. a Colab ``/content`` cwd).

Typical use::

    from experiments.paths import DATA_CSV, setup_mosek_license, require_file
    setup_mosek_license()                         # set MOSEKLM_LICENSE_FILE if available
    require_file(DATA_CSV, "healthcare cohort")   # clear error instead of a deep crash

This module is intentionally dependency-free (stdlib only) so it is safe to import
at the very top of a driver, *before* numpy / torch / pandas.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout
# ---------------------------------------------------------------------------
# experiments/paths.py -> parents[0] = experiments/, parents[1] = repo root.
REPO_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = REPO_ROOT / "data"
DATA_CSV: Path = DATA_DIR / "data_processed.csv"   # healthcare cohort (gitignored)
MOSEK_LIC: Path = DATA_DIR / "mosek.lic"           # solver license (gitignored)
GUROBI_LIC: Path = DATA_DIR / "gurobi.lic"         # solver license (gitignored)

RESULTS_DIR: Path = REPO_ROOT / "results"

# ---------------------------------------------------------------------------
# Result generations reported in the paper.
# ---------------------------------------------------------------------------
HC_GRID: Path = RESULTS_DIR / "healthcare" / "main_v5_multiinstance"
MD_GRID: Path = RESULTS_DIR / "md_knapsack" / "main_v6_rowsum"

HC_MASTER_CSV: Path = HC_GRID / "grid" / "tables" / "master_tagged.csv"
MD_MASTER_CSV: Path = MD_GRID / "grid" / "tables" / "master_tagged.csv"

# Generated tables / figures land here.
TABLES_OUT: Path = REPO_ROOT / "tables"
PLOTS_OUT: Path = REPO_ROOT / "figures"
SUPP_TABLES_OUT: Path = REPO_ROOT / "tables" / "supplement"
SUPP_PLOTS_OUT: Path = REPO_ROOT / "figures" / "supplement"

def setup_mosek_license() -> bool:
    """Point ``MOSEKLM_LICENSE_FILE`` at ``data/mosek.lic`` if it is not already set.

    Only sets the env var when it is currently unset/empty AND the repo-root
    license file actually exists, so a pre-set env var or a genuinely-missing file is left untouched.

    Returns ``True`` if a MOSEK license is now available (env var set or file found).
    """
    if not os.environ.get("MOSEKLM_LICENSE_FILE") and MOSEK_LIC.exists():
        os.environ["MOSEKLM_LICENSE_FILE"] = str(MOSEK_LIC)
    return bool(os.environ.get("MOSEKLM_LICENSE_FILE"))


def require_mosek_license() -> str:
    """Ensure a MOSEK license is available and return its path, else raise.

    Opt-in: call this only from scripts that genuinely require the MOSEK-backed
    solver (the analytic alpha-fair group solve / MD exact eval). Scripts that can
    fall back to SCS should call :func:`setup_mosek_license` instead so they keep
    working without a license.
    """
    if setup_mosek_license():
        return os.environ["MOSEKLM_LICENSE_FILE"]
    raise FileNotFoundError(
        "MOSEK license not found. Set MOSEKLM_LICENSE_FILE, or place the license at "
        f"{MOSEK_LIC}.\n  (Only needed for the optional MOSEK-backed solver; the "
        "reported pipeline runs on the open-source SCS/ECOS solvers.)"
    )


def require_file(path: str | os.PathLike, what: str = "") -> Path:
    """Return ``path`` as a :class:`~pathlib.Path`, or raise a clear FileNotFoundError.

    Use at the entry of a run to fail fast with an actionable message instead of a
    deep crash inside the data loader when the file is missing (e.g. the script was
    launched from the wrong working directory).
    """
    p = Path(path)
    if not p.exists():
        label = f" ({what})" if what else ""
        raise FileNotFoundError(
            f"Required file{label} not found: {p}\n"
            f"  cwd = {Path.cwd()}\n"
            f"  Run from the repo root ({REPO_ROOT}) or pass an absolute path."
        )
    return p
