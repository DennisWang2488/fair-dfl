"""Shared paper figure style (2026-07-14 unified redesign).

Single source for the rcParams and the method palette used by ALL paper and
supplement figures, so every figure reads as one system: STIX serif to match
the manuscript body, Okabe--Ito colorblind-safe palette with FIXED per-family
semantics, light y-grid, no top/right spines. Import and call ``apply()``
before creating any figure.

Palette semantics (same as the Sec 5.3 main-text figures):
  prediction-focused family -> blues; decision-only -> gray;
  fairness-augmented scalarized family -> vermillion / orange;
  dynamic MOO handlers -> greens; FPLG (hybrid, supplement-only) -> black.
"""
from __future__ import annotations

import matplotlib.pyplot as plt

RC = {
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.grid.axis": "y",
    "grid.alpha": 0.25,
    "grid.linewidth": 0.6,
    "savefig.bbox": "tight",
}

# Canonical display-label -> color. Includes the legacy label aliases (FDFL)
# so existing lookup code keeps working during the naming migration.
COLOR = {
    "PTO": "#0072B2", "FPTO": "#0072B2", "SAA": "#56B4E9", "WDRO": "#CC79A7",
    "DFL": "#7F7F7F",
    "Regret-and-MAD": "#D55E00", "FDFL": "#D55E00",
    "Regret-and-MSE": "#E69F00", "FDFL-Scal": "#E69F00",
    "FPLG": "#000000",
    "PCGrad": "#009E73", "MGDA": "#117733", "NashMTL": "#44AA99",
}
MARKER = {
    "PTO": "o", "FPTO": "o", "SAA": "v", "WDRO": "<", "DFL": "^",
    "Regret-and-MAD": "h", "FDFL": "h",
    "Regret-and-MSE": "s", "FDFL-Scal": "s", "FPLG": "D",
    "PCGrad": "*", "MGDA": "X", "NashMTL": "P",
}

TCRIT5 = 2.776  # 95% two-sided t critical value, df=4 (5 seeds)


def apply() -> None:
    plt.rcParams.update(RC)
