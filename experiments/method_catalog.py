"""Central display metadata for the paper-figure method pool.

Single source of truth for the per-method *display* attributes -- short label,
plot color, plot marker, and family -- of the 10-method pool used by the FINAL
v5/v6 paper figures and tables. Today these tuples are copy-pasted into
``plotters/plot_healthcare.py``, ``plotters/plot_knapsack.py``, ``plotters/plot_hc_v4.py``
and several analyzers; importing them from here lets a label/color/marker change
happen in exactly one place.

The values below reproduce ``plot_healthcare.METHOD_CFG`` *exactly* (and therefore the
identical palette/markers used by ``plot_knapsack``), so migrating a plotter is a
drop-in::

    from experiments.method_catalog import paper_cfg
    METHOD_CFG = paper_cfg()                    # plot_healthcare: (label, color, marker, family)
    METHOD_CFG = paper_cfg(include_family=False)  # plot_knapsack: (label, color, marker)

Scope notes:
  * Keys are the lowercase method keys the grid masters use (``df.method.str.lower()``).
  * This is the *paper-figure* pool. Other exploration variants
    the FDFL-mu variants, ...) lives in ``experiments.configs.COLOR_MAP`` /
    ``MARKER_MAP`` and uses a different palette by design -- it is intentionally not
    merged here, so neither set of figures shifts color.
  * Method *training* semantics (flags, mu, handlers) stay in
    ``experiments.configs.ALL_METHOD_CONFIGS``; this module is display-only.

Dependency-free (stdlib only) so it is safe to import anywhere.
"""
from __future__ import annotations

from typing import NamedTuple


class MethodStyle(NamedTuple):
    """Display attributes for one method in the paper-figure pool."""
    label: str
    color: str
    marker: str
    family: str


# Canonical 10-method paper pool -- transcribed verbatim from plot_healthcare.METHOD_CFG.
# Order is the canonical plotting/legend order.
METHOD_CFG: dict[str, MethodStyle] = {
    "fpto":      MethodStyle("FPTO",      "#0072B2", "o", "PTO"),
    "saa":       MethodStyle("SAA",       "#56B4E9", "v", "PTO"),
    "wdro":      MethodStyle("WDRO",      "#CC79A7", "<", "PTO"),
    "dfl":       MethodStyle("DFL",       "#7F7F7F", "^", "DFL"),
    "fdfl":      MethodStyle("Regret-and-MAD", "#D55E00", "h", "DFL"),
    "fdfl-scal": MethodStyle("FDFL-Scal", "#E69F00", "s", "Scal"),
    "fplg":      MethodStyle("FPLG",      "#000000", "D", "Scal"),
    "pcgrad":    MethodStyle("PCGrad",    "#009E73", "*", "MOO"),
    "mgda":      MethodStyle("MGDA",      "#117733", "X", "MOO"),
    "nashmtl":   MethodStyle("NashMTL",   "#44AA99", "P", "MOO"),
}

METHOD_ORDER: list[str] = list(METHOD_CFG)

# Convenience flat maps (derived -- never edit directly).
LABEL: dict[str, str] = {k: v.label for k, v in METHOD_CFG.items()}
COLOR: dict[str, str] = {k: v.color for k, v in METHOD_CFG.items()}
MARKER: dict[str, str] = {k: v.marker for k, v in METHOD_CFG.items()}
FAMILY: dict[str, str] = {k: v.family for k, v in METHOD_CFG.items()}

# The reduced "core" set used by the v5/v6 headline figures (DFL = FDFL@lambda0,
# FDFL = FDFL@lambda>0). Convenience only; tables define their own row order in
# the table builders under pipeline/analyzers/.
CORE_KEYS: tuple[str, ...] = ("fpto", "dfl", "fdfl", "fdfl-scal", "fplg", "pcgrad")


def paper_cfg(include_family: bool = True) -> dict[str, tuple]:
    """Return the paper-pool style dict as plain tuples (drop-in for the plotters).

    With ``include_family=True`` (default) each value is ``(label, color, marker,
    family)`` -- exactly ``plot_healthcare.METHOD_CFG``. With ``include_family=False``
    each value is ``(label, color, marker)`` -- exactly ``plot_knapsack.METHOD_CFG``.
    """
    if include_family:
        return {k: (v.label, v.color, v.marker, v.family) for k, v in METHOD_CFG.items()}
    return {k: (v.label, v.color, v.marker) for k, v in METHOD_CFG.items()}
