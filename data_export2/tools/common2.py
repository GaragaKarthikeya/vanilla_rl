"""Shared setup for data_export2 drivers.

Reuses data_export/tools/common.py verbatim (VTR path fixes, pinned universe
dims, CsvSink, baseline loader) and only redirects the output folder and the
isolated cache suffix so data_export2 never writes into data_export's CSVs or
the paper's cache DBs.

Must be run inside `distrobox enter ubuntu-work`.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "data_export" / "tools"))

from common import *  # noqa: F401,F403  (REPO, MAX_*, HELDOUT, SEEDS, CKPT, baseline, adp, CsvSink, now, json, csv, os)
import common as _c

OUT = _REPO / "data_export2"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

# Separate cache namespace from data_export's "_dataexport".
CACHE_SUFFIX = "_dataexport2"

CONTROL_CSV = _REPO / "data_export" / "results_per_circuit_seed.csv"


def control_rows():
    """Held-out deterministic control results from Tier B (no re-run)."""
    out = {}
    with open(CONTROL_CSV) as fh:
        for r in _c.csv.DictReader(fh):
            if r["split"] == "heldout":
                out[(r["circuit"], int(r["seed"]))] = r
    return out
