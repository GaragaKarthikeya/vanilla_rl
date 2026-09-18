"""Shared setup for data_export3 drivers.

Reuses data_export/tools/common.py (VTR path fixes, pinned universe dims,
CsvSink, baseline loader) and data_export/tools/c345.py (env construction,
model loading), and only redirects the output folder and the cache namespace.

One isolated cache is shared across F1/F2/F3 of this request
(runs/vtr_layout_cache_<circuit>_dataexport3.db), separate from the paper's
caches and from data_export / data_export2.

Must be run inside `distrobox enter ubuntu-work`.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "data_export" / "tools"))
# common.py chdir()s to the repo root, so this package's own directory has to be
# on sys.path explicitly for sibling imports to keep working from any cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import *  # noqa: F401,F403
import common as _c
import c345

OUT = _REPO / "data_export3"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

CACHE_SUFFIX = "_dataexport3"

DET_JSON = _REPO / "data_export" / "logs" / "det_actions_and_timing.json"
CONTROL_CSV = _REPO / "data_export" / "results_per_circuit_seed.csv"


def make_env(name, timeout=1800):
    return c345.make_env(name, cache_suffix=CACHE_SUFFIX, timeout=timeout)


load_model = c345.load_model
set_paper_weights = c345.set_paper_weights
run_pool = c345.run_pool


def det_actions():
    """circuit-seed -> the paper's deterministic action sequence."""
    return {(r["circuit"], r["seed"]): r["actions"]
            for r in _c.json.loads(DET_JSON.read_text())}


def control_rows():
    out = {}
    with open(CONTROL_CSV) as fh:
        for r in _c.csv.DictReader(fh):
            if r["split"] == "heldout":
                out[(r["circuit"], int(r["seed"]))] = r
    return out


def a2xy(a):
    """Action index -> (x, y) on the core grid, as FPGAEnv decodes it."""
    return 1 + a // MAX_H, 1 + a % MAX_H   # noqa: F405


def xy2a(x, y):
    return (x - 1) * MAX_H + (y - 1)       # noqa: F405


def decode_rollout(name, acts):
    """Replay an action sequence WITHOUT triggering VTR.

    Returns (aspect_ratio, placed_dsps, placed_brams, block_types, core_w, core_h)
    where block_types[i] is the type of the i-th H-block in the environment's own
    placement order (1 = DSP, else BRAM), matching FPGAEnv._blocks_to_place.
    """
    env = make_env(name)
    env.reset()
    for a in acts[:-1]:
        env.step(a)
    # decode the terminal placement the way FPGAEnv.step does, without evaluating
    x, y = a2xy(acts[-1])
    bt = env._blocks_to_place[env._current_step - 1]
    (env._placed_dsps if bt == 1 else env._placed_brams).append((x, y))
    cfg = env._active_config
    return (env._chosen_aspect_ratio, list(env._placed_dsps), list(env._placed_brams),
            list(env._blocks_to_place), cfg.width, cfg.height)


def replay(name, acts):
    """Evaluate an action sequence end-to-end (bake -> pin -> VTR -> metrics),
    on this request's isolated cache."""
    env = make_env(name)
    set_paper_weights(env)
    env.reset()
    for a in acts[:-1]:
        env.step(a)
    t0 = now()        # noqa: F405
    _, reward, term, _, info = env.step(acts[-1])
    secs = now() - t0  # noqa: F405
    assert term
    m, _ = baseline(name)   # noqa: F405
    row = {"aspect_ratio": info.get("aspect_ratio"), "vtr_success": bool(info.get("success")),
           "cached": bool(info.get("cached")), "vtr_seconds": round(secs, 2), "reward": reward,
           "dsp_coords": json.dumps(info.get("placed_dsps")),     # noqa: F405
           "bram_coords": json.dumps(info.get("placed_brams"))}   # noqa: F405
    if info.get("success"):
        a = adp(info["routing_area"], info["delay_ns"], info["power_w"])  # noqa: F405
        b = adp(m["routing_area"], m["delay_ns"], m["power_w"])           # noqa: F405
        row.update(area_mwta=info["routing_area"], delay_ns=info["delay_ns"],
                   power_w=info["power_w"], adp=a, adp_reduction_pct=(b - a) / b * 100,
                   grid_w=info["grid_W"], grid_h=info["grid_H"])
    return row
