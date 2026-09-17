"""E1b: hand-coded heuristic floorplan, no learning. One per held-out circuit
(6 VTR calls).

Rule (fixed in advance, applied identically to all 6 circuits):

  1. Aspect ratio from the CLB-to-IO ratio r = num_clbs_packed / num_io
     (data_export/circuits.csv). A design with few CLBs per IO pad is
     IO-bound: it needs perimeter rather than area, and an elongated
     rectangle buys more perimeter per tile than a square. So

         r <  1.0  -> IO-bound  -> aspect ratio 0.5 (elongated)
         r >= 1.0  -> logic-bound -> aspect ratio 1.0 (square)

  2. H-blocks are placed in the environment's own order (the policy's order:
     BRAMs then DSPs by descending net count) and each one takes the LEGAL
     action closest to the grid centre (Euclidean distance, ties broken by the
     lower action index), i.e. clustered around the centre.

  3. Legality is the paper's own FPGAEnv.get_action_mask(), so capacity,
     collisions and column rules are identical to the policy's.

Usage (inside distrobox ubuntu-work):
    python e1b.py WORKERS
"""
import sys

import numpy as np

from elib import *  # noqa: F401,F403
import elib
from src.env.fpga_env import ASPECT_RATIOS

HDR = ["circuit", "seed", "variant", "draw_index", "aspect_ratio", "grid_w", "grid_h", "area_mwta",
       "delay_ns", "power_w", "adp", "adp_reduction_pct", "vtr_success", "cached", "vtr_seconds",
       "dsp_coords", "bram_coords", "clb_io_ratio", "rule", "source"]

RULE = ("AR = 0.5 if num_clbs_packed/num_io < 1 (IO-bound, elongate) else 1.0 (square); "
        "H-blocks in FPGAEnv order (BRAMs then DSPs by descending net count), each taking the "
        "legal action nearest the grid centre by Euclidean distance (ties -> lower action index).")


def circuits_csv():
    with open(REPO / "data_export" / "circuits.csv") as fh:   # noqa: F405
        return {r["circuit"]: r for r in csv.DictReader(fh)}  # noqa: F405


def heuristic_actions(env, ar_action):
    env.reset()
    acts = [int(ar_action)]
    env.step(acts[0])
    cfg = env._active_config
    cx, cy = (1 + cfg.width) / 2.0, (1 + cfg.height) / 2.0
    while True:
        valid = np.flatnonzero(env.get_action_mask())
        x = 1 + valid // MAX_H          # noqa: F405
        y = 1 + valid % MAX_H           # noqa: F405
        d = (x - cx) ** 2 + (y - cy) ** 2
        acts.append(int(valid[int(np.argmin(d))]))   # argmin -> lowest index on ties
        if env._current_step == env._total_blocks:
            return acts
        env.step(acts[-1])


def main(workers):
    cc = circuits_csv()
    sink = CsvSink(OUT / "e1b_heuristic.csv", HDR, ["circuit"])   # noqa: F405
    jobs = []
    for name in HELDOUT:            # noqa: F405
        if sink.has((name,)):
            continue
        r = cc[name]
        ratio = float(r["num_clbs_packed"]) / float(r["num_io"])
        ar = 0.5 if ratio < 1.0 else 1.0
        env = elib.make_env(name)
        acts = heuristic_actions(env, ASPECT_RATIOS.index(ar))
        print(f"{name:20s} clb/io={ratio:8.3f} -> AR {ar}", flush=True)
        jobs.append(({"circuit": name, "seed": "", "variant": "heuristic", "draw_index": 0,
                      "clb_io_ratio": ratio, "rule": RULE,
                      "source": "data_export2/tools/e1b.py; clb/io from data_export/circuits.csv; "
                                "evaluated through FPGAEnv (same bake/pin/VTR path as the policy)"},
                     (name, acts)))
    run_pool(jobs, elib.replay, sink, workers, "E1b")   # noqa: F405


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
