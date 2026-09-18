"""F1: does a floorplan transfer geometrically?

Apply circuit A's deterministic floorplan to circuit B without running the
policy for B at all, for all 30 ordered pairs of distinct held-out circuits at
seed 42.

Transfer rule (implemented once, in transfer_actions() below):
  1. Take A's chosen aspect ratio and its H-block coordinates on A's CORE grid.
  2. Rescale each coordinate to B's core grid:
         x_B = round(1 + (x_A - 1) * (W_B - 1) / (W_A - 1))
     and the same for y, then clamp into B's legal range.
  3. Apply B's own legality mask (FPGAEnv.get_action_mask()). If a rescaled
     coordinate is illegal or collides with a block already placed, take the
     nearest legal tile by Manhattan distance, ties broken by lower x then
     lower y.
  4. Place min(n_A, n_B) blocks from the rescaled list, in the environment's own
     order (BRAMs then DSPs, each by descending net count). Any remaining blocks
     B needs are placed uniformly at random over the legal mask, using
     numpy.default_rng([seed, circuit_index]); the count is recorded in
     n_blocks_random.
  5. B's aspect ratio is A's chosen ratio.

Usage (inside distrobox ubuntu-work):
    python f1.py pattern        # f1_pattern.csv + f1_reference.csv, no VTR
    python f1.py transfer W     # the 30 transfer evaluations
"""
import sys

import numpy as np

from common3 import *  # noqa: F401,F403
import common3 as c3

HDR = ["source_circuit", "target_circuit", "seed", "source_aspect_ratio", "target_grid_w",
       "target_grid_h", "n_blocks_transferred", "n_blocks_random", "area_mwta", "delay_ns",
       "power_w", "adp", "adp_reduction_pct", "vtr_success", "vtr_seconds", "cached",
       "dsp_coords", "bram_coords", "source"]

PAT_HDR = ["circuit", "seed", "grid_w", "grid_h", "block_type", "x", "y", "x_norm", "y_norm",
           "dist_to_center_norm", "dist_to_edge_norm", "source"]

REF_HDR = ["target_circuit", "seed", "method", "adp_reduction_pct", "aspect_ratio", "source"]

SEED = 42


def nearest_legal(mask, x, y, W, H):
    """Nearest legal tile to (x, y) by Manhattan distance; ties -> lower x, then lower y."""
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        return None
    lx = 1 + legal // MAX_H          # noqa: F405
    ly = 1 + legal % MAX_H           # noqa: F405
    d = np.abs(lx - x) + np.abs(ly - y)
    # lexicographic: distance, then x, then y
    best = np.lexsort((ly, lx, d))[0]
    return int(legal[best])


def transfer_actions(src, dst, rng):
    """Build dst's action sequence from src's deterministic floorplan."""
    a_acts = c3.det_actions()[(src, SEED)]
    ar_action = a_acts[0]
    ar_a, dsps_a, brams_a, types_a, W_a, H_a = c3.decode_rollout(src, a_acts)

    # A's coordinates in the environment's own placement order
    di = iter(dsps_a)
    bi = iter(brams_a)
    coords_a = [next(di) if t == 1 else next(bi) for t in types_a]

    env = c3.make_env(dst)
    env.reset()
    cfg = env._active_config
    W_b, H_b = cfg.width, cfg.height

    def rescale(v, n_a, n_b):
        if n_a <= 1:            # degenerate source extent: keep the tile at the origin
            return 1
        return int(round(1 + (v - 1) * (n_b - 1) / (n_a - 1)))

    env.step(ar_action)                       # step 0: A's aspect ratio
    acts = [ar_action]
    n_b = env._total_blocks
    n_a = len(coords_a)
    n_transfer = min(n_a, n_b)
    n_random = n_b - n_transfer

    for i in range(n_b):
        mask = env.get_action_mask()
        if i < n_transfer:
            xa, ya = coords_a[i]
            x = min(max(rescale(xa, W_a, W_b), 1), W_b)
            y = min(max(rescale(ya, H_a, H_b), 1), H_b)
            a = nearest_legal(mask, x, y, W_b, H_b)
        else:
            a = int(rng.choice(np.flatnonzero(mask)))
        acts.append(a)
        if env._current_step == env._total_blocks:
            break
        env.step(a)
    return acts, ar_a, W_b, H_b, n_transfer, n_random


def transfer(workers):
    sink = CsvSink(OUT / "f1_geometric_transfer.csv", HDR,          # noqa: F405
                   ["source_circuit", "target_circuit", "seed"])
    jobs = []
    for src in HELDOUT:            # noqa: F405
        for ci, dst in enumerate(HELDOUT):    # noqa: F405
            if src == dst or sink.has((src, dst, SEED)):
                continue
            rng = np.random.default_rng([SEED, ci])
            acts, ar_a, W_b, H_b, nt, nr = transfer_actions(src, dst, rng)
            jobs.append(({"source_circuit": src, "target_circuit": dst, "seed": SEED,
                          "source_aspect_ratio": ar_a, "target_grid_w": W_b,
                          "target_grid_h": H_b, "n_blocks_transferred": nt,
                          "n_blocks_random": nr,
                          "source": "data_export3/tools/f1.py transfer_actions(); source floorplan "
                                    "from data_export/logs/det_actions_and_timing.json"},
                         (dst, acts)))
            print(f"{src:20s} -> {dst:20s} AR={ar_a} grid={W_b}x{H_b} "
                  f"transferred={nt} random={nr}", flush=True)
    run_pool(jobs, c3.replay, sink, workers, "F1")    # noqa: F405


def pattern():
    """f1_pattern.csv: where the policy puts blocks, from the existing
    deterministic rollouts. No new VTR runs.

    dist_to_center_norm = Euclidean distance to the core-grid centre, divided by
      the distance from the centre to the (1,1) corner, so 0 = centre, 1 = corner.
    dist_to_edge_norm  = Chebyshev-style distance to the nearest edge,
      min(x-1, W-x, y-1, H-y), divided by its maximum possible value
      floor((min(W,H)-1)/2), so 0 = on the edge, 1 = deepest interior.
    """
    det = c3.det_actions()
    sink = CsvSink(OUT / "f1_pattern.csv", PAT_HDR,      # noqa: F405
                   ["circuit", "seed", "block_type", "x", "y"])
    for name in HELDOUT:         # noqa: F405
        for seed in SEEDS:       # noqa: F405
            _ar, dsps, brams, _t, W, H = c3.decode_rollout(name, det[(name, seed)])
            cx, cy = (1 + W) / 2.0, (1 + H) / 2.0
            corner = ((cx - 1) ** 2 + (cy - 1) ** 2) ** 0.5
            edge_max = ((min(W, H) - 1) // 2) or 1
            for bt, coords in (("DSP", dsps), ("BRAM", brams)):
                for (x, y) in coords:
                    if sink.has((name, seed, bt, x, y)):
                        continue
                    sink.write({
                        "circuit": name, "seed": seed, "grid_w": W, "grid_h": H,
                        "block_type": bt, "x": x, "y": y,
                        "x_norm": x / W, "y_norm": y / H,
                        "dist_to_center_norm": (((x - cx) ** 2 + (y - cy) ** 2) ** 0.5) / corner,
                        "dist_to_edge_norm": min(x - 1, W - x, y - 1, H - y) / edge_max,
                        "source": "data_export/logs/det_actions_and_timing.json decoded through "
                                  "FPGAEnv (no VTR); core grid from baselines/*_resources.txt",
                    })
    print("wrote f1_pattern.csv")


def reference():
    """Per-target reference numbers at seed 42, all reused from data_export/."""
    ctrl = c3.control_rows()
    sweep, rnd = {}, {}
    with open(REPO / "data_export" / "baseline_ar_sweep.csv") as fh:    # noqa: F405
        for r in csv.DictReader(fh):                                    # noqa: F405
            if r["vtr_success"] == "True":
                sweep.setdefault(r["circuit"], []).append(
                    (float(r["adp_reduction_pct_vs_baseline"]), float(r["aspect_ratio"])))
    with open(REPO / "data_export" / "random_floorplans.csv") as fh:    # noqa: F405
        for r in csv.DictReader(fh):                                    # noqa: F405
            if r["vtr_success"] == "True" and int(r["seed"]) == SEED:
                rnd.setdefault(r["circuit"], []).append(float(r["adp_reduction_pct"]))

    sink = CsvSink(OUT / "f1_reference.csv", REF_HDR,      # noqa: F405
                   ["target_circuit", "seed", "method"])
    for name in HELDOUT:        # noqa: F405
        rows = [("policy_deterministic", float(ctrl[(name, SEED)]["adp_reduction_pct"]),
                 ctrl[(name, SEED)]["policy_aspect_ratio"],
                 "data_export/results_per_circuit_seed.csv"),
                ("trimmed_baseline_ar_sweep_best", max(sweep[name])[0], max(sweep[name])[1],
                 "data_export/baseline_ar_sweep.csv (best of 20 aspect ratios)"),
                ("random_best_of_20", max(rnd[name]), "",
                 "data_export/random_floorplans.csv (best of the 20 random legal floorplans at seed 42)")]
        for method, val, ar, src in rows:
            if not sink.has((name, SEED, method)):
                sink.write({"target_circuit": name, "seed": SEED, "method": method,
                            "adp_reduction_pct": val, "aspect_ratio": ar,
                            "source": src + "; reused, not re-run"})
    print("wrote f1_reference.csv")


if __name__ == "__main__":
    if sys.argv[1] == "pattern":
        pattern()
        reference()
    elif sys.argv[1] == "transfer":
        transfer(int(sys.argv[2]))
