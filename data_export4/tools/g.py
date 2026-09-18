"""Data request 4: shape vs. placement as separate levers.

Subcommands (inside distrobox ubuntu-work, run from anywhere):
    python g.py sweep W   # corrected AR sweep for the new six (120 VTR calls)
    python g.py g1 W      # sweep's best shape + policy placement (36 calls)
    python g.py g2 W      # new six at AR 1.0: columns / policy / random
    python g.py g3 W      # original six, seed 42, policy at all 20 shapes (120)

Everything goes through the paper's own paths:
  * the sweep reuses data_export/tools/c2_perar.py:run_one unchanged (per-AR
    tile planning; direct VTRRunner call, no layout cache -> cached=False);
  * policy and random floorplans are evaluated through FPGAEnv.step via
    common3.replay (same bake, atom pinning, VTR invocation and metrics as the
    paper), on this request's isolated cache.
"""
import sys

import numpy as np

from common4 import *  # noqa: F401,F403
import common3 as c3
import c2_perar  # data_export/tools/c2_perar.py
from src.env.fpga_env import ASPECT_RATIOS

SWEEP_HDR = c2_perar.HDR + ["cached", "source"]
POL_HDR = ["circuit", "set", "seed", "forced_aspect_ratio", "grid_w", "grid_h", "area_mwta",
           "delay_ns", "power_w", "adp", "adp_reduction_pct", "vtr_success", "vtr_seconds",
           "cached", "dsp_coords", "bram_coords", "source"]
G2_HDR = ["circuit", "seed", "variant", "draw_index", "grid_w", "grid_h", "area_mwta", "delay_ns",
          "power_w", "adp", "adp_reduction_pct", "vtr_success", "vtr_seconds", "cached", "source"]

PERAR = REPO / "data_export" / "baseline_ar_sweep_perar.csv"      # noqa: F405
NEW_SWEEP = OUT / "g1_sweep_new_circuits.csv"                      # noqa: F405


def ar_index(ar):
    return ASPECT_RATIOS.index(round(float(ar), 1))


# ---------------------------------------------------------------- sweep (new six)
def sweep(workers):
    sink = CsvSink(NEW_SWEEP, SWEEP_HDR, ["circuit", "aspect_ratio"])     # noqa: F405

    def one(name, ar):
        row = c2_perar.run_one(name, ar)
        row["cached"] = False
        row["source"] = ("data_export/tools/c2_perar.py:run_one (per-AR tile placement), "
                         "baseline from data_export3/tools/f5_build.py")
        return row

    jobs = [({"circuit": n, "aspect_ratio": ar}, (n, ar))
            for n in NEW for ar in ASPECT_RATIOS if not sink.has((n, ar))]   # noqa: F405
    run_pool(jobs, one, sink, workers, "sweep")                          # noqa: F405


def best_ar():
    """circuit -> (best aspect ratio, its reduction %, source file)."""
    out = {}
    for path in (PERAR, NEW_SWEEP):
        with open(path) as fh:
            for r in csv.DictReader(fh):                                 # noqa: F405
                if r["vtr_success"] != "True":
                    continue
                v = float(r["adp_reduction_pct_vs_baseline"])
                if r["circuit"] not in out or v > out[r["circuit"]][1]:
                    out[r["circuit"]] = (float(r["aspect_ratio"]), v, path.name)
    return out


# ---------------------------------------------------------------- policy at a forced shape
def forced_jobs(circuits, seeds, ar_of, extra):
    """Build (key, (circuit, actions)) jobs: step 0 forced, policy places blocks."""
    jobs = []
    for name in circuits:
        env = c3.make_env(name)
        for seed in seeds:
            model = c3.load_model(seed, env)
            for ar in ar_of(name):
                acts = policy_after_forced_ar(model, env, ar_index(ar))     # noqa: F405
                jobs.append((extra(name, seed, ar), (name, acts)))
    return jobs


def pol_row(name, acts):
    r = c3.replay(name, acts)
    return {k: r.get(k) for k in ("grid_w", "grid_h", "area_mwta", "delay_ns", "power_w", "adp",
                                  "adp_reduction_pct", "vtr_success", "vtr_seconds", "cached",
                                  "dsp_coords", "bram_coords")}


def g1(workers):
    best = best_ar()
    missing = [c for c in ORIGINAL + NEW if c not in best]              # noqa: F405
    if missing:
        raise SystemExit(f"no sweep result for {missing}; run `g.py sweep` first")
    sink = CsvSink(OUT / "g1_sweep_shape_policy_placement.csv", POL_HDR,   # noqa: F405
                   ["circuit", "seed", "forced_aspect_ratio"])
    todo = [c for c in ORIGINAL + NEW                                    # noqa: F405
            if not all(sink.has((c, s, best[c][0])) for s in SEEDS)]    # noqa: F405
    jobs = forced_jobs(
        todo, SEEDS, lambda n: [best[n][0]],                             # noqa: F405
        lambda n, s, ar: {"circuit": n, "set": "original" if n in ORIGINAL else "new",   # noqa: F405
                          "seed": s, "forced_aspect_ratio": ar,
                          "source": f"step 0 forced to the best AR of {best[n][2]}; policy "
                                    f"(runs/ checkpoint for seed {s}) places H-blocks deterministically"})
    jobs = [j for j in jobs if not sink.has((j[0]["circuit"], j[0]["seed"], j[0]["forced_aspect_ratio"]))]
    run_pool(jobs, pol_row, sink, workers, "G1")                          # noqa: F405


def g3(workers):
    sink = CsvSink(OUT / "g3_policy_all_shapes.csv", POL_HDR,             # noqa: F405
                   ["circuit", "seed", "forced_aspect_ratio"])
    jobs = forced_jobs(
        ORIGINAL, [42], lambda n: ASPECT_RATIOS,                         # noqa: F405
        lambda n, s, ar: {"circuit": n, "set": "original", "seed": s, "forced_aspect_ratio": ar,
                          "source": "step 0 forced to this AR; seed-42 policy places H-blocks "
                                    "deterministically"})
    jobs = [j for j in jobs if not sink.has((j[0]["circuit"], j[0]["seed"], j[0]["forced_aspect_ratio"]))]
    run_pool(jobs, pol_row, sink, workers, "G3")                          # noqa: F405


# ---------------------------------------------------------------- G2
def random_after_ar1(env, rng):
    env.reset()
    acts = [ar_index(1.0)]
    env.step(acts[0])
    while True:
        acts.append(int(rng.choice(np.flatnonzero(env.get_action_mask()))))
        if env._current_step == env._total_blocks:
            return acts
        env.step(acts[-1])


def g2(workers):
    sink = CsvSink(OUT / "g2_fixed_shape_new.csv", G2_HDR,                # noqa: F405
                   ["circuit", "seed", "variant", "draw_index"])

    # columns: reused from the new-circuit sweep at AR 1.0 (no re-run)
    with open(NEW_SWEEP) as fh:
        for r in csv.DictReader(fh):                                     # noqa: F405
            if float(r["aspect_ratio"]) == 1.0 and not sink.has((r["circuit"], "", "columns", 0)):
                sink.write({"circuit": r["circuit"], "seed": "", "variant": "columns", "draw_index": 0,
                            "grid_w": r["grid_w"], "grid_h": r["grid_h"], "area_mwta": r["area_mwta"],
                            "delay_ns": r["delay_ns"], "power_w": r["power_w"], "adp": r["adp"],
                            "adp_reduction_pct": r["adp_reduction_pct_vs_baseline"],
                            "vtr_success": r["vtr_success"], "vtr_seconds": r["vtr_seconds"],
                            "cached": r["cached"],
                            "source": "reused from data_export4/g1_sweep_new_circuits.csv at AR 1.0 "
                                      "(unseeded; seed column empty)"})

    jobs = []
    for ci, name in enumerate(NEW):                                      # noqa: F405
        env = c3.make_env(name)
        for seed in SEEDS:                                               # noqa: F405
            if not sink.has((name, seed, "policy", 0)):
                model = c3.load_model(seed, env)
                acts = policy_after_forced_ar(model, env, ar_index(1.0))  # noqa: F405
                jobs.append(({"circuit": name, "seed": seed, "variant": "policy", "draw_index": 0,
                              "source": "step 0 forced to 1.0; policy places H-blocks deterministically"},
                             (name, acts)))
            for draw in range(3):
                rng = np.random.default_rng([seed, ci, draw])
                acts = random_after_ar1(env, rng)       # always drawn so indices stay reproducible
                if not sink.has((name, seed, "random", draw)):
                    jobs.append(({"circuit": name, "seed": seed, "variant": "random", "draw_index": draw,
                                  "source": "AR 1.0 + uniform legal placement, "
                                            f"numpy.default_rng([{seed}, {ci}, {draw}]), circuit_index "
                                            "= position in the new-six list"},
                                 (name, acts)))
    run_pool(jobs, pol_row, sink, workers, "G2")                          # noqa: F405


if __name__ == "__main__":
    cmd, w = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
    {"sweep": sweep, "g1": g1, "g2": g2, "g3": g3}[cmd](w)
