"""F5 stage 2: zero-shot evaluation on the extra held-out circuits.

For every candidate that built a baseline AND fits the checkpoint's universe
caps, run one deterministic rollout per seed plus one VTR evaluation, exactly as
the paper's held-out evaluation does.

Writes data_export3/f5_extra_heldout.csv with the columns of
data_export/results_per_circuit_seed.csv plus the circuit's resource profile and
occupancy.

Usage (inside distrobox ubuntu-work):
    python f5_eval.py WORKERS
"""
import sys

from common3 import *  # noqa: F401,F403
import common3 as c3

HDR = ["circuit", "split", "seed", "baseline_area_mwta", "baseline_delay_ns", "baseline_power_w",
       "baseline_adp", "policy_area_mwta", "policy_delay_ns", "policy_power_w", "policy_adp",
       "adp_reduction_pct", "policy_aspect_ratio", "policy_grid_w", "policy_grid_h",
       "baseline_grid_w", "baseline_grid_h", "num_clbs_packed", "num_dsp", "num_bram", "num_io",
       "core_grid_w", "core_grid_h", "occupancy", "vtr_success", "vtr_seconds", "cached",
       "how_obtained", "source"]


def eligible():
    """Candidates that built a baseline, fit the checkpoint's universe caps, and
    have at least one H-block to place.

    A circuit that packs to 0 DSP + 0 BRAM has an empty floorplanning action
    space -- the episode is the aspect-ratio step and nothing else -- so it
    cannot test placement and is excluded here (recorded in notes.md).
    """
    out = []
    for f in ("f5_candidates.csv", "f5_candidates_b2.csv"):
        p = OUT / f                                  # noqa: F405
        if not p.exists():
            continue
        with open(p) as fh:
            for r in csv.DictReader(fh):             # noqa: F405
                if (r["baseline_built"] == "True" and r["fits_universe"] == "True"
                        and int(r["num_dsp"] or 0) + int(r["num_bram"] or 0) > 0):
                    out.append(r)
    return out


def main(workers):
    cands = {r["circuit"]: r for r in eligible()}
    sink = CsvSink(OUT / "f5_extra_heldout.csv", HDR, ["circuit", "seed"])   # noqa: F405
    jobs = []
    for name, meta in cands.items():
        env = c3.make_env(name)
        for seed in SEEDS:           # noqa: F405
            if sink.has((name, seed)):
                continue
            model = c3.load_model(seed, env)
            acts, _ = rollout(model, env)
            m, res = baseline(name)   # noqa: F405
            jobs.append(({
                "circuit": name, "split": "heldout_extra", "seed": seed,
                "baseline_area_mwta": m["routing_area"], "baseline_delay_ns": m["delay_ns"],
                "baseline_power_w": m["power_w"],
                "baseline_adp": adp(m["routing_area"], m["delay_ns"], m["power_w"]),  # noqa: F405
                "baseline_grid_w": res["fpga_size"][0] + 2,
                "baseline_grid_h": res["fpga_size"][1] + 2,
                "num_clbs_packed": meta["num_clbs_packed"], "num_dsp": meta["num_dsp"],
                "num_bram": meta["num_bram"], "num_io": meta["num_io"],
                "core_grid_w": meta["core_grid_w"], "core_grid_h": meta["core_grid_h"],
                "occupancy": meta["occupancy"],
                "how_obtained": "final_checkpoint_deterministic",
                "source": "data_export3/tools/f5_eval.py; baseline from "
                          "data_export3/tools/f5_build.py (traditional VTR flow, stock arch)",
            }, (name, acts)))
            print(f"{name:20s} s{seed} actions={len(acts)}", flush=True)

    def work(name, acts):
        r = c3.replay(name, acts)
        return {"policy_area_mwta": r.get("area_mwta"), "policy_delay_ns": r.get("delay_ns"),
                "policy_power_w": r.get("power_w"), "policy_adp": r.get("adp"),
                "adp_reduction_pct": r.get("adp_reduction_pct"),
                "policy_aspect_ratio": r.get("aspect_ratio"),
                "policy_grid_w": r.get("grid_w"), "policy_grid_h": r.get("grid_h"),
                "vtr_success": r["vtr_success"], "vtr_seconds": r["vtr_seconds"],
                "cached": r["cached"]}

    run_pool(jobs, work, sink, workers, "F5eval")    # noqa: F405


def rollout(model, env):
    obs, _ = env.reset()
    acts = []
    while True:
        a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        acts.append(int(a))
        if env._current_step == env._total_blocks:
            return acts, None
        obs, _, term, _, _ = env.step(acts[-1])
        assert not term


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 12)
