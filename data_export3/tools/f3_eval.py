"""F3 stage 2: zero-shot evaluation of the no-GCN checkpoint.

Deterministic rollout + one VTR evaluation per held-out circuit, exactly as the
paper's held-out evaluation (and as data_export's det rows), but loading
runs/f3_no_gcn_seed42.zip with NoGCNFeaturesExtractor.

Writes data_export3/f3_no_gcn.csv: the columns of
data_export/results_per_circuit_seed.csv plus training_wall_clock_hours and
total_episodes.

Usage (inside distrobox ubuntu-work):
    python f3_eval.py WORKERS
"""
import sys

from common3 import *  # noqa: F401,F403
import common3 as c3
from no_gcn_extractor import NoGCNFeaturesExtractor
from src.training.ppo import CustomMaskablePPO

CKPT = "runs/f3_no_gcn_seed42.zip"
SEED = 42
EPISODE_LOG = REPO / "all_layouts_multi_seed_42_f3_no_gcn_seed42.jsonl"   # noqa: F405
WALL = OUT / "logs" / "f3_train_wallclock.json"                            # noqa: F405

HDR = ["circuit", "split", "seed", "baseline_area_mwta", "baseline_delay_ns", "baseline_power_w",
       "baseline_adp", "policy_area_mwta", "policy_delay_ns", "policy_power_w", "policy_adp",
       "adp_reduction_pct", "policy_aspect_ratio", "policy_grid_w", "policy_grid_h",
       "baseline_grid_w", "baseline_grid_h", "how_obtained", "training_wall_clock_hours",
       "total_episodes", "vtr_success", "vtr_seconds", "cached", "source"]


def rollout(model, env):
    obs, _ = env.reset()
    acts = []
    while True:
        a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        acts.append(int(a))
        if env._current_step == env._total_blocks:
            return acts
        obs, _, term, _, _ = env.step(acts[-1])
        assert not term


def main(workers):
    hours = json.loads(WALL.read_text())["training_wall_clock_hours"] if WALL.exists() else None  # noqa: F405
    episodes = sum(1 for _ in open(EPISODE_LOG)) if EPISODE_LOG.exists() else None
    sink = CsvSink(OUT / "f3_no_gcn.csv", HDR, ["circuit", "seed"])     # noqa: F405
    jobs = []
    for name in HELDOUT:                                                # noqa: F405
        if sink.has((name, SEED)):
            continue
        env = c3.make_env(name)
        model = CustomMaskablePPO.load(CKPT, env=env, device="cpu",
                                       policy_kwargs={"features_extractor_class": NoGCNFeaturesExtractor})
        acts = rollout(model, env)
        m, res = baseline(name)                                         # noqa: F405
        jobs.append(({
            "circuit": name, "split": "heldout", "seed": SEED,
            "baseline_area_mwta": m["routing_area"], "baseline_delay_ns": m["delay_ns"],
            "baseline_power_w": m["power_w"],
            "baseline_adp": adp(m["routing_area"], m["delay_ns"], m["power_w"]),   # noqa: F405
            "baseline_grid_w": res["fpga_size"][0] + 2, "baseline_grid_h": res["fpga_size"][1] + 2,
            "how_obtained": "final_checkpoint_deterministic (no-GCN retrain)",
            "training_wall_clock_hours": hours, "total_episodes": episodes,
            "source": f"{CKPT} via data_export3/tools/f3_eval.py; wall-clock from "
                      f"data_export3/logs/f3_train_wallclock.json; episodes = lines of {EPISODE_LOG.name}",
        }, (name, acts)))

    def work(name, acts):
        r = c3.replay(name, acts)
        return {"policy_area_mwta": r.get("area_mwta"), "policy_delay_ns": r.get("delay_ns"),
                "policy_power_w": r.get("power_w"), "policy_adp": r.get("adp"),
                "adp_reduction_pct": r.get("adp_reduction_pct"),
                "policy_aspect_ratio": r.get("aspect_ratio"),
                "policy_grid_w": r.get("grid_w"), "policy_grid_h": r.get("grid_h"),
                "vtr_success": r["vtr_success"], "vtr_seconds": r["vtr_seconds"], "cached": r["cached"]}

    run_pool(jobs, work, sink, workers, "F3eval")                       # noqa: F405


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 6)
