"""E1: does the policy actually use the netlist encoding?

Three inference-time corruptions of the netlist pathway only (capacity,
masking, block counts, block order and VTR are untouched), deterministic
rollout + one VTR evaluation each, 6 held-out circuits x 3 seeds.

The control is the existing deterministic rollout: actions from
data_export/logs/det_actions_and_timing.json, metrics from
data_export/results_per_circuit_seed.csv. Nothing is re-run for it.

Usage (inside distrobox ubuntu-work):
    python e1.py rollouts        # rollouts + confidence, no VTR
    python e1.py vtr WORKERS     # the 54 VTR evaluations (resumable)
"""
import sys

import numpy as np

from elib import *  # noqa: F401,F403
import elib

VARIANTS = ["mismatched_netlist", "zero_graph_embedding", "shuffled_netcount", "zero_edges"]

# A <- B: donor pairs with clearly different H-block profiles (an in-distribution but
# WRONG netlist, so it separates "the netlist mattered" from "the input was OOD").
DONOR = {
    "custom_macbuf": "softmax",
    "lightweight_cipher": "softmax",
    "softmax": "arm_core",
    "arm_core": "reduction_layer",
    "reduction_layer": "mkDelayWorker32B",
    "mkDelayWorker32B": "reduction_layer",
}

HDR = ["circuit", "seed", "variant", "donor_circuit", "aspect_ratio", "grid_w", "grid_h",
       "area_mwta", "delay_ns", "power_w", "adp", "adp_reduction_pct",
       "action_sequence_identical_to_control", "n_actions_differing", "permutation_changed",
       "vtr_success", "cached", "vtr_seconds", "source"]
CONF_HDR = ["circuit", "seed", "step_index", "step_kind", "chosen_action", "chosen_action_prob",
            "top1_prob", "entropy_over_legal_actions", "n_legal_actions"]

ACTIONS_JSON = OUT / "logs" / "e1_actions.json"      # noqa: F405
DET_JSON = REPO / "data_export" / "logs" / "det_actions_and_timing.json"   # noqa: F405


def control_actions():
    return {(r["circuit"], r["seed"]): r["actions"] for r in json.loads(DET_JSON.read_text())}  # noqa: F405


def rollouts():
    """Generate every action sequence single-threaded and deterministically, plus
    control-rollout confidence. Writes e1_actions.json and e1_action_confidence.csv."""
    ctrl = control_actions()
    store, conf_rows = {}, []
    for seed in SEEDS:                                   # noqa: F405
        for ci, name in enumerate(HELDOUT):              # noqa: F405
            env = make_env(name)
            model = load_model(seed, env)
            nf = env.reset()[0]["node_features"]

            # control: re-derive actions to confirm they match the logged ones,
            # and collect the per-step confidence (no VTR, no new floorplan).
            acts_c, conf = elib.rollout(model, env, collect_conf=True)
            assert acts_c == ctrl[(name, seed)], (name, seed, acts_c, ctrl[(name, seed)])
            for i, c in enumerate(conf):
                conf_rows.append({"circuit": name, "seed": seed, "step_index": i,
                                  "step_kind": "aspect_ratio" if i == 0 else "hblock", **c})

            n_a = elib.n_valid_nodes(nf) - 1
            for v in VARIANTS:
                rng = np.random.default_rng([seed, ci])
                if v == "mismatched_netlist":
                    fn = elib.make_donor_transform(elib.donor_graph(DONOR[name]), n_a)
                else:
                    fn = elib.make_obs_transform(v, rng, nf)
                if v == "zero_graph_embedding":
                    with elib.ZeroGraphDims(model.policy.features_extractor):
                        acts, _ = elib.rollout(model, env, fn)
                else:
                    acts, _ = elib.rollout(model, env, fn)
                store[f"{name}|{seed}|{v}"] = acts
                nd = sum(1 for a, b in zip(acts, acts_c) if a != b) + abs(len(acts) - len(acts_c))
                print(f"{name:20s} s{seed:<4d} {v:22s} diff={nd}/{len(acts_c)}", flush=True)
            store[f"{name}|{seed}|control"] = acts_c

    ACTIONS_JSON.write_text(json.dumps(store, indent=1))   # noqa: F405
    sink = CsvSink(OUT / "e1_action_confidence.csv", CONF_HDR, ["circuit", "seed", "step_index"])  # noqa: F405
    for r in conf_rows:
        if not sink.has((r["circuit"], r["seed"], r["step_index"])):
            sink.write(r)
    print("wrote", ACTIONS_JSON, "and e1_action_confidence.csv", flush=True)


def perm_changed():
    """circuit-seed -> did the shuffled_netcount permutation change any value?
    (With 3-4 H-blocks many permutations are the identity, and some circuits give
    every H-block the same net count, so the permutation can be a no-op.)"""
    p = OUT / "e1_netcount_perm_check.csv"                 # noqa: F405
    if not p.exists():
        return {}
    with open(p) as fh:
        return {(r["circuit"], int(r["seed"])): r["permutation_is_noop"] == "False"
                for r in csv.DictReader(fh)}               # noqa: F405


def vtr(workers):
    store = json.loads(ACTIONS_JSON.read_text())          # noqa: F405
    ctrl = control_rows()                                  # noqa: F405
    pchg = perm_changed()
    sink = CsvSink(OUT / "e1_encoder_corruption.csv", HDR, ["circuit", "seed", "variant"])  # noqa: F405

    # control rows: reuse Tier B numbers, no VTR run
    for seed in SEEDS:                                     # noqa: F405
        for name in HELDOUT:                               # noqa: F405
            if sink.has((name, seed, "control")):
                continue
            r = ctrl[(name, seed)]
            sink.write({
                "circuit": name, "seed": seed, "variant": "control",
                "aspect_ratio": r["policy_aspect_ratio"], "grid_w": r["policy_grid_w"],
                "grid_h": r["policy_grid_h"], "area_mwta": r["policy_area_mwta"],
                "delay_ns": r["policy_delay_ns"], "power_w": r["policy_power_w"],
                "adp": r["policy_adp"], "adp_reduction_pct": r["adp_reduction_pct"],
                "action_sequence_identical_to_control": True, "n_actions_differing": 0,
                "donor_circuit": "", "permutation_changed": "",
                "vtr_success": True, "cached": "", "vtr_seconds": "",
                "source": "data_export/results_per_circuit_seed.csv (paper zero-shot eval; not re-run). "
                          "vtr_seconds null: the original eval did not log it.",
            })

    jobs = []
    for seed in SEEDS:                                     # noqa: F405
        for name in HELDOUT:                               # noqa: F405
            ca = store[f"{name}|{seed}|control"]
            for v in VARIANTS:
                if sink.has((name, seed, v)):
                    continue
                acts = store[f"{name}|{seed}|{v}"]
                nd = sum(1 for a, b in zip(acts, ca) if a != b) + abs(len(acts) - len(ca))
                jobs.append(({"circuit": name, "seed": seed, "variant": v,
                              "donor_circuit": DONOR[name] if v == "mismatched_netlist" else "",
                              "permutation_changed": (pchg.get((name, seed), "")
                                                      if v == "shuffled_netcount" else ""),
                              "action_sequence_identical_to_control": acts == ca,
                              "n_actions_differing": nd,
                              "source": "data_export2/tools/e1.py rollout + FPGAEnv VTR eval; "
                                        "actions in data_export2/logs/e1_actions.json"},
                             (name, acts)))
    run_pool(jobs, elib.replay, sink, workers, "E1")       # noqa: F405


if __name__ == "__main__":
    if sys.argv[1] == "rollouts":
        rollouts()
    elif sys.argv[1] == "vtr":
        vtr(int(sys.argv[2]))
