"""E2: aspect ratio vs. H-block placement.

Decomposes the zero-shot gain into the step-0 aspect-ratio choice and the
H-block placement, over the 6 held-out circuits x 3 seeds:

  policy_ar_random_blocks : policy's step-0 AR, then every H-block drawn
                            uniformly over the legal mask, policy's order.
                            3 draws per circuit-seed.
  fixed_ar_policy_blocks  : AR forced to 1.0, policy places H-blocks
                            deterministically. 1 run per circuit-seed.
  fixed_ar_random_blocks  : AR 1.0 + random legal placement. 3 draws.

The policy's own deterministic rollout (AR + blocks) is the control and is
already in data_export/results_per_circuit_seed.csv; it is not re-run.

Usage (inside distrobox ubuntu-work):
    python e2.py rollouts
    python e2.py vtr WORKERS
"""
import sys

import numpy as np

from elib import *  # noqa: F401,F403
import elib
from src.env.fpga_env import ASPECT_RATIOS

HDR = ["circuit", "seed", "variant", "draw_index", "aspect_ratio", "grid_w", "grid_h", "area_mwta",
       "delay_ns", "power_w", "adp", "adp_reduction_pct", "vtr_success", "cached", "vtr_seconds",
       "dsp_coords", "bram_coords", "source"]

# 5 random draws where the flow is fast enough; 3 on the three slowest circuits
# (each of their VTR runs is minutes, and 2 extra draws x 2 random variants x 3
# seeds x 3 circuits would add ~36 long runs).
SLOW = {"reduction_layer", "arm_core", "softmax"}


def n_draws(name):
    return 3 if name in SLOW else 5

ACTIONS_JSON = OUT / "logs" / "e2_actions.json"   # noqa: F405
AR_ONE = ASPECT_RATIOS.index(1.0)                 # step-0 action selecting aspect ratio 1.0


def random_blocks_after(env, ar_action, rng):
    """Force step 0 to ar_action, then fill every H-block uniformly over the
    legal mask, in the env's own block order."""
    env.reset()
    acts = [int(ar_action)]
    env.step(acts[0])
    while True:
        valid = np.flatnonzero(env.get_action_mask())
        acts.append(int(rng.choice(valid)))
        if env._current_step == env._total_blocks:
            return acts
        env.step(acts[-1])


def policy_blocks_after(model, env, ar_action):
    """Force step 0 to ar_action, then let the policy place H-blocks deterministically."""
    obs, _ = env.reset()
    acts = [int(ar_action)]
    obs, _, term, _, _ = env.step(acts[0])
    assert not term
    while True:
        a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        acts.append(int(a))
        if env._current_step == env._total_blocks:
            return acts
        obs, _, term, _, _ = env.step(acts[-1])
        assert not term


def rollouts():
    store = {}
    for seed in SEEDS:                               # noqa: F405
        for ci, name in enumerate(HELDOUT):          # noqa: F405
            env = elib.make_env(name)
            model = load_model(seed, env)            # noqa: F405
            det, _ = elib.rollout(model, env)        # policy's own AR is det[0]
            ar_policy = det[0]

            for draw in range(n_draws(name)):
                rng = np.random.default_rng([seed, ci, draw])
                store[f"{name}|{seed}|policy_ar_random_blocks|{draw}"] = \
                    random_blocks_after(env, ar_policy, rng)
                rng = np.random.default_rng([seed, ci, draw])
                store[f"{name}|{seed}|fixed_ar_random_blocks|{draw}"] = \
                    random_blocks_after(env, AR_ONE, rng)

            store[f"{name}|{seed}|fixed_ar_policy_blocks|0"] = policy_blocks_after(model, env, AR_ONE)
            print(f"{name:20s} s{seed:<4d} policy_ar_action={ar_policy} "
                  f"(={ASPECT_RATIOS[ar_policy]})  ar1.0_action={AR_ONE}", flush=True)
    ACTIONS_JSON.write_text(json.dumps(store, indent=1))   # noqa: F405
    print("wrote", ACTIONS_JSON, flush=True)


def vtr(workers):
    store = json.loads(ACTIONS_JSON.read_text())           # noqa: F405
    sink = CsvSink(OUT / "e2_decomposition.csv", HDR, ["circuit", "seed", "variant", "draw_index"])  # noqa: F405
    jobs = []
    for key, acts in store.items():
        name, seed, variant, draw = key.split("|")
        seed, draw = int(seed), int(draw)
        if sink.has((name, seed, variant, draw)):
            continue
        jobs.append(({"circuit": name, "seed": seed, "variant": variant, "draw_index": draw,
                      "source": "data_export2/tools/e2.py; actions in data_export2/logs/e2_actions.json"},
                     (name, acts)))
    run_pool(jobs, elib.replay, sink, workers, "E2")       # noqa: F405


if __name__ == "__main__":
    if sys.argv[1] == "rollouts":
        rollouts()
    elif sys.argv[1] == "vtr":
        vtr(int(sys.argv[2]))
