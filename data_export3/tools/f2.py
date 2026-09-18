"""F2: is the flat placement distribution real?

Part 1 (no VTR): the full shape of the action distribution at every placement
step of every control rollout -> f2_action_distribution.csv.

Part 2 (60 VTR calls): at the FIRST placement step only, seed 42, evaluate the
top-5 legal actions and 5 random legal actions, each completed deterministically
by the policy and evaluated -> f2_rank_vs_adp.csv. This tests whether higher
probability actually means a better floorplan.

Usage (inside distrobox ubuntu-work):
    python f2.py dist
    python f2.py rank W
"""
import math
import sys

import numpy as np
import torch

from common3 import *  # noqa: F401,F403
import common3 as c3

DIST_HDR = ["circuit", "seed", "step_index", "n_legal_actions", "chosen_action_prob", "top1_prob",
            "top5_mass", "top10_mass", "top1pct_mass", "uniform_reference_top10_mass", "entropy",
            "entropy_over_ln_n_legal", "gini", "source"]

RANK_HDR = ["circuit", "seed", "candidate_kind", "action_rank_if_topk", "action_prob", "action",
            "adp", "adp_reduction_pct", "vtr_success", "vtr_seconds", "cached", "source"]

SEED = 42
SRC = "policy distribution at the control rollout's own observations (no VTR)"


def gini(p):
    """Gini coefficient of the legal-action probability vector."""
    x = np.sort(np.asarray(p, dtype=np.float64))
    n = x.size
    s = x.sum()
    if n == 0 or s <= 0:
        return None
    return float((2.0 * np.arange(1, n + 1).dot(x)) / (n * s) - (n + 1.0) / n)


def step_probs(model, obs, mask):
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs)
        d = model.policy.get_distribution(t, action_masks=mask)
        return d.distribution.probs.squeeze(0).cpu().numpy().astype(np.float64)


def dist():
    sink = CsvSink(OUT / "f2_action_distribution.csv", DIST_HDR,    # noqa: F405
                   ["circuit", "seed", "step_index"])
    det = c3.det_actions()
    for seed in SEEDS:              # noqa: F405
        for name in HELDOUT:        # noqa: F405
            env = c3.make_env(name)
            model = c3.load_model(seed, env)
            obs, _ = env.reset()
            acts = det[(name, seed)]
            for i, a in enumerate(acts):
                mask = env.get_action_mask()
                if i > 0:   # placement steps only; step 0 is the aspect ratio
                    p = step_probs(model, obs, mask)
                    legal = np.flatnonzero(mask)
                    pl = p[legal]
                    pl = pl / pl.sum()
                    n = pl.size
                    srt = np.sort(pl)[::-1]
                    k1 = max(1, int(math.ceil(0.01 * n)))
                    ent = float(-(srt * np.log(np.clip(srt, 1e-30, None))).sum())
                    if not sink.has((name, seed, i)):
                        sink.write({
                            "circuit": name, "seed": seed, "step_index": i,
                            "n_legal_actions": int(n), "chosen_action_prob": float(p[a]),
                            "top1_prob": float(srt[0]),
                            "top5_mass": float(srt[:5].sum()), "top10_mass": float(srt[:10].sum()),
                            "top1pct_mass": float(srt[:k1].sum()),
                            "uniform_reference_top10_mass": min(10.0 / n, 1.0),
                            "entropy": ent,
                            "entropy_over_ln_n_legal": ent / math.log(n) if n > 1 else None,
                            "gini": gini(pl), "source": SRC})
                if i == len(acts) - 1:
                    break
                obs, _, term, _, _ = env.step(a)
                assert not term
            print(f"{name:20s} s{seed} steps={len(acts) - 1}", flush=True)
    print("wrote f2_action_distribution.csv")


def rank_candidates():
    """For each circuit at seed 42: the top-5 legal first-placement actions and 5
    random legal ones, each completed deterministically by the policy."""
    out = {}
    for ci, name in enumerate(HELDOUT):     # noqa: F405
        env = c3.make_env(name)
        model = c3.load_model(SEED, env)
        acts0 = c3.det_actions()[(name, SEED)]
        obs, _ = env.reset()
        obs, _, _, _, _ = env.step(acts0[0])       # step 0: the policy's aspect ratio
        mask = env.get_action_mask()
        p = step_probs(model, obs, mask)
        legal = np.flatnonzero(mask)
        order = legal[np.argsort(-p[legal])]
        top5 = [int(a) for a in order[:5]]
        rng = np.random.default_rng([SEED, ci])
        pool = [int(a) for a in legal if int(a) not in top5]
        rnd = [int(a) for a in rng.choice(pool, size=5, replace=False)]

        cands = [("top_k", r + 1, a) for r, a in enumerate(top5)] + \
                [("random", "", a) for a in rnd]
        for kind, rank, a in cands:
            # complete the rest of the rollout deterministically from this action
            e2 = c3.make_env(name)
            e2.reset()
            e2.step(acts0[0])
            seq = [acts0[0], a]
            if e2._current_step != e2._total_blocks:
                o, _, term, _, _ = e2.step(a)
                while True:
                    m = e2.get_action_mask()
                    na, _ = model.predict(o, action_masks=m, deterministic=True)
                    seq.append(int(na))
                    if e2._current_step == e2._total_blocks:
                        break
                    o, _, term, _, _ = e2.step(int(na))
            out[(name, kind, rank, a)] = (seq, float(p[a]))
        print(f"{name:20s} top5={top5} rnd={rnd}", flush=True)
    return out


def rank(workers):
    cands = rank_candidates()
    sink = CsvSink(OUT / "f2_rank_vs_adp.csv", RANK_HDR,    # noqa: F405
                   ["circuit", "seed", "candidate_kind", "action_rank_if_topk", "action"])
    jobs = []
    for (name, kind, rankk, a), (seq, prob) in cands.items():
        if sink.has((name, SEED, kind, rankk, a)):
            continue
        jobs.append(({"circuit": name, "seed": SEED, "candidate_kind": kind,
                      "action_rank_if_topk": rankk, "action_prob": prob, "action": a,
                      "source": "data_export3/tools/f2.py; first placement step overridden, "
                                "rest of the rollout deterministic"},
                     (name, seq)))
    run_pool(jobs, c3.replay, sink, workers, "F2")    # noqa: F405


if __name__ == "__main__":
    if sys.argv[1] == "dist":
        dist()
    elif sys.argv[1] == "rank":
        rank(int(sys.argv[2]))
