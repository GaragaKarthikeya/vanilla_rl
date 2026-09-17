"""Rollout / evaluation helpers shared by E1 and E2.

Everything here goes through the paper's own FPGAEnv, baker, VTRRunner and
metric path. No training or eval code is modified: the only intervention is a
forward-pre-hook on the features extractor's fuse layer (E1
zero_graph_embedding) and observation-dict rewriting before model.predict
(E1 shuffled_netcount / zero_edges), both applied at inference only.
"""
import numpy as np
import torch

from common2 import *  # noqa: F401,F403
import common2 as _c2
import c345  # data_export/tools/c345.py (on sys.path via common2)

from src.env.fpga_env import ASPECT_RATIOS  # noqa: F401

CACHE = _c2.CACHE_SUFFIX


def make_env(name, timeout=1800):
    return c345.make_env(name, cache_suffix=CACHE, timeout=timeout)


load_model = c345.load_model
set_paper_weights = c345.set_paper_weights
run_pool = c345.run_pool


def replay(name, acts):
    """Evaluate an action sequence end-to-end (bake -> pin -> VTR -> metrics).

    Same body as data_export/tools/c345.py:replay, but on the data_export2
    isolated cache.
    """
    env = make_env(name)
    set_paper_weights(env)
    env.reset()
    for a in acts[:-1]:
        env.step(a)
    t0 = now()  # noqa: F405
    _, reward, term, _, info = env.step(acts[-1])
    secs = now() - t0  # noqa: F405
    assert term
    m, _ = baseline(name)  # noqa: F405
    row = {"aspect_ratio": info.get("aspect_ratio"), "vtr_success": bool(info.get("success")),
           "cached": bool(info.get("cached")), "vtr_seconds": round(secs, 2), "reward": reward,
           "dsp_coords": json.dumps(info.get("placed_dsps")),        # noqa: F405
           "bram_coords": json.dumps(info.get("placed_brams"))}      # noqa: F405
    if info.get("success"):
        a = adp(info["routing_area"], info["delay_ns"], info["power_w"])   # noqa: F405
        b = adp(m["routing_area"], m["delay_ns"], m["power_w"])            # noqa: F405
        row.update(area_mwta=info["routing_area"], delay_ns=info["delay_ns"], power_w=info["power_w"],
                   adp=a, adp_reduction_pct=(b - a) / b * 100,
                   grid_w=info["grid_W"], grid_h=info["grid_H"])
    return row


# ---------------------------------------------------------------- E1 corruptions

def n_valid_nodes(node_features):
    """Valid graph nodes = rows with a type indicator set (graph_reduction.py:206-207
    pads with all-zero rows)."""
    return int((node_features[:, :3].sum(axis=-1) > 0).sum())


def make_obs_transform(variant, rng, node_features):
    """Return f(obs)->obs applying the E1 netlist-pathway corruption.

    Node feature layout is [is_dsp, is_bram, is_fabric, net_count_normalized]
    (graph_reduction.py:29). H-block nodes are rows 0..n_valid-2; the fabric
    node is the last valid row (graph_reduction.py:138-148).
    """
    if variant in ("control", "zero_graph_embedding"):
        return lambda obs: obs

    if variant == "shuffled_netcount":
        n = n_valid_nodes(node_features)
        perm = rng.permutation(n - 1)  # H-blocks only; fabric node untouched

        def f(obs):
            o = dict(obs)
            nf = np.array(obs["node_features"], copy=True)
            nf[: n - 1, 3] = nf[perm, 3]
            o["node_features"] = nf
            return o
        return f

    if variant == "zero_edges":
        def f(obs):
            o = dict(obs)
            o["edge_index"] = np.full_like(np.asarray(obs["edge_index"]), -1.0)
            o["edge_weight"] = np.zeros_like(np.asarray(obs["edge_weight"]))
            return o
        return f

    raise ValueError(variant)


def donor_graph(donor_name):
    """Padded graph tensors of the donor circuit B, plus its H-block node count."""
    env = make_env(donor_name)
    o = env.reset()[0]
    nf = np.array(o["node_features"], copy=True)
    return {"node_features": nf,
            "edge_index": np.array(o["edge_index"], copy=True),
            "edge_weight": np.array(o["edge_weight"], copy=True),
            "n_hblocks": n_valid_nodes(nf) - 1}


def make_donor_transform(donor, n_hblocks_a):
    """E1 mismatched_netlist: swap in donor circuit B's whole netlist graph while
    circuit A's capacity, mask, H-block count and placement order are untouched.

    current_block_idx is remapped so it stays a valid node of B:
      * an H-block node i of A  -> B's H-block node (i mod n_B)
      * A's fabric node (id n_A) -> B's fabric node (id n_B)
    """
    n_b = donor["n_hblocks"]

    def f(obs):
        o = dict(obs)
        o["node_features"] = donor["node_features"]
        o["edge_index"] = donor["edge_index"]
        o["edge_weight"] = donor["edge_weight"]
        i = int(np.asarray(obs["current_block_idx"]).ravel()[0])
        j = n_b if i >= n_hblocks_a else i % n_b
        o["current_block_idx"] = np.array([j], dtype=np.float32)
        return o
    return f


class ZeroGraphDims:
    """Zero the 128 graph-derived dims of the 194-d fused vector, after the GCN
    and before the fuse layer.

    gnn_extractor.py:102 builds
        combined = cat([cnn_out(64), graph_embed(64), current_embed(64), valid_wh(2)])
    and gnn_extractor.py:56-59 feeds it to self.final. Slots 64:192 are exactly
    the mean-pooled netlist embedding and the current-node embedding.
    """

    def __init__(self, extractor, lo=64, hi=192):
        self.h = None
        self.lo, self.hi = lo, hi
        self.extractor = extractor

    def __enter__(self):
        def hook(_mod, inputs):
            x = inputs[0].clone()
            x[..., self.lo:self.hi] = 0.0
            return (x,)
        self.h = self.extractor.final.register_forward_pre_hook(hook)
        return self

    def __exit__(self, *exc):
        self.h.remove()
        return False


def rollout(model, env, obs_fn=lambda o: o, deterministic=True, collect_conf=False):
    """Deterministic rollout with no terminal VTR call. Returns (actions, conf_rows)."""
    obs, _ = env.reset()
    acts, conf = [], []
    while True:
        mask = env.get_action_mask()
        o = obs_fn(obs)
        if collect_conf:
            conf.append(action_confidence(model, o, mask))
        a, _ = model.predict(o, action_masks=mask, deterministic=deterministic)
        a = int(a)
        acts.append(a)
        if collect_conf:
            conf[-1]["chosen_action"] = a
            conf[-1]["chosen_action_prob"] = float(conf[-1].pop("_probs")[a])
        if env._current_step == env._total_blocks:
            return acts, conf
        obs, _, term, _, _ = env.step(a)
        assert not term


def action_confidence(model, obs, mask):
    """Probabilities over the legal action set at this step."""
    from stable_baselines3.common.utils import obs_as_tensor
    with torch.no_grad():
        t, _ = model.policy.obs_to_tensor(obs)
        dist = model.policy.get_distribution(t, action_masks=mask)
        p = dist.distribution.probs.squeeze(0).cpu().numpy().astype(np.float64)
    legal = np.flatnonzero(mask)
    pl = p[legal]
    pl = pl / pl.sum()
    ent = float(-(pl * np.log(np.clip(pl, 1e-30, None))).sum())
    return {"_probs": p, "top1_prob": float(p.max()), "entropy_over_legal_actions": ent,
            "n_legal_actions": int(len(legal))}
