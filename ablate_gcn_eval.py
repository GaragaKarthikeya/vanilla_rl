#!/usr/bin/env python3
"""Ablation: zero the GCN branch (graph_embed + current_embed) at inference,
re-run zero-shot eval, compare ADP to the intact policy. Answers: does the
graph encoding actually earn its keep, or do CNN+scalars carry the result?"""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from src.env.fpga_env import FPGAEnv, build_benchmark_configs, compute_max_dims
from src.training.gnn_extractor import GNNFeaturesExtractor
from src.training.ppo import CustomMaskablePPO
from src.utils.config import load_env_file

PROJECT_ROOT = Path(__file__).resolve().parent
ABLATE = {"on": False}

# Monkeypatch forward to optionally zero the two GCN-derived embeddings.
_orig_forward = GNNFeaturesExtractor.forward
def ablated_forward(self, observations):
    grid = observations["grid"]
    node_features = observations["node_features"]
    edge_index = observations["edge_index"]
    edge_weight = observations["edge_weight"]
    current_block_idx = observations["current_block_idx"].long().squeeze(-1).clamp(min=0)
    valid_wh = observations["valid_wh"]
    device = grid.device
    batch = self._build_batch(node_features, edge_index, edge_weight).to(device)
    h = F.relu(self.conv1(batch.x, batch.edge_index, batch.edge_attr))
    h = F.relu(self.conv2(h, batch.edge_index, batch.edge_attr))
    graph_embed = global_mean_pool(h, batch.batch)
    current_global_idx = batch.ptr[:-1] + current_block_idx
    current_embed = h[current_global_idx]
    if ABLATE["on"]:
        graph_embed = torch.zeros_like(graph_embed)
        current_embed = torch.zeros_like(current_embed)
    grid_chw = grid.permute(0, 3, 1, 2)
    cnn_out = self.cnn(grid_chw)
    combined = torch.cat([cnn_out, graph_embed, current_embed, valid_wh], dim=-1)
    return self.final(combined)
GNNFeaturesExtractor.forward = ablated_forward

def run_episode(env, model, max_steps=200):
    obs, _ = env.reset()
    for _ in range(max_steps):
        mask = env.get_action_mask()
        action, _ = model.predict(obs, action_masks=mask, deterministic=True)
        obs, reward, term, trunc, info = env.step(int(action))
        if term or trunc:
            return {"reward": float(reward), **info}
    return {"reward": 0.0, "status": "max_steps", "success": False}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--eval_benchmarks", required=True)
    ap.add_argument("--universe_benchmarks", required=True)
    ap.add_argument("--ablate", action="store_true")
    ap.add_argument("--vtr_timeout", type=int, default=1200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ABLATE["on"] = a.ablate
    try: load_env_file(PROJECT_ROOT/".env")
    except FileNotFoundError: pass
    ev = [b.strip() for b in a.eval_benchmarks.split(",") if b.strip()]
    un = [b.strip() for b in a.universe_benchmarks.split(",") if b.strip()]
    mw,mh,mn,me = compute_max_dims(un)
    print(f"dims W={mw} H={mh} N={mn} E={me}  ABLATE={a.ablate}")
    suffix = "_ablate" if a.ablate else "_intact"
    cfgs = build_benchmark_configs(ev, mw,mh,mn,me, cache_suffix=suffix)
    env = FPGAEnv(cfgs, mw,mh,mn,me, vtr_timeout=a.vtr_timeout)
    model = CustomMaskablePPO.load(a.model_path, env=env,
        policy_kwargs={"features_extractor_class": GNNFeaturesExtractor})
    res = {}
    for cfg in cfgs:
        env._active_config = cfg
        env.benchmark_configs = [cfg]
        info = run_episode(env, model)
        env.benchmark_configs = cfgs
        base = json.loads((PROJECT_ROOT/"baselines"/f"{cfg.name}_traditional_metric.txt").read_text())
        row = {"benchmark":cfg.name,"reward":info.get("reward"),"success":info.get("success"),
               "status":info.get("status"),"delay_ns":info.get("delay_ns"),
               "power_w":info.get("power_w"),"routing_area":info.get("routing_area")}
        if info.get("success") and info.get("routing_area") not in (None,"?"):
            adp = info["routing_area"]*info["delay_ns"]*info["power_w"]
            badp = base["routing_area"]*base["delay_ns"]*base["power_w"]
            row["adp_reduction_pct"] = (badp-adp)/badp*100.0
        res[cfg.name]=row
        print(f"{cfg.name:20s} success={row.get('success')} adp_red%={row.get('adp_reduction_pct')}")
    Path(a.out).write_text(json.dumps(res,indent=2))
    print("wrote", a.out)

if __name__=="__main__":
    main()
