"""C3 (random legal floorplans), C4 (VPR placer-seed noise), C5 (sampled rollouts),
plus policy-only rollout timing for inference_timing.csv.

Usage (inside distrobox ubuntu-work, from anywhere):
    python c345.py c3 N_SAMPLES WORKERS
    python c345.py c4 WORKERS
    python c345.py c5 K WORKERS
    python c345.py timing
"""
import shutil
import subprocess
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch

from common import *  # noqa
from src.env.fpga_env import ASPECT_RATIOS, FPGAEnv, build_benchmark_configs
from src.evaluation.vtr_runner import VTRRunner
from src.layout.baker import bake_layout
from src.training.gnn_extractor import GNNFeaturesExtractor
from src.training.ppo import CustomMaskablePPO


def make_env(name, cache_suffix=CACHE_SUFFIX, timeout=1800):
    cfgs = build_benchmark_configs([name], MAX_W, MAX_H, MAX_NODES, MAX_EDGES, cache_suffix=cache_suffix)
    return FPGAEnv(cfgs, MAX_W, MAX_H, MAX_NODES, MAX_EDGES, vtr_timeout=timeout)  # weights set below


def set_paper_weights(env):
    # Paper runs: --ar_weight 1 --dl_weight 1 --pw_weight 1 --wl_weight 0
    env.ar_weight, env.dl_weight, env.pw_weight, env.wl_weight = 1.0, 1.0, 1.0, 0.0


def load_model(seed, env):
    return CustomMaskablePPO.load(CKPT[seed], env=env, device="cpu",
                                  policy_kwargs={"features_extractor_class": GNNFeaturesExtractor})


def policy_actions(model, env, deterministic, torch_seed=None):
    """Roll out the policy WITHOUT the terminal step (no VTR). Returns action list."""
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    obs, _ = env.reset()
    acts = []
    while True:
        a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=deterministic)
        a = int(a)
        acts.append(a)
        if env._current_step == env._total_blocks:  # this action is the terminal placement
            return acts
        obs, _, term, _, _ = env.step(a)
        assert not term


def random_actions(env, rng):
    env.reset()
    acts = [int(rng.integers(0, len(ASPECT_RATIOS)))]
    env.step(acts[0])
    while True:
        valid = np.flatnonzero(env.get_action_mask())
        a = int(rng.choice(valid))
        acts.append(a)
        if env._current_step == env._total_blocks:
            return acts
        env.step(a)


def replay(name, acts):
    """Evaluate an action sequence through FPGAEnv (same bake/pin/VTR/cache path as training)."""
    env = make_env(name)
    set_paper_weights(env)
    env.reset()
    for a in acts[:-1]:
        env.step(a)
    t0 = now()
    _, reward, term, _, info = env.step(acts[-1])
    secs = now() - t0
    assert term
    m, _ = baseline(name)
    row = {"aspect_ratio": info.get("aspect_ratio"), "vtr_success": bool(info.get("success")),
           "cached": bool(info.get("cached")), "vtr_seconds": round(secs, 2), "reward": reward,
           "dsp_coords": json.dumps(info.get("placed_dsps")), "bram_coords": json.dumps(info.get("placed_brams"))}
    if info.get("success"):
        a = adp(info["routing_area"], info["delay_ns"], info["power_w"])
        b = adp(m["routing_area"], m["delay_ns"], m["power_w"])
        row.update(area_mwta=info["routing_area"], delay_ns=info["delay_ns"], power_w=info["power_w"],
                   adp=a, adp_reduction_pct=(b - a) / b * 100, grid_w=info["grid_W"], grid_h=info["grid_H"])
    return row


def run_pool(jobs, fn, sink, workers, label):
    print(f"{label}: {len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(fn, *j[1]): j[0] for j in jobs}
        for f in as_completed(futs):
            base = futs[f]
            try:
                row = {**base, **f.result()}
            except Exception as exc:
                row = {**base, "vtr_success": False}
                print("ERR", base, repr(exc), flush=True)
            sink.write(row)
            print("done", label, base, row.get("vtr_success"), row.get("vtr_seconds"), flush=True)


# ---------------------------------------------------------------- C3
C3_HDR = ["circuit", "seed", "sample_index", "aspect_ratio", "area_mwta", "delay_ns", "power_w", "adp",
          "adp_reduction_pct", "vtr_success", "vtr_seconds", "cached", "grid_w", "grid_h", "dsp_coords", "bram_coords"]


def c3(n_samples, workers):
    sink = CsvSink(OUT / "random_floorplans.csv", C3_HDR, ["circuit", "seed", "sample_index"])
    jobs = []
    for name in HELDOUT:
        env = make_env(name)
        for seed in SEEDS:
            rng = np.random.default_rng([seed, HELDOUT.index(name)])
            for i in range(n_samples):
                acts = random_actions(env, rng)  # always generated so indices stay reproducible
                if not sink.has((name, seed, i)):
                    jobs.append(({"circuit": name, "seed": seed, "sample_index": i}, (name, acts)))
    run_pool(jobs, replay, sink, workers, "C3")


# ---------------------------------------------------------------- C5
C5_HDR = ["circuit", "seed", "sample_index", "adp_reduction_pct", "vtr_success", "aspect_ratio", "area_mwta",
          "delay_ns", "power_w", "adp", "vtr_seconds", "cached", "grid_w", "grid_h", "dsp_coords", "bram_coords"]


def c5(k, workers):
    sink = CsvSink(OUT / "policy_sampled.csv", C5_HDR, ["circuit", "seed", "sample_index"])
    jobs = []
    for seed in SEEDS:
        for name in HELDOUT:
            env = make_env(name)
            model = load_model(seed, env)
            for i in range(k):
                acts = policy_actions(model, env, deterministic=False, torch_seed=seed * 1000 + i)
                if not sink.has((name, seed, i)):
                    jobs.append(({"circuit": name, "seed": seed, "sample_index": i}, (name, acts)))
    run_pool(jobs, replay, sink, workers, "C5")


# ---------------------------------------------------------------- timing (B7)
def timing():
    rows = []
    for seed in SEEDS:
        for name in HELDOUT:
            env = make_env(name)
            model = load_model(seed, env)
            policy_actions(model, env, deterministic=True)  # warm-up (first call pays lazy init)
            t0 = now()
            acts = policy_actions(model, env, deterministic=True)
            secs = now() - t0
            rows.append({"circuit": name, "seed": seed, "rollout_seconds_policy_only": round(secs, 4),
                         "actions": acts})
            print(name, seed, secs, flush=True)
    (OUT / "logs" / "det_actions_and_timing.json").write_text(json.dumps(rows, indent=1))


# ---------------------------------------------------------------- C4
C4_HDR = ["circuit", "floorplan", "vpr_seed", "area_mwta", "delay_ns", "power_w", "adp", "vtr_seconds",
          "vtr_success", "aspect_ratio", "grid_w", "grid_h"]


def c4_one(name, floorplan, vpr_seed, acts):
    wid = uuid.uuid4().hex[:8]
    rundir = REPO / "runs" / f"temp_run_dx_{wid}"
    runner = VTRRunner()
    extra = {}
    if floorplan == "baseline":
        arch, cons = REPO / "arch" / "k6_frac_N10_mem32K_40nm.xml", None
    else:
        env = make_env(name)
        env.reset()
        for a in acts[:-1]:
            env.step(a)
        # decode terminal placement exactly as FPGAEnv.step does, without triggering VTR
        a = acts[-1]
        x, y = 1 + a // MAX_H, 1 + a % MAX_H
        bt = env._blocks_to_place[env._current_step - 1]
        (env._placed_dsps if bt == 1 else env._placed_brams).append((x, y))
        cfg = env._active_config
        names = cfg.dsp_block_names + cfg.bram_block_names
        arch = REPO / f"temp_arch_dx_{wid}.xml"
        cons = REPO / f"temp_constraints_dx_{wid}.xml"
        bake_layout(name, env._placed_dsps, env._placed_brams, width=cfg.width + 2, height=cfg.height + 2,
                    output_path=str(arch), aspect_ratio=env._chosen_aspect_ratio,
                    block_names=names or None, constraints_output_path=str(cons) if names else None)
        extra["aspect_ratio"] = env._chosen_aspect_ratio
        if not names:
            cons = None
    rundir.mkdir(parents=True, exist_ok=True)
    cmd = runner._build_cmd(REPO / "benchmarks" / f"{name}.v", arch, rundir, True, cons) + ["--seed", str(vpr_seed)]
    t0 = now()
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=3600).returncode
    secs = now() - t0
    row = {"vtr_seconds": round(secs, 2), "vtr_success": False, **extra}
    vm = VTRRunner.parse_metrics(rundir / "vpr.out", rundir / "vpr.crit_path.out", rundir / f"{name}.power")
    # confirm the seed actually reached VPR
    vo = (rundir / "vpr.out").read_text(errors="ignore") if (rundir / "vpr.out").exists() else ""
    row["seed_seen"] = f"placer_opts.seed: {vpr_seed}" in vo
    if rc == 0 and vm.is_complete():
        res = VTRRunner.parse_resources(rundir / "vpr.out")
        row.update(area_mwta=vm.routing_area, delay_ns=vm.delay_ns, power_w=vm.power_w,
                   adp=adp(vm.routing_area, vm.delay_ns, vm.power_w), vtr_success=True,
                   grid_w=res.fpga_size[0] + 2, grid_h=res.fpga_size[1] + 2)
    if floorplan != "baseline":
        arch.unlink(missing_ok=True)
        if cons:
            cons.unlink(missing_ok=True)
    shutil.rmtree(rundir, ignore_errors=True)
    if not row.pop("seed_seen"):
        raise RuntimeError("VPR seed not applied")
    return row


def c4(workers):
    det = {(r["circuit"], r["seed"]): r["actions"]
           for r in json.loads((OUT / "logs" / "det_actions_and_timing.json").read_text())}
    sink = CsvSink(OUT / "vpr_seed_noise.csv", C4_HDR, ["circuit", "floorplan", "vpr_seed"])
    jobs = []
    for name in HELDOUT:
        for fp in ["baseline", "policy_seed42"]:
            for s in [1, 2, 3, 4, 5]:
                if not sink.has((name, fp, s)):
                    jobs.append(({"circuit": name, "floorplan": fp, "vpr_seed": s},
                                 (name, fp, s, det[(name, 42)])))
    run_pool(jobs, c4_one, sink, workers, "C4")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "c3":
        c3(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "c5":
        c5(int(sys.argv[2]), int(sys.argv[3]))
    elif cmd == "c4":
        c4(int(sys.argv[2]))
    elif cmd == "timing":
        timing()
