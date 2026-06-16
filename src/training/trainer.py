#!/usr/bin/env python3

import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from src.env.fpga_env import FPGAEnv
from src.training.callbacks import BestLayoutCallback
from src.training.ppo import CustomMaskablePPO

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class TrainConfig:
    benchmark_name: str = "diffeq1"
    n_envs: Optional[int] = None
    timesteps: int = 10_000
    lr: float = 3e-4
    ent_coef: float = 0.0
    batch_size: int = 64
    n_steps: int = 80
    n_epochs: int = 10
    gamma: float = 0.99
    seed: int = 42
    max_episodes: int = 256
    wl_weight: float = 0.00
    pw_weight: float = 0.34
    dl_weight: float = 0.33
    ar_weight: float = 0.33
    cache_db_path: Optional[str] = None
    save_path: Optional[str] = None
    load_path: Optional[str] = None
    log_suffix: str = ""


def train(cfg: TrainConfig) -> None:
    """Entry point for a full training run."""
    benchmark_name = cfg.benchmark_name.removesuffix(".v")

    res_file = PROJECT_ROOT / f"{benchmark_name}_traditional_resources.txt"
    metric_file = PROJECT_ROOT / f"{benchmark_name}_traditional_metric.txt"

    if not res_file.is_file() or not metric_file.is_file():
        print(
            f"Error: Traditional baseline files not found for '{benchmark_name}' in {PROJECT_ROOT}",
            file=sys.stderr,
        )
        sys.exit(1)

    res_data = json.loads(res_file.read_text())
    metric_data = json.loads(metric_file.read_text())

    fpga_size = res_data.get("fpga_size", [0, 0])
    if len(fpga_size) != 2 or not all(fpga_size):
        print(f"Error: Invalid 'fpga_size' in {res_file}", file=sys.stderr)
        sys.exit(1)

    width, height = int(fpga_size[0]), int(fpga_size[1])
    reqs = res_data.get("requirements", {})
    req_dsp = reqs.get("dsp", 0)
    req_bram = reqs.get("bram", 0)

    print("=" * 60)
    print(f"Benchmark          : {benchmark_name}")
    print(f"Core grid          : {width}×{height}  ({width+2}×{height+2} with IO ring)")
    print(f"Required DSPs/BRAMs: {req_dsp} / {req_bram}")
    print(f"Baseline wirelength: {metric_data.get('wirelength')}")
    print(f"Baseline delay (ns): {metric_data.get('delay_ns')}")
    print(f"Baseline power (W) : {metric_data.get('power_w')}")
    print(f"Reward weights     : wl={cfg.wl_weight} pw={cfg.pw_weight} dl={cfg.dl_weight} ar={cfg.ar_weight}")
    print("=" * 60)

    n_envs = cfg.n_envs or min(8, os.cpu_count() or 1)
    print(f"Parallel workers   : {n_envs}")

    db_path = cfg.cache_db_path or str(PROJECT_ROOT / "runs" / f"vtr_layout_cache_{benchmark_name}.db")

    env_kwargs = {
        "benchmark_name": benchmark_name,
        "width": width,
        "height": height,
        "req_dsp": req_dsp,
        "req_bram": req_bram,
        "traditional_metrics": metric_data,
        "cache_db_path": db_path,
        "wl_weight": cfg.wl_weight,
        "pw_weight": cfg.pw_weight,
        "dl_weight": cfg.dl_weight,
        "ar_weight": cfg.ar_weight,
    }

    env = make_vec_env(FPGAEnv, n_envs=n_envs, seed=cfg.seed, vec_env_cls=SubprocVecEnv, env_kwargs=env_kwargs)

    best_layout_cb = BestLayoutCallback(
        benchmark_name=benchmark_name,
        width=width,
        height=height,
        seed_val=cfg.seed,
        trad_metrics=metric_data,
        max_episodes=cfg.max_episodes,
        log_suffix=cfg.log_suffix,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, cfg.n_steps * 5),
        save_path=str(PROJECT_ROOT / "runs" / f"checkpoints_seed_{cfg.seed}"),
        name_prefix=f"{benchmark_name}_model",
    )

    try:
        import tensorboard  # noqa: F401
        tb_log = str(PROJECT_ROOT / "runs" / "tb_logs")
    except ImportError:
        tb_log = None
        print("Warning: TensorBoard not installed — logging disabled.")

    ppo_kwargs = dict(
        learning_rate=cfg.lr,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        ent_coef=cfg.ent_coef,
        seed=cfg.seed,
        verbose=1,
        tensorboard_log=tb_log,
    )

    if cfg.load_path and os.path.exists(cfg.load_path):
        print(f"Loading model from {cfg.load_path}")
        model = CustomMaskablePPO.load(cfg.load_path, env=env, **ppo_kwargs)
    else:
        model = CustomMaskablePPO("MlpPolicy", env, **ppo_kwargs)

    print(f"\nStarting training — target timesteps: {cfg.timesteps}")
    print(f"Rollout buffer size: {n_envs * cfg.n_steps} steps per update")
    print("=" * 60)

    try:
        model.learn(total_timesteps=cfg.timesteps, callback=CallbackList([best_layout_cb, checkpoint_cb]))
        print("\nTraining complete.")
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        if cfg.save_path:
            model.save(cfg.save_path)
            print(f"Model saved to {cfg.save_path}")

        _save_all_layouts(best_layout_cb, benchmark_name, cfg)
        env.close()


def _save_all_layouts(callback: BestLayoutCallback, benchmark_name: str, cfg: TrainConfig) -> None:
    layouts = callback.all_completed_layouts[: cfg.max_episodes]
    dest = PROJECT_ROOT / f"layouts_{benchmark_name}_seed_{cfg.seed}{cfg.log_suffix}.json"
    try:
        dest.write_text(json.dumps(layouts, indent=4))
        print(f"Saved {len(layouts)} layouts → {dest.name}")
    except Exception as exc:
        print(f"Error saving layouts: {exc}", file=sys.stderr)
