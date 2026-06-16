#!/usr/bin/env python3
"""
Vanilla RL training entry point for FPGA placement optimisation.

Usage:
    python train.py --benchmark diffeq1
    python train.py --benchmark diffeq1 --seed 1337 --max_episodes 500 --n_envs 8
    python train.py --help
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.utils.config import load_env_file
from src.training.trainer import TrainConfig, train

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser(description="Vanilla RL — FPGA placement optimisation")
    p.add_argument("--benchmark",    default="diffeq1",  help="Benchmark name (e.g. diffeq1)")
    p.add_argument("--n_envs",       type=int,           help="Parallel env workers (default: min(8, cpu_count))")
    p.add_argument("--timesteps",    type=int, default=10_000, help="Total training timesteps")
    p.add_argument("--lr",           type=float, default=3e-4,  help="PPO learning rate")
    p.add_argument("--ent_coef",     type=float, default=0.0,   help="Entropy coefficient")
    p.add_argument("--batch_size",   type=int,   default=64,    help="Minibatch size")
    p.add_argument("--n_steps",      type=int,   default=80,    help="Steps per env per rollout")
    p.add_argument("--seed",         type=int,   default=42,    help="Random seed")
    p.add_argument("--max_episodes", type=int,   default=256,   help="Episode budget")
    p.add_argument("--ar_weight",    type=float, default=1.0,   help="Routing-area reward weight")
    p.add_argument("--dl_weight",    type=float, default=1.0,   help="Delay reward weight")
    p.add_argument("--pw_weight",    type=float, default=1.0,   help="Power reward weight")
    p.add_argument("--wl_weight",    type=float, default=0.0,   help="Wirelength reward weight")
    p.add_argument("--cache_db_path", default=None,             help="SQLite cache path")
    p.add_argument("--save_path",    default=None,              help="Save trained model (.zip)")
    p.add_argument("--load_path",    default=None,              help="Load existing model (.zip)")
    p.add_argument("--log_suffix",   default="",                help="Suffix for output log files")
    args = p.parse_args()

    return TrainConfig(
        benchmark_name=args.benchmark,
        n_envs=args.n_envs,
        timesteps=args.timesteps,
        lr=args.lr,
        ent_coef=args.ent_coef,
        batch_size=args.batch_size,
        n_steps=args.n_steps,
        seed=args.seed,
        max_episodes=args.max_episodes,
        wl_weight=args.wl_weight,
        pw_weight=args.pw_weight,
        dl_weight=args.dl_weight,
        ar_weight=args.ar_weight,
        cache_db_path=args.cache_db_path,
        save_path=args.save_path,
        load_path=args.load_path,
        log_suffix=args.log_suffix,
    )


if __name__ == "__main__":
    try:
        load_env_file(PROJECT_ROOT / ".env")
    except FileNotFoundError:
        pass  # optional; paths can be set via env vars

    train(parse_args())
