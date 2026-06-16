# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Quick Start

This is a clean RL training pipeline for FPGA placement optimization using VTR (Verilog-to-Routing). All commands must use `/home/digital-2/.venv/bin/python3` since the system Python doesn't have required packages.

**Essential environment setup:**
- `.env` file contains paths to VTR toolchain (venv, flow script, power tech file)
- Pre-computed baseline metric and resource files required before training: `{benchmark}_traditional_metric.txt`, `{benchmark}_traditional_resources.txt`

## Commands

### Training
```bash
# Full training run with defaults (diffeq1 benchmark)
/home/digital-2/.venv/bin/python3 train.py

# Training with specific benchmark and seed
/home/digital-2/.venv/bin/python3 train.py --benchmark diffeq1 --seed 42 --max_episodes 256

# With parallelization and custom hyperparameters
/home/digital-2/.venv/bin/python3 train.py --benchmark conv_layer --n_envs 8 --timesteps 50000 --lr 1e-4

# Save trained model
/home/digital-2/.venv/bin/python3 train.py --benchmark diffeq1 --save_path runs/model.zip

# Load and continue training
/home/digital-2/.venv/bin/python3 train.py --benchmark diffeq1 --load_path runs/model.zip --timesteps 20000

# With custom reward weights (default: ar=0.33, dl=0.33, pw=0.34, wl=0.00)
/home/digital-2/.venv/bin/python3 train.py --benchmark diffeq1 --ar_weight 0.4 --dl_weight 0.3 --pw_weight 0.3
```

### Baseline evaluation
```bash
# Generate traditional (non-RL) baseline for a benchmark
/home/digital-2/.venv/bin/python3 run_traditional_flow.py --benchmark diffeq1

# With custom architecture file
/home/digital-2/.venv/bin/python3 run_traditional_flow.py --benchmark diffeq1 --arch arch/k6_N10_I40_Fi6_L4_frac0_ff1_C5_45nm.xml

# Without power estimation (faster)
/home/digital-2/.venv/bin/python3 run_traditional_flow.py --benchmark diffeq1 --no-power
```

### Available benchmarks
37 Verilog designs in `benchmarks/`: diffeq1, diffeq2, conv_layer, arm_core, aes_cipher, bfly, bgm, blob_merge, boundtop, ch_intrinsics, etc.

## Architecture

### High-level workflow
1. **Baseline generation**: `run_traditional_flow.py` runs VTR synthesis/placement/routing on default architecture, extracts delay/power/wirelength/routing-area metrics
2. **RL training**: `train.py` creates parallel `FPGAEnv` instances, trains `CustomMaskablePPO` agent to place DSP/BRAM blocks on custom-generated architectures
3. **Evaluation**: Each episode runs VTR flow on agent-generated architecture XML via `vtr_runner.py`; results cached in SQLite

### Key modules

**`src/utils/`**
- `config.py`: `VTRPaths` (resolves venv/flow/power paths from .env), `load_env_file()`
- `cache.py`: `LayoutCache` (SQLite wrapper for VTR results), `CacheRow` (metrics struct)

**`src/evaluation/`**
- `vtr_runner.py`: `VTRRunner` (runs VTR flow subprocess), `VTRMetrics` (delay/wirelength/power/routing-area), `VTRResources` (FPGA size and requirements)

**`src/env/`**
- `fpga_env.py`: `FPGAEnv` — Gymnasium environment for heterogeneous block placement
  - Observation: Box(W, H, 3) with occupancy grid, block-type hint, aspect-ratio channel
  - Action space: Discrete(W * H); step 0 selects aspect ratio, steps 1–N place blocks
  - Episode: select aspect ratio → place all DSPs → place all BRAMs → VTR evaluate → return reward

**`src/layout/`**
- `baker.py`: `bake_layout()` — renders DSP/BRAM placement + aspect ratio → Jinja2-templated VTR architecture XML

**`src/training/`**
- `ppo.py`: `CustomMaskablePPO` — extends `sb3_contrib.MaskablePPO` with per-environment gradient variance diagnostics (batch_reward_variance, gradient_variance_norm, avg_cosine_similarity, global_grad_norm, policy_entropy, value_loss, explained_variance)
- `trainer.py`: `TrainConfig` (dataclass for all hyperparameters), `train()` (orchestrates env creation, model training, callbacks, tensorboard logging)
- `callbacks.py`: custom callbacks (e.g., `BestLayoutCallback`)

### Data flow for a training step
1. Agent selects aspect ratio (action 0)
2. Agent places DSPs then BRAMs sequentially (actions 1 onward)
3. `FPGAEnv` calls `bake_layout()` to generate architecture XML
4. `VTRRunner` executes VTR flow script, parses metrics from output
5. `LayoutCache` caches results by (dsp_coords, bram_coords, aspect_ratio) key
6. Reward computed as: `-(ar_w * log(area/baseline) + dl_w * log(delay/baseline) + pw_w * log(power/baseline) + wl_w * log(wirelength/baseline))`
7. Metrics returned as observation channels + reward

### Reward function
Negative log-ratio against VTR baseline:
```
reward = -(
    ar_weight * log(routing_area / baseline_area) +
    dl_weight * log(delay / baseline_delay) +
    pw_weight * log(power / baseline_power) +
    wl_weight * log(wirelength / baseline_wirelength)
)
```
Default weights: ar=0.33, dl=0.33, pw=0.34, wl=0.00 (no wirelength in reward).

### Key hyperparameters (trainable via CLI)
- `--timesteps`: total environment steps (default 10000)
- `--n_envs`: parallel workers (default 8, capped at CPU count)
- `--lr`: PPO learning rate (default 3e-4)
- `--batch_size`: minibatch size (default 64)
- `--n_steps`: rollout steps per env (default 80)
- `--seed`: random seed
- `--max_episodes`: episode budget before termination
- Reward weights: `--ar_weight`, `--dl_weight`, `--pw_weight`, `--wl_weight`

### Cache behavior
- SQLite cache at `runs/vtr_layout_cache_{benchmark}.db`
- Keyed by `{dsp_coords}|{bram_coords}|{aspect_ratio}`
- Shared across all training runs for the same benchmark
- Avoids re-running expensive VTR evaluations

### Block constraints (from VTR architecture)
- DSP tile height: 4 rows
- BRAM tile height: 6 rows
- These are architecture-specific; templates in `template/` define the layout

## Development notes

### Adding a new benchmark
1. Place Verilog file in `benchmarks/{name}.v`
2. Run `run_traditional_flow.py --benchmark {name}` to generate baseline files
3. Train: `train.py --benchmark {name}`

### Modifying the reward function
Edit reward computation in `src/env/fpga_env.py` (typically in `FPGAEnv._calculate_reward()`) or adjust weights via CLI `--*_weight` flags.

### Tuning PPO stability
The `CustomMaskablePPO` class computes 7 diagnostics at each train step:
1. batch_reward_variance
2. gradient_variance_norm
3. avg_cosine_similarity
4. global_grad_norm
5. policy_entropy
6. value_loss
7. explained_variance

These are logged to TensorBoard under `runs/tb_logs/` for inspection.

### Architecture templates
Three pre-defined template XMLs in `template/`:
- `k6_frac_N10_mem32K_40nm.xml.j2` (default; 10-bit LUT, 32KB BRAM)
- Others available; customize block types/positions via Jinja2 variables

Bake a custom layout: `bake_layout(benchmark, dsps, brams, width, height, template_name=...)`

### Environment variables (set in .env)
- `VTR_VENV_PATH`: path to Python venv (default /home/digital-2/.venv)
- `VTR_FLOW_SCRIPT`: path to VTR run_vtr_flow.py
- `VTR_POWER_TECH_FILE`: path to 45nm.xml power model
