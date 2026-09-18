# data_export: FPGA'27 revision data

Location: `~/workspace/rl_gnn/vanilla_rl/data_export/`. Paths below are relative to `~/workspace/rl_gnn/vanilla_rl` unless prefixed.

The paper's numbers came from **this** repo (`vanilla_rl`, git HEAD `fa1bd3a`, no uncommitted changes in `src/` or `train.py`) and from the GA in `~/workspace/vtr_exp/ga_agent.py`. The session was started in `~/something_new`, a later fork that did not produce them.

No training code, eval code, checkpoints, or paper cache DBs were modified. New code lives only in `data_export/tools/`.

## Checkpoints used

| seed | checkpoint |
|---|---|
| 7 | `runs/multi11_long_seed7.zip` |
| 42 | `runs/multi11_long_seed42_v2.zip` |
| 123 | `runs/multi11_long_seed123.zip` |

These are the final `model.save()` of each run, and the same files the paper's zero-shot evals loaded (`runs/det_seed*_*.log`).

Caveat: the paper's seed-42 in-pool **pooled** number (35.73%) comes from an older seed-42 run whose checkpoint is gone. See `mismatches.md` §1.

## Files

| file | tier | status |
|---|---|---|
| `facts.json` | A1–A12 | done; every leaf is `{value, source, note}` |
| `results_per_circuit_seed.csv` | B1 | done, 51 rows. Train = `best_during_training` (what the paper used); held-out = `final_checkpoint_deterministic` |
| `circuits.csv` | B2 | done, 17 rows (fifo `source_suite` null: not found in the VTR tree) |
| `training_accounting.csv` | B3 | done; cache hits/misses, cpu_model, cpu_cores_used are null (not logged; see source column) |
| `vtr_runtime.csv` | B4 | logged timings exist only for the 6 held-out circuits (GA stdout logs, 8 concurrent flows); the 11 train circuits are null. For all 17 circuits, measured timings are in the `vtr_seconds` column of `baseline_ar_sweep.csv` (C2) |
| `ga_runs.csv` | B5 | done, 18 rows; wall_clock_hours and best grid null (not logged; log file timestamps are unreliable) |
| `ga_convergence.csv` | B6 | done; GA logs once per generation, so `vtr_call_index` = (generation+1) × 8 |
| `inference_timing.csv` | B7 | NOT originally logged. Rollout time measured now for all 18 (CPU); `vtr_seconds` only for seed 42 (from C4, VPR seed 1) |
| `learning_curves.csv` | B8 | done, 39,012 episodes |
| `figures.md`, `fig_seed_variance.pdf` | B9 | done; new PDF is exactly 3.33 in wide, all text 8 pt, verified programmatically + visually |
| `inpool_final_checkpoint.csv` | C1 | done with **no new runs**: exactly this eval already existed (`runs/inpool_seed{7,42_v2,123}.json`, final checkpoints, deterministic). Aspect ratio/grid were not recorded there → null |
| `baseline_ar_sweep_perar.csv` | C2 (corrected) | done 2026-09-18, 340/340; per-AR tile placement, held-out mean 28.56% |
| `baseline_ar_sweep.csv` | C2 | done, 340/340 VTR runs succeeded. **Methodologically flawed; see the C2 caveat below and `mismatches.md` §7** |
| `random_floorplans.csv` | C3 | done at **20 samples per circuit-seed** (360 runs, all succeeded), **not 720** |
| `vpr_seed_noise.csv` | C4 | done, 60/60 |
| `policy_sampled.csv` | C5 | done, k=10: 180/180 |
| `mismatches.md` | — | paper-number cross-check + code-vs-text discrepancies |
| `logs/` | — | per-stage logs, `tier_c_timeline.txt`, deterministic action sequences + timings, figure preview |

## What was skipped and why

- **C3 at 720 samples.** 18 circuit-seeds × 720 = 12,960 VTR runs. The held-out circuits take 35–520 s per run (reduction_layer/arm_core/softmax median 150–250 s), which is about 1.5 days at 16 concurrent flows. Per the instructions, 20 per circuit-seed were run, so best-of-720 cannot be computed. `tools/c345.py c3 N W` resumes and extends to larger N with the same reproducible sample sequence (samples 0–19 stay identical).
- **Nothing else was skipped.** Null fields are explained in the relevant `source`/`note`.

## Tier C method notes

- **Toolchain.** VTR 9.0.0-dev `22a09d39ef`, the same build as the paper. Run inside the `ubuntu-work` distrobox; the host RHEL 8 glibc cannot run `vpr`.
  - The repo `.env` points to the pre-move path `/home/digital-2/vtr-verilog-to-routing`. The drivers set `VTR_*` env vars to `~/workspace/vtr-verilog-to-routing` instead of calling `load_env_file()`.
  - Sanity check: at VPR seed 1, all 12 C4 floorplans (6 baselines + 6 seed-42 policy floorplans) reproduce the paper's recorded ADP exactly (ratio 1.0000).
- **Universe dims.** Pinned to the logged training values (48, 48, 67, 2694), because `baselines/robot_rl_*` is now deleted in the working tree and `compute_max_dims` cannot be re-run.
- **Caching.** Every Tier C evaluation used an **isolated** cache (`runs/vtr_layout_cache_<circuit>_dataexport.db`) so the paper's cache DBs were not written to and every run is a real VTR run with a wall-clock time. There were 0 cache hits in C3/C5.
  - C1 reused existing results, which came from the paper's shared caches.
- **C2 caveat (found after delivery): this sweep is a strawman.** The H-block
  tile coordinates below are computed once on the square baseline grid and do
  not change with the aspect ratio. VPR must then stretch the grid to keep every
  fixed tile on the fabric, adding CLB area. For example mkDelayWorker32B goes
  to 44x440 at AR 0.1 and 95x50 at AR 1.9, while the policy fits the same 43
  BRAMs in 34x38. So `baseline_ar_sweep.csv`, and the 14.86% held-out mean
  derived from it, **understate a fair non-learning AR sweep**. A corrected
  sweep with per-aspect-ratio tile placement is `baseline_ar_sweep_perar.csv`
  (340/340 succeeded, `tools/c2_perar.py`): its held-out mean is **28.56%**, not
  14.86%, against the policy's 31.23%. See `mismatches.md` §7a.
- **C2 trimmed baseline, H-block positioning.**
  - Same column pattern as `arch/k6_frac_N10_mem32K_40nm.xml`: DSP columns x = 6, 14, 22, … and BRAM columns x = 2, 10, 18, …, tiles stacked from y = 1 at pitch 4 (DSP) / 6 (BRAM).
  - Only the netlist's required count is kept, filling the lowest slot of the leftmost column first, within the baseline core grid (x ≤ W, y+h−1 ≤ H), the same bounds as the policy mask.
  - Rendered through the paper template (`auto_layout aspect_ratio=AR` with `<single>` tiles).
  - No atom-pinning constraints file (the baseline flow pins nothing); default VPR seed.
  - Exact coordinates are in the `dsp_coords`/`bram_coords` columns.
- **C3 random floorplans.**
  - Aspect ratio uniform over the 20 values.
  - Each H-block uniform over the currently legal actions of the paper's own `FPGAEnv.get_action_mask()`, in the policy's order (BRAMs, then DSPs).
  - Evaluated through `FPGAEnv.step` (same bake, atom pinning, VTR, metric path as the policy).
  - RNG `numpy.default_rng([seed, circuit_index])`.
- **C4 VPR seed noise.**
  - `--seed N` appended to `run_vtr_flow.py`, which passes it to VPR; each run checks that `placer_opts.seed: N` appears in `vpr.out`.
  - The seed changes placement in the min-W VPR run; the relaxed-W run is route-only.
  - Baseline = stock arch; policy = the deterministic seed-42 floorplan re-baked with its constraints file.
- **C5 sampled rollouts.** `model.predict(deterministic=False)` on CPU, `torch.manual_seed(seed*1000 + k)`, evaluated like C3.

## Tier C headline numbers (computed from the CSVs; no new claims beyond these)

ADP reduction %, held-out circuits. "best-of-20" and "best-of-10" are the mean over the 3 seeds of the per-seed best.

| circuit | zero-shot det. (paper) | C2 AR-sweep best (20 calls, no learning) | C3 random mean | C3 random best-of-20 | C5 sampled mean | C5 sampled best-of-10 |
|---|---|---|---|---|---|---|
| custom_macbuf | 55.46 | 48.20 | −466.5 | 50.70 | 51.33 | 67.57 |
| mkDelayWorker32B | 58.73 | 10.89 | −712.1 | −2.78 | −270.8 | −2.79 |
| lightweight_cipher | 33.49 | −0.08 | −236.4 | 31.27 | 25.98 | 34.43 |
| reduction_layer | 23.30 | 8.47 | −483.9 | −20.00 | −236.7 | −8.03 |
| arm_core | 9.11 | **11.80** | −532.1 | −17.73 | −27.43 | −0.56 |
| softmax | 7.27 | **9.89** | −211.1 | −4.48 | −410.0 | 6.78 |

- **C2.** On arm_core and softmax, the non-learning trimmed-baseline AR sweep exceeds the policy's zero-shot mean.
  - Across all 17 circuits the sweep's best reaches e.g. fifo 60.0, boundtop 50.3, macbuf 48.2, or1200 33.2 (see CSV).
- **C4, placer-seed noise across VPR seeds 1–5** (ADP coefficient of variation):
  - baseline 1.4–5.3% (max−min range up to 14.5% of mean, softmax);
  - seed-42 policy floorplans 0.3–2.1% (range up to 6.2%).
- **C5.** Stochastic sampling from the policy is much worse on average than deterministic rollout on the large circuits: its mean is dominated by a few very bad samples.

## Hardware and wall-clock (Tier C)

- **CPU:** 12th Gen Intel Core i9-12900K, 16 cores / 24 threads, 31 GiB RAM, RHEL 8.10 host, Ubuntu 24.04 distrobox. No GPU used (policy inference on CPU). Host `localhost.localdomain`.
- **Concurrency:** 16 concurrent VTR flows for C2, C3, C5; 10 for C4.

| stage | start | end | wall-clock |
|---|---|---|---|
| C1 | — | — | 0 (reused) |
| C2 | 12:50:13 | 13:16:10 | 26 min |
| C3 (20/circuit-seed) | 13:16:10 | 14:30:04 | 1 h 14 min |
| C4 | 14:30:04 | 14:40:21 | 10 min |
| C5 | 14:40:21 | 15:14:41 | 34 min |
| **total** | | | **2 h 24 min 28 s** (2026-09-17, IST) |

Plus about 20 s for the policy-only rollout timing pass. Per-run times are in each CSV's `vtr_seconds`; all runs ran under concurrent load, so single-run times would be lower.

## Reproducing

```bash
distrobox enter ubuntu-work -- bash ~/workspace/rl_gnn/vanilla_rl/data_export/tools/run_tier_c.sh   # resumable
distrobox enter ubuntu-work -- ~/.venv/bin/python ~/workspace/rl_gnn/vanilla_rl/data_export/tools/tier_b.py
distrobox enter ubuntu-work -- ~/.venv/bin/python ~/workspace/rl_gnn/vanilla_rl/data_export/tools/plot_seed_variance_8pt.py
```

`tier_b.py b7` needs `logs/det_actions_and_timing.json` (`c345.py timing`) and `vpr_seed_noise.csv`.
