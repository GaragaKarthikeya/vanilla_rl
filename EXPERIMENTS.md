# RL FPGA Placement — Experiment Log

This document summarizes the training runs and experiments conducted on the vanilla RL pipeline (MaskablePPO agent placing DSP blocks for VTR synthesis). All runs use the `diffeq1` / `diffeq2` benchmarks (14×14 core grid, 5 DSPs, 0 BRAMs) and the reward function:

```
reward = -(ar_weight * log(area / baseline_area)
         + dl_weight * log(delay / baseline_delay)
         + pw_weight * log(power / baseline_power)
         + wl_weight * log(wirelength / baseline_wirelength))
```

with weights `ar=1.0, dl=1.0, pw=1.0, wl=0.0` throughout (equal weight on area, delay, and power; wirelength excluded). A reward of 0 means "matches the traditional VTR baseline exactly"; positive means better.

Area×Delay×Power (**ADP**) is used throughout as a single combined figure of merit, since it compounds all three metrics multiplicatively.

---

## 0. Bug fixes preceding the experiments

Three bugs were found and fixed before any of the results below were generated:

1. **Net-count sort silently disabled** (`src/training/trainer.py`) — a string key-matching check (`k.endswith("')")`) failed for keys like `"('DSP', 0)"` because they end in a digit, not `')`. This meant DSP blocks were placed in identity order instead of being sorted by net count, defeating the purpose of the net-count-aware placement feature. Fixed by checking `k.startswith("('")` instead.
2. **JSON serialization crash** (`src/training/callbacks.py`) — `placed_dsps`/`placed_brams` were stored as NumPy `int64` tuples, which `json.dumps` cannot serialize. Fixed with a recursive `_to_py()` converter applied before writing.
3. **Best-layout artifacts not namespaced by run** (`src/training/callbacks.py`) — `best_baked_layout_{benchmark}.xml`, `best_layout_coordinates_{benchmark}.txt`, and `runs/best_run_{benchmark}/` were keyed only by benchmark name, not by `--log_suffix`. Two runs on the same benchmark would silently overwrite each other's saved best layout. This bug was actually triggered once (Experiment 2 below) before being fixed by appending `self.log_suffix` to all four output paths.

---

## 1. Full-scale training on diffeq2 (6000 episodes)

**Config:** `n_envs=24, n_steps=30, batch_size=120, n_epochs=15, ent_coef=0.01, max_episodes=6000, seed=42`. Runtime: ~1675s (28 min).

| Metric | Traditional baseline | RL best (6000 ep) | Change |
|---|---|---|---|
| Routing area | 666,210 | 298,474 | **-55.2%** |
| Power (W) | 0.007747 | 0.005506 | **-28.9%** |
| Delay (ns) | 17.948 | 17.276 | **-3.7%** |
| **ADP** | **92,632** | **28,392** | **-69.4% (3.26x better)** |
| Reward | — | **1.1825** | — |

The best layout used aspect ratio 1.0 (9×9 final grid) with all 5 DSPs clustered into adjacent columns, versus the traditional placer's sparse 14×12 active region. Average reward was still climbing at episode 6000 with no clear plateau, suggesting a longer run would find an even better layout.

---

## 2. Update frequency: 24 vs. 120 episodes per policy update (diffeq2)

A follow-up question: does updating the PPO policy more frequently (smaller rollout buffer) help or hurt, given this environment's reward is sparse (zero except at the terminal step) and high-variance? Two configurations were compared head-to-head at a matched 2000-episode budget, both seed 42, sharing the same VTR layout cache (a confound — see caveat below).

| Episodes | Avg reward, 120/update | Avg reward, 24/update |
|---|---|---|
| 1–200 | -0.74 | -0.34 |
| 1201–1400 | 0.27 | **0.90** |
| 1801–2000 | 0.48 | **0.94** |
| Running-max by ep 2000 | 0.85 | **1.06** |

The 24-episodes/update configuration converged dramatically faster at a matched episode count — the opposite of the a-priori concern that smaller batches would be too noisy given the high-variance terminal reward.

**Caveat:** the second run reused the first run's VTR cache (already populated with ~6000 evaluated layouts), so part of its fast climb may be cache hits replaying known-good layouts rather than purely faster learning. A clean ablation would need separate cache databases (this was corrected in later experiments). This run is also where the best-layout file-overwrite bug (§0.3) was discovered — the 24/update run's lower final best (reward 1.0635, ADP 31,982, -65.5% vs. baseline) silently overwrote the 120/update run's better saved artifacts on disk; the 120/update numbers above were preserved only because they were recorded before the second run started.

---

## 3. Zero-shot cross-benchmark transfer: diffeq2 → diffeq1

The diffeq2-trained policy (no further training) was run once, deterministically, on **diffeq1** — a different benchmark with the same grid size and DSP count, making the observation/action spaces compatible. Compared against the diffeq1 traditional baseline and 20 trials of a random-valid-placement policy.

| | Area | Delay (ns) | Power (W) | ADP | Reward |
|---|---|---|---|---|---|
| Traditional | 694,168 | 22.340 | 0.007714 | 119,624 | — |
| Random (avg of 20) | — | — | — | — | -1.02 |
| Random (best of 20) | 575,810 | 22.138 | 0.006892 | 87,854 | 0.31 |
| **RL zero-shot (1 episode)** | **449,975** | **21.574** | **0.006178** | **59,976** | **0.69** |

The zero-shot policy beat the traditional baseline by **49.9% on ADP** and beat the best of 20 random trials by **31.7% on ADP**, despite never training on diffeq1. The typical random placement actually performed *worse* than the traditional baseline (avg reward -1.02), underscoring that the RL policy learned a genuine, transferable packing heuristic rather than memorizing diffeq2-specific layout details.

---

## 4. Training from scratch vs. fine-tuning on diffeq1

Given the zero-shot result, the next question was whether *actually training* on diffeq1, starting from the diffeq2-pretrained weights, beats training from scratch. Both agents ran 6000 episodes with identical hyperparameters (matching §1) and **separate** fresh VTR caches to avoid the confound from §2.

| | From scratch | Fine-tuned from diffeq2 |
|---|---|---|
| Final best reward | **0.851** | 0.815 |
| Final best ADP vs. baseline | **-57.3% (2.34x)** | -55.8% (2.26x) |
| Avg reward, final 200-ep window | 0.736 | **0.785** |
| Wall-clock for 6000 episodes | 1969s (33 min) | **828s (14 min, 2.4x faster)** |
| Policy entropy: start → end | 4.63 → 2.42 | 2.00 → 0.79 |
| Unique VTR evaluations (cache misses) | 5,664 / 6,000 (94.4%) | 1,892 / 6,000 (31.5%) |

**Fine-tuning dominated on typical performance and speed for essentially the whole run** — its average reward was higher than scratch's in nearly every 200-episode window, from the very first (0.64 vs. -0.83) to the very last (0.785 vs. 0.736), and it finished 2.4x faster in wall-clock time.

**But scratch found the better single layout.** Fine-tuning's best-ever result plateaued by around episode 3000 and never improved again, while scratch kept finding small improvements all the way to ~episode 5300, eventually edging ahead by about 4.4% in reward.

**Root cause — verified programmatically, not just inferred:** the fine-tuned policy's entropy collapsed quickly (2.00 → 0.79, vs. scratch's 4.63 → 2.42), meaning its action distribution concentrated onto a small set of placements. This was confirmed directly by querying each run's SQLite cache: fine-tuning only triggered a genuinely new VTR evaluation for **31.5%** of its episodes — the other 68.5% were repeats of already-cached layouts, resolved almost instantly instead of via a fresh multi-second VTR run. Scratch, still exploring, generated novel layouts 94.4% of the time. Wall-clock-per-rollout data confirms this directly: fine-tuning's rollouts cost ~23s each early on (mostly fresh evaluations) but dropped to as little as 0-3s each by the end (mostly cache hits).

**Practical takeaway:** pretraining gives a large head start in typical performance and wall-clock efficiency, but the resulting policy is overconfident and stops exploring, capping how far it can improve within a fixed episode budget. If the goal is "good, fast," fine-tune. If the goal is the single best possible layout and episode budget isn't the binding constraint, train from scratch — or fine-tune with a higher entropy coefficient to counteract the inherited overconfidence (untested; a natural next experiment).

---

## 5. Visualizer

`src/visualization/plot_layout.py` renders VPR `.place` + architecture XML files into PNGs, with support for 1, 2, or 3 side-by-side panels (`--compare`, optional `--place3/--arch3/--title3`). Originally a dark theme; switched to a light, professional palette (white background; amber I/O pads, green CLBs, blue DSPs, red BRAMs, each with a matching darker edge color) at 300 DPI for presentation-quality output. DSP tiles render 4 cells tall, BRAM tiles 6 cells tall, with an "active-logic bounding box" annotation showing the footprint actually used by placed blocks.

---

## Artifact index

| Item | Path |
|---|---|
| diffeq2 full-scale model | `runs/diffeq2_fullscale_seed42.zip` |
| diffeq1 scratch / fine-tuned models | `runs/diffeq1_scratch_seed42.zip`, `runs/diffeq1_finetuned_seed42.zip` |
| Per-episode logs | `all_layouts_{benchmark}_seed_42{_suffix}.jsonl` |
| Best layout per run | `best_layout_coordinates_{benchmark}{_suffix}.txt`, `best_baked_layout_{benchmark}{_suffix}.xml` |
| VTR re-run of each best layout | `runs/best_run_{benchmark}{_suffix}/` |
| Zero-shot transfer results | `zero_shot_transfer_diffeq1_results.json` |
| Visual comparisons | `visualizations/*.png`, `visualizations/analysis/*.png` |
| Traditional baselines | `baselines/{benchmark}_traditional_{metric,resources}.txt` |
| VTR layout caches (reusable across future runs on the same benchmark) | `runs/vtr_layout_cache_diffeq2.db`, `runs/vtr_layout_cache_diffeq1_scratch.db`, `runs/vtr_layout_cache_diffeq1_finetuned.db` |
| TensorBoard logs (required by `plot_training_analysis.py` for the entropy plot) | `runs/tb_logs/MaskablePPO_{3,4,5,6}` (diffeq2 120/update, diffeq2 24/update, diffeq1 scratch, diffeq1 fine-tuned, respectively) |
| Training analysis plots | `src/visualization/plot_training_analysis.py` → `visualizations/analysis/` |
