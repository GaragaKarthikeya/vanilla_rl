# Mismatches against the known paper numbers

Neither side has been "fixed". Recomputed values use the files named here. Paths are relative to `~/workspace/rl_gnn/vanilla_rl` unless prefixed.

## 1. In-pool pooled best ADP reduction, seed 42: paper 35.73%, final seed-42 run 34.98% (MISMATCH)

| seed | paper | recomputed | source |
|---|---|---|---|
| 7 | 23.11 | 23.11 | `best_layout_coordinates_{11 circuits}_multi11_long_seed7.txt` (= min ADP in `all_layouts_multi_seed_7_multi11_long_seed7.jsonl`) |
| 42 | **35.73** | **34.98** | `best_layout_coordinates_*_multi11_long_seed42_v2.txt` (= `all_layouts_multi_seed_42_multi11_long_seed42_v2.jsonl`) |
| 123 | 23.97 | 23.97 | `best_layout_coordinates_*_multi11_long_seed123.txt` |

Pooled = 1 - sum(best ADP) / sum(baseline ADP) over the 11 training circuits.

**35.73% is exactly the pooled value of a different, older seed-42 run**, `multi11_long` (no `_seed42_v2` suffix, 2026-06-18): `best_layout_coordinates_*_multi11_long.txt` gives 35.73. That run's checkpoint is not on disk (`runs/multi11_long.zip` does not exist).

Everything else for seed 42 uses v2:
- the held-out zero-shot numbers load `runs/multi11_long_seed42_v2.zip` (`runs/det_seed42_*.log`);
- the per-circuit seed-42 in-pool dots in `fig_seed_variance.pdf` match v2 on every circuit.

The v1/v2 per-circuit bests differ:

| circuit | v1 | v2 |
|---|---|---|
| diffeq1 | 57.40 | 55.92 |
| spree | 38.61 | 38.89 |
| diffeq2 | 68.21 | 68.24 |
| raygentop | -2.92 | -1.70 |
| mkSMAdapter4B | 36.19 | 35.51 |
| mkPktMerge | 40.23 | 37.78 |
| ch_intrinsics | 36.25 | 35.08 |

The pooled seed average in the paper (27.60%) inherits this: with v2 it would be (23.11+34.98+23.97)/3 = 27.35%.

`PAPER_PLAN.md:24-25` records "seed7 23.11% vs seed42's 35.73%", written when v1 was the seed-42 run.

## 2. The in-pool numbers are best-during-training, not final-checkpoint (clarification, not a numeric mismatch)

The paper's in-pool numbers (pooled text and figure dots) are the best layout found during training. The final checkpoints, run deterministically (`runs/inpool_seed{7,42_v2,123}.json`, reproduced in `data_export/inpool_final_checkpoint.csv`), give pooled reductions of **1.89% / 5.84% / 4.72%**, with several circuits negative:
- or1200: -14.9 / -16.9 / -17.2
- raygentop: -18.3 / -11.9 / -18.1
- boundtop seed 7: -8.4

## 3. Held-out zero-shot means (MATCH under the paper's rounding; 2 entries differ if unrounded values are averaged)

Recomputed from `det_seed{7,42,123}_{circuit}.json`, mean over 3 seeds:

| circuit | paper | mean of unrounded per-seed | mean of per-seed values rounded to 2 dp (figure data) |
|---|---|---|---|
| macbuf (custom_macbuf) | 55.46 | 55.4606 | 55.4600 |
| mkDelayWorker32B | 58.73 | **58.7247 -> 58.72** | 58.7267 -> 58.73 |
| cipher (lightweight_cipher) | 33.49 | 33.4892 | 33.4867 |
| reduction_layer | 23.30 | 23.2984 | 23.2967 |
| arm_core | 9.11 | **9.1158 -> 9.12** | 9.1133 -> 9.11 |
| softmax | 7.27 | 7.2687 | 7.2700 |
| overall mean | 31.23 | 31.226 | 31.227 |

All six paper values equal the mean of the 2-dp-rounded per-seed values. Averaging the unrounded values changes mkDelayW32B to 58.72 and arm_core to 9.12.

## 4. GA table (ADP MATCH; one VTR-call count truncated instead of rounded)

Recomputed from `~/workspace/vtr_exp/ga_output_*_scratch_seed{7,42,123}.txt` (arm_core seed 7 = `ga_output_scratch_seed7.txt`). Calls = logged generations x 8 individuals, equal to the number of `OK (took` lines in the stdout logs.

| circuit | paper ADP / calls | recomputed ADP (per seed) | recomputed calls (per seed) |
|---|---|---|---|
| macbuf | 65.33 / 728 | 65.33 (60.80, 68.00, 67.18) | 728.0 (800, 824, 560) |
| mkDelayW32B | 17.92 / **746** | 17.92 (22.64, 14.86, 16.26) | **746.67** (656, 1016, 568) -> rounds to 747 |
| cipher | 44.13 / 576 | 44.13 (43.63, 43.18, 45.59) | 576.0 (704, 472, 552) |
| reduction_layer | 7.18 / 731 | 7.18 (6.02, 7.47, 8.07) | 730.67 (712, 888, 592) -> 731 |
| arm_core | 17.88 / 736 | 17.88 (17.20, 15.78, 20.67) | 736.0 (1120, 424, 664) |
| softmax | 15.61 / 784 | 15.61 (16.11, 16.57, 14.16) | 784.0 (488, 840, 1024) |

The rounding is inconsistent: 746.67 was truncated to 746, while 730.67 was rounded to 731.

## 5. spree seed-123 timeline (MATCH)

`runs/spree_timeline/manifest.json` plus each milestone's re-run `vpr.out`:

| panel | paper | recomputed |
|---|---|---|
| milestone_14 | 10x13 / +39.1% | 10x13 / 39.106% |
| milestone_10 | 11x12 / +36.9% | 11x12 / 36.915% |
| milestone_09 | 12x11 / +33.3% | 12x11 / 33.312% |

## 6. Methodology claims in the paper text that the code does not support

These are not numbers from the list above; they were found while collecting Tier A and are listed so the revision can address them.

- **W_max/H_max "taken over both the training pool and the unseen, zero-shot benchmark circuits".**
  - The sizing universe was the 11 training circuits + softmax + reduction_layer + robot_rl (wandb-metadata `--universe_benchmarks`).
  - 4 of the 6 held-out circuits (custom_macbuf, lightweight_cipher, arm_core, mkDelayWorker32B) were not in it.
  - All caps (48x48, MAX_NODES 67, MAX_EDGES 2694) come from robot_rl, which is in neither split.
  - The held-out circuits fit only because they happen to be <= robot_rl.
- **"Four of the six are larger than anything in the training pool (up to 3.4x)".** Not verified here. The largest training core grid is mkPktMerge 26x26 = 676 tiles and the largest held-out is mkDelayWorker32B 48x48 = 2304 tiles, a ratio of 3.41 by tile count. So 3.4x is consistent with a tile-count ratio; no source for the paper's definition was found.
- **"(BRAMs then DSPs, both ordered by descending shared-net connectivity)".** The code sorts by each block's own unique-net count (`net_count_normalized`), not by shared nets (`src/env/fpga_env.py:263-284`).
- **"Edges are weighted by shared-net count, normalized to [0,1]".** The code applies log compression, `min(log1p(w)/log1p(C), 1)`, with C = 80 for block-block edges and C = 1300 for block-fabric edges (`src/netlist/graph_reduction.py:41-42,170`).
- **"GA ... under the same oracle".**
  - Same template, VTR build, flow, metrics, and baseline.
  - But the GA writes no VPR atom-pinning constraints file. The policy pins each H-block atom to its coordinate (`--read_vpr_constraints`); the GA lets VPR assign atoms to the placed tiles.
  - The GA has no cache.
- **"each of its generations requires its own VTR call".** Each generation is 8 VTR calls (population 8, run in parallel).
- **Spare H-block as a node with net count 0** (task list A4). Not supported by the current code path. See `facts.json` graph.zero_net_node_supported.
- **"Delay is VPR's critical-path delay ... Area is VPR's routing area"** (reported at the same channel width, by implication).
  - Routing area comes from the min-W binary-search VPR run.
  - Delay and power come from the relaxed (1.3 x W_min) re-route.
  - For the diffeq1 baseline, the routing area at relaxed W would be 904,549, not the reported 694,168 (`runs/diffeq1_traditional/vpr.out` vs `vpr_stdout.log`).
- **"Power is VPR's total power: the dynamic and static power"** vs footnote-style "dynamic power" in the Evaluation Oracle section. The parsed value is the `Total` row of the `.power` report (`src/evaluation/vtr_runner.py:140-144`). The paper uses both wordings.
- **Reward "An invalid layout ... receives a strict penalty of -10".** Matches the code. But the `reward` field stored in `det_seed*.json` / `inpool_seed*.json` was computed with FPGAEnv's constructor-default weights (wl .1, pw .3, dl .6, ar 0), because `evaluate_held_out.py:68` does not pass the training weights. ADP numbers are unaffected.

## 7. The C2 trimmed-baseline sweep (14.86%) is a strawman (data_export's own error)

This is a flaw in `data_export`'s C2 design, not in the paper's code. But the
paper's abstract cites the number, so it is recorded here.

`tools/c2_ar_sweep.py:trimmed_positions` computes DSP/BRAM tile coordinates
once, from the square baseline core grid, and reuses them at every aspect ratio.
The template renders `auto_layout aspect_ratio=AR` with those fixed `<single>`
tiles, so VPR has to grow the grid until every fixed tile fits and then stretch
the other dimension to hold the ratio. The extra area fills with unused CLBs.
From `baseline_ar_sweep.csv`:

| circuit | AR 0.1 | AR 0.7 | AR 1.0 | AR 1.9 |
|---|---|---|---|---|
| mkDelayWorker32B | 44x440 | 44x63 | 50x50 | 95x50 |
| reduction_layer | 36x360 | | | |

The policy places the same 43 BRAMs of mkDelayWorker32B in 34x38.

Consequence: the held-out mean of the sweep's best, **14.86%**, is a lower bound
for a non-learning aspect-ratio sweep, and it should not be cited as a fair
baseline. A corrected sweep, with tile coordinates recomputed on each aspect
ratio's own grid, is in `baseline_ar_sweep_perar.csv`; its result is in §7a.

### 7a. Result of the corrected sweep (`baseline_ar_sweep_perar.csv`, 340/340 succeeded)

Tile coordinates recomputed on each aspect ratio's own grid
(`tools/c2_perar.py`). The stretching is gone: mkDelayWorker32B at AR 0.1 is
built at 14x140 (planned core 13x132), against 44x440 in the flawed sweep.

Best ADP reduction over the 20 aspect ratios, held-out circuits:

| circuit | flawed sweep | corrected sweep (best AR) | policy zero-shot |
|---|---|---|---|
| custom_macbuf | 48.20 | 48.20 (0.6) | 55.46 |
| mkDelayWorker32B | 10.89 | 37.60 (0.1) | 58.72 |
| lightweight_cipher | -0.08 | 35.35 (0.3) | 33.49 |
| reduction_layer | 8.47 | 27.40 (0.6) | 23.30 |
| arm_core | 11.80 | 11.80 (1.2) | 9.12 |
| softmax | 9.89 | 11.03 (1.4) | 7.27 |
| **mean** | **14.86** | **28.56** | **31.23** |

A fair 20-call non-learning sweep reaches 28.56%, not 14.86%. The policy's
single call still leads on the mean by 2.67 points, but the sweep beats it on
4 of 6 circuits (lightweight_cipher, reduction_layer, arm_core, softmax), and
the policy's lead rests mainly on mkDelayWorker32B. Over all 17 circuits the
corrected sweep's mean best is 26.24% (flawed: 18.59%).

The abstract's "exceeds an aspect-ratio sweep over a trimmed baseline (14.86% at
20 calls)" should be revised to the corrected figure.
