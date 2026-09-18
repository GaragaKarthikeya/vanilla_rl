# data_export4: shape vs. placement as separate levers

Location: `~/workspace/rl_gnn/vanilla_rl/data_export4/`. Paths are relative to
`~/workspace/rl_gnn/vanilla_rl` unless prefixed.

Same repo, same three final checkpoints (seeds 7, 42, 123), same VTR 9.0.0-dev
build `22a09d39ef`, same flow as `data_export/`–`data_export3/`. No training
code, evaluation code, checkpoints or paper cache DBs were modified. New code is
in `data_export4/tools/`. One isolated cache is shared across this request:
`runs/vtr_layout_cache_<circuit>_dataexport4.db`.

Context numbers verified before starting (`notes.md` §0): at AR 1.0 on the
original six, the policy's placement averages 29.32% and the column placement
−5.90%; the corrected sweep's held-out mean is 28.56%.

## Headline findings

1. **G1: the two levers do not stack as specified.** Forcing each circuit's
   sweep-best shape and letting the policy place blocks averages **−64.1%** on
   the original six (median 9.38) and **−12.32%** on the new six (median 6.19).
   G1 beats both the sweep best and the policy's own result on **2 of 12**
   circuits (uriscv_core, lenet), and on none of the original six.
2. **The cause is in the paper's action space, not in placement quality.** The
   placement mask uses the circuit's square baseline grid at every aspect ratio
   (`src/env/fpga_env.py:367-369`). At an extreme shape, VPR must stretch the
   grid to reach the policy's tiles, which is the same artifact that made the
   original C2 sweep a strawman. mkDelayWorker32B at AR 0.1 is built at
   27×270–28×280 under the policy, against 14×140 for the column placement, and
   loses 347–463%.
3. **G3: at moderate shapes the levers do stack.** Across all 20 shapes at seed
   42, the policy's placement is poor only at the extremes (every circuit's worst
   result is at AR 0.1). Its best forced shape beats its own chosen shape on 5 of
   6 circuits. mkDelayWorker32B reaches **65.0% at AR 0.5**, against 37.60 for the
   sweep best and 54.88 for the policy's own result at seed 42.
4. **G2: on the new six, the policy's placement advantage mostly disappears.**
   At AR 1.0 the policy averages −2.20% (median 6.93), against 3.94% (median
   7.92) for the column placement and −16.01% for random. The policy beats the
   columns on 3 of 6 circuits and beats random on 5 of 6. On the original six
   the same comparison is 29.32% vs −5.90%.
5. **On the new six, the column sweep beats the policy outright:** sweep best
   averages 14.10% (median 15.34), the policy's own result −1.24% (median
   10.08).

Details and caveats are in `notes.md`.

## Files

| file | experiment | status |
|---|---|---|
| `g1_sweep_new_circuits.csv` | G1 prerequisite | done, 120 rows (6 circuits × 20 ARs), 116 succeeded; the 4 failures are lenet at AR 0.1–0.4 |
| `g1_sweep_shape_policy_placement.csv` | G1 | done, 36/36 |
| `g2_fixed_shape_new.csv` | G2 | done, 78 rows: 6 `columns` (reused from the sweep at AR 1.0) + 18 `policy` + 54 `random`, all succeeded |
| `g3_policy_all_shapes.csv` | G3 | done, 120/120 |
| `summary.json` | summary | done; per set, means and medians, plus per-circuit G1 comparison |
| `logs/` | — | per-stage logs, runner log with all interruptions |

`g1_sweep_new_circuits.csv` has the columns of
`data_export/baseline_ar_sweep_perar.csv` plus `cached` and `source`, because
the request's rules require `cached` for every run.

## Method notes

- **Sweep (new six).** `data_export/tools/c2_perar.py:run_one`, unchanged: tile
  coordinates planned on each aspect ratio's own grid. It calls VTR directly with
  no layout cache, so `cached` is always False.
- **Forced shape + policy placement (G1, G2 `policy`, G3).** Step 0 is set to
  the aspect ratio's action index. The seed's final checkpoint then places every
  H-block deterministically with the paper's own mask, and the floorplan is
  evaluated through `FPGAEnv.step`: same bake, atom pinning, VTR invocation and
  metrics as the paper's evaluation.
- **G1 shape choice.** For each circuit, the AR with the highest
  `adp_reduction_pct_vs_baseline` among successful rows of
  `data_export/baseline_ar_sweep_perar.csv` (original six) or
  `g1_sweep_new_circuits.csv` (new six).
- **G2 `random`.** AR 1.0, then each H-block uniform over the legal mask in the
  policy's block order, `numpy.default_rng([seed, circuit_index, draw])`,
  `circuit_index` = position in `[usb_uart_core, uriscv_core, lenet,
  aes_inv_cipher, 8051, stereovision1]`.
- **Summary rule.** Seeded quantities are averaged over seeds (and draws) per
  circuit first, then summarized over circuits, as in the paper's held-out table.
  For the original six, the AR-1.0 comparison is reused from
  `data_export/baseline_ar_sweep_perar.csv` (columns) and
  `data_export2/e2_decomposition.csv` (policy, random), not re-run.

## Runs, cache hits and failures

| file | rows | real VTR runs | cache hits | reused | failed | summed vtr_seconds |
|---|---|---|---|---|---|---|
| sweep (new six) | 120 | 120 | 0 | 0 | 4 | 6.24 h |
| G1 | 36 | 36 | 0 | 0 | 0 | 1.36 h |
| G2 | 78 | 72 | 0 | 6 (`columns`) | 0 | 4.43 h* |
| G3 | 120 | 114 | 6 | 0 | 0 | 2.18 h |
| **total** | **354** | **342** | **6** | **6** | **4** | **~13.9 h** |

\* G2's summed seconds include the 6 reused `columns` rows, which were timed in
the sweep; the ~13.9 h total counts them once.

G3's 6 cache hits are the seed-42 original-six floorplans G1 had already
evaluated at the same shapes.

**Failures:** 4 sweep runs, all lenet at AR 0.1, 0.2, 0.3 and 0.4. Each failed
at the same point (565.1 s) with VTR returning non-zero or incomplete metrics, so
they are consistent flow failures at extreme elongation, not interruptions. The
driver's note text on those rows mentions a possible memory cap; no cap was in
use for this request, so that wording does not apply.

## Hardware, concurrency, wall-clock

- **CPU:** 12th Gen Intel Core i9-12900K, 16 cores / 24 threads, 31 GiB RAM.
  RHEL 8.10 host, Ubuntu 24.04 distrobox. No GPU; policy inference on CPU.
- **Concurrency:** 2 VTR flows for the sweep, G1 and G2 (the new circuits'
  synthesis peaks at ~4.5–6 GB per yosys process); 4 for G3 (original six only).
  Run alone, never alongside F3 training.
- **Wall-clock, final successful pass:** sweep 20:10–21:26, G1 21:26–22:11, G2
  22:11–00:15, G3 00:15–00:49, i.e. **4 h 39 min** (2026-09-18 20:10 to
  2026-09-19 00:49 IST).

Interruptions before that pass (resumable CsvSinks, so no finished row was lost
or duplicated):

1. 11:46: started alongside F3 at 6 workers; the memory watchdog stopped it at
   11:47 with 5 GB free. That also stopped F3 (see `data_export3/`).
2. 12:00: restarted at 4 workers; memory dipped to 4 GB and it was stopped by
   hand. Then restarted at 2 workers.
3. 16:36: a power cut rebooted the host mid-sweep; relaunched at 20:10.

## Reproducing

```bash
cd ~/workspace/rl_gnn/vanilla_rl/data_export4/tools
distrobox enter ubuntu-work -- bash run_g.sh      # sweep -> G1 -> G2 -> G3, resumable
distrobox enter ubuntu-work -- ~/.venv/bin/python summary.py
```
