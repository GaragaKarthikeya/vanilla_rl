# notes.md — data request 4

Paths are relative to `~/workspace/rl_gnn/vanilla_rl`. Nothing here "fixes"
either the code or the paper.

## 0. Context numbers checked before running

- Column placement at AR 1.0, original six: −6.23, −4.44, −10.97, −10.54,
  −6.46, +3.22 → mean **−5.90** (`data_export/baseline_ar_sweep_perar.csv`).
- Policy placement at AR 1.0, original six, 18 circuit-seeds: mean **29.32**
  (`data_export2/e2_decomposition.csv`, `fixed_ar_policy_blocks`).
- `summary.json` reproduces both, plus the sweep's 28.56 and the policy's
  31.23, from the files.

## 1. G1: forcing the sweep's best shape breaks the policy's placement

Mean over seeds, per circuit (`summary.json`, `per_circuit_g1_vs_sweep_and_policy`):

| circuit | set | sweep best (AR) | policy own | G1 | G1 beats both |
|---|---|---|---|---|---|
| custom_macbuf | original | 48.20 (0.6) | 55.46 | 51.54 | no |
| mkDelayWorker32B | original | 37.60 (0.1) | 58.72 | **−411.67** | no |
| lightweight_cipher | original | 35.35 (0.3) | 33.49 | −54.14 | no |
| reduction_layer | original | 27.40 (0.6) | 23.30 | 10.55 | no |
| arm_core | original | 11.80 (1.2) | 9.12 | 10.92 | no |
| softmax | original | 11.03 (1.4) | 7.27 | 8.21 | no |
| usb_uart_core | new | 15.04 (1.3) | 14.28 | 2.25 | no |
| uriscv_core | new | 14.46 (0.8) | 12.80 | 14.50 | **yes** |
| lenet | new | 15.64 (1.7) | 11.03 | 16.35 | **yes** |
| aes_inv_cipher | new | 18.58 (2.0) | 9.12 | 10.13 | no |
| 8051 | new | 21.17 (1.2) | 4.68 | −40.20 | no |
| stereovision1 | new | −0.28 (0.7) | −59.33 | −76.94 | no |

| set | sweep best mean / median | policy own mean / median | G1 mean / median |
|---|---|---|---|
| original | 28.56 / 31.37 | 31.23 / 28.39 | **−64.10 / 9.38** |
| new | 14.10 / 15.34 | −1.24 / 10.08 | **−12.32 / 6.19** |

**The mean is dominated by one circuit.** mkDelayWorker32B alone contributes
−411.67 to the original-six mean; the median (9.38) is the more honest summary.
Even so, G1 wins over both references on only 2 of 12 circuits, and those two
(uriscv_core, lenet) win by 0.04 and 0.71 points over the sweep.

### Why: the placement mask does not know the shape

`FPGAEnv.get_action_mask()` accepts an H-block at (x, y) only if
`1 <= x <= cfg.width` and `y + h - 1 <= cfg.height`
(`src/env/fpga_env.py:367-369`). `cfg.width` and `cfg.height` are the circuit's
**square baseline** core grid, and they do not change with the aspect ratio
chosen at step 0. So at any aspect ratio the policy places tiles as if the
fabric were square, and VPR's `auto_layout` must then grow the grid until every
fixed tile fits while holding the requested ratio. The extra area fills with
unused CLBs.

This is the same mechanism that made the original C2 sweep a strawman
(`data_export/mismatches.md` §7). The corrected sweep avoids it because it plans
tiles on each ratio's own grid; the policy cannot, by construction.

Evidence from the grids VPR actually built:

| circuit | AR | column placement (sweep) | policy placement (G1, 3 seeds) |
|---|---|---|---|
| mkDelayWorker32B | 0.1 | 14×140 | 27×270, 28×280, 25×250 |
| lightweight_cipher | 0.3 | 6×20 | 8×27, 9×30, 8×27 |
| 8051 | 1.2 | 18×15 | 23×19, 19×16, 23×19 |

Near square, the grids match (softmax 1.4: 45×32 both; lenet 1.7: 49×29 both;
uriscv_core 0.8: 14×18 both), and G1 behaves reasonably there.

So G1 as specified does not measure "policy placement quality at a good shape".
It measures what happens when a shape-blind action space is paired with a shape
it was never trained to use. That is a genuine limitation of the paper's
method, and worth stating: the policy cannot exploit extreme aspect ratios, and
the aspect-ratio decision and the placement are coupled through the canvas.

## 2. G3: at moderate shapes the two levers do stack

`g3_policy_all_shapes.csv`, original six, seed 42, all 20 shapes, 120/120
succeeded.

| circuit | best forced AR → result | policy own (seed 42) | at 0.8 / 0.9 / 1.0 | worst (at AR 0.1) | beats columns at |
|---|---|---|---|---|---|
| custom_macbuf | 0.8 → 59.5 | 59.5 | 59.5 / 56.5 / 51.1 | −985.0 | 6/20 ARs |
| mkDelayWorker32B | 0.5 → **65.0** | 54.9 | 53.2 / 54.9 / 52.6 | −463.3 | 17/20 |
| lightweight_cipher | 1.3 → 39.9 | 39.7 | 27.0 / 39.7 / 33.6 | −774.7 | 14/20 |
| reduction_layer | 1.1 → 23.3 | 21.5 | 20.0 / 21.5 / 22.8 | −2021.6 | 10/20 |
| arm_core | 1.8 → 15.4 | 11.8 | 8.1 / 11.8 / 12.1 | −927.8 | 14/20 |
| softmax | 1.4 → 9.4 | 5.4 | 8.8 / 5.4 / 8.7 | −94.9 | 9/20 |

- Every circuit's worst result is at AR 0.1, consistent with §1.
- The best forced shape beats the policy's own choice on **5 of 6** circuits
  (custom_macbuf ties at 59.5, because its best forced shape *is* its own 0.8).
- On mkDelayWorker32B, AR 0.5 with policy placement gives **65.0%**, above both
  the sweep's best (37.60) and the policy's own seed-42 result (54.88). That is
  the stacking the request looked for; it appears at moderate shapes, not at the
  sweep's extreme 0.1.
- "Beats columns at N/20 ARs" compares against the corrected sweep at the same
  shape. The policy wins at most shapes on mkDelayWorker32B, lightweight_cipher
  and arm_core, and at fewer than half on custom_macbuf and softmax.
- This is seed 42 only, as requested, and picking the best of 20 shapes after
  seeing the results is an oracle; the paper should not present 65.0 as
  something the policy achieves unaided.

## 3. G2: on the new six, the placement advantage mostly disappears

`g2_fixed_shape_new.csv`, AR 1.0, per-circuit mean over seeds (and draws):

| circuit | columns | policy | random | policy per seed |
|---|---|---|---|---|
| usb_uart_core | 5.20 | 2.66 | −6.21 | 4.9, −1.7, 4.8 |
| uriscv_core | 9.22 | **11.20** | −6.59 | 9.8, 10.6, 13.2 |
| lenet | 8.46 | **13.64** | 3.52 | 15.4, 13.9, 11.7 |
| aes_inv_cipher | 7.37 | **13.14** | −10.13 | 11.1, 13.2, 15.1 |
| 8051 | 14.36 | −10.40 | −0.52 | −24.3, 14.9, −21.9 |
| stereovision1 | −20.96 | −43.43 | −76.13 | −30.6, −51.6, −48.1 |
| **mean / median** | **3.94 / 7.92** | **−2.20 / 6.93** | **−16.01 / −6.40** | |

| set | columns | policy | random |
|---|---|---|---|
| original six (reused, see README) | −5.90 | 29.32 | −10.74 |
| new six (G2) | 3.94 | −2.20 | −16.01 |

On the original six, the policy's placement beats the columns by 35 points at a
fixed shape. On the new six it beats them on 3 of 6 circuits and trails on the
mean by 6 points. It still beats random placement on 5 of 6, so it is not
placing blindly; its advantage over a simple regular pattern just does not
transfer to these circuits.

Two circuits drive the mean:

- **8051** is unstable across seeds (−24.3, +14.9, −21.9), the same pattern seen
  in `data_export3` F5 (+18.8, +17.1, −21.9 with the policy's own shape).
- **stereovision1** (44 DSPs) is bad for every method, and worst for the
  policy. It is the only new circuit with a substantial H-block count, so it is
  the closest test of placement quality, and the policy's placement is worse
  than the column pattern there by 22 points.

Note also that on the original six the column placement at 1.0 is negative
(−5.90), while on the new six it is positive (3.94): the regular pattern suits
these circuits better, which narrows the room for any learned placement.

## 4. Random-placement caveats

- 3 draws per seed, 9 per circuit. `random` is a floor, not a tuned baseline.
- The original-six `random` numbers come from `data_export2` and used 5 draws on
  custom_macbuf, mkDelayWorker32B and lightweight_cipher and 3 on the other
  three, so the two sets' `random` rows are not strictly like-for-like.

## 5. The lenet failures

4 of lenet's 20 sweep shapes failed: AR 0.1, 0.2, 0.3 and 0.4. All four stopped
at 565.1 s with VTR returning non-zero or incomplete metrics. That identical
timing across shapes points to a consistent failure in the flow at extreme
elongation (the planned grids are very tall and narrow), not to an interruption.
lenet's sweep best (15.64 at AR 1.7) is taken over the 16 successful shapes. The
rows' `note` text mentions a possible memory cap; `c2_perar.py` writes that
default text on any failure, and no memory cap was used in this request.

## 6. Interruptions and memory

This request was interrupted three times before the clean pass (README). The
important lesson for the method section, not just for logistics: synthesizing
the larger new circuits (lenet, stereovision1) takes ~4.5–6 GB per yosys
process, against well under 1 GB for the original six. That is a practical
limit on evaluating the policy on larger fabrics, separate from the checkpoint's
48×48 canvas.
