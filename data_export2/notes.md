# notes.md — surprising and inconsistent things found in data request 2

Paths are relative to `~/workspace/rl_gnn/vanilla_rl`. Nothing below "fixes"
either the code or the paper; it records what the runs show.

## 1. E1 is a negative result: the netlist encoding barely changes the output

Mean held-out ADP reduction over 6 circuits x 3 seeds (`e1_encoder_corruption.csv`):

| variant | mean ADP reduction % | action sequence identical to control | mean actions differing |
|---|---|---|---|
| control (paper zero-shot) | 31.23 | 18/18 | 0.00 |
| **mismatched_netlist** | **30.71** | **9/18** | 3.28 |
| zero_graph_embedding | 31.25 | 3/18 | 8.11 |
| shuffled_netcount | 31.29 | 17/18 | 0.11 |
| zero_edges | 30.91 | 9/18 | 1.78 |

`mismatched_netlist` is the informative one: the policy is given another
held-out circuit's *entire* netlist graph (an in-distribution but wrong input)
while capacity, masking, H-block count and placement order stay the circuit's
own. In 9 of 18 circuit-seeds it emits a **byte-identical action sequence**, and
the mean result moves by only **−0.52 points**. On custom_macbuf every seed is
identical; on seed 42, five of the six circuits are identical.

Read together with §3 and §4, the straightforward reading is that most of what
the checkpoint contributes is a **largely circuit-independent placement rule**,
keyed off what the policy can see without the graph: the grid channel,
`valid_wh`, the H-block count and the legality mask. Note this is *not* a mere
shape prior — §4 shows the aspect ratio contributes almost nothing and the
placement contributes nearly all of the gain. The netlist encoder changes the
answer at the margin, and mostly on seed 123.

Seed 123 is the exception and is consistently the most netlist-sensitive
(e.g. reduction_layer 16/33 actions differ under a donor netlist, arm_core
10/25); seed 42 is the least (5 of 6 circuits unchanged).

Caveats that keep this from being a stronger claim than it is:

- `zero_graph_embedding` and `zero_edges` are only **sanity checks**. Zeroing
  128 of the 194 fused dimensions feeds the fuse layer an input it never saw in
  training, so a change there does not prove the netlist was *used*, and the
  near-zero mean change does not prove it was not. They are reported for
  completeness.
- ADP is not a monotone function of "how right" the input is. Several corrupted
  runs score *better* than control (arm_core seed 7: 9.07 vs 6.99;
  softmax seed 7: 8.51 vs 5.65), which is within the placer-seed noise band
  measured in C4 (baseline CV 1.4–5.3%).

## 2. shuffled_netcount is partly vacuous, and the ordering leak

Two separate problems, both recorded in `e1_netcount_perm_check.csv`:

1. **The permutation is sometimes a mathematical no-op.** `reduction_layer`
   (32 H-blocks) and `softmax` (8) give *every* H-block the same
   `net_count_normalized` (0.45 and 0.23), so permuting it cannot change the
   observation at all. custom_macbuf seed 7 and lightweight_cipher seed 42 draw
   the identity permutation over 4 and 3 blocks. That is 8 of 18 circuit-seeds
   where "actions unchanged" is vacuous.
   Restricted to the **10 circuit-seeds where a value actually moved**, the
   result is unchanged in substance: 9/10 still emit identical actions, mean ADP
   38.39 vs control 38.29.
2. **Block ordering still carries the true net counts.** `FPGAEnv` sorts the
   blocks to place by each block's own net count at `reset()`
   (`src/env/fpga_env.py:263-284`), from the benchmark config and *not* from the
   observation. The request fixed the order deliberately, so permuting the
   feature leaves that channel intact: the policy can still infer the ranking
   from the order in which blocks arrive. `shuffled_netcount` therefore bounds
   the value of the *feature*, not of the net-count information.

## 3. E1b: a hand-coded heuristic does not match the policy

`e1b_heuristic.csv`, 6 VTR calls, no learning. Aspect ratio from the CLB-to-IO
ratio (elongate to 0.5 when IO-bound, square 1.0 otherwise), H-blocks clustered
at the grid centre in the environment's own order under the policy's mask.

| circuit | clb/io | heuristic AR | heuristic ADP % | policy (mean of 3 seeds) % |
|---|---|---|---|---|
| custom_macbuf | 0.18 | 0.5 | −11.42 | 55.46 |
| mkDelayWorker32B | 0.44 | 0.5 | 2.93 | 58.72 |
| lightweight_cipher | 0.15 | 0.5 | 15.06 | 33.49 |
| reduction_layer | 13.85 | 1.0 | 8.92 | 23.30 |
| arm_core | 2.78 | 1.0 | 6.20 | 9.12 |
| softmax | 2.22 | 1.0 | 1.40 | 7.27 |
| **mean** | | | **3.85** | **31.23** |

The policy beats this heuristic on all 6 circuits, by 27.4 points on average.

This is **one** hand-coded rule, fixed before the runs and not tuned, so it is
evidence that *this* heuristic is not competitive, not that no heuristic is. The
damage is concentrated in the aspect-ratio rule: elongating to 0.5 is what sinks
custom_macbuf (−11.42, against 48.20 for the best AR in the C2 sweep). A
heuristic that swept the aspect ratio instead of deriving it would land near the
C2 number (14.86), still well below the policy.

## 4. E2: the aspect ratio does *not* carry the gain — placement does

`e2_decomposition.csv`, 162/162 runs succeeded. Mean ADP reduction %, per
circuit averaged over seeds and draws:

| circuit | control | policy AR + random blocks | AR 1.0 + policy blocks | AR 1.0 + random blocks |
|---|---|---|---|---|
| custom_macbuf | 55.46 | −11.24 | 40.99 | 6.36 |
| mkDelayWorker32B | 58.72 | −17.94 | 56.54 | −8.90 |
| lightweight_cipher | 33.49 | 10.83 | 33.87 | 12.59 |
| reduction_layer | 23.30 | −46.31 | 28.98 | −29.69 |
| arm_core | 9.12 | −33.09 | 8.41 | −21.04 |
| softmax | 7.27 | −43.75 | 7.13 | −23.76 |
| **mean** | **31.23** | **−23.58** | **29.32** | **−10.74** |

This contradicts the premise of the reviewer's Figure 2 objection. Throwing away
the policy's aspect ratio entirely and forcing 1.0 costs only **1.9 points**
(31.23 → 29.32), and on reduction_layer and lightweight_cipher the forced square
is actually *better* than the policy's own choice. Throwing away the policy's
placement instead, while keeping its aspect ratio, costs **54.8 points** and
lands far below the traditional baseline.

The H-block placement is doing essentially all of the work.

Two further observations:

- Random placement is **better** at AR 1.0 (−10.74) than at the policy's own
  0.8–0.9 (−23.58). The policy's aspect ratio is only an asset in combination
  with its own placement; on its own it is a liability. So the AR and the
  placement are not separable contributions that add up.
- `fixed_ar_policy_blocks` beats the control on reduction_layer (28.98 vs 23.30)
  and lightweight_cipher (33.87 vs 33.49), i.e. the policy's aspect-ratio choice
  is actively costing it on two of six circuits.

Combined with §1, the picture is a policy that has learned a strong and largely
*circuit-independent* placement rule — good enough to beat a hand-coded
heuristic comfortably (§3) and robust to being handed the wrong netlist (§1),
but not evidently reasoning per-circuit from the graph.

## 5. E2: the policy's aspect ratio is not the sweep's best

`e2_ar_rank.csv`, no new runs. Rank of the policy's chosen AR among the 20
aspect ratios of the trimmed-baseline sweep (1 = best):

| circuit | policy AR | rank | sweep best AR |
|---|---|---|---|
| custom_macbuf | 0.8 | 4 | 0.6 |
| mkDelayWorker32B | 0.9 | 2 | 0.8 |
| lightweight_cipher | 0.8 / 0.9 | 2 / 1 | 0.9 |
| reduction_layer | 0.9 | 2 | 0.8 |
| arm_core | 0.9 | 6 | 1.2 |
| softmax | 0.9 | 9 | 0.5 |

The policy picks 0.8–0.9 almost everywhere, across all circuits and seeds — only
custom_macbuf (0.8 vs 0.9) and lightweight_cipher vary at all. That near-constant
choice is consistent with §1: the aspect-ratio decision looks like a learned
prior rather than a per-circuit inference. It ranks well on the circuits where
0.8–0.9 happens to be right and poorly on softmax (9/20) and arm_core (6/20),
which are exactly the two circuits where the C2 sweep beats the policy.

The ranking is by the sweep's own measure, which holds H-block placement fixed
to the trimmed pattern, so it is not a statement about the policy's own ADP.

## 6. The policy is confident about the aspect ratio and almost uniform over placements

`e1_action_confidence.csv`, control rollouts only:

| step kind | n | mean prob of chosen action | mean entropy over legal actions | entropy / ln(n_legal) |
|---|---|---|---|---|
| aspect_ratio (step 0) | 18 | 0.845 | 0.63 | 0.21 |
| H-block placement | 342 | 0.026 | 6.71 | 0.94 |

The aspect-ratio decision is nearly deterministic (median chosen probability
0.906), while each placement step is close to uniform over its legal actions —
94% of the maximum possible entropy, with the chosen action typically at
p ≈ 0.007.

This is the reverse of what §4 would suggest on its own, and the two together
explain C5: the placement *argmax* carries almost all the value even though the
placement *distribution* is nearly flat, so sampling instead of taking the argmax
destroys the result (`data_export/policy_sampled.csv`: sampled rollouts are far
worse than deterministic on the large circuits). The confident-looking decision
contributes little; the low-confidence decisions contribute nearly everything.

A caveat: entropy is measured over the full legal action set, which is large
(a mostly empty grid), so a near-uniform distribution is the expected shape and
this is not by itself evidence of a poorly trained head.

## 7. Cache accounting for E1

The `cached` column was added after E1's first pass had already run, so it is
empty for the 54 corrupted rows of the first pass (the 18 control rows are
reused Tier B numbers and were never re-run). What can be stated exactly:

- 13 of those 54 evaluations repeat a floorplan already evaluated elsewhere in
  the same pass (computed from the action sequences in `logs/e1_actions.json`).
- Only 2 of the 54 completed in under 2 s. The other 11 repeats were submitted
  to the 16-worker pool concurrently with their twin, so both were real VTR runs
  and neither could hit the cache.

`cached` is recorded properly for the 18 `mismatched_netlist` rows, for E1b and
for E2, all of which share one isolated cache (`*_dataexport2.db`).

## 8. E3: the GA's reported number is the best individual ever evaluated

Confirmed from the code, for the paper to state directly. In `ga_agent.py`
(`~/workspace/vtr_exp/ga_agent.py`), `run()` keeps `best_fit_global`, which is
initialised to infinity (`:278`), updated only when a generation's minimum
improves on it (`:291-292`), and returned both on early stop (`:312`) and on
exhausting the generations (`:337`). The per-generation log line writes the same
running best (`:300`).

So the absence of elitism cannot cost the GA its reported answer — the incumbent
is never lost from the *result*, only from the *population*. Elitism can only
change search progress, which is what E3 measures.
