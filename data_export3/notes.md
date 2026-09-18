# notes.md — data request 3

Paths are relative to `~/workspace/rl_gnn/vanilla_rl`. Nothing here "fixes"
either the code or the paper; it records what the runs show.

## 1. F1: the learned rule IS extractable as a geometric pattern

`f1_geometric_transfer.csv`, 30/30 runs succeeded, seed 42.

The headline split is by whether the source circuit had enough H-blocks to fill
the target. Where it did (`n_blocks_random` = 0), the transferred floorplan
holds up; where it did not, the random fill dominates and the result collapses.

| subset | n | mean ADP reduction % |
|---|---|---|
| transfers with no random fill | 15 | **32.72** |
| transfers needing random fill | 15 | −40.27 |
| the policy itself (seed 42, same 6 circuits) | 6 | 32.15 |

Per target, restricted to transfers with no random fill:

| target | policy (seed 42) | transferred, no random fill | n |
|---|---|---|---|
| custom_macbuf | 59.53 | 61.34 | 4 |
| lightweight_cipher | 39.73 | 36.56 | 5 |
| reduction_layer | 21.51 | 12.38 | 1 |
| arm_core | 11.83 | 14.58 | 2 |
| softmax | 5.39 | 7.02 | 3 |
| mkDelayWorker32B | 54.88 | — | 0 |
| **mean over the 5 targets with data** | **27.60** | **26.38** | |

Averaged per target so the easy circuits do not dominate, a floorplan copied
from *another circuit* and rescaled gives up only **1.2 points** against the
policy's own. On three of five targets the transfer is *better* than the policy.

This is the direct answer to the question the request posed: the policy's output
is largely a portable geometric pattern, and the paper can print it. It is
consistent with `data_export2`'s finding that the policy is largely
circuit-independent.

Two honest limits:

- **The per-target sample is small and unbalanced** (1 to 5 sources), because
  only sources with at least as many H-blocks as the target qualify.
  mkDelayWorker32B (43 H-blocks) has **no** qualifying source among the other
  five, so it contributes nothing to the clean comparison — and it is the
  circuit where the policy's margin over every baseline is largest.
- **The random-fill subset is not a transfer measurement.** When a 4-block
  circuit is transferred onto a 43-block one, 39 of the 43 blocks are random, so
  −40.27 mostly measures random placement (`data_export/random_floorplans.csv`
  gives a comparable figure). It should not be read as "transfer fails".

## 2. F2: the flat distribution is real, but it is not a masking artifact

`f2_action_distribution.csv`, 342 placement steps, no VTR.

| quantity | median | mean |
|---|---|---|
| n_legal_actions | 1496 | 1405 |
| chosen_action_prob | 0.0073 | 0.0263 |
| top10_mass | 0.0607 | 0.1163 |
| uniform_reference_top10_mass | 0.0067 | 0.0141 |
| entropy / ln(n_legal) | 0.966 | 0.936 |
| gini | 0.285 | 0.314 |

The distribution *is* close to uniform by entropy (96.6% of the maximum), but it
is **not** uniform: the ten highest-probability legal actions carry a median
**7.6x** the mass a uniform distribution would put there, and the Gini
coefficient is 0.285. So Section 5.4's "median chosen-action probability 0.007"
is a consequence of there being ~1500 legal actions, not of a degenerate or
masked-out head.

`chosen_action_prob` equals `top1_prob` in every row, as it must: these are
deterministic (argmax) rollouts. That is a consistency check on the pipeline.

### The probabilities do rank floorplans, but only coarsely

`f2_rank_vs_adp.csv`, 60 VTR runs, seed 42, first placement step only.

| circuit | top-5 mean | random-5 mean |
|---|---|---|
| custom_macbuf | 53.82 | 21.19 |
| lightweight_cipher | 34.42 | 19.57 |
| mkDelayWorker32B | 54.86 | 31.57 |
| reduction_layer | 21.06 | 15.33 |
| arm_core | 11.90 | 8.08 |
| softmax | 5.94 | −9.13 |
| **pooled** | **30.33** | **14.43** |

The policy's top-5 actions beat random legal actions by **15.9 points** on every
one of the six circuits. That supports what Section 5.4 claims implicitly.

But **within** the top 5 the ordering carries no information: of 360 comparable
pairs, 189 are concordant (**52.5%**, i.e. chance). The head separates good
regions from bad ones; it does not finely rank inside the good region. Worth
stating plainly rather than implying a sharper ordering than exists.

## 3. F4: instance pinning handicaps the policy, it does not flatter it

`f4_no_pinning.csv`, 18 unpinned VTR runs (pinned rows reused, not re-run),
0 failures.

| circuit | pinned (paper) | unpinned | delta |
|---|---|---|---|
| custom_macbuf | 55.46 | 55.33 | −0.13 |
| mkDelayWorker32B | 58.72 | 62.91 | +4.19 |
| lightweight_cipher | 33.49 | 33.73 | +0.24 |
| reduction_layer | 23.30 | 28.11 | +4.81 |
| arm_core | 9.12 | 21.40 | **+12.28** |
| softmax | 7.27 | 6.41 | −0.86 |
| **mean** | **31.23** | **34.65** | **+3.42** |

The reviewer's confound is real but points the other way. Letting VPR assign
instances to the policy's tiles — exactly the freedom the GA has — makes the
policy **better**, by 3.4 points on average and 12.3 points on arm_core. So the
policy-vs-GA comparison as published is *conservative*: removing the asymmetry
widens the gap rather than closing it. The 40-point mkDelayWorker32B result is
not an artifact of pinning; unpinned it grows to 62.91 vs the GA's 17.92.

Note the floorplan is identical in both variants; only the constraints file is
omitted. So this measures the cost of over-constraining VPR's packer/placer, not
a different floorplan.

## 4. F5: the reviewer was right — the original six were favourable

This is the most consequential result in this request, and it goes against the
paper.

`f5_extra_heldout.csv`. Zero-shot, deterministic, 3 seeds, same flow as the
paper's held-out evaluation.

| circuit | occupancy | H-blocks | mean ADP reduction % | per seed |
|---|---|---|---|---|
| usb_uart_core | 0.636 | 2 | 14.28 | 14.7, 14.1, 14.0 |
| uriscv_core | 0.719 | 2 | 12.80 | 14.0, 13.1, 11.4 |
| lenet | 0.744 | 1 | 11.03 | 13.1, 11.2, 8.8 |
| aes_inv_cipher | 0.719 | 4 | 9.12 | 9.8, 5.2, 12.3 |
| 8051 | 0.675 | 2 | 4.68 | 18.8, 17.1, **−21.9** |
| stereovision1 | 0.569 | 44 | **−59.33** | −43.7, −69.0, −65.3 |

| set | mean | median |
|---|---|---|
| **new circuits (6)** | **−1.24** | **10.08** |
| new circuits excluding stereovision1 | 10.38 | 11.03 |
| original six | 31.23 | 28.40 |
| original four, excluding the authored probes | 24.60 | — |
| all twelve combined | 15.00 | — |

Even discarding stereovision1 entirely, the new circuits average **10.38%**
against the original six's 31.23% and the four non-authored originals' 24.60%.
The held-out set in the paper is not representative, and the zero-shot number
falls by roughly half to two-thirds on circuits chosen without reference to it.

**stereovision1 is the informative failure.** It is the only new circuit with a
substantial H-block count (44 DSPs), and it is the one that collapses. The
policy shrinks the grid from the baseline's 40x40 to 32x36 (1152 tiles for 821
required blocks) and the result is 59 points *worse* than the baseline, on all
three seeds. That is the same "shrink the canvas" behaviour that wins on the
sparse original circuits, applied where there is no slack to reclaim. High
occupancy is exactly the regime the reviewer asked about, and the policy's rule
is counter-productive there.

8051 seed 123 (−21.9 against +18.8 and +17.1 on the other two seeds) is a second
instance of the same instability, on a 17x17 grid.

### Why only 6 circuits, not the 8–10 requested

The checkpoint's universe caps and the occupancy criterion are in direct
tension, and this is structural rather than a search failure:

- The canvas is fixed at 48x48 with MAX_NODES=67 and MAX_EDGES=2694
  (`data_export/facts.json`). Any well-packed modern design exceeds it:
  spmv needs 98x98 and 40,190 edges, attention_layer 74x74 and 23,508 edges,
  conv_layer_hls 103x103. These cannot be evaluated by this checkpoint at all.
- Of the circuits that *do* fit, most pack to **zero DSPs and zero BRAMs**
  (blob_merge 0.72 occupancy, stereovision0 0.74, bgm 0.75, aes_cipher 0.74,
  xtea 0.75, soc_core 0.27, mips_16). With no H-blocks the floorplanning action
  space is empty — the episode is the aspect-ratio step and nothing else — so
  they cannot test placement and were excluded. They are recorded in
  `f5_candidates.csv` / `f5_candidates_b2.csv` with their resource profiles.
- Two candidates failed to synthesize: `eltwise_layer` and `conv_layer`
  (traditional VTR flow rc=1). Two more, `ethmac` and `enet_core`, also failed
  to build.
- `bnn` and `gemm_layer` were still in ABC logic optimization after ~2.5 hours
  when this request was written up; they are not in the table. Both are large
  Koios designs and would very likely have exceeded the caps in any case.

So 6 circuits, of which 5 of 6 (83%) have occupancy above 60%, which does meet
the "at least half" criterion. `robot_rl` was deliberately excluded as a
candidate: it is not in the training pool but it *was* in the checkpoint's
sizing universe, so it is not genuinely unseen.

The honest summary for the paper is that the eligible-circuit population for
this checkpoint is small and skewed sparse, and that the sparse regime is where
the method works.

## 5. F6: the policy is converged, not undertrained

`f6_entropy.csv`, 181 PPO updates across the three runs, parsed from the
training logs.

| seed | updates | policy_entropy first → last | explained_variance first → last | ep_rew_mean first → last |
|---|---|---|---|---|
| 7 | 1–901 | 4.781 → 3.187 | −0.005 → 0.925 | −0.649 → 0.247 |
| 42 | 1–886 | 4.926 → 3.069 | 0.035 → 0.925 | −0.791 → 0.260 |
| 123 | 1–886 | 4.896 → 3.379 | 0.013 → 0.951 | −0.773 → 0.271 |

Entropy falls by about 35%, explained variance rises from ~0 to ~0.93, and mean
episode reward crosses from negative to positive. This is a converged run, not a
policy still at its initialization. So the near-uniform *held-out* distribution
in §2 is not undertraining.

**A discrepancy worth flagging.** The converged training entropy is ~3.1, while
the held-out placement entropy measured in §2 is ~7.06. These are not directly
comparable — the training circuits are much smaller (core grids 6x6 to 26x26, so
far fewer legal actions) than the held-out ones (up to 48x48, ~1500 legal
actions), and the F6 number is a training-time average over the 11-circuit pool
while the F2 number is an inference-time measurement on 6 unseen circuits.
Taken together they are *suggestive* that the policy is sharply peaked in
distribution and diffuse out of distribution, but the two numbers do not
establish that on their own; a like-for-like measurement would have to evaluate
the checkpoint's entropy on the training circuits, which was not requested and
was not run.
