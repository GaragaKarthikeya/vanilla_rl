# data_export3: FPGA'27 revision, data request 3

Location: `~/workspace/rl_gnn/vanilla_rl/data_export3/`. Paths below are relative
to `~/workspace/rl_gnn/vanilla_rl` unless prefixed.

Follow-up to `data_export/` and `data_export2/`. Same repo, same three final
checkpoints (seeds 7, 42, 123), same VTR 9.0.0-dev build `22a09d39ef`, same flow.

No training code, evaluation code, checkpoints or paper cache DBs were modified.
All new code is in `data_export3/tools/`. One isolated cache
(`runs/vtr_layout_cache_<circuit>_dataexport3.db`) is shared across this request.

Experiments were run in the priority order given in the follow-up message:
**F1, F4, F2, F5, F6, then F3.**

## Headline findings

1. **F1 — the learned rule is a portable geometric pattern.** A floorplan copied
   from another circuit and rescaled gives up only **1.2 points** against the
   policy's own (26.38 vs 27.60, averaged per target over transfers that needed
   no random fill). The paper can print the pattern.
2. **F5 — but the held-out set was favourable, and this is the bad news.** On
   six circuits chosen without reference to the paper, zero-shot averages
   **−1.24%** (median 10.08), or **10.38%** excluding the one high-H-block
   circuit, against the original six's **31.23%**. stereovision1 (44 DSPs)
   collapses to **−59.33%** on all three seeds.
3. **F4 — instance pinning handicaps the policy.** Removing the constraints file
   *improves* the policy to **34.65%** from 31.23%. The published GA comparison
   is conservative, not flattering.
4. **F2 — the flat placement distribution is real but not an artifact.** Top-10
   actions carry 7.6x the uniform mass; top-5 actions beat random legal actions
   by 15.9 points on all six circuits. Within the top 5, ordering is chance.
5. **F6 — the policy is converged**, not undertrained (entropy 4.9 → 3.1,
   explained variance 0.03 → 0.93).

Full detail, caveats and the vacuous/degenerate cases are in `notes.md`.

## Files

| file | experiment | status |
|---|---|---|
| `f1_geometric_transfer.csv` | F1 | done, 30/30 ordered pairs at seed 42 |
| `f1_pattern.csv` | F1 | done, 342 block positions (18 circuit-seeds), no new runs |
| `f1_reference.csv` | F1 | done, per-target policy / AR-sweep / random-best-of-20, all reused |
| `f2_action_distribution.csv` | F2 | done, 342 placement steps, no new runs |
| `f2_rank_vs_adp.csv` | F2 | done, 60/60 |
| `f4_no_pinning.csv` | F4 | done, 18 unpinned runs + 18 reused pinned rows |
| `f5_candidates.csv`, `f5_candidates_b2.csv` | F5 | 23 candidates attempted: 21 recorded, 2 killed by an out-of-memory reboot |
| `f5_extra_heldout.csv` | F5 | done, 6 circuits x 3 seeds = 18/18 |
| `f6_entropy.csv` | F6 | done, 181 PPO updates, no new runs |
| `logs/` | — | per-stage logs, raw action sequences |

**F3 was not run.** See below.

## Method notes

### F1 transfer rule (implemented once, `tools/f1.py:transfer_actions`)

1. Take A's chosen aspect ratio and its H-block coordinates on A's **core** grid.
2. Rescale to B's core grid, `x_B = round(1 + (x_A - 1) * (W_B - 1) / (W_A - 1))`
   and likewise for y, then clamp into B's legal range. A source with a
   degenerate extent (`W_A == 1`) maps to 1.
3. Apply B's own legality mask (`FPGAEnv.get_action_mask()`). If the rescaled
   coordinate is illegal or collides, take the nearest legal tile by **Manhattan**
   distance, ties broken by lower x then lower y (`numpy.lexsort` on
   distance, x, y).
4. Place `min(n_A, n_B)` blocks in the environment's own order (BRAMs then DSPs,
   each by descending net count). Any remainder is placed uniformly at random
   over the legal mask with `numpy.default_rng([seed, circuit_index])`, and
   counted in `n_blocks_random`.
5. B's aspect ratio is A's chosen ratio.

`f1_pattern.csv` normalizations: `dist_to_center_norm` is the Euclidean distance
to the core-grid centre divided by the centre-to-(1,1) distance (0 = centre,
1 = corner); `dist_to_edge_norm` is `min(x-1, W-x, y-1, H-y)` divided by
`floor((min(W,H)-1)/2)` (0 = on the edge, 1 = deepest interior).

### F4

The floorplan is byte-identical to the policy's; only the VPR constraints file is
omitted (`bake_layout(..., block_names=None, constraints_output_path=None)`),
default VPR seed. So F4 measures the cost of over-constraining VPR's packer and
placer, not a different floorplan.

### F5 circuit selection

Candidates were drawn from the VTR and Koios suites plus the freecores and
ultraembedded collections in the VTR tree, excluding every circuit in the
training pool, the six held-out circuits, and `robot_rl` (not in the training
pool, but it *was* in the checkpoint's sizing universe, so it is not unseen).

A candidate is eligible only if it (a) builds an island-style baseline through
the traditional VTR flow on the stock arch, (b) fits the checkpoint's universe
caps (core grid <= 48x48, graph nodes <= 67, edges <= 2694) and (c) packs to at
least one DSP or BRAM. Criterion (c) matters: many circuits that fit pack to
zero H-blocks, which leaves the floorplanning action space empty. All 21
surveyed candidates, eligible or not, are in `f5_candidates*.csv` with their
resource profiles and the reason for exclusion. `notes.md` §4 explains each.

Baselines were built with `tools/f5_build.py` rather than
`run_traditional_flow.py`, because the latter calls `load_env_file()` and would
pick up the repo's stale `.env` paths. The arch, flow and metric parsing are the
same.

## What was skipped

- **F3 (retrain seed 42 without the GCN) was not run.** It is last in the stated
  priority order and is a full ~11 h training run (the paper's seed-42 run took
  11.0 h, `data_export/training_accounting.csv`). `tools/f3_train.py` and
  `tools/no_gcn_extractor.py` are written and smoke-tested: the fuse layer takes
  66 inputs (64 CNN + 2 fabric dims, no zero-padding), has no `conv1`/`conv2`,
  and still emits 128 features; the paper's extractor takes 194. Running it needs
  two runtime rebindings, both documented in the script: the extractor class
  (`trainer.py:176` hard-codes it) and `compute_max_dims` (the paper's universe
  includes `robot_rl`, whose files are deleted, so the dims are pinned to the
  logged 48/48/67/2694).
- **`bnn` and `gemm_layer` baselines never finished.** Both were still in ABC
  logic optimization after ~2.5 h when the host ran out of memory and rebooted.
  The builds had no per-job memory limit; that was a driver mistake. They have
  no row in `f5_candidates*.csv` and appear in no result. Both are large Koios
  designs and would very likely exceed the universe caps.
- **`eltwise_layer`, `conv_layer`, `ethmac`, `enet_core`** failed to build a
  baseline (traditional VTR flow returned rc=1). Recorded with that reason.

## Runs, cache hits and failures

| stage | rows | real VTR runs | cache hits | failures |
|---|---|---|---|---|
| F1 transfer | 30 | 30 | 0 | 0 |
| F1 pattern / reference | 342 / 18 | 0 (reused) | — | 0 |
| F2 distribution | 342 | 0 (no VTR) | — | 0 |
| F2 rank | 60 | 60 | 0 | 0 |
| F4 unpinned | 18 | 18 | 0 | 0 |
| F4 pinned | 18 | 0 (reused) | — | 0 |
| F5 baseline builds | 21 | 21 | — | 4 build failures; 2 more killed by OOM reboot |
| F5 eval | 18 | 18 | 0 | 0 |
| F6 | 181 | 0 (no VTR) | — | 0 |
| **total** | | **147 timed VTR runs** | **0** | **0 evaluation failures** |

No *evaluation* run failed. The only failures are the four candidate baselines
that would not synthesize, listed above with their return code.

Cache hits were 0 throughout: every experiment evaluates floorplans the others
do not, so the shared isolated cache never served a repeat.

## Hardware, concurrency, wall-clock

- **CPU:** 12th Gen Intel Core i9-12900K, 16 cores / 24 threads, 31 GiB RAM.
  RHEL 8.10 host, Ubuntu 24.04 distrobox. No GPU; policy inference on CPU.
- **Concurrency:** 16 concurrent VTR flows for F1 and F2; 16 for F4; 7 then 5 for
  the F5 baseline builds; 12 for the F5 evaluation.

| stage | end time | notes |
|---|---|---|
| F1 pattern + reference | 04:39 | no VTR |
| F1 transfer (30) | 04:47 | |
| F2 distribution | 04:50 | no VTR |
| F2 rank (60) | 04:58 | |
| F4 (18) | 05:03 | |
| F6 | 05:05 | no VTR |
| F5 builds + eval | 06:11 | bnn / gemm_layer later killed by an OOM reboot |
| **total** | | **~3 h 10 min** (2026-09-18, IST), 04:30–06:11 wall-clock excluding the two killed builds |

Summed `vtr_seconds` over the 147 timed runs is **5.90 h** of VTR time,
compressed by the concurrency above. All runs ran under concurrent load, so
single-run times would be lower.

## Reproducing

```bash
cd ~/workspace/rl_gnn/vanilla_rl/data_export3/tools
distrobox enter ubuntu-work -- ~/.venv/bin/python f1.py pattern
distrobox enter ubuntu-work -- ~/.venv/bin/python f1.py transfer 16
distrobox enter ubuntu-work -- ~/.venv/bin/python f4.py 16
distrobox enter ubuntu-work -- ~/.venv/bin/python f2.py dist
distrobox enter ubuntu-work -- ~/.venv/bin/python f2.py rank 16
distrobox enter ubuntu-work -- ~/.venv/bin/python f5_build.py 7          # batch 1
distrobox enter ubuntu-work -- ~/.venv/bin/python f5_build.py 5 batch2   # batch 2
distrobox enter ubuntu-work -- ~/.venv/bin/python f5_eval.py 12
distrobox enter ubuntu-work -- ~/.venv/bin/python f6.py
# not run:
distrobox enter ubuntu-work -- ~/.venv/bin/python f3_train.py
```

All drivers are resumable: each writes through an append-only `CsvSink` that
skips rows already present.
