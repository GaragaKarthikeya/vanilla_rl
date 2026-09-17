# data_export2: FPGA'27 revision, data request 2

Location: `~/workspace/rl_gnn/vanilla_rl/data_export2/`. Paths below are relative
to `~/workspace/rl_gnn/vanilla_rl` unless prefixed.

Follow-up to `data_export/`. Same repo (git HEAD `f969f2a`), same three final
checkpoints, same VTR 9.0.0-dev build `22a09d39ef`, same flow.

No training code, evaluation code, checkpoints or paper cache DBs were modified.
All new code is in `data_export2/tools/`. The GA is subclassed, not edited.

| seed | checkpoint |
|---|---|
| 7 | `runs/multi11_long_seed7.zip` |
| 42 | `runs/multi11_long_seed42_v2.zip` |
| 123 | `runs/multi11_long_seed123.zip` |

## Headline findings

1. **E2 answers the Figure 2 objection, in the paper's favour.** H-block
   placement carries essentially all of the gain, not the aspect ratio.
   Forcing the aspect ratio to 1.0 and letting the policy place blocks keeps
   **29.32%** against the control's 31.23% — a 1.9-point loss. Keeping the
   policy's aspect ratio but placing blocks at random collapses to **−23.58%**.
2. **E1 is a negative result for the netlist encoder.** Given another circuit's
   entire netlist, the policy emits an identical action sequence in 9 of 18
   circuit-seeds and the mean moves by −0.52 points (31.23 → 30.71).
3. **E1b: the hand-coded heuristic does not match the policy** (3.85% vs
   31.23%, losing on all 6 circuits).

Taken together: the checkpoint has learned a strong, largely
*circuit-independent* placement policy. It is not a shape prior (E2 rules that
out) and not a re-expressed heuristic (E1b rules that out), but it is also not
doing much per-circuit netlist reasoning (E1). See `notes.md` §1–§4.

## Files

| file | experiment | status |
|---|---|---|
| `e1_encoder_corruption.csv` | E1 | done, 90 rows = 18 control (reused) + 4 variants x 18 |
| `e1_action_confidence.csv` | E1 | done, control rollouts only, one row per step |
| `e1_netcount_perm_check.csv` | E1 | extra: whether the `shuffled_netcount` permutation was a no-op |
| `e1b_heuristic.csv` | E1b | done, 6/6 |
| `e2_decomposition.csv` | E2 | done, 162/162 succeeded |
| `e2_ar_rank.csv` | E2 | done, no new runs |
| `e4_summary.json` | E4 | done, no new runs |
| `logs/e1_actions.json` | E1 | raw action sequences, all variants incl. control |
| `logs/e2_actions.json` | E2 | raw action sequences |
| `notes.md` | — | caveats, vacuous cases, GA elitism code check |

**E3 was not run.** See "What was skipped".

### E1 variants

| variant | what is corrupted | how |
|---|---|---|
| `control` | nothing | reused from `data_export/results_per_circuit_seed.csv`, not re-run |
| `mismatched_netlist` | whole netlist graph replaced by a donor circuit's | primary test; see donor table below |
| `zero_graph_embedding` | the 128 graph dims of the 194-d fused vector | forward-pre-hook on `final`, zeroing slots 64:192 (`gnn_extractor.py:102`) |
| `shuffled_netcount` | `net_count_normalized` permuted across H-block nodes | fabric node untouched; `default_rng([seed, circuit_index])` |
| `zero_edges` | all edges dropped | `edge_index` set to −1, `edge_weight` to 0 |

`zero_graph_embedding` and `zero_edges` are **sanity checks only** — they feed
the fuse layer an input never seen in training, so a change there does not show
the netlist was used. `mismatched_netlist` is the primary test because a donor
netlist is in-distribution.

Donor pairs (A ← B) and the current-node rule: the donor's whole padded graph
(`node_features`, `edge_index`, `edge_weight`) replaces A's, while A's capacity,
mask, H-block count and placement order are untouched. `current_block_idx` is
remapped to stay a valid node of B — **H-block node i of A → B's node
`i mod n_B`**, and A's fabric node → B's fabric node (`tools/elib.py`,
`make_donor_transform`).

| A | B (donor) |
|---|---|
| custom_macbuf | softmax |
| lightweight_cipher | softmax |
| softmax | arm_core |
| arm_core | reduction_layer |
| reduction_layer | mkDelayWorker32B |
| mkDelayWorker32B | reduction_layer |

### E2 variants

`policy_ar_random_blocks` and `fixed_ar_random_blocks` use **5 random draws** per
circuit-seed on custom_macbuf, mkDelayWorker32B and lightweight_cipher, and
**3 draws** on reduction_layer, arm_core and softmax, whose VTR flows are
minutes each. `fixed_ar_policy_blocks` is deterministic (`draw_index` 0).
Fixed aspect ratio is 1.0 (`ASPECT_RATIOS` index 9). Random placement draws
uniformly over the legal actions of the paper's own
`FPGAEnv.get_action_mask()`, in the policy's block order, with
`default_rng([seed, circuit_index, draw])`.

## What was skipped and why

- **E3 (GA with elitism) was not run.** It is last in the stated priority order
  and the two target runs are the most expensive work in the request: the
  non-elitist seed-42 runs took 888 VTR calls (reduction_layer) and 1016
  (mkDelayWorker32B), and those circuits average 110–330 s per call, so at
  population-8 concurrency the pair is roughly 12–20 hours. `tools/e3_ga_elitism.py`
  is written, subclasses `GA_Agent` without editing it, and is ready to run as
  `python e3_ga_elitism.py reduction_layer 42`. Nothing in `e3_*.csv` exists yet.
  The code check E3 was meant to support is already settled — see `notes.md` §6:
  the GA's reported result is the best individual ever evaluated
  (`ga_agent.py:278`, `:291-292`, `:312`, `:337`), so the missing elitism cannot
  cost the GA its reported answer.
- **E5 (Bayesian optimization) was not run**, per the request's own instruction
  to skip rather than guess. The decision space is variable-length and
  masked — the number of H-blocks differs per circuit (3 to 43) and legality
  depends on every earlier placement — so it has no fixed-dimension
  parameterization that scikit-optimize or BoTorch would accept at defaults.
  Encoding it would require choices (padding, repair of illegal points, a
  penalty for infeasibility) that would determine the result, and no such
  parameterization is specified in the request or the codebase.

## Method notes

- **Toolchain.** Run inside `distrobox enter ubuntu-work`; the host RHEL 8 glibc
  cannot run `vpr`. `tools/common2.py` reuses `data_export/tools/common.py`,
  which points `VTR_*` at `~/workspace/vtr-verilog-to-routing` rather than
  calling `load_env_file()` (the repo `.env` still names the pre-move path).
- **Universe dims** pinned to the logged training values (48, 48, 67, 2694).
- **Caching.** E1, E1b and E2 share one isolated cache
  (`runs/vtr_layout_cache_<circuit>_dataexport2.db`), separate from both the
  paper's caches and `data_export`'s. Cache hits are reported in the `cached`
  column and counted separately from VTR runs below.
- **Evaluation path.** Every run goes through `FPGAEnv.step`, i.e. the same
  bake, atom-pinning constraints, VTR invocation and metric parsing as training,
  with the paper's reward weights (ar/dl/pw = 1/1/1, wl = 0).
- **Control validation.** The E1 control rollouts were regenerated and asserted
  equal to the logged deterministic actions in
  `data_export/logs/det_actions_and_timing.json` for all 18 circuit-seeds. The
  assertion passed, so the corrupted variants are compared against a rollout
  path proven to reproduce the paper's.

## Runs, cache hits and failures

| stage | rows | real VTR runs | cache hits | failures |
|---|---|---|---|---|
| E1 control | 18 | 0 (reused) | — | 0 |
| E1 corruptions | 72 | 72 | see `notes.md` §5 | 0 |
| E1b | 6 | 6 | 0 | 0 |
| E2 | 162 | 162 | 0 | 0 |
| **total** | **258** | **240** | | **0** |

**No run failed.** All 240 VTR runs returned complete metrics.

The `cached` flag was added after E1's first pass, so it is blank for those 54
rows; `notes.md` §5 gives the exact duplicate-floorplan count (13) and what can
be said about them. E2's 162 runs were all cache misses despite sharing the
cache, because each variant draws distinct floorplans.

## Hardware, concurrency, wall-clock

- **CPU:** 12th Gen Intel Core i9-12900K, 16 cores / 24 threads, 31 GiB RAM.
  RHEL 8.10 host, Ubuntu 24.04 distrobox. No GPU; policy inference on CPU.
- **Concurrency:** 16 concurrent VTR flows for E1 and E2, 6 for E1b.

| stage | start | end | wall-clock |
|---|---|---|---|
| E1 rollouts + confidence | 03:05 | 03:09 | 4 min |
| E1 corruption VTR (54) | 03:10 | 03:16 | 6 min |
| E4 (no runs) | 03:12 | 03:13 | <1 min |
| E1 mismatched_netlist VTR (18) | 03:22 | 03:25 | 3 min |
| E1b (6) | 03:25 | 03:27 | 2 min |
| E2 ar rank + rollouts (no VTR) | 03:27 | 03:28 | <1 min |
| E2 VTR (162) | 03:28 | 03:50 | 22 min |
| **total** | 03:05 | 03:50 | **45 min** (2026-09-18, IST) |

Summed `vtr_seconds` over the 240 timed runs is 7.46 h of VTR time, compressed
into 45 min of wall-clock by the 16-way concurrency. All runs ran under
concurrent load, so single-run times would be lower.

## Reproducing

```bash
cd ~/workspace/rl_gnn/vanilla_rl/data_export2/tools
distrobox enter ubuntu-work -- ~/.venv/bin/python e1.py rollouts
distrobox enter ubuntu-work -- bash run_rest.sh      # E1 VTR -> E1b -> E2, resumable
distrobox enter ubuntu-work -- ~/.venv/bin/python e1_permcheck.py
distrobox enter ubuntu-work -- ~/.venv/bin/python e4.py
# not run:
distrobox enter ubuntu-work -- ~/.venv/bin/python e3_ga_elitism.py reduction_layer 42
```
