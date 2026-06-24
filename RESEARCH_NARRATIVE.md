# RL for eFPGA Placement — Research Narrative

*Self-contained project summary, written so a fresh AI agent (with or without access to this repo) can pick up the full research context without re-deriving it. Last updated 2026-06-19.*

**Status (2026-06-19): experimental work is paused, deliberately, to move to paper-writing.** §5.8 is the final settled comparison between the two trained models — don't run more one-off zero-shot probes expecting to resolve the open question there; it needs multi-seed training or a much larger held-out set, both out of scope for now. If picking this up cold for paper-writing, venue selection, or a critical-thinking pass: read §6 (honest assessment) and §5.8 (final scorecard) first, they're the load-bearing sections.

---

## 1. The one-paragraph pitch

This project embeds a small reconfigurable fabric (eFPGA) inside an ASIC to hide one critical IP block's function from GDSII-level reverse engineering (a hardware-security technique: logic masking). A generic FPGA architecture is over-provisioned for any single fixed design, so a custom-tailored architecture/placement sized for that one netlist can recover most of the area/delay/power ("ADP") overhead. Two prior papers solved this with **per-instance evolutionary search** (genetic algorithms). This project asks: can a **learned RL policy** do the same job, and — critically — can one trained policy **amortize across multiple IP blocks**, transferring/fine-tuning onto a new design instead of restarting search from zero every time? That second question is the one evolutionary search structurally cannot answer, and it's the actual novelty claim.

---

## 2. Background and research lineage

**Why ADP is the target metric:** Area×Delay×Power is the size of the "masking tax" — the PPA overhead of implementing an IP block on reconfigurable fabric instead of hardening it into standard cells. Minimizing ADP is minimizing the cost of the security technique.

**The two papers this work continues**, same benchmark suite (diffeq1, diffeq2, etc.), same VTR-based evaluation methodology:

- **GOLDS** (Nandi, Mishra, Rao — GLSVLSI '24): Genetic Algorithm over DSP/BRAM placement on a fixed eFPGA layout. Fitness function = **Area×Delay only (2-term, no power)**. Reports up to ~35% improvement over a traditional layout on diffeq1.
- **Meta-heuristic / NSGA-II** (Bhargav, Pradyumna, Rao — VLSID '25): NSGA-II, multi-objective (delay, power), *additionally* co-evolves CLB LUT size and custom DSP/BRAM sizing (not just placement, unlike GOLDS and unlike this project so far). Reports up to ~44.6% improvement.

Both share a structural limitation: every new IP block restarts the search population from scratch. There is no concept of a policy that carries knowledge from one design to the next.

---

## 3. The two core research questions

1. **Can RL go head-to-head with genetic/evolutionary search on a single design**, using the same VTR-based evaluation methodology and benchmarks?
2. **Does an RL policy generalize across benchmarks well enough to make fine-tuning on a new target design faster/better than training from scratch** — i.e., does the "amortization" GA/NSGA-II structurally lack actually materialize in practice?

Both are good, well-posed, falsifiable questions, distinct from what the prior papers could even ask. See §6 for an honest assessment of how far the evidence currently goes toward answering each — **promising, not yet proven**, for reasons that matter for how any paper draft should hedge its claims.

---

## 4. System overview

- **Pipeline:** Verilog → VTR synthesis/placement/routing on a custom-generated FPGA architecture XML → extract delay/power/wirelength/routing-area → reward.
- **Reward:** `-(ar_w·log(area/baseline_area) + dl_w·log(delay/baseline_delay) + pw_w·log(power/baseline_power) + wl_w·log(wirelength/baseline_wirelength))`, weights `ar=dl=pw=1.0, wl=0.0` throughout. Zero = matches traditional VTR baseline exactly; positive = better. Log-ratio chosen for scale-invariance across differently-sized benchmarks sharing one rollout buffer.
- **Agent:** `CustomMaskablePPO` (extends `sb3_contrib.MaskablePPO`) with 7 custom training-stability diagnostics (gradient variance, cosine similarity between per-env gradients, policy entropy, explained variance, etc.), logged to Weights & Biases.
- **Architecture evolution (important):** the system was originally locked to a single benchmark's exact grid shape and block count — loading a trained model into a differently-shaped benchmark threw a hard `ValueError` (confirmed via `conv_layer`). It was rebuilt around:
  - `graph_reduction.py`: collapses a benchmark's full netlist into one node per DSP/BRAM block + one aggregate "fabric" node.
  - A 2-layer GCN feature extractor + CNN over the occupancy grid, feeding a **shared, oversized** `Dict`/`Discrete` observation/action space sized from a benchmark "universe" superset, with per-benchmark action masking.
  - This is what makes cross-benchmark training and zero-shot evaluation possible at all — without it, "generalization" isn't a question the system could even attempt.

---

## 5. Results, in chronological/logical order

### 5.1 Single-benchmark RL (old architecture, before the GNN rebuild)

- **diffeq2 full run** (6000 episodes): ADP **-69.4%** vs. traditional baseline (3.26x better). Reward still climbing at episode 6000, no clear plateau.
- **Zero-shot transfer, diffeq2(trained)→diffeq1(never trained), one deterministic episode:** beat the traditional baseline by **49.9% on ADP**, beat the best of 20 random-valid-placement trials by **31.7% on ADP** — and this is the one rigorous head-to-head data point against the literature: recomputed on **GOLDS' own Area×Delay (2-term) metric**, it beat GOLDS' published GA result on diffeq1, with **zero training on diffeq1 at all**.
- **diffeq1: scratch vs. fine-tuned from diffeq2** (genuine transfer — diffeq1 was *not* in diffeq2's training), 6000 episodes each, identical hyperparameters, separate VTR caches:
  - Fine-tuned dominated typical reward almost the entire run (0.64 vs -0.83 in the first 200 episodes; 0.785 vs 0.736 in the last 200) and finished **2.4x faster** in wall-clock (828s vs 1969s).
  - **But scratch found the better final layout**: best reward 0.851 (ADP -57.3%) vs. fine-tuned's 0.815 (ADP -55.8%).
  - Root cause, confirmed directly via cache-hit-rate analysis: fine-tuned's policy entropy collapsed fast (2.00→0.79 vs scratch's 4.63→2.42), so only 31.5% of its episodes were genuinely novel VTR evaluations (vs. 94.4% for scratch) — it converged to a good region and stopped exploring, capping its ceiling.

### 5.2 GNN rebuild and joint multi-benchmark training

- One policy trained jointly across **11 "light-tier" benchmarks** simultaneously (fifo, ch_intrinsics, spree, boundtop, mmc_core, diffeq1, diffeq2, raygentop, mkSMAdapter4B, or1200, mkPktMerge), 13,000 episodes, ~10.7 hours, seed 42.
- **35.7% aggregate ADP reduction** vs. traditional baselines across those 11 benchmarks (Area×Delay×Power, the 3-term metric — not directly comparable to GOLDS' 2-term number; "comparable ballpark," not a verified head-to-head win — see §6).
- 3 "heavy-tier" benchmarks (softmax, reduction_layer, robot_rl) were held out of training entirely, used only for zero-shot eval. The canvas/graph-size caps were computed from the union of all 14 benchmarks specifically so the trained checkpoint stays loadable against the held-out ones.

### 5.3 Zero-shot generalization tiers (the actual point of the GNN rebuild)

| Benchmark | Relationship to training | Result |
|---|---|---|
| softmax | held out, in canvas-sizing universe | **+5.9% ADP** |
| reduction_layer | held out | **+21.7% ADP** |
| robot_rl | held out, largest (1006 CLB) | zero-shot episode hit an eval-script timeout; *separately* confirmed the traditional baseline itself is correct and fast (90s VTR run, bit-for-bit reproducible) — **the zero-shot RL number itself was never successfully re-obtained after the timeout was identified; still an open gap** |
| arm_core | never referenced anywhere in this project before this test | **+12.2% ADP** |
| custom_macbuf | hand-written 3-DSP/2-BRAM design, created specifically for this test, did not exist before it | **+61.2% ADP** — strongest result, and the cleanest evidence that this isn't just memorizing the training set's distribution |
| custom_5ch_mac | hand-written 5-DSP/5-BRAM design (upper bound of both resource types), created specifically for this test, did not exist before it | **+18.0% ADP** (area -15.3%, power -4.1%, delay essentially flat/+1.0% worse) |
| custom_3dsp_8bram | hand-written 3-DSP/8-BRAM design, created to probe a resource combination (3 DSP *and* 8 BRAM simultaneously) the 11-benchmark training pool never saw together | **-3.8% ADP.** Worse than baseline on every term (area +5.1%, delay +3.8%, power -4.9% i.e. the only term that improved). |
| custom_8dsp_3bram | hand-written 8-DSP/3-BRAM design — the inverse resource ratio of the above, also absent from the training pool, also at the architecture's DSP ceiling | **-10.25% ADP.** Area +11.3% (the dominant regression), delay -0.8%, power ~flat. Reward came out slightly *positive* (+0.0099) despite ADP being clearly worse — reward is a log-ratio **sum**, ADP a **product**; a few small per-term wins can offset one large loss in the sum without doing so in the product. |

The wide spread (-10.25% to +61.2%) across 7-8 real data points is itself a finding worth noting honestly: generalization clearly happens, often very well, but it is not universal — custom_3dsp_8bram and custom_8dsp_3bram are two confirmed failure cases — and the variance is large. The working hypothesis, now strengthened by a second failure in the *inverse* resource direction (8 DSP/3 BRAM, not just 3 DSP/8 BRAM), is that failures specifically cluster around **DSP+BRAM joint combinations absent from the training pool**, regardless of which resource dominates — not benchmark size, not either resource count in isolation. Two consistent failures in opposite-imbalance directions is meaningfully stronger evidence than one, though still not a rigorously tested hypothesis (no controlled sweep, just two hand-picked probe points). Both benchmarks have since been added to the next training pool specifically to close this gap (§ trained_benchmarks.txt, now 14 benchmarks).

### 5.4 A real bug, found and fixed mid-project

`_cache_key()` (the VTR-result cache lookup) used `sorted(self._placed_dsps)`/`sorted(self._placed_brams)` — but placement **order is semantically meaningful**: position `j` is pinned to the `j`-th highest-net-count instance (via a `zip(block_names, coords)` step that generates the VPR constraint file), so two placements using the same grid cells in a different order are genuinely different constraint files, not equivalent layouts. Sorting collapsed them into one cache key, causing **false cache hits**: the second episode silently got the *first* episode's (different) VTR result instead of a fresh evaluation.

- Present since the **very first commit** of the project (confirmed via `git log`), so it affected every experiment run before the fix, not just the GNN-era ones.
- Quantified on the 13,000-episode joint training run: **410 of 13,003 episodes (≈3.15%)** received a wrong cached reward, concentrated entirely in the 3 of 11 benchmarks with ≥2 same-type blocks (diffeq1: 150, spree: 137, diffeq2: 123) — benchmarks with ≤1 block per type have no possible order permutation, hence zero affected episodes.
- Fixed by using placement order directly in the cache key instead of sorting. Verified post-fix on a live run: 178 groups had multiple distinct orderings, and **100% of them produced distinct metrics** (vs. 0% before) — the fix works as intended.

### 5.5 Post-fix re-validation experiments

- **lightweight_cipher** (12×12, 0 DSP, 3 BRAM — genuinely held out of the 11-benchmark joint training): fine-tuned-from-multi-model vs scratch, both 6000 episodes. Post-fix fine-tuned best reward 0.6019 (ADP -45.22%) vs. its own pre-fix result of 0.5786 (ADP -43.93%) — **the fix didn't just remove noise, it improved the result**, meaning the bug was injecting genuinely wrong reward signal, not just harmless duplication. (Scratch was only run pre-fix for this benchmark, at reward 0.6048/ADP -45.38% — not yet re-validated post-fix; minor known gap.)
- **diffeq1 post-fix re-run** — ⚠️ **methodologically distinct from every other fine-tuning result above, and must not be conflated with them:** diffeq1 *was already inside* the 11-benchmark joint training set. So "fine-tuning from the multi-benchmark model on diffeq1" here means **continuing training on a benchmark the policy already partially knows**, not transfer to something unseen (unlike the diffeq2→diffeq1 experiment in §5.1, or the lightweight_cipher/softmax/reduction_layer/arm_core/custom_macbuf results in §5.3/5.5, which are genuine transfer).
  - Under this setup, fine-tuned won on **every axis**: best reward 0.890 vs scratch's 0.834; ADP **-58.95%** vs scratch's **-56.55%**; typical reward dominant throughout (0.81 final vs scratch's 0.65, still climbing).
  - The more interesting reframing (prompted by catching that raw wall-clock-for-a-fixed-budget wasn't the right comparison): fine-tuned **reaches its convergence plateau by episode ~1,600-1,800** (≈7 minutes) and barely moves after that, while scratch **never plateaus within the full 6,000-episode budget** — its reward is still climbing at the very last window. That's a sharper way to state the amortization benefit: not "X times faster for a fixed budget," but "reaches the same destination in an order of magnitude fewer episodes."
  - **Why this result can't be used as evidence for the cross-benchmark generalization claim** (§3, question 2): it's evidence for a different, narrower, still-useful claim — that a multi-task policy can keep improving sample-efficiently on a benchmark it already partially learned, without the fine-tuning ceiling tradeoff seen in genuine-transfer cases. Don't cite this number where a "transfer to unseen design" claim is needed; use §5.3 or the lightweight_cipher result instead.

### 5.6 Expanded retraining run closes the generalization gap found in §5.3

Directly motivated by the two negative zero-shot results in §5.3 (`custom_3dsp_8bram` -3.8%, `custom_8dsp_3bram` -10.25%), the training pool was expanded from 11 to 13 benchmarks — adding both gap-probe benchmarks (`custom_5ch_mac` was also added, for a total of 3 new custom designs) — and retrained from scratch (`multi13_ambitious`, seed 42, 30,003 episodes, ~25.9 hours wall-clock).

**Final result: 42.73% aggregate ADP reduction** (up from the original 11-benchmark run's 35.7%), with **12 of 13 benchmarks finishing positive**:

| Benchmark | Best reward | ADP reduction |
|---|---|---|
| diffeq2 | 1.176 | **+69.1%** |
| custom_5ch_mac | 1.133 | **+67.8%** |
| fifo | 0.921 | **+60.2%** |
| diffeq1 | 0.854 | **+57.4%** |
| boundtop | 0.743 | **+52.4%** |
| custom_8dsp_3bram | 0.734 | **+52.0%** |
| custom_3dsp_8bram | 0.727 | **+51.6%** |
| or1200 | 0.500 | **+39.4%** |
| mkPktMerge | 0.500 | **+39.3%** |
| spree | 0.489 | **+38.7%** |
| mkSMAdapter4B | 0.469 | **+37.5%** |
| mmc_core | 0.288 | **+25.0%** |
| raygentop | -0.020 | **-2.0%** |

**The key finding: both gap-probe benchmarks flipped from negative zero-shot to strongly positive once trained on** (`custom_3dsp_8bram`: -3.8% → +51.6%; `custom_8dsp_3bram`: -10.25% → +52.0%) — direct, clean confirmation that the earlier failures were genuinely about the training pool never having seen that DSP+BRAM joint combination, not some other deeper limitation of the architecture. This is a satisfying close of the loop from §5.3's negative results through §7's gap-closing recommendation.

`raygentop` remains the one laggard (also the slowest-converging benchmark in the original 11-benchmark run), improved from roughly -3% mid-run to -2.0% final but still not crossing zero — worth a closer look in a future session (longer episode budget specifically for this benchmark, or checking whether its 6-DSP/1-BRAM profile is itself a thinner-represented combination in the pool).

⚠️ Same methodological caveat as §5.5 applies to reading these per-benchmark numbers as "zero-shot" or "transfer" results — they are not; every benchmark listed here was directly trained on in this run. The genuine zero-shot/transfer evidence remains §5.3, updated below in §5.7.

### 5.7 Re-testing genuine zero-shot generalization after the expanded retraining — a real regression

The natural follow-up to §5.6: does the new `multi13_ambitious_seed42.zip` checkpoint still generalize well to benchmarks genuinely outside its training pool, or did fixing the two gap-probe failures cost something? Re-ran the zero-shot eval on `softmax`, `reduction_layer`, and `arm_core` (the three held-out benchmarks from §5.3 that are still outside this expanded 13-benchmark pool).

**`robot_rl` could not be re-tested at all** — the training launch's `--universe_benchmarks` list for this run omitted `robot_rl` (it used `mkDelayWorker32B`, `lightweight_cipher`, `custom_macbuf` instead as the "extra" universe padding), so the model's canvas was sized at `MAX_NODES=44`, below `robot_rl`'s required 67. This is a configuration oversight, not a model limitation — future training launches should always carry `robot_rl` (or the largest benchmark in the full benchmark set) in `--universe_benchmarks` regardless of which benchmarks are actually trained on, specifically so the checkpoint stays testable against it.

| Benchmark | Original 11-bench model (§5.3) | New 13-bench model | Change |
|---|---|---|---|
| softmax | +5.9% | **-1.27%** | flipped to negative |
| reduction_layer | +21.7% | **+6.21%** | lost most of its margin |
| arm_core | +12.2% | **+9.15%** | smaller decline |

**All three declined, in the same direction** — that's a meaningfully different signal than a mixed bag would be; with only single-seed runs throughout this project, a uniform decline across three independent benchmarks is much less likely to be pure noise than if results had gone in different directions. This is a genuine, currently-unexplained regression: **the expanded pool's in-pool gains (35.7%→42.73% aggregate, both gap-probe benchmarks fixed) came at a real cost to generalization on benchmarks still outside the pool.**

Plausible explanations, none yet tested:
- **Fixed model capacity, more pool diversity to fit.** The GNN feature extractor's size didn't change between the 11- and 13-benchmark runs. Packing in 3 more benchmarks (2 of them deliberately extreme DSP/BRAM combinations) may have traded general-purpose representation capacity for more in-pool-specific fitting — a capacity/diversity tradeoff that hadn't been measured before this comparison.
- **The 2 added "extreme" benchmarks may have pulled learned representations toward unusual joint regions**, at the expense of the smoother interpolation behavior that supported generalization to softmax/reduction_layer/arm_core in the original run.
- **Single-seed variance** — can't be fully ruled out despite the consistent-direction argument above; this needs a multi-seed re-run to actually confirm as a real effect rather than confirm-by-plausibility.

This result directly updates the Question 2 assessment in §6 below — read that section's revision before citing either the §5.6 or §5.7 numbers in isolation.

### 5.8 Final settled comparison — old (11-benchmark) vs. new (13-benchmark) model, all fronts

**Experimental work on this comparison stopped here** (deliberate decision, 2026-06-19) to move to paper-writing. Three more zero-shot probes were run after §5.7 to chase the regression's cause (a small hand-written benchmark, a large hand-written benchmark, and a real large lopsided benchmark — `mkDelayWorker32B`), and the pattern got *less* explicable with each one, not more — see the full 6-point table below. The honest call: this is a 6-point sample under a single training seed per model, and no further single-example probing was going to resolve it. A real answer needs either multi-seed training or a much larger held-out set — both bigger asks than another Verilog file, and out of scope for now. This section is the final word on the comparison as it stands.

**In-pool performance** (benchmarks each model was actually trained on — not comparable to each other's pools, see caveat below):

| | Old model | New model |
|---|---|---|
| Training pool | 11 benchmarks | 13 benchmarks |
| Episodes | 13,000 | 30,003 |
| Wall-clock | ~10.7 hrs | ~25.9 hrs |
| Aggregate ADP reduction | 35.7% | **42.73%** |

⚠️ Caveat: the pools are not a clean superset — the new pool *dropped* `ch_intrinsics` while adding 3 new custom benchmarks, and the new model trained *after* the §5.4 cache-key bug fix while the old model trained before it (~3.15% of its episodes had corrupted reward). Both confounds mean "new model's in-pool number is better" can't be cleanly attributed to "bigger pool is better" alone.

**Zero-shot generalization — genuine head-to-head, neither model trained on these benchmarks:**

| Benchmark | Old model | New model | Winner | Margin |
|---|---|---|---|---|
| mkDelayWorker32B (real, 48×48, 0 DSP/43 BRAM) | +56.75% | +57.37% | New | 0.6pp |
| custom_2dsp_4bram (hand-written, 12×12, 2/4) | +24.60% | +28.67% | New | 4.1pp |
| custom_4dsp_6bram_big (hand-written, 30×30, 4/6) | +7.89% | +9.06% | New | 1.2pp |
| arm_core (real, 35×35, 0/24) | +12.2% | +9.15% | Old | 3.1pp |
| softmax (real, 41×41, 8/0) | +5.9% | -1.27% | Old | 7.2pp |
| reduction_layer (real, 42×42, 0/32) | +21.7% | +6.21% | Old | 15.5pp |

**3-3 split on win count. Old model wins by larger average margin: 21.5% mean ADP reduction vs. new model's 18.2%.** No single hypothesis tried (hand-written-vs-real, resource-type lopsidedness, grid scale) cleanly separates the wins from the losses — `mkDelayWorker32B` in particular is real, lopsided, and the largest grid in the set, which should have been the strongest case *for* the old model under every hypothesis tested, and instead favored the new model. Treat this as **benchmark-specific/idiosyncratic variation under single-seed training**, not a characterized structural property, until a multi-seed study says otherwise.

**Capability — benchmarks the new model structurally cannot be evaluated on at all** (not a performance gap, a hard `ValueError` on load):

| Benchmark | Nodes required | Old model max (67) | New model max (44) |
|---|---|---|---|
| robot_rl | 67 | ✓ | ✗ |
| stereovision1 | 45 | ✓ | ✗ |

Caused by a configuration mistake, not a fundamental limitation: the `multi13_ambitious` launch's `--universe_benchmarks` omitted `robot_rl`, so the canvas was sized smaller than necessary. Trivially fixable in the next training launch (§7, gap #10), but as the checkpoint exists today, the new model is strictly less capable in this one respect.

**The two benchmarks that motivated the whole retraining (not a fair zero-shot comparison — new model trained directly on these):**

| Benchmark | Old model (zero-shot) | New model (trained on it) |
|---|---|---|
| custom_3dsp_8bram | -3.8% (failure) | +51.6% |
| custom_8dsp_3bram | -10.25% (failure) | +52.0% |

**Net assessment:** neither model is a strict upgrade. New model wins decisively on in-pool performance and fixes the two failures it was built to fix; old model is slightly ahead on average zero-shot generalization margin and retains a real capability the new model lost. Which one is "the" model for the paper depends on which question is being argued — Question 1 (beats traditional tooling) favors citing the new model's in-pool numbers; Question 2 (generalizes to unseen designs) is better served by the old model's numbers, which are both slightly stronger on average and free of the robot_rl/stereovision1 blind spot.

---

## 6. Honest assessment: what's actually proven vs. promising

### Question 1 (RL vs. GA head-to-head): **not proven, one real data point exists**

- The only true apples-to-apples comparison is the diffeq2→diffeq1 zero-shot result recomputed on GOLDS' exact Area×Delay (2-term) metric (§5.1) — and it's one benchmark, one seed, likely against GOLDS' best-of-population result (unconfirmed whether GOLDS reports single-run or averaged-over-trials).
- Every other number in this project (35.7% aggregate, the various 50-69% single-benchmark reductions) uses this project's own 3-term Area×Delay×Power metric, which is **not** the metric either prior paper reports. "Comparable ballpark" ≠ verified win — this exact conflation was caught and corrected mid-project once already; don't let it recur in paper drafts.
- **Unresolved and load-bearing:** the traditional-FPGA baseline architecture assumptions have never been matched against GOLDS' setup. If the baselines differ, percentage-improvement comparisons aren't comparing the same absolute thing on either side. This is the single biggest rigor gap blocking Question 1.
- No variance bars anywhere (single seed throughout); GA/NSGA-II are themselves stochastic.

### Question 2 (generalization speeding up fine-tuning): **directionally supported on fine-tuning speed; generalization itself is less robust than it looked**

- Two genuine-transfer experiments (diffeq2→diffeq1 pre-fix, and multi-model→lightweight_cipher) both show the same pattern: fine-tuned dominates typical reward and converges faster, but **scratch finds a better final ceiling** in both cases. The mechanism (entropy collapse → less exploration → capped ceiling) is consistent and was verified programmatically (cache-hit-rate), not just inferred.
- So the safely provable claim is **"fine-tuning reaches a good design faster,"** not **"fine-tuning reaches the best design."** An untested follow-up flagged twice now and never run: fine-tuning with an elevated `ent_coef` to counteract the entropy collapse and see if it closes the ceiling gap.
- Zero-shot generalization itself (§5.3) is real but the variance across benchmarks (-10.25% to +61.2%) is large and unexplained, and the largest held-out benchmark's (robot_rl) actual zero-shot ADP number is still unresolved against any checkpoint.
- The diffeq1 post-fix result (§5.5) looks like a clean win for fine-tuning on every axis, but per the warning above, it's not a fair test of this question — diffeq1 was already in the pretraining set.
- **New, important complication (§5.7):** expanding the training pool from 11→13 benchmarks specifically to fix two generalization failures (§5.3, §5.6) *worked* for those two benchmarks, but **reduced zero-shot performance on every other held-out benchmark re-tested** (softmax +5.9%→-1.27%, reduction_layer +21.7%→+6.21%, arm_core +12.2%→+9.15%). This means "add more training diversity to fix a generalization gap" is not a free action — at fixed model capacity, it appears to trade general-purpose representation for in-pool fitting. This significantly weakens confidence in the generalization claim's robustness: the policy's zero-shot behavior is evidently sensitive to exactly what's in the training pool in ways that aren't yet predictable, which is the opposite of what a strong generalization claim needs. Any paper draft citing the §5.3 zero-shot numbers as evidence for Question 2 must now also address this regression, not cite §5.3 in isolation as if it were the final word.

### Are the research questions themselves well-posed?

Yes. Both are legitimate, falsifiable, and ask something the prior evolutionary-search papers structurally cannot — that's a sound basis for a paper. The gap is entirely in rigor of evidence, not in the validity of the questions.

---

## 7. Concrete gaps to close before this is publication-ready

1. **Match the traditional baseline architecture to GOLDS' exact setup** — currently unverified, and blocks any rigorous Question-1 claim.
2. **Multi-seed runs** for every headline number — currently single-seed throughout, on both the RL side and (implicitly) when comparing to GA/NSGA-II's own stochastic results.
3. **Resolve robot_rl's zero-shot ADP number** — baseline confirmed correct/fast, but the actual RL episode result was never successfully re-obtained after the original timeout.
4. **Test fine-tuning with elevated `ent_coef`** to see whether the speed-vs-ceiling tradeoff (§5.1, §5.5) is fundamental or just an artifact of using the same entropy coefficient for both conditions.
5. **Broaden benchmark coverage to match GOLDS/NSGA-II's own benchmark list** (sha, ch_intrinsics✓, boundtop✓, spree✓, raygentop✓, the Koios DL suite) for more directly comparable points — several are already in the 11-benchmark training set, so this is largely re-scoring, not new training.
6. **Characterize the generalization-variance** across held-out benchmarks (§5.3) instead of just reporting the spread.
7. **Co-optimize CLB LUT size / custom arithmetic-block sizing**, matching NSGA-II's action space — currently this project only covers DSP/BRAM placement + aspect ratio.
8. **Re-validate lightweight_cipher's scratch run post-fix** for full consistency with its already-re-validated fine-tuned counterpart.
9. **Investigate the §5.7 generalization regression** — does adding training-pool diversity fundamentally trade off against zero-shot generalization at this model's fixed capacity, or is this specific to the 2 "extreme" custom benchmarks added? Test by retraining with a larger feature extractor (more capacity) on the same 13-benchmark pool, and/or retraining with only `custom_5ch_mac` added (the one new benchmark that didn't show a pre-fix generalization failure) to isolate whether the extreme benchmarks specifically are the cause.
10. **Re-run training with `robot_rl` correctly included in `--universe_benchmarks`** — the `multi13_ambitious` launch omitted it, making the resulting checkpoint untestable against robot_rl entirely. Trivial to fix, but blocks closing out gap #3 above against any future checkpoint until corrected.

---

## 8. Possible venues (open question, not yet researched)

The two predecessor papers were published at **GLSVLSI** (GOLDS) and **VLSID** (the NSGA-II follow-up) — VLSI-design/hardware-security-adjacent venues, not general ML venues. A natural starting point would be the same or sibling venues (e.g. ICCAD, DATE, ASP-DAC, HOST for the security angle specifically), but this hasn't been researched yet — worth a dedicated literature/venue-fit session before committing to a target, considering both the hardware-security angle (logic masking, IP protection) and the ML-for-EDA angle (learned policies for physical design) separately, since they may point to different venues.

---

## 9. Where to look for more detail (if working in the repo)

- `CLAUDE.md` — engineering/architecture reference (modules, CLI commands, file layout).
- `EXPERIMENTS.md` — detailed log of the pre-GNN single-benchmark experiments (§5.1 here, with full hyperparameter tables).
- This project's Claude Code memory store (`memory/MEMORY.md` and linked files) — session-level detail behind every result summarized here, including exact commands used.
- W&B project `fpga-placement-gnn` — all training curves, the 7 PPO stability diagnostics, per-benchmark episode/best-reward tracking.
