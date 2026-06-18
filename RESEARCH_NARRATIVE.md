# RL for eFPGA Placement — Research Narrative

*Self-contained project summary, written so a fresh AI agent (with or without access to this repo) can pick up the full research context without re-deriving it. Last updated 2026-06-18.*

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

The wide spread (+5.9% to +61.2%) across only 4-5 real data points is itself a finding worth noting honestly: generalization clearly happens, sometimes very well, but the variance is large and currently uncharacterized — no analysis yet of *why* some benchmarks transfer better than others.

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

---

## 6. Honest assessment: what's actually proven vs. promising

### Question 1 (RL vs. GA head-to-head): **not proven, one real data point exists**

- The only true apples-to-apples comparison is the diffeq2→diffeq1 zero-shot result recomputed on GOLDS' exact Area×Delay (2-term) metric (§5.1) — and it's one benchmark, one seed, likely against GOLDS' best-of-population result (unconfirmed whether GOLDS reports single-run or averaged-over-trials).
- Every other number in this project (35.7% aggregate, the various 50-69% single-benchmark reductions) uses this project's own 3-term Area×Delay×Power metric, which is **not** the metric either prior paper reports. "Comparable ballpark" ≠ verified win — this exact conflation was caught and corrected mid-project once already; don't let it recur in paper drafts.
- **Unresolved and load-bearing:** the traditional-FPGA baseline architecture assumptions have never been matched against GOLDS' setup. If the baselines differ, percentage-improvement comparisons aren't comparing the same absolute thing on either side. This is the single biggest rigor gap blocking Question 1.
- No variance bars anywhere (single seed throughout); GA/NSGA-II are themselves stochastic.

### Question 2 (generalization speeding up fine-tuning): **directionally supported, with a real nuance**

- Two genuine-transfer experiments (diffeq2→diffeq1 pre-fix, and multi-model→lightweight_cipher) both show the same pattern: fine-tuned dominates typical reward and converges faster, but **scratch finds a better final ceiling** in both cases. The mechanism (entropy collapse → less exploration → capped ceiling) is consistent and was verified programmatically (cache-hit-rate), not just inferred.
- So the safely provable claim is **"fine-tuning reaches a good design faster,"** not **"fine-tuning reaches the best design."** An untested follow-up flagged twice now and never run: fine-tuning with an elevated `ent_coef` to counteract the entropy collapse and see if it closes the ceiling gap.
- Zero-shot generalization itself (§5.3) is real but the variance across benchmarks (+5.9% to +61.2%) is large and unexplained, and the largest held-out benchmark's (robot_rl) actual zero-shot ADP number is still unresolved.
- The diffeq1 post-fix result (§5.5) looks like a clean win for fine-tuning on every axis, but per the warning above, it's not a fair test of this question — diffeq1 was already in the pretraining set.

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

---

## 8. Possible venues (open question, not yet researched)

The two predecessor papers were published at **GLSVLSI** (GOLDS) and **VLSID** (the NSGA-II follow-up) — VLSI-design/hardware-security-adjacent venues, not general ML venues. A natural starting point would be the same or sibling venues (e.g. ICCAD, DATE, ASP-DAC, HOST for the security angle specifically), but this hasn't been researched yet — worth a dedicated literature/venue-fit session before committing to a target, considering both the hardware-security angle (logic masking, IP protection) and the ML-for-EDA angle (learned policies for physical design) separately, since they may point to different venues.

---

## 9. Where to look for more detail (if working in the repo)

- `CLAUDE.md` — engineering/architecture reference (modules, CLI commands, file layout).
- `EXPERIMENTS.md` — detailed log of the pre-GNN single-benchmark experiments (§5.1 here, with full hyperparameter tables).
- This project's Claude Code memory store (`memory/MEMORY.md` and linked files) — session-level detail behind every result summarized here, including exact commands used.
- W&B project `fpga-placement-gnn` — all training curves, the 7 PPO stability diagnostics, per-benchmark episode/best-reward tracking.
