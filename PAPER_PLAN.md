# ASP-DAC 2027 — Paper Implementation Plan
# RL-based eFPGA Tile Placement for Logic Masking
#
# This document is the single source of truth for all tasks between now and July 18.
# Feed this to your local Claude agent at the start of every session.
# Update status markers as tasks complete: [ ] → [x]
#
# Key dates (confirmed against official CFP, 2026-06-24):
#   Abstract deadline       : July 11, 2026 (5 PM AOE)
#   PDF deadline            : July 18, 2026 (5 PM AOE)
#   Accepted IDs announced  : Sept 1, 2026
#   Notification of accept. : Sept 4, 2026
#   Updated manuscript (copyright notice + DOI/ISBN) : within 2 weeks of acceptance
#                             notification — i.e. by ~Sept 18, 2026 if notified Sept 4
#   Final version deadline  : Oct 27, 2026 (5 PM AOE)
#
# DECISION (2026-06-25): the paper uses the 11-BENCHMARK model as its sole
# headline model, not the 13-benchmark one. The 13-bench results below are
# kept as historical log entries (what was tried, what was learned) but are
# NOT used in aspdac2027_draft.tex anymore. Reason: the 11-bench policy's
# two-seed zero-shot eval (31.30% vs 30.32% avg, no sign flips on 6 held-out
# benchmarks) is far more seed-stable than the 13-bench policy's was (sign
# flips on 2/6 benchmarks) — a cleaner, more defensible story for the paper.
# In-pool seed7 numbers for the 11-bench model: 23.11% pooled aggregate vs
# seed42's 35.73%, driven almost entirely by or1200 — see
# Section~\ref{sec:results-tradeoff} in the draft for the write-up.
#   Today                   : June 24, 2026
#   Days remaining (to PDF) : 24
#
# CRITICAL — confirmed from official Author's Guide (2026-06-24):
#   Title, author names, author list, and author order are LOCKED at submission —
#   NO CHANGES PERMITTED AFTER SUBMISSION, full stop. This means Open Question #4
#   (author order) and the paper title must be finalized BEFORE the July 11 abstract
#   submission, not decided later. Treat this as a Phase 0 blocker, not a Phase 5 item.
#   Valid ORCID + valid email required for EVERY author AT THE TIME OF SUBMISSION
#   (i.e. by July 11), not just "before EasyChair" loosely — Phase 0.5 is more urgent
#   than originally scoped.
#   (Refined 2026-06-24 from official Preparation Guide): author-order lock is not
#   absolute — TPC CAN approve an exception post-submission — but treat it as locked
#   for planning purposes; don't rely on getting an exception.
#
# TEMPLATE (confirmed from official Preparation Guide, 2026-06-24):
#   Official ACM Primary Article Template, LaTeX file = sample-sigconf.tex
#   (confirms our acmart `sigconf` choice). Template support: acmtexsupport@aptaracorp.com
#   Initial-submission PDF must leave the title-block blank-space area for
#   authors/affiliations untouched (added back only in the final/camera-ready version) —
#   the default sample-sigconf.tex layout already reserves this, just don't remove it.
#   ACM rights text + bibliographic strip (bottom-left, page 1) is sent by ACM AFTER
#   acceptance for the final manuscript — leave that corner area as the template
#   default, don't preemptively edit it.
#
# PAPER TYPE RISK (new info): authors do NOT choose "regular" vs "short" — TPC
#   decides after acceptance. Short papers are capped at 4 pages at camera-ready,
#   which could force cuts to a paper written and accepted at 6 pages. Not actionable
#   now, but keep prose modular (Results/Related Work especially) so cutting later
#   doesn't require a rewrite. Camera-ready DOES allow up to 2 extra pages (8 max)
#   for regular papers with extra payment, if more room is needed later.
#
# NOT APPLICABLE: ACM's policy on research involving human participants/subjects
#   (mentioned in the guide) doesn't apply — this work has no human-subject data.

---

## CONTEXT (read this first, every session)

The paper presents an RL-based framework (MaskablePPO + GCN feature extractor) for
optimizing DSP/BRAM hard-block tile placement on custom eFPGA fabrics for IP protection
via logic masking. The primary novelty is cross-benchmark zero-shot generalization —
a single trained policy transfers to unseen benchmarks, which GA/NSGA-II cannot do
structurally.

Two trained models exist (status updated 2026-06-24, see Daily Log):
- 11-benchmark model  : 13,000 episodes, ~10.7 hrs, 35.7% avg ADP reduction (in-pool)
                        Single seed (42) zero-shot result on 6 held-out benchmarks:
                        +31.30% avg, all 6 positive — best result seen so far, but
                        UNCONFIRMED at n=1 seed. A second seed (seed 7, same config)
                        is running now to check if this holds up.
- 13-benchmark model  : 30,003 episodes, ~25.9 hrs, 42.73% avg ADP reduction (in-pool)
                        Two valid seeds (42, 7) on the SAME 6 held-out benchmarks show
                        enormous seed variance: seed42 avg +27.50%, seed7 avg -18.60%,
                        with sign flips on 2/6 benchmarks and one benchmark
                        (mkDelayWorker32B) swinging +57% to -122% between seeds.
                        DO NOT claim either model is "stronger on zero-shot" until the
                        11-benchmark model also has ≥2 seeds — right now that comparison
                        is 1 seed vs 2 seeds and not apples-to-apples.
                        A third seed (123) was attempted but invalidated — see Daily Log,
                        it trained entirely on a broken VTR toolchain in this environment
                        and was deleted, not a real data point.

Prior work to beat:
- GOLDS (GLSVLSI 2024)    : GA, 2-term Area×Delay, 35% ADP gain on diffeq1
- NSGA-II (VLSID 2025)    : NSGA-II, 3-term, 44.6% improvement
- ARIANNA (TODAES 2025)   : branch-and-bound heuristic, security angle, no RL, no transfer

Target venue : ASP-DAC 2027, Track 9 (Physical Design and Timing Analysis),
               Sub-track 9.1 (Floorplanning, partitioning, placement and routing optimization)
Format       : ACM sigconf, double-blind, max 6 pages (incl. abstract/figs/tables)
               + 1 extra page of references (does not count toward the 6)
Submission   : EasyChair https://easychair.org/conferences/?conf=aspdac2027

CONFIRMED FROM OFFICIAL CFP (2026-06-24) — items not previously in this plan:
- Double-blind scope is strict: no author name/affiliation ANYWHERE, including
  abstract, references, AND bibliographic citation text itself — so "prior work [X]"
  phrasing (Phase 4.3) must also avoid any citation wording that would reveal which
  cited work is the authors' own (e.g. don't write "in our earlier conference paper [X]").
- No double/parallel submission of similar work to ANY other conference, symposium,
  or journal — this directly resolves Open Question #5: HOST 2027 parallel submission
  is NOT allowed under the general policy, not just an ASP-DAC-specific rule. Treat as
  resolved unless Madhav has specific information otherwise.
- Open-source code release is encouraged (GitHub or similar) but the repo/identity must
  stay anonymized until after double-blind review — do not link the actual repo in the
  submitted PDF; mention "code will be released upon publication" instead if desired.
- NEW (ACM policy, effective Jan 1 2026): ACM is fully Open Access now. Either an
  author's institution participates in "ACM Open" (no charge), or an Article
  Processing Charge (APC) applies, unless a financial waiver is granted. ACTION ITEM:
  check with Madhav/institution whether ACM Open membership applies, otherwise budget
  for the APC or prepare a waiver application — see acm.org APC waiver policy page.
- Each accepted paper requires at least one author to register at the (non-student)
  speaker rate — budget/logistics item for after acceptance (Sept 2026).
- ACM/IEEE can pull a paper from the Digital Library/Xplore post-conference if no
  author actually presents it — logistics reminder for whoever attends.

---

## PHASE 0 — RESOLVE BEFORE WRITING ANYTHING
# These are blockers. Do these first, ideally with Madhav.
# Estimated time: 1-2 days

[x] 0.1  BASELINE CLARIFICATION — RESOLVED 2026-06-24 (confirmed directly, no
         Madhav meeting needed for this one):
         - GOLDS used the SAME exact VTR architecture template as us — confirmed.
         - GOLDS reports best-of-population only (not averaged) — confirmed. This
           makes our single deterministic zero-shot diffeq1 result a valid
           best-vs-best comparison, not just "comparable ballpark."
         - NSGA-II dropped as a rigor-matched comparison target (decision: our
           work succeeds GOLDS specifically, not the NSGA-II co-optimization paper,
           which optimizes a different/larger action space — CLB+sizing, not just
           placement). Still cited in related work for context, just not held to
           the same baseline-matching bar.
         - One small loose end: whether GOLDS' "diffeq1" is the exact same Verilog
           source as ours (standard VTR-suite version) vs. a variant — low risk,
           worth a quick confirm with Madhav if it comes up, not blocking.
         - Output: see this entry + Phase 0.3 below (golds_comparison.txt not
           written as a separate file — number is now in PAPER_PLAN.md directly)

[ ] 0.2  METRIC CLARIFICATION
         - Confirm: GOLDS uses 2-term Area×Delay; your work uses 3-term Area×Delay×Power
         - Write one sentence that will appear in the paper clarifying this distinction
         - Never directly compare your % numbers to GOLDS' % numbers without this caveat
         - Output: one sentence saved to  →  metric_caveat.txt

[x] 0.3  RETRIEVE THE DIFFEQ1 HEAD-TO-HEAD NUMBER — DONE 2026-06-24
         - Source: zero_shot_transfer_diffeq1_results.json (diffeq2-trained policy,
           zero-shot on diffeq1, one deterministic episode, zero training on diffeq1).
         - Baseline (2-term, Area×Delay): 694168 × 22.3395 ≈ 15.51M
         - RL zero-shot (2-term): 449975 × 21.5744 ≈ 9.71M
         - Reduction: 37.4% vs. GOLDS' published ~35% on the SAME 2-term metric,
           SAME architecture (0.1), SAME best-vs-best comparison basis (0.1).
         - DECISION (2026-06-24, asked directly): this is now a SECONDARY/supporting
           comparison point in the paper, not the primary one — the primary RL-vs-GA
           comparison is the in-house custom_macbuf result (Phase 2.5, 74.0% vs 68.2%,
           bigger margin and zero metric-conversion caveat). Use this diffeq1 number
           as the literature-facing head-to-head alongside it.

[ ] 0.4  CONFIRM HELD-OUT BENCHMARKS WITH MADHAV
         - Original proposed 4: softmax, reduction_layer, arm_core, custom_macbuf
         - We now have actual multi-seed results for 6: the above 4 PLUS
           lightweight_cipher and mkDelayWorker32B (data already exists, see
           phaseD_zeroshot6_seed7.json / phaseD_zeroshot6_seed42.json / phaseD_zeroshot6_multi11.json)
         - mkDelayWorker32B is the most volatile benchmark by far (+57% to -122%
           across seeds) — decide with Madhav whether to: (a) include it because
           it's the most informative about generalization limits, or (b) omit it
           because reviewers may read large swings as instability/cherry-picking risk
         - robot_rl: still omit (timeout issue, canvas sizing bug — not resolved)
         - Get Madhav's sign-off on final set (4 or 6)
         - Output: confirmed list saved to  →  heldout_benchmarks.txt

[ ] 0.5  ORCID + AUTHOR INFO (UPGRADED TO HARD BLOCKER — confirmed from Author's Guide)
         - Valid ORCID AND valid email required for every author AT THE TIME OF
           SUBMISSION (i.e. by July 11 abstract deadline) — not "before EasyChair"
           loosely, this is enforced at submission time
         - Register ORCID for yourself at https://orcid.org if not already done
         - Get Madhav's ORCID + confirm his email for the submission form
         - ALSO LOCK DOWN NOW, not later: paper title, author list, and author order —
           the Author's Guide states NONE of these can change after submission.
           Decide this with Madhav before July 11, not as a Phase 5 afterthought.
         - Output: ORCIDs/emails saved to → orcids.txt; title + author order recorded here

---

## PHASE 1 — FIGURES (do before writing prose)
# In a physical design paper, figures carry the argument.
# Write the figures first; prose explains the figures.
# Estimated time: 3-4 days

[ ] 1.1  FIGURE: System Architecture Diagram
         - Show the full pipeline end-to-end:
           Verilog netlist → graph_reduction.py → GCN feature extractor →
           MaskablePPO agent → action (tile placement) → Jinja2 XML generation →
           VTR (Yosys + VPR) → reward (ADP vs baseline)
         - Should be one clean block diagram
         - Tool: draw in matplotlib / tikz / draw.io — export as PDF or high-res PNG
         - Output: fig_system_architecture.pdf

[ ] 1.2  FIGURE: eFPGA Fabric Diagram
         - Show a tile grid with DSP/BRAM hard blocks placed on it
         - Show before (traditional island-style) vs after (RL-optimized placement)
         - Keep it simple — 2 subfigures side by side
         - Output: fig_fabric_before_after.pdf

[ ] 1.3  FIGURE: Training Reward Curves
         - Plot: typical reward vs episode number for joint multi-benchmark training
         - Show convergence behavior
         - If possible show per-benchmark reward lines (or average + shaded range)
         - Data source: W&B logs from 13-benchmark training run (multi13_ambitious)
         - Tool: matplotlib, export PDF
         - Output: fig_training_curves.pdf

[ ] 1.4  FIGURE: Fine-tuning vs Scratch Comparison
         - Plot: typical reward vs episode for (a) fine-tuned from pretrained policy
           and (b) training from scratch, on a held-out benchmark
         - Highlight: fine-tuned reaches plateau by ~1600-1800 episodes;
           scratch hasn't plateaued at 6000 episodes
         - Data source: lightweight_cipher fine-tuning experiment logs
         - Output: fig_finetuning_comparison.pdf

[ ] 1.5  FIGURE OR TABLE: Zero-Shot Generalization Results
         - Bar chart showing ADP improvement (%) on 4 held-out benchmarks
           for the 11-benchmark model (zero-shot, no fine-tuning)
         - Alternatively this can be Table 2 in the paper — decide with Madhav
         - Benchmarks: softmax (+5.9%), reduction_layer (+21.7%),
           arm_core (+12.2%), custom_macbuf (+61.2%)
         - Output: fig_zeroshot_results.pdf  OR  included in LaTeX table directly

[ ] 1.6  FIGURE: Pareto / Comparison Bar Chart (optional but strong)
         - If you can run GOLDS on the same benchmarks, show side-by-side comparison
         - If not: show your method vs random valid placement vs traditional baseline
         - This is the figure reviewers will look at first
         - Output: fig_baseline_comparison.pdf

---

## PHASE 2 — EXPERIMENTS (run in parallel with writing)
# Only the experiments that strengthen the paper before July 18.
# Do not chase perfection — scope is fixed.
# Estimated time: 3-5 days of compute + analysis

[ ] 2.1  MULTI-SEED ZERO-SHOT VALIDATION (highest priority) — PARTIALLY DONE 2026-06-24
         - Done: 13-benchmark model, seeds 42 + 7, all 6 held-out benchmarks
           (softmax, reduction_layer, lightweight_cipher, custom_macbuf,
           mkDelayWorker32B, arm_core). Result: huge variance (avg +27.50% vs -18.60%,
           sign flips on 2/6). Files: phaseD_zeroshot6_seed42.json, phaseD_zeroshot6_seed7.json.
         - Done: 11-benchmark model, seed 42 only, same 6 benchmarks: avg +31.30%,
           all positive. File: phaseD_zeroshot6_multi11.json.
         - A 3rd seed for the 13-bench model (123) was attempted but invalidated —
           trained entirely on a broken VTR toolchain in this dev environment, deleted.
         - IN PROGRESS: 2nd seed (7) for the 11-benchmark model, same config as seed42,
           launched 2026-06-24, running in `ubuntu-work` distrobox
           (save path: runs/multi11_long_seed7.zip). Needed before the
           "11-bench is stronger on zero-shot" framing can be stated with confidence.
         - Still TODO: 3rd seed for whichever model(s) end up in the paper, time permitting
         - Output: zeroshot_multiseed_results.csv (still needs to be compiled from the
           phaseD_zeroshot6_*.json files above)

[ ] 2.2  RETRIEVE/RECOMPUTE GOLDS COMPARISON (from existing logs)
         - Do not run new experiments for this — it should already exist from §5.1
         - Find the diffeq1 zero-shot result recomputed on 2-term Area×Delay metric
         - Confirm the exact number and save it
         - Output: append to golds_comparison.txt

[ ] 2.3  RAYGENTOP INVESTIGATION (low priority, time permitting)
         - raygentop is the one negative result in the 13-benchmark in-pool table (-2.0%)
         - At minimum: write one sentence explaining why (slowest converging, 6-DSP/1-BRAM
           profile, may need longer budget)
         - Optionally: run 500 more episodes to see if it crosses zero
         - Do not delay the paper for this — one negative result in 13 is acceptable
         - Output: raygentop_note.txt (for the paper's discussion)

[ ] 2.5  NEW (2026-06-24): DIRECT GA-VS-RL HEAD-TO-HEAD ON custom_macbuf — DONE,
         CONFIRMED STRONG RESULT — candidate headline comparison for the paper.

         Traditional baseline (custom_macbuf): routing_area=488146, delay_ns=14.3481,
         power_w=0.006354  →  baseline ADP = 44503.21

         GA  (ga_agent.py, 500-gen cap, patience=50, converged via patience @ gen ~186):
           best layout: dsps=[(4,1),(1,1),(7,2)], brams=[(5,2)], aspect_ratio=1.0
           best ADP = 14164.52  →  68.17% reduction vs baseline

         RL  (11-benchmark pretrained model, fine-tuned on custom_macbuf only,
              5 parallel envs, ent_coef=0.01, loaded from multi_light_tier_model.zip):
           best layout: dsps=[(4,4),(2,2),(3,3)], brams=[(6,2)], aspect_ratio=0.9,
           grid 8x9, wirelength=3408, delay_ns=13.2385, power_w=0.004732
           routing_area=184424  →  ADP = 11553.16  →  74.06% reduction vs baseline

         RL beats GA by ~5.9 percentage points on this benchmark, same metric, same
         VTR toolchain, both run end-to-end by us (not a re-derived number from
         someone else's paper). This is a clean, fully-reproducible, in-house
         baseline comparison — strongly consider using this as the paper's primary
         head-to-head figure/table (Phase 1.6 / 3.3b), since it sidesteps the
         2-term-vs-3-term metric caveat (0.2) entirely.
         - Output: numbers above are final for this run; still need fig_baseline_comparison
           (1.6) built from this if adopted as the headline comparison.

[ ] 2.4  BASELINE ARCHITECTURE VERIFICATION (from Phase 0, if unresolved)
         - Once Madhav confirms the GOLDS baseline parameters, verify your baseline
           uses the same settings
         - Document the exact VTR architecture XML parameters used as baseline
         - Output: baseline_architecture_params.txt

---

## PHASE 3 — WRITING (in this order — do NOT write linearly)
# Write sections in this exact order. Introduction and Abstract come last.
# Every section has a target length in double-column ACM format.
# Estimated time: 8-10 days

[ ] 3.1  SECTION: METHODOLOGY (~1.5 pages)
         Write in this sub-order:
         [ ] 3.1a  Problem formulation — define placement as sequential decision process,
                   state the ADP objective equation (derive fresh — see Note below,
                   no pre-existing draft is being used)
         [ ] 3.1b  RL formulation — state, action, reward (Eq. 2), episode structure
         [ ] 3.1c  GCN feature extractor — architecture details, node features,
                   adjacency construction, output dimension, how it feeds PPO heads
         [ ] 3.1d  Universe superset + action masking — explain how one checkpoint
                   loads against any benchmark within canvas size
         [ ] 3.1e  Training setup — hyperparameters table, joint multi-benchmark
                   sampling strategy, W&B diagnostics

[ ] 3.2  SECTION: EXPERIMENTAL SETUP (~0.5 pages)
         [ ] 3.2a  Evaluation oracle — VTR pipeline description, how architecture XML
                   is generated from placement via Jinja2, what metrics are extracted
         [ ] 3.2b  Baseline description — traditional island-style FPGA, exact parameters
                   (from Phase 0.1), why this is the right comparison point
         [ ] 3.2c  Benchmark table — training benchmarks (11 or 13), held-out benchmarks,
                   grid sizes, DSP/BRAM counts for each
         [ ] 3.2d  Baselines listed — traditional VTR, GOLDS (GA), random valid placement

[ ] 3.3  SECTION: RESULTS (~2 pages)
         Write in this sub-order:
         [ ] 3.3a  In-pool performance — Table 1 (13-benchmark model results),
                   note 12/13 positive, briefly address raygentop
         [ ] 3.3b  Head-to-head vs GOLDS — the diffeq1 2-term number,
                   include the metric caveat (from Phase 0.2), keep this tight
         [ ] 3.3c  Zero-shot generalization — Table 2 or Figure 1.5,
                   explain custom_macbuf result specifically (strongest evidence),
                   include multi-seed results if Phase 2.1 is done
         [ ] 3.3d  Fine-tuning efficiency — the episode count comparison,
                   explain entropy collapse mechanism (this is a finding, not a failure)
         [ ] 3.3e  Capacity/diversity tradeoff — frame as a characterization result,
                   not a limitation; cite §5.7 findings honestly

[ ] 3.4  SECTION: BACKGROUND AND RELATED WORK (~0.75 pages)
         [ ] 3.4a  Logic masking and eFPGA security — 2-3 sentences, cite foundational
                   papers + ARIANNA
         [ ] 3.4b  Evolutionary eFPGA architecture search — GOLDS, NSGA-II,
                   their structural limitation (per-benchmark restart)
         [ ] 3.4c  RL for physical design — AlphaChip (Nature 2021), ChiPFormer,
                   RL for fixed-FPGA placement — explicitly contrast with your work
                   (fabric architecture design ≠ placement on fixed fabric)
         [ ] 3.4d  ARIANNA positioning paragraph — this is the most important paragraph
                   in related work; one crisp statement of what ARIANNA does,
                   what it cannot do (no RL, no transfer), where you pick up

[ ] 3.5  SECTION: INTRODUCTION (~0.75 pages)
         Write this AFTER sections 3.1-3.4 are drafted.
         [ ] 3.5a  Hook — eFPGA logic masking, ADP overhead problem
         [ ] 3.5b  Prior work limitation — GA/NSGA-II per-benchmark restart
         [ ] 3.5c  Your approach — one sentence
         [ ] 3.5d  Contributions — exactly 4 bullet points, written fresh from the
                   CONTEXT section above and finalized experiment numbers (no
                   pre-existing draft — see Note below)

[ ] 3.6  SECTION: CONCLUSION (~0.25 pages)
         [ ] 3.6a  Summary of contributions — 3-4 sentences
         [ ] 3.6b  Limitations — single seed, no CLB co-optimization, raygentop
         [ ] 3.6c  Future work — multi-seed validation, elevated ent_coef for fine-tuning,
                   CLB LUT size co-optimization to match NSGA-II action space

[ ] 3.7  ABSTRACT (~150 words)
         Write this LAST.
         [ ] 3.7a  Problem (1 sentence)
         [ ] 3.7b  Prior work limitation (1 sentence)
         [ ] 3.7c  Your method (1-2 sentences)
         [ ] 3.7d  Key results — 3 specific numbers (in-pool avg, zero-shot range, fine-tuning speedup)
         [ ] 3.7e  Implication (1 sentence)
         Target: exactly 150 words, not more

---

## PHASE 4 — REFERENCES
# Estimated time: 1 day

[ ] 4.1  Create references.bib locally (writing the whole paper from scratch in this
         repo, NOT Overleaf — see Note below) with these mandatory entries:
         - GOLDS (GLSVLSI 2024) — Nandi, Mishra, Rao
         - NSGA-II paper (VLSID 2025) — Bhargav, Pradyumna, Rao
         - ARIANNA (ACM TODAES 2025) — Collini et al.
         - VTR/VPR — Luu et al. (get exact citation from vtr-verilog-to-routing repo)
         - MaskablePPO — Huang & Ontañón (sb3_contrib)
         - AlphaChip — Mirhoseini et al. (Nature 2021)
         - ChiPFormer — Lai et al. (ICML 2023)
         - Yosys — Wolf et al.
         - At least 1-2 foundational logic masking / eFPGA redaction papers
         - REDACTOR (arXiv 2025) — eFPGA for DNN security (cite to show breadth of problem)

[ ] 4.2  Verify all citations are correct (title, venue, year, authors)
[ ] 4.3  Verify no self-citation written as "our prior work" — write "prior work [X]" instead
         (double-blind requirement)

---

## PHASE 5 — POLISH AND SUBMISSION
# Estimated time: 3 days

[ ] 5.1  PAGE COUNT CHECK
         - Target: exactly 6 pages of content + up to 1 page references
         - If over: cut related work first, then conclusion, then tighten results prose
         - If under: expand results analysis, add a figure

[ ] 5.2  DOUBLE-BLIND CHECKLIST (scope confirmed strict per official CFP)
         [ ] No author names anywhere — manuscript, abstract, references, AND
             bibliographic citation text itself
         [ ] No affiliations anywhere in the PDF
         [ ] No "our prior work" — use "prior work [X]", and phrase citations so they
             don't reveal which cited work is the authors' own
         [ ] No GitHub links to a personal/lab account (ok to say "code will be
             released upon publication" if you want to signal open-source intent)
         [ ] No acknowledgments section
         [ ] PDF metadata stripped — check with: pdfinfo paper.pdf
             (Author and Creator fields must be empty or generic — verified working
             locally: pdfinfo is installed and confirmed functional as of 2026-06-24)

[ ] 5.3  FORMATTING CHECKLIST
         [ ] ACM sigconf two-column format confirmed
         [ ] Compiler: pdfLaTeX
         [ ] No Asian or special fonts
         [ ] All figures are vector PDF or high-res PNG (≥300 DPI)
         [ ] Figure captions are self-contained (reader should understand figure without
             reading the body text)
         [ ] All tables have \toprule \midrule \bottomrule (booktabs style)
         [ ] Equations are numbered
         [ ] All citations appear in references and vice versa

[ ] 5.1b OPEN ACCESS / APC CHECK (new ACM policy, effective Jan 1 2026)
         - Confirm whether Madhav's or your institution participates in ACM Open
         - If not, either budget for the Article Processing Charge or apply for an
           ACM financial waiver (see acm.org APC waiver policy) — do this well before
           the Oct 27 final-version deadline, not last-minute
         - Output: confirmed funding path noted here

[ ] 5.4  MADHAV REVIEW
         - Send full draft to Madhav at least 5 days before July 18 (i.e., by July 13)
         - Give him a specific list of things to check:
           (a) baseline setup description accuracy
           (b) GOLDS comparison framing
           (c) contribution claims — are they overclaimed or underclaimed
           (d) related work completeness

[ ] 5.4b POST-ACCEPTANCE PROCEDURE (confirmed from official Author's Guide,
         2026-06-24 — only relevant after Sept 4 notification, noted now so it
         isn't missed later):
         - Complete ACM copyright transfer ASAP after acceptance notification
           (Sept 4) — ACM only issues DOI/ISBN after this, and that itself takes
           "a few days," so delaying the copyright transfer delays everything downstream
         - Once DOI/ISBN are issued: submit an UPDATED manuscript with the copyright
           notice + DOI/ISBN included, due within 2 weeks of acceptance notification
           (~Sept 18, 2026 if notified Sept 4) — this is a real, separate deadline,
           not the same as the Oct 27 final-version deadline
         - Final manuscript deadline: Oct 27, 2026
         - Missing any of these steps/deadlines can get the paper excluded from
           proceedings even after acceptance — per the Author's Guide's explicit warning

[ ] 5.5  EASYCHAIR SUBMISSION
         [ ] Create EasyChair account at https://easychair.org/conferences/?conf=aspdac2027
         [ ] Submit abstract text by July 11 (can be updated before PDF deadline)
         [ ] Upload final PDF by July 18
         [ ] Enter all author ORCIDs correctly
         [ ] Select track: Track 9 — Physical Design and Timing Analysis
             Sub-track: 9.1 Floorplanning, partitioning, placement and routing optimization

---

## DAILY LOG
# Update this every session so your local agent knows where you left off.

2026-06-24 : Plan created. Phase 0 not started. Awaiting Madhav meeting.

2026-06-24 (later same day) : Major environment + experiment session.
  - ROOT CAUSE FOUND: VTR's `yosys` binary was built inside the `ubuntu-work`
    distrobox container (newer glibc userland) and does not run on the bare RHEL8
    host at all — not a missing-library issue, a full ABI mismatch. Any VTR run
    launched outside `distrobox enter ubuntu-work -- ...` silently fails synthesis
    and returns reward=-10 / VTR failure for every fresh (non-cached) placement.
    Fix: always run training/eval through `distrobox enter ubuntu-work --`.
  - Seed 123 (3rd seed attempt for the 13-benchmark model) was launched outside
    the distrobox and trained for its full 30,000 episodes on ~100% broken VTR
    feedback. Confirmed via reward log (consistently -10.0) and ~22,000
    contaminated failure rows found in the shared per-benchmark VTR caches.
    Deleted: the model, checkpoints, logs, best-layout files, and cleaned the
    poisoned cache rows (left seed 7 / seed 42's valid cache entries untouched).
  - Recomputed real (non-broken) zero-shot results for seed 7 + seed 42 of the
    13-benchmark model, and seed 42 of the 11-benchmark model, across 6 held-out
    benchmarks (see Phase 2.1 above). Found large seed-to-seed variance on the
    13-benchmark model — this is now the single most important open question
    for the paper's generalization claim (see Open Questions #1, now answered
    in part: variance is real and large, not a hypothetical).
  - Launched a 2nd seed (7) of the 11-benchmark config to get the same multi-seed
    coverage for that model before deciding which one's zero-shot numbers go in
    the paper. Still running as of this entry.
  - Ran a controlled in-house GA baseline (500 gens, patience=50) on custom_macbuf
    and a from-pretrained RL fine-tune on the same benchmark — direct head-to-head,
    RL ahead by ~6 points (74.0% vs 68.2% ADP reduction). See Phase 2.5 (new).
  - DECIDED: writing the entire paper from scratch in this repo, no Overleaf draft
    will be reused — removed all references to a pre-existing LaTeX draft from this
    plan (Phase 3.1a, 3.5d, 4.1, Files-to-produce list all updated accordingly).
  - Got the official Author's Guide (submission policy, separate from the CFP).
    Important new constraints: title/author-list/author-order are LOCKED at
    submission with NO changes permitted afterward — moved this from a vague Phase 5
    item to a Phase 0 blocker (0.5). ORCID + email required for every author AT
    submission time (July 11), not loosely "before EasyChair." Also surfaced a real
    deadline that wasn't tracked before: updated manuscript with copyright notice +
    DOI/ISBN due within 2 weeks of acceptance notification (~Sept 18) — added as
    Phase 5.4b, separate from the Oct 27 final-version deadline.
  - Started setting up a local LaTeX environment (acmart class, TeX Live 2018 base)
    for writing the paper outside Overleaf — paused mid-setup, hit a LaTeX2e kernel
    version mismatch (current acmart needs a 2020+ kernel; this machine had a 2017
    kernel). RESOLVED later same day: user installed a proper TeX Live 2023 toolchain.
    Verified working end-to-end — acmart 2024/02/04 v2.03 compiles cleanly, including
    ACM-Reference-Format.bst. Cleaned up earlier manual package patches (now redundant/
    shadowing the proper system versions). Local LaTeX environment is ready to write in.
  - Got the official ASP-DAC 2027 CFP. Confirmed track (9.1), page limits (6+1),
    and full review timeline (notification Sept 4, final version Oct 27 2026).
    New items the CFP surfaced that weren't in this plan before: double-blind scope
    is stricter than assumed (covers citation text, not just author-name mentions);
    parallel-submission ban resolves Open Question #5 directly (no HOST 2027 overlap,
    full stop); and a new ACM Open Access / APC cost decision is now needed before the
    Oct 27 final-version deadline (Phase 5.1b, new).
  - Got the official Preparation Guide (the actual author kit). Confirmed template:
    ACM Primary Article Template, LaTeX file sample-sigconf.tex — matches our acmart
    `sigconf` setup, no changes needed there. New operational details added to the
    header note block above: title-block blank-space handling, ACM's post-acceptance
    rights-text/bib-strip insertion, the short-vs-regular paper type risk (TPC decides,
    not authors), and the 2-extra-page camera-ready allowance. Confirmed human-subjects
    policy doesn't apply to this work. Author-order lock is technically TPC-overridable
    but plan as if it's fixed at submission regardless.
  - User downloaded the real official ACM Primary Article Template (acmart-primary.zip,
    v2.18, May 2026) plus GOLDS.pdf, the NSGA-II eFPGA paper, and google_paper.pdf
    (AlphaChip) into ~/Downloads — covers most of the Phase 4 bibliography sourcing.
  - Extracted the package, generated sigconf.tex via samples.ins, verified it compiles
    clean with the official v2.18 cls (this venue's required template — ASP-DAC calls
    it "sample-sigconf.tex," same file, older naming).
  - SET UP paper/ in this repo: acmart.cls (v2.18), ACM-Reference-Format.bst,
    acm-jdslogo.png copied in locally; aspdac2027_draft.tex scaffolded with
    [sigconf,review,anonymous] options and the Phase 3 section skeleton (no content
    yet — writing from scratch per earlier decision); references.bib placeholder with
    the Phase 4 mandatory-entry list as comments. Compiles clean, pdfinfo confirms no
    author metadata leaks. Environment is fully ready — next real step is Phase 3
    content writing (Methodology first, per the plan's writing order).

---

## OPEN QUESTIONS (resolve with Madhav)

1. [PARTLY ANSWERED 2026-06-24] Does Madhav want multi-seed runs before submission?
   We now have direct evidence the answer must be yes for the zero-shot claim:
   2 seeds of the 13-benchmark model disagree in sign on 2/6 held-out benchmarks,
   with one swinging +57% to -122%. Single-seed + honest framing is not really an
   option anymore for that specific claim — we've already seen both outcomes and
   can't claim ignorance. Still open: how many seeds is "enough" given the July 18
   deadline, and whether this changes which model anchors the zero-shot table.
2. Exact GOLDS baseline VTR architecture parameters — needed to write experiments section
3. Does Madhav want CLB LUT size co-optimization included or deferred to future work?
4. Which author order? (conventionally: first author = main contributor, last = PI)
5. [RESOLVED 2026-06-24] Is parallel submission to HOST 2027 allowed? No — official
   CFP confirms ASP-DAC prohibits parallel submission of similar work to ANY other
   conference/symposium/journal, not specific to HOST. Do not submit overlapping
   content to both.
6. [NEW 2026-06-24] Include mkDelayWorker32B in the held-out set? It's the most
   volatile benchmark by far (see Phase 0.4) — informative but risky-looking.
7. [NEW 2026-06-24] Use the in-house GA-vs-RL custom_macbuf head-to-head (Phase 2.5)
   as a second baseline comparison alongside/instead of the GOLDS diffeq1 number?
   It avoids the 2-term-vs-3-term metric caveat entirely since it's our own GA run
   on our own metric, but it's only 1 benchmark.

---

## FILES TO PRODUCE (final checklist)

[ ] aspdac2027_draft.tex       — LaTeX source, written from scratch locally
                                  (decision 2026-06-24: no Overleaf draft is being
                                  used/reused — everything in this repo is fresh)
[ ] references.bib             — bibliography file
[ ] fig_system_architecture.pdf
[ ] fig_fabric_before_after.pdf
[ ] fig_training_curves.pdf
[ ] fig_finetuning_comparison.pdf
[ ] fig_zeroshot_results.pdf
[ ] baseline_setup.txt
[ ] metric_caveat.txt
[ ] golds_comparison.txt
[ ] heldout_benchmarks.txt
[ ] orcids.txt
[ ] zeroshot_multiseed_results.csv  (if Phase 2.1 completed)
[ ] aspdac2027_final.pdf       — final submission PDF
