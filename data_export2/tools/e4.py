"""E4: summary statistics for the write-up. No new VTR runs.

Everything is recomputed from data_export/ CSVs; each block carries the file it
came from and the exact aggregation rule used.
"""
import statistics as st

from elib import *  # noqa: F401,F403

DX = REPO / "data_export"     # noqa: F405
PROBES = {"custom_macbuf", "lightweight_cipher"}


def rd(p):
    with open(DX / p) as fh:
        return list(csv.DictReader(fh))   # noqa: F405


def stats(vals, src, note=""):
    vals = [v for v in vals if v is not None]
    if not vals:
        return {"value": None, "source": src, "note": note or "no values"}
    return {"mean": st.mean(vals), "median": st.median(vals), "min": min(vals), "max": max(vals),
            "stdev": st.stdev(vals) if len(vals) > 1 else None,
            "n": len(vals), "source": src,
            "note": note + (" stdev null: only one value." if len(vals) == 1 else "")}


res = rd("results_per_circuit_seed.csv")
held = [r for r in res if r["split"] == "heldout"]
zs = {}
for r in held:
    zs.setdefault(r["circuit"], {})[int(r["seed"])] = float(r["adp_reduction_pct"])

out = {"_meta": {
    "generated_by": "data_export2/tools/e4.py",
    "circuits": HELDOUT,      # noqa: F405
    "seeds": SEEDS,           # noqa: F405
    "authored_probes_excluded_in_*_no_probes": sorted(PROBES),
    "note": "All ADP reductions are percent vs. the traditional VTR baseline. "
            "'across the 6 circuits' statistics are taken over one value per circuit; "
            "for seeded methods that per-circuit value is the mean over seeds 7/42/123, "
            "which is the same rule the paper uses for its held-out table.",
}}

# ---------------------------------------------------------------- per-circuit zero-shot
SRC_RES = "data_export/results_per_circuit_seed.csv (split=heldout)"
out["per_circuit_zero_shot_over_seeds"] = {
    c: stats(list(zs[c].values()), SRC_RES, "over seeds 7/42/123.") for c in HELDOUT}  # noqa: F405

# ---------------------------------------------------------------- per-method per-circuit values
percircuit = {}

percircuit["zero_shot_policy"] = ({c: st.mean(zs[c].values()) for c in HELDOUT},   # noqa: F405
                                  SRC_RES, "per circuit = mean over 3 seeds.")

sweep = {}
for r in rd("baseline_ar_sweep.csv"):
    if r["vtr_success"] == "True":
        sweep.setdefault(r["circuit"], []).append(float(r["adp_reduction_pct_vs_baseline"]))
percircuit["trimmed_baseline_ar_sweep_best"] = (
    {c: max(sweep[c]) for c in HELDOUT if c in sweep}, "data_export/baseline_ar_sweep.csv",
    "per circuit = best over the 20 aspect ratios (a 20-VTR-call, no-learning search). "
    "Not seeded. CAVEAT: see data_export2/notes.md - this sweep fixes H-block tile "
    "coordinates on the square baseline grid, so VPR stretches the other dimension; "
    "it understates a fair non-learning baseline.")

rnd = {}
for r in rd("random_floorplans.csv"):
    if r["vtr_success"] == "True":
        rnd.setdefault((r["circuit"], int(r["seed"])), []).append(float(r["adp_reduction_pct"]))
percircuit["random_best_of_20"] = (
    {c: st.mean([max(rnd[(c, s)]) for s in SEEDS if (c, s) in rnd]) for c in HELDOUT},  # noqa: F405
    "data_export/random_floorplans.csv",
    "per circuit-seed = best of the 20 random legal floorplans; per circuit = mean over seeds.")

smp = {}
for r in rd("policy_sampled.csv"):
    if r["vtr_success"] == "True":
        smp.setdefault((r["circuit"], int(r["seed"])), []).append(float(r["adp_reduction_pct"]))
percircuit["sampled_best_of_10"] = (
    {c: st.mean([max(smp[(c, s)]) for s in SEEDS if (c, s) in smp]) for c in HELDOUT},  # noqa: F405
    "data_export/policy_sampled.csv",
    "per circuit-seed = best of 10 stochastic rollouts; per circuit = mean over seeds.")

conv = {}
for r in rd("ga_convergence.csv"):
    conv.setdefault((r["circuit"], int(r["seed"])), []).append(
        (int(r["vtr_call_index"]), float(r["best_so_far_adp_reduction_pct"])))
for k in conv:
    conv[k].sort()


def ga_at(c, s, n):
    """Best-so-far at or before n VTR calls; None if the run logged nothing that early."""
    pts = [v for i, v in conv.get((c, s), []) if i <= n]
    return max(pts) if pts else None


for n in (8, 24, 96, 200):
    d = {}
    for c in HELDOUT:            # noqa: F405
        v = [ga_at(c, s, n) for s in SEEDS]      # noqa: F405
        v = [x for x in v if x is not None]
        if v:
            d[c] = st.mean(v)
    percircuit[f"ga_at_{n}_calls"] = (
        d, "data_export/ga_convergence.csv",
        f"per circuit-seed = GA best-so-far at <= {n} VTR calls (the GA logs once per "
        "generation = 8 calls); per circuit = mean over seeds.")

ga_final = {}
for r in rd("ga_runs.csv"):
    ga_final.setdefault(r["circuit"], []).append(float(r["adp_reduction_pct"]))
percircuit["ga_converged"] = (
    {c: st.mean(ga_final[c]) for c in HELDOUT if c in ga_final}, "data_export/ga_runs.csv",
    "per circuit = mean over the 3 seeds of each run's best individual at early-stop.")

out["per_circuit_values_by_method"] = {k: {"values": v[0], "source": v[1], "note": v[2]}
                                       for k, v in percircuit.items()}
out["across_6_circuits"] = {
    k: stats(list(v[0].values()), v[1], v[2]) for k, v in percircuit.items()}
out["across_4_circuits_no_probes"] = {
    k: stats([x for c, x in v[0].items() if c not in PROBES], v[1],
             "Excludes custom_macbuf and lightweight_cipher. " + v[2])
    for k, v in percircuit.items()}

# ---------------------------------------------------------------- grids and tile counts
grids = {}
missing = []
for r in held:
    c, s = r["circuit"], int(r["seed"])
    pw, ph = r["policy_grid_w"], r["policy_grid_h"]
    bw, bh = r["baseline_grid_w"], r["baseline_grid_h"]
    e = {"policy_grid_w": int(pw) if pw else None, "policy_grid_h": int(ph) if ph else None,
         "baseline_grid_w": int(bw) if bw else None, "baseline_grid_h": int(bh) if bh else None}
    e["policy_total_tiles"] = (e["policy_grid_w"] * e["policy_grid_h"]
                               if e["policy_grid_w"] and e["policy_grid_h"] else None)
    e["baseline_total_tiles"] = (e["baseline_grid_w"] * e["baseline_grid_h"]
                                 if e["baseline_grid_w"] and e["baseline_grid_h"] else None)
    if e["policy_total_tiles"] is None:
        missing.append(f"{c} seed {s}")
    grids.setdefault(c, {})[s] = e
out["grids_and_tile_counts"] = {
    "values": grids, "source": SRC_RES,
    "note": "Grid dimensions as reported by VPR including the IO ring; total tiles = w*h. "
            + ("Missing policy grids: " + ", ".join(missing) if missing
               else "A policy grid was recorded for every circuit-seed; none missing."),
}

# ---------------------------------------------------------------- sampled-rollout fractions
det = {(r["circuit"], int(r["seed"])): float(r["adp_reduction_pct"]) for r in held}
sampled = {}
for c in HELDOUT:            # noqa: F405
    for s in SEEDS:          # noqa: F405
        v = smp.get((c, s))
        if not v:
            continue
        sampled.setdefault(c, {})[s] = {
            "n_successful_samples": len(v),
            "frac_beating_baseline": sum(x > 0 for x in v) / len(v),
            "frac_beating_deterministic": sum(x > det[(c, s)] for x in v) / len(v),
            "min": min(v), "median": st.median(v), "max": max(v),
            "deterministic_rollout": det[(c, s)],
        }
out["sampled_rollouts"] = {
    "values": sampled, "source": "data_export/policy_sampled.csv + " + SRC_RES,
    "note": "10 stochastic rollouts per circuit-seed. 'beating baseline' = adp_reduction_pct > 0; "
            "'beating deterministic' = strictly greater than the deterministic rollout's reduction. "
            "Fractions are over successful VTR runs only (n_successful_samples).",
}

(OUT / "e4_summary.json").write_text(json.dumps(out, indent=1))   # noqa: F405
print("wrote e4_summary.json")
for k, v in out["across_6_circuits"].items():
    print(f"  {k:32s} mean={v.get('mean')}")
