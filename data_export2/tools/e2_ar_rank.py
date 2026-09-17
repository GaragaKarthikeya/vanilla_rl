"""E2 addition, no new VTR runs: where does the policy's chosen aspect ratio
rank among the 20 aspect ratios of the trimmed-baseline sweep?

Ranks come from data_export/baseline_ar_sweep.csv (1 = best ADP reduction).
This says whether the policy picked a good AR *by the sweep's own measure*; it
does not use the policy's own ADP, since the sweep holds H-block placement
fixed to the trimmed pattern.
"""
from elib import *  # noqa: F401,F403

HDR = ["circuit", "seed", "policy_aspect_ratio", "rank_of_that_ar_in_sweep", "n_ars_in_sweep",
       "adp_reduction_pct_of_that_ar_in_sweep", "sweep_best_ar", "sweep_best_adp_reduction_pct",
       "source"]

sweep = {}
with open(REPO / "data_export" / "baseline_ar_sweep.csv") as fh:   # noqa: F405
    for r in csv.DictReader(fh):                                   # noqa: F405
        if r["vtr_success"] == "True":
            sweep.setdefault(r["circuit"], []).append(
                (float(r["aspect_ratio"]), float(r["adp_reduction_pct_vs_baseline"])))

ctrl = control_rows()   # noqa: F405
sink = CsvSink(OUT / "e2_ar_rank.csv", HDR, ["circuit", "seed"])   # noqa: F405
for name in HELDOUT:        # noqa: F405
    ranked = sorted(sweep[name], key=lambda t: -t[1])   # best reduction first
    best_ar, best_v = ranked[0]
    for seed in SEEDS:      # noqa: F405
        if sink.has((name, seed)):
            continue
        ar = float(ctrl[(name, seed)]["policy_aspect_ratio"])
        hit = [i for i, (a, _) in enumerate(ranked) if abs(a - ar) < 1e-9]
        sink.write({
            "circuit": name, "seed": seed, "policy_aspect_ratio": ar,
            "rank_of_that_ar_in_sweep": hit[0] + 1 if hit else "",
            "n_ars_in_sweep": len(ranked),
            "adp_reduction_pct_of_that_ar_in_sweep": ranked[hit[0]][1] if hit else "",
            "sweep_best_ar": best_ar, "sweep_best_adp_reduction_pct": best_v,
            "source": "data_export/baseline_ar_sweep.csv (rank, 1=best) + "
                      "data_export/results_per_circuit_seed.csv (policy AR)"
                      + ("" if hit else "; policy AR not present among the sweep's successful runs"),
        })
print("wrote e2_ar_rank.csv")
