"""data_export4/summary.json -- no new runs.

Per set (original six / new six), mean and median over circuits of:
  sweep_best      best of the 20 aspect ratios in the corrected sweep
  policy_own      the policy's own zero-shot result (its own shape + placement)
  g1              sweep's best shape + policy placement
  at_ar_1.0       columns / policy / random placement at aspect ratio 1.0

Per-circuit value rule: seeded quantities are averaged over seeds (and draws)
first, then summarized over circuits -- the rule the paper uses for its held-out
table. Unseeded quantities (sweep, columns) are used as is.

AR-1.0 comparison for the ORIGINAL six is not re-run: it already exists --
columns = data_export/baseline_ar_sweep_perar.csv at AR 1.0,
policy   = data_export2/e2_decomposition.csv fixed_ar_policy_blocks,
random   = data_export2/e2_decomposition.csv fixed_ar_random_blocks
(5 draws on the three fast circuits, 3 on the slow ones; see data_export2).
For the new six it is G2.
"""
import statistics as st

from common4 import *  # noqa: F401,F403

DX = REPO     # noqa: F405


def rows(rel):
    p = DX / rel
    if not p.exists():
        return []
    with open(p) as fh:
        return list(csv.DictReader(fh))          # noqa: F405


def ok(r):
    return r.get("vtr_success") == "True"


def mean_by(rs, key, val):
    d = {}
    for r in rs:
        if ok(r) and r[val] not in ("", None):
            d.setdefault(r[key], []).append(float(r[val]))
    return {k: st.mean(v) for k, v in d.items()}


def summarize(percircuit, circuits, src, note=""):
    v = [percircuit[c] for c in circuits if c in percircuit]
    missing = [c for c in circuits if c not in percircuit]
    if not v:
        return {"mean": None, "median": None, "n": 0, "source": src,
                "note": (note + " no values yet").strip()}
    return {"mean": st.mean(v), "median": st.median(v), "n": len(v), "per_circuit": {
        c: percircuit[c] for c in circuits if c in percircuit}, "source": src,
        "note": (note + (f" Missing: {missing}." if missing else "")).strip()}


# ---------------------------------------------------------------- per-circuit values
sweep = {}
for rel in ("data_export/baseline_ar_sweep_perar.csv", "data_export4/g1_sweep_new_circuits.csv"):
    for r in rows(rel):
        if ok(r):
            v = float(r["adp_reduction_pct_vs_baseline"])
            if v > sweep.get(r["circuit"], (-1e18,))[0]:
                sweep[r["circuit"]] = (v, float(r["aspect_ratio"]), rel)
sweep_best = {c: t[0] for c, t in sweep.items()}

# Original six: results_per_circuit_seed.csv has no vtr_success column; every
# held-out row there is a successful paper evaluation, so mark them as such.
# New six: f5_extra_heldout.csv carries its own vtr_success.
own = mean_by([dict(r, vtr_success="True") for r in rows("data_export/results_per_circuit_seed.csv")
               if r["split"] == "heldout"], "circuit", "adp_reduction_pct")
own.update(mean_by(rows("data_export3/f5_extra_heldout.csv"), "circuit", "adp_reduction_pct"))

g1 = mean_by(rows("data_export4/g1_sweep_shape_policy_placement.csv"), "circuit", "adp_reduction_pct")

cols_orig = {r["circuit"]: float(r["adp_reduction_pct_vs_baseline"])
             for r in rows("data_export/baseline_ar_sweep_perar.csv")
             if ok(r) and float(r["aspect_ratio"]) == 1.0 and r["circuit"] in ORIGINAL}   # noqa: F405
e2 = rows("data_export2/e2_decomposition.csv")
pol1_orig = mean_by([r for r in e2 if r["variant"] == "fixed_ar_policy_blocks"], "circuit", "adp_reduction_pct")
rnd1_orig = mean_by([r for r in e2 if r["variant"] == "fixed_ar_random_blocks"], "circuit", "adp_reduction_pct")

g2 = rows("data_export4/g2_fixed_shape_new.csv")
cols_new = mean_by([r for r in g2 if r["variant"] == "columns"], "circuit", "adp_reduction_pct")
pol1_new = mean_by([r for r in g2 if r["variant"] == "policy"], "circuit", "adp_reduction_pct")
rnd1_new = mean_by([r for r in g2 if r["variant"] == "random"], "circuit", "adp_reduction_pct")

out = {"_meta": {
    "generated_by": "data_export4/tools/summary.py",
    "rule": "per-circuit value = mean over seeds (and draws) for seeded quantities; "
            "then mean/median over the circuits of the set",
    "original": ORIGINAL, "new": NEW}}     # noqa: F405

for name, circuits, ar1 in (
        ("original", ORIGINAL, (cols_orig, pol1_orig, rnd1_orig,               # noqa: F405
                                "data_export/baseline_ar_sweep_perar.csv (AR 1.0)",
                                "data_export2/e2_decomposition.csv fixed_ar_policy_blocks",
                                "data_export2/e2_decomposition.csv fixed_ar_random_blocks")),
        ("new", NEW, (cols_new, pol1_new, rnd1_new,                            # noqa: F405
                      "data_export4/g2_fixed_shape_new.csv columns",
                      "data_export4/g2_fixed_shape_new.csv policy",
                      "data_export4/g2_fixed_shape_new.csv random"))):
    blk = {
        "sweep_best": summarize(sweep_best, circuits,
                                "data_export/baseline_ar_sweep_perar.csv + data_export4/g1_sweep_new_circuits.csv"),
        "policy_own": summarize(own, circuits,
                                "data_export/results_per_circuit_seed.csv (original) / "
                                "data_export3/f5_extra_heldout.csv (new)"),
        "g1_sweep_shape_policy_placement": summarize(
            g1, circuits, "data_export4/g1_sweep_shape_policy_placement.csv"),
        "at_ar_1.0": {"columns": summarize(ar1[0], circuits, ar1[3], "Unseeded."),
                      "policy": summarize(ar1[1], circuits, ar1[4]),
                      "random": summarize(ar1[2], circuits, ar1[5])},
    }
    per = {}
    for c in circuits:
        if c in g1 and c in sweep_best and c in own:
            per[c] = {"g1": g1[c], "sweep_best": sweep_best[c], "sweep_best_ar": sweep[c][1],
                      "policy_own": own[c],
                      "g1_beats_sweep_best": g1[c] > sweep_best[c],
                      "g1_beats_policy_own": g1[c] > own[c],
                      "g1_beats_both": g1[c] > sweep_best[c] and g1[c] > own[c]}
        else:
            per[c] = {"value": None, "note": "G1, sweep or policy result not available yet"}
    blk["per_circuit_g1_vs_sweep_and_policy"] = per
    out[name] = blk

(OUT / "summary.json").write_text(json.dumps(out, indent=1))     # noqa: F405
print("wrote summary.json")
for name in ("original", "new"):
    b = out[name]
    print(name, {k: (round(b[k]["mean"], 2) if b[k].get("mean") is not None else None)
                 for k in ("sweep_best", "policy_own", "g1_sweep_shape_policy_placement")},
          {k: (round(v["mean"], 2) if v.get("mean") is not None else None)
           for k, v in b["at_ar_1.0"].items()})
