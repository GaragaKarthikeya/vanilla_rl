"""Validity check for the E1 shuffled_netcount variant.

If every H-block of a circuit carries the same net_count_normalized value, the
permutation is a mathematical no-op and that circuit's "actions unchanged"
result is vacuous rather than evidence that the policy ignores net counts.
Writes data_export2/e1_netcount_perm_check.csv.
"""
import numpy as np

from elib import *  # noqa: F401,F403
import elib

HDR = ["circuit", "seed", "n_hblock_nodes", "n_distinct_netcounts", "netcount_min", "netcount_max",
       "n_entries_changed_by_permutation", "permutation_is_noop", "source"]

sink = CsvSink(OUT / "e1_netcount_perm_check.csv", HDR, ["circuit", "seed"])  # noqa: F405
for ci, name in enumerate(HELDOUT):   # noqa: F405
    env = elib.make_env(name)
    nf = env.reset()[0]["node_features"]
    n = elib.n_valid_nodes(nf)
    v = nf[: n - 1, 3]
    for seed in SEEDS:                # noqa: F405
        if sink.has((name, seed)):
            continue
        perm = np.random.default_rng([seed, ci]).permutation(n - 1)
        ch = int((v[perm] != v).sum())
        sink.write({"circuit": name, "seed": seed, "n_hblock_nodes": n - 1,
                    "n_distinct_netcounts": int(len(np.unique(np.round(v, 9)))),
                    "netcount_min": float(v.min()), "netcount_max": float(v.max()),
                    "n_entries_changed_by_permutation": ch, "permutation_is_noop": ch == 0,
                    "source": "FPGAEnv obs node_features[:,3] (graph_reduction.py:29,138-148); "
                              "perm = numpy.default_rng([seed, circuit_index]).permutation(n_hblocks)"})
print("wrote e1_netcount_perm_check.csv")
