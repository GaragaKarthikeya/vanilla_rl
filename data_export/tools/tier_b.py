"""Tier B: compile CSVs from existing logs/results only (no VTR runs)."""
import ast
import csv
import json
import re
import sqlite3
import statistics as st
from pathlib import Path

REPO = Path("/home/digital-2/workspace/rl_gnn/vanilla_rl")
GA = Path("/home/digital-2/workspace/vtr_exp")
OUT = REPO / "data_export"
TRAIN = ["fifo", "ch_intrinsics", "spree", "boundtop", "mmc_core", "diffeq1", "diffeq2",
         "raygentop", "mkSMAdapter4B", "or1200", "mkPktMerge"]
HELDOUT = ["custom_macbuf", "mkDelayWorker32B", "lightweight_cipher", "reduction_layer", "arm_core", "softmax"]
SEEDS = [7, 42, 123]
SUFFIX = {7: "_multi11_long_seed7", 42: "_multi11_long_seed42_v2", 123: "_multi11_long_seed123"}
MAX_H = 48
AR = [round(0.1 * i, 1) for i in range(1, 21)]


def rel(p):
    return str(Path(p)).replace("/home/digital-2/workspace/", "~/workspace/")


def jload(p):
    return json.loads(Path(p).read_text())


def base(name):
    return jload(REPO / f"baselines/{name}_traditional_metric.txt"), jload(REPO / f"baselines/{name}_traditional_resources.txt")


def adp(d):
    return d["routing_area"] * d["delay_ns"] * d["power_w"]


def write(name, header, rows):
    with open(OUT / name, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(["" if r.get(h) is None else r.get(h) for h in header])
    print("wrote", name, len(rows))


def cache_grid(name, dsps, brams, ar):
    key = f"dsps:{[tuple(x) for x in dsps]}|brams:{[tuple(x) for x in brams]}|ratio:{ar}"
    con = sqlite3.connect(f"file:{REPO}/runs/vtr_layout_cache_{name}.db?mode=ro", uri=True)
    row = con.execute("SELECT grid_w, grid_h, routing_area, delay_ns, power_w FROM layout_cache WHERE cache_key=?", (key,)).fetchone()
    con.close()
    return row, key


# ------------------------------------------------------------------ B1
B1_HDR = ["circuit", "split", "seed", "baseline_area_mwta", "baseline_delay_ns", "baseline_power_w", "baseline_adp",
          "policy_area_mwta", "policy_delay_ns", "policy_power_w", "policy_adp", "adp_reduction_pct",
          "policy_aspect_ratio", "policy_grid_w", "policy_grid_h", "baseline_grid_w", "baseline_grid_h",
          "how_obtained", "source"]


def b1():
    rows = []
    det_actions = {(r["circuit"], r["seed"]): r["actions"] for r in jload(OUT / "logs/det_actions_and_timing.json")}
    for name in TRAIN + HELDOUT:
        m, r = base(name)
        for s in SEEDS:
            row = {"circuit": name, "seed": s, "baseline_area_mwta": m["routing_area"], "baseline_delay_ns": m["delay_ns"],
                   "baseline_power_w": m["power_w"], "baseline_adp": adp(m),
                   "baseline_grid_w": r["fpga_size"][0] + 2, "baseline_grid_h": r["fpga_size"][1] + 2}
            src_base = f"baselines/{name}_traditional_metric.txt; baselines/{name}_traditional_resources.txt (fpga_size+2 IO ring)"
            if name in TRAIN:
                f = REPO / f"best_layout_coordinates_{name}{SUFFIX[s]}.txt"
                c = jload(f)
                row.update(split="train", policy_area_mwta=c["routing_area"], policy_delay_ns=c["delay_ns"],
                           policy_power_w=c["power_w"], policy_adp=adp(c), policy_aspect_ratio=c["aspect_ratio"],
                           policy_grid_w=c["grid_W"], policy_grid_h=c["grid_H"], how_obtained="best_during_training",
                           source=f"{f.name}; {src_base}")
            else:
                f = REPO / f"det_seed{s}_{name}.json"
                d = jload(f)[name]
                acts = det_actions[(name, s)]
                ar = AR[acts[0]]
                coords = [(1 + a // MAX_H, 1 + a % MAX_H) for a in acts[1:]]
                nb = r["requirements"]["bram"]
                brams, dsps = coords[:nb], coords[nb:]  # env order: BRAMs then DSPs
                crow, _ = cache_grid(name, dsps, brams, ar)
                assert crow and abs(crow[2] - d["routing_area"]) < 1e-6, (name, s, crow)
                row.update(split="heldout", policy_area_mwta=d["routing_area"], policy_delay_ns=d["delay_ns"],
                           policy_power_w=d["power_w"], policy_adp=d["adp"], policy_aspect_ratio=ar,
                           policy_grid_w=int(crow[0]), policy_grid_h=int(crow[1]),
                           how_obtained="final_checkpoint_deterministic",
                           source=(f"{f.name} (metrics); runs/det_seed{s}_{name}.log; aspect ratio from re-played "
                                   f"deterministic actions data_export/logs/det_actions_and_timing.json (matches logged "
                                   f"placements); grid from runs/vtr_layout_cache_{name}.db row for that key; {src_base}"))
            row["adp_reduction_pct"] = (row["baseline_adp"] - row["policy_adp"]) / row["baseline_adp"] * 100
            rows.append(row)
    write("results_per_circuit_seed.csv", B1_HDR, rows)
    return rows


# ------------------------------------------------------------------ C1 (existing final-checkpoint in-pool evals)
def c1():
    rows = []
    files = {7: "runs/inpool_seed7.json", 42: "runs/inpool_seed42_v2.json", 123: "runs/inpool_seed123.json"}
    for name in TRAIN:
        m, r = base(name)
        for s in SEEDS:
            d = jload(REPO / files[s])[name]
            rows.append({"circuit": name, "split": "train", "seed": s, "baseline_area_mwta": m["routing_area"],
                         "baseline_delay_ns": m["delay_ns"], "baseline_power_w": m["power_w"], "baseline_adp": adp(m),
                         "policy_area_mwta": d.get("routing_area"), "policy_delay_ns": d.get("delay_ns"),
                         "policy_power_w": d.get("power_w"), "policy_adp": d.get("adp"),
                         "adp_reduction_pct": d.get("adp_reduction_pct"),
                         "policy_aspect_ratio": None, "policy_grid_w": None, "policy_grid_h": None,
                         "baseline_grid_w": r["fpga_size"][0] + 2, "baseline_grid_h": r["fpga_size"][1] + 2,
                         "how_obtained": "final_checkpoint_deterministic",
                         "source": f"{files[s]} / {files[s].replace('.json', '.log')} (evaluate_held_out.py run with "
                                   f"{ {7: 'runs/multi11_long_seed7.zip', 42: 'runs/multi11_long_seed42_v2.zip', 123: 'runs/multi11_long_seed123.zip'}[s]}, deterministic=True; "
                                   f"aspect ratio/grid not recorded in that output)"})
    write("inpool_final_checkpoint.csv", B1_HDR, rows)


# ------------------------------------------------------------------ B2
def b2():
    suite = {"fifo": (None, "not found in VTR 22a09d39 tree; nearest file libs/EXTERNAL/yosys/docs/source/code_examples/fifo/fifo.v differs"),
             "mmc_core": ("VTR-other", "vtr_flow/benchmarks/ultraembedded/mmc_core.v (identical)"),
             "reduction_layer": ("Koios", "vtr_flow/benchmarks/verilog/koios/reduction_layer.v (identical)"),
             "softmax": ("Koios", "vtr_flow/benchmarks/verilog/koios/softmax.v (identical)"),
             "custom_macbuf": ("authored", "benchmarks/custom_macbuf.v header comment"),
             "lightweight_cipher": ("authored", "benchmarks/lightweight_cipher.v header comment")}
    identical = {"ch_intrinsics", "boundtop", "diffeq1", "diffeq2", "raygentop"}
    rows = []
    for name in TRAIN + HELDOUT:
        vo = (REPO / f"runs/{name}_traditional/vpr.out").read_text(errors="ignore")
        so = (REPO / f"runs/{name}_traditional/vpr_stdout.log").read_text(errors="ignore")
        _, r = base(name)
        stats = vo[vo.index("Circuit Statistics:"):]
        stats = stats[:stats.index("Netlist Clocks")]
        g = lambda k: int(re.search(rf"\.{k}\s*:\s*(\d+)", stats).group(1)) if re.search(rf"\.{k}\s*:\s*(\d+)", stats) else 0
        nets = int(re.search(r"Nets\s*:\s*(\d+)", stats).group(1))
        wmin = int(float(re.search(r"successfully routed with a channel width factor of (\d+)", vo).group(1)))
        wrel = int(float(re.search(r"successfully routed with a channel width factor of (\d+)", so).group(1)))
        if name in suite:
            s, note = suite[name]
        else:
            s = "VTR"
            note = ("vtr_flow/benchmarks/verilog/%s.v (identical)" % name) if name in identical else \
                   ("vtr_flow/benchmarks/verilog/%s.v exists but repo copy is NOT byte-identical" % name)
        rows.append({"circuit": name, "split": "train" if name in TRAIN else "heldout", "source_suite": s,
                     "num_luts": g("names"), "num_ffs": g("latch"), "num_clbs_packed": r["requirements"]["clb"],
                     "num_dsp": r["requirements"]["dsp"], "num_bram": r["requirements"]["bram"],
                     "num_io": r["requirements"]["io"], "num_nets": nets, "baseline_channel_width": wmin,
                     "source": (f"runs/{name}_traditional/vpr.out 'Circuit Statistics' (.names=LUTs incl. 0-LUT, .latch=FFs, "
                                f"Nets = pre-pack atom nets) and min-W search result; relaxed routing W used for "
                                f"delay/power = {wrel} (vpr_stdout.log); counts clb/dsp/bram/io from "
                                f"baselines/{name}_traditional_resources.txt; suite: {note}")})
    write("circuits.csv", ["circuit", "split", "source_suite", "num_luts", "num_ffs", "num_clbs_packed", "num_dsp",
                           "num_bram", "num_io", "num_nets", "baseline_channel_width", "source"], rows)


# ------------------------------------------------------------------ B3
def b3():
    runs = {7: "run-20260624_160804-w1y8bh7a", 42: "run-20260625_040212-7y4orlsm", 123: "run-20260625_040228-0imgwjus"}
    rows = []
    for s in SEEDS:
        jl = REPO / f"all_layouts_multi_seed_{s}{SUFFIX[s]}.jsonl"
        recs = [json.loads(l) for l in open(jl)]
        summ = jload(REPO / f"runs/wandb/{runs[s]}/files/wandb-summary.json")
        meta = jload(REPO / f"runs/wandb/{runs[s]}/files/wandb-metadata.json")
        rows.append({"seed": s, "episodes": len(recs), "vtr_calls_requested": len(recs),
                     "vtr_runs_executed_cache_misses": None, "cache_hits": None,
                     "failed_floorplans": sum(1 for x in recs if not x["success"]),
                     "training_wall_clock_hours": round(summ["_runtime"] / 3600, 3),
                     "cpu_model": None, "cpu_cores_used": None, "parallel_envs": 24,
                     "gpu_model_or_none": meta.get("gpu"),
                     "source": (f"{jl.name} (one line per completed episode; each reaches the terminal VTR evaluation "
                                f"or a cache lookup); runs/wandb/{runs[s]}/files/wandb-summary.json _runtime; "
                                f"wandb-metadata.json (args --n_envs 24, gpu, cpu_count 16 physical / "
                                f"{meta.get('cpu_count_logical')} logical, host {meta.get('host')}); runs/train_multi11_seed{s if s != 42 else '42_v2'}.log. "
                                f"NOTES: cache hits/misses not logged per episode (the 'Cached' flag is printed only on new-best "
                                f"events), and the per-benchmark cache DBs are shared with other runs, so misses cannot be "
                                f"reconstructed. vtr_calls_requested = completed episodes (every episode ends in a VTR call or "
                                f"cache lookup). cpu_model not recorded (wandb metadata has no CPU model string; host "
                                f"localhost.localdomain with 16 physical/24 logical CPUs and {int(meta['memory']['total'])/1e9:.2f} GB "
                                f"RAM, the same specs as the current i9-12900K machine, but that is not proof it is the same "
                                f"machine). cpu_cores_used not measured (24 SubprocVecEnv workers, each launching a VTR flow). "
                                f"Wall clock = W&B _runtime (process start to finish, includes W&B setup).")})
    write("training_accounting.csv", ["seed", "episodes", "vtr_calls_requested", "vtr_runs_executed_cache_misses",
                                      "cache_hits", "failed_floorplans", "training_wall_clock_hours", "cpu_model",
                                      "cpu_cores_used", "parallel_envs", "gpu_model_or_none", "source"], rows)


# ------------------------------------------------------------------ GA helpers
GA_OUT = {("arm_core", 7): "ga_output_scratch_seed7.txt"}
GA_STDOUT = {("reduction_layer", 7): "runs/ga_reduction_layer_scratch_seed7_v2.log",
             ("reduction_layer", 42): "runs/ga_reduction_layer_scratch_seed42_v2.log",
             ("reduction_layer", 123): "ga_run_reduction_layer_seed123_stdout.log",
             ("arm_core", 123): "ga_run_arm_core_seed123_stdout.log"}


def ga_files(name, s):
    return GA / GA_OUT.get((name, s), f"ga_output_{name}_scratch_seed{s}.txt"), \
           GA / GA_STDOUT.get((name, s), f"runs/ga_{name}_scratch_seed{s}.log")


def ga_gens(path):
    out = []
    for line in open(path):
        if not line.startswith("Generation"):
            continue
        gen = int(re.match(r"Generation (\d+)", line).group(1))
        fit = float(re.search(r"Best Fitness: ([0-9.e+inf]+)", line).group(1))
        ar = re.search(r"Aspect Ratio: ([0-9.]+|N/A)", line).group(1)
        out.append((gen, fit, ar))
    return out


def b5_b6_b4():
    rows5, rows6 = [], []
    timings = {}
    for name in HELDOUT:
        m, _ = base(name)
        B = adp(m)
        for s in SEEDS:
            fo, so = ga_files(name, s)
            gens = ga_gens(fo)
            txt = so.read_text(errors="ignore")
            took = [float(x) for x in re.findall(r"OK \(took ([0-9.]+) seconds", txt)]
            timings.setdefault(name, []).append((so, took))
            n_gen = len(gens)
            inds = re.findall(r"Evaluating generation of (\d+) individuals", txt)
            assert len(set(inds)) == 1 and n_gen == len(inds), (name, s)
            k = int(inds[0])
            assert len(took) == n_gen * k, (name, s, len(took), n_gen * k)
            best = gens[-1][1]
            last_improve = max(g for g, f, _ in gens if f == best and all(f2 > best for g2, f2, _ in gens if g2 < g)) \
                if any(f > best for _, f, _ in gens) else 0
            broke = "Breaking condition reached" in txt
            stop = "early_stop" if broke else ("max_gen" if n_gen == 500 else None)
            rows5.append({"circuit": name, "seed": s, "best_adp": best, "adp_reduction_pct": (B - best) / B * 100,
                          "generations_run": n_gen, "vtr_calls_reported": n_gen * k,
                          "vtr_runs_executed_cache_misses": len(took), "cache_hits": 0, "wall_clock_hours": None,
                          "best_aspect_ratio": gens[-1][2], "best_grid_w": None, "best_grid_h": None,
                          "stop_reason": stop,
                          "source": (f"{rel(fo)} (per-generation best-so-far); {rel(so)} ({k} individuals/generation, "
                                     f"{len(took)} 'OK (took' VTR completions, 'Breaking condition' line; last "
                                     f"improvement at generation {last_improve}); baseline baselines/{name}_traditional_metric.txt; "
                                     f"GA has no cache (ga_agent.py fitness_function runs VTR for every individual)")})
            prev = None
            for g, f, _ in gens:
                if prev is None or f < prev:
                    rows6.append({"circuit": name, "seed": s, "vtr_call_index": (g + 1) * k,
                                  "best_so_far_adp_reduction_pct": (B - f) / B * 100})
                    prev = f
            if rows6[-1]["vtr_call_index"] != n_gen * k:
                rows6.append({"circuit": name, "seed": s, "vtr_call_index": n_gen * k,
                              "best_so_far_adp_reduction_pct": (B - prev) / B * 100})
    write("ga_runs.csv", ["circuit", "seed", "best_adp", "adp_reduction_pct", "generations_run", "vtr_calls_reported",
                          "vtr_runs_executed_cache_misses", "cache_hits", "wall_clock_hours", "best_aspect_ratio",
                          "best_grid_w", "best_grid_h", "stop_reason", "source"], rows5)
    write("ga_convergence.csv", ["circuit", "seed", "vtr_call_index", "best_so_far_adp_reduction_pct"], rows6)

    rows4 = []
    for name in TRAIN + HELDOUT:
        if name not in timings:
            rows4.append({"circuit": name, "vtr_runs_logged": 0, "source": "no per-run VTR wall-clock logged for this circuit "
                          "(training env runs VTR with silent=True and records no duration; baseline run dir output.txt has "
                          "vpr_seconds only, VPR stage, not full flow)"})
            continue
        allt = [t for _, ts in timings[name] for t in ts]
        rows4.append({"circuit": name, "vtr_runs_logged": len(allt), "vtr_seconds_median": st.median(allt),
                      "vtr_seconds_min": min(allt), "vtr_seconds_max": max(allt),
                      "source": "full run_vtr_flow.py wall-clock 'OK (took N seconds' lines in GA stdout logs "
                                + "; ".join(rel(p) for p, _ in timings[name])
                                + " -- measured with 8 VTR flows running concurrently (GA population evaluated in parallel)"})
    write("vtr_runtime.csv", ["circuit", "vtr_runs_logged", "vtr_seconds_median", "vtr_seconds_min", "vtr_seconds_max",
                              "source"], rows4)


# ------------------------------------------------------------------ B8
def b8():
    rows = []
    for s in SEEDS:
        base_adp = {n: adp(base(n)[0]) for n in TRAIN}
        for i, l in enumerate(open(REPO / f"all_layouts_multi_seed_{s}{SUFFIX[s]}.jsonl")):
            x = json.loads(l)
            red = (base_adp[x["benchmark_name"]] - adp(x)) / base_adp[x["benchmark_name"]] * 100 if x["success"] else None
            rows.append({"seed": s, "episode": i, "circuit": x["benchmark_name"], "episode_reward": x["reward"],
                         "adp_reduction_pct_or_null": red})
    write("learning_curves.csv", ["seed", "episode", "circuit", "episode_reward", "adp_reduction_pct_or_null"], rows)


# ------------------------------------------------------------------ B7
def b7():
    t = {(r["circuit"], r["seed"]): r for r in jload(OUT / "logs/det_actions_and_timing.json")}
    vtr = {}
    p = OUT / "vpr_seed_noise.csv"
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r["floorplan"] == "policy_seed42" and r["vpr_seed"] == "1" and r["vtr_seconds"]:
                vtr[r["circuit"]] = float(r["vtr_seconds"])
    rows = []
    for name in HELDOUT:
        for s in SEEDS:
            rows.append({"circuit": name, "seed": s,
                         "rollout_seconds_policy_only": t[(name, s)]["rollout_seconds_policy_only"],
                         "vtr_seconds": vtr.get(name) if s == 42 else None,
                         "source": ("NOT logged originally. rollout time measured in Tier C (data_export/tools/c345.py timing: "
                                    "CPU, torch, 2nd rollout after warm-up, excludes the terminal VTR call) -> "
                                    "data_export/logs/det_actions_and_timing.json"
                                    + ("; vtr_seconds = full VTR flow of this exact floorplan, VPR seed 1, from vpr_seed_noise.csv (C4)"
                                       if s == 42 else "; vtr_seconds not measured for this seed"))})
    write("inference_timing.csv", ["circuit", "seed", "rollout_seconds_policy_only", "vtr_seconds", "source"], rows)


if __name__ == "__main__":
    import sys
    what = sys.argv[1:] or ["b1", "c1", "b2", "b3", "b56", "b8", "b7"]
    for w in what:
        {"b1": b1, "c1": c1, "b2": b2, "b3": b3, "b56": b5_b6_b4, "b8": b8, "b7": b7}[w]()
