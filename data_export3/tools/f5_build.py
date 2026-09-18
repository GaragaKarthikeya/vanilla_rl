"""F5 stage 1: build island-style baselines for candidate held-out circuits.

For each candidate: copy the Verilog into benchmarks/, run the traditional VTR
flow on the stock architecture into runs/<name>_traditional/ (same arch, build
and flow as the existing baselines), write
baselines/<name>_traditional_{metric,resources}.txt in the existing format, and
extract the reduced netlist graph.

run_traditional_flow.py is not reused because it calls load_env_file(), which
would overwrite the environment with the repo's stale .env paths; this driver
uses the same VTRRunner with the corrected paths from common3.

Writes data_export3/f5_candidates.csv with, for every candidate, whether the
baseline built and whether it fits the checkpoint's universe caps
(core grid <= 48x48, graph nodes <= 67, edges <= 2694).

Usage (inside distrobox ubuntu-work):
    python f5_build.py WORKERS
"""
import shutil
import subprocess
import sys

from common3 import *  # noqa: F401,F403
from src.evaluation.vtr_runner import VTRRunner
from src.netlist.analyzer import analyze_netlist, build_netlist_graph, save_netlist_graph_json

VTR_BENCH = Path("/home/digital-2/workspace/vtr-verilog-to-routing/vtr_flow/benchmarks")  # noqa: F405
ARCH = "arch/k6_frac_N10_mem32K_40nm.xml"

# name -> source .v. Chosen from the VTR and Koios suites, excluding every
# circuit in the training pool, the six held-out circuits, and robot_rl (which
# was in the checkpoint's sizing universe, so it is not unseen).
CANDIDATES = {
    "bnn": "verilog/koios/bnn.v",
    "lenet": "verilog/koios/lenet.v",
    "conv_layer": "verilog/koios/conv_layer.v",
    "conv_layer_hls": "verilog/koios/conv_layer_hls.v",
    "eltwise_layer": "verilog/koios/eltwise_layer.v",
    "gemm_layer": "verilog/koios/gemm_layer.v",
    "spmv": "verilog/koios/spmv.v",
    "attention_layer": "verilog/koios/attention_layer.v",
    "blob_merge": "verilog/blob_merge.v",
    "sha": "verilog/sha.v",
    "stereovision0": "verilog/stereovision0.v",
    "stereovision1": "verilog/stereovision1.v",
    "bgm": "verilog/bgm.v",
    # batch 2: mid-size cores, added because most of batch 1 either exceeded the
    # 48x48 canvas / 67-node graph cap or packs to zero DSP+BRAM (which leaves
    # the policy nothing to place but the aspect ratio).
    "8051": "freecores/8051.v",
    "aes_cipher": "freecores/aes_cipher.v",
    "aes_inv_cipher": "freecores/aes_inv_cipher.v",
    "ethmac": "freecores/ethmac.v",
    "mips_16": "freecores/mips_16.v",
    "xtea": "freecores/xtea.v",
    "enet_core": "ultraembedded/enet_core.v",
    "soc_core": "ultraembedded/soc_core.v",
    "uriscv_core": "ultraembedded/uriscv_core.v",
    "usb_uart_core": "ultraembedded/usb_uart_core.v",
}

HDR = ["circuit", "source_verilog", "baseline_built", "core_grid_w", "core_grid_h",
       "num_clbs_packed", "num_dsp", "num_bram", "num_io", "occupancy", "graph_nodes",
       "graph_edges", "fits_universe", "reason_if_excluded", "vtr_seconds", "source"]


def build(name, rel):
    src = VTR_BENCH / rel
    dst = REPO / "benchmarks" / f"{name}.v"        # noqa: F405
    if not dst.exists():
        shutil.copy(src, dst)

    rundir = REPO / "runs" / f"{name}_traditional"  # noqa: F405
    row = {"vtr_seconds": "", "baseline_built": False}
    metric_f = REPO / "baselines" / f"{name}_traditional_metric.txt"      # noqa: F405
    res_f = REPO / "baselines" / f"{name}_traditional_resources.txt"      # noqa: F405

    if not (metric_f.exists() and res_f.exists() and (rundir / f"{name}.net").exists()):
        rundir.mkdir(parents=True, exist_ok=True)
        runner = VTRRunner()
        cmd = runner._build_cmd(dst, REPO / ARCH, rundir, True, None)     # noqa: F405
        t0 = now()       # noqa: F405
        rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=14400).returncode
        row["vtr_seconds"] = round(now() - t0, 2)      # noqa: F405
        vm = VTRRunner.parse_metrics(rundir / "vpr.out", rundir / "vpr.crit_path.out",
                                     rundir / f"{name}.power")
        if rc != 0 or not vm.is_complete():
            row["reason_if_excluded"] = f"traditional VTR flow failed (rc={rc})"
            return row
        res = VTRRunner.parse_resources(rundir / "vpr.out")
        metric_f.write_text(json.dumps({          # noqa: F405
            "delay_ns": vm.delay_ns, "wirelength": vm.wirelength,
            "power_w": vm.power_w, "routing_area": vm.routing_area}, indent=4))
        res_f.write_text(json.dumps({             # noqa: F405
            "fpga_size": list(res.fpga_size), "requirements": res.requirements,
            "limits": res.limits}, indent=4))

    rd = json.loads(res_f.read_text())            # noqa: F405
    W, H = rd["fpga_size"]
    reqs = rd["requirements"]
    row.update(baseline_built=True, core_grid_w=W, core_grid_h=H,
               num_clbs_packed=reqs.get("clb", 0), num_dsp=reqs.get("dsp", 0),
               num_bram=reqs.get("bram", 0), num_io=reqs.get("io", 0),
               occupancy=(reqs.get("clb", 0) + reqs.get("dsp", 0) + reqs.get("bram", 0)) / (W * H))

    # reduced netlist graph
    gf = REPO / f"{name}_netlist_graph.json"      # noqa: F405
    if not gf.exists():
        st = analyze_netlist(name, rundir / f"{name}.net")
        save_netlist_graph_json(build_netlist_graph(name, rundir / f"{name}.net",
                                                    include_clbs=True), gf)
        del st
    from src.netlist.graph_reduction import reduce_netlist_graph
    g = reduce_netlist_graph(gf)
    row.update(graph_nodes=g.num_nodes, graph_edges=g.num_edges)

    bad = []
    if W > MAX_W or H > MAX_H:            # noqa: F405
        bad.append(f"core grid {W}x{H} exceeds the {MAX_W}x{MAX_H} canvas")   # noqa: F405
    if g.num_nodes > MAX_NODES:           # noqa: F405
        bad.append(f"{g.num_nodes} graph nodes exceeds MAX_NODES={MAX_NODES}")  # noqa: F405
    if g.num_edges > MAX_EDGES:           # noqa: F405
        bad.append(f"{g.num_edges} graph edges exceeds MAX_EDGES={MAX_EDGES}")  # noqa: F405
    row["fits_universe"] = not bad
    row["reason_if_excluded"] = "; ".join(bad)
    return row


def main(workers, out_name="f5_candidates.csv", only=None):
    """`only` restricts the batch, so a second batch can run in its own CSV
    while a first batch is still busy (two processes must not append to the
    same file)."""
    sink = CsvSink(OUT / out_name, HDR, ["circuit"])    # noqa: F405
    done = set()
    for f in ("f5_candidates.csv", "f5_candidates_b2.csv"):
        p = OUT / f                                      # noqa: F405
        if p.exists():
            with open(p) as fh:
                done |= {r["circuit"] for r in csv.DictReader(fh)}   # noqa: F405
    jobs = [({"circuit": n, "source_verilog": str(VTR_BENCH / r),
              "source": "data_export3/tools/f5_build.py; traditional VTR flow on " + ARCH},
             (n, r)) for n, r in CANDIDATES.items()
            if n not in done and (only is None or n in only)]
    run_pool(jobs, build, sink, workers, "F5build")   # noqa: F405


BATCH2 = ["8051", "aes_cipher", "aes_inv_cipher", "ethmac", "mips_16", "xtea",
          "enet_core", "soc_core", "uriscv_core", "usb_uart_core"]

if __name__ == "__main__":
    w = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    if len(sys.argv) > 2 and sys.argv[2] == "batch2":
        main(w, "f5_candidates_b2.csv", set(BATCH2))
    else:
        main(w)
