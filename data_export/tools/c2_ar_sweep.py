"""C2: aspect-ratio sweep of the trimmed island-style baseline (no learning).

H-block positioning after trimming (also written to the README):
  The baseline arch (arch/k6_frac_N10_mem32K_40nm.xml) has mult_36 columns at
  startx=6 repeatx=8 and memory columns at startx=2 repeatx=8, starty=1, i.e.
  tiles stacked from y=1 with pitch = block height (DSP 4, BRAM 6). We keep
  that exact column pattern but instantiate only the netlist's required
  count: blocks fill the lowest free slot of the leftmost pattern column
  (x=6,14,22,... for DSP; x=2,10,18,... for BRAM; y=1,1+h,1+2h,...),
  restricted to the benchmark's baseline core grid (x<=W, y+h-1<=H) -- the
  same bounds the policy's action mask uses. Rendered through the paper's
  own template (src/layout/baker.py, auto_layout aspect_ratio=AR, <single>
  tiles), no atom-pinning constraints file (the baseline flow pins nothing),
  VPR placer seed left at the flow default.
"""
import sys
import uuid
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import *  # noqa
from src.layout.baker import bake_layout
from src.evaluation.vtr_runner import VTRRunner
from src.env.fpga_env import ASPECT_RATIOS

HDR = ["circuit", "aspect_ratio", "grid_w", "grid_h", "area_mwta", "delay_ns", "power_w", "adp",
       "adp_reduction_pct_vs_baseline", "vtr_success", "vtr_seconds", "dsp_coords", "bram_coords"]


def trimmed_positions(W, H, n_dsp, n_bram):
    def fill(n, startx, h):
        out = []
        x = startx
        while len(out) < n and x <= W:
            y = 1
            while len(out) < n and y + h - 1 <= H:
                out.append((x, y))
                y += h
            x += 8
        if len(out) < n:
            raise ValueError(f"pattern capacity {len(out)} < required {n}")
        return out
    return fill(n_dsp, 6, 4), fill(n_bram, 2, 6)


def run_one(name, ar):
    m, r = baseline(name)
    W, H = r["fpga_size"]
    dsps, brams = trimmed_positions(W, H, r["requirements"]["dsp"], r["requirements"]["bram"])
    wid = uuid.uuid4().hex[:8]
    arch = REPO / f"temp_arch_dx_{wid}.xml"
    rundir = REPO / "runs" / f"temp_run_dx_{wid}"
    bake_layout(name, dsps, brams, width=W + 2, height=H + 2, output_path=str(arch), aspect_ratio=ar)
    runner = VTRRunner()
    t0 = now()
    rc = runner.run(REPO / "benchmarks" / f"{name}.v", arch, rundir, silent=True, timeout=1800)
    secs = now() - t0
    row = {"circuit": name, "aspect_ratio": ar, "vtr_seconds": round(secs, 2), "vtr_success": False,
           "dsp_coords": json.dumps(dsps), "bram_coords": json.dumps(brams)}
    vm = VTRRunner.parse_metrics(rundir / "vpr.out", rundir / "vpr.crit_path.out", rundir / f"{name}.power")
    if rc == 0 and vm.is_complete():
        res = VTRRunner.parse_resources(rundir / "vpr.out")
        a = adp(vm.routing_area, vm.delay_ns, vm.power_w)
        b = adp(m["routing_area"], m["delay_ns"], m["power_w"])
        row.update(grid_w=res.fpga_size[0] + 2, grid_h=res.fpga_size[1] + 2, area_mwta=vm.routing_area,
                   delay_ns=vm.delay_ns, power_w=vm.power_w, adp=a,
                   adp_reduction_pct_vs_baseline=(b - a) / b * 100, vtr_success=True)
    arch.unlink(missing_ok=True)
    shutil.rmtree(rundir, ignore_errors=True)
    return row


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    sink = CsvSink(OUT / "baseline_ar_sweep.csv", HDR, ["circuit", "aspect_ratio"])
    jobs = [(n, ar) for n in TRAIN + HELDOUT for ar in ASPECT_RATIOS if not sink.has((n, ar))]
    print(f"C2: {len(jobs)} jobs", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(run_one, n, ar): (n, ar) for n, ar in jobs}
        for f in as_completed(futs):
            try:
                row = f.result()
            except Exception as exc:
                n, ar = futs[f]
                row = {"circuit": n, "aspect_ratio": ar, "vtr_success": False, "vtr_seconds": ""}
                print("ERR", n, ar, exc, flush=True)
            sink.write(row)
            print("done", row["circuit"], row["aspect_ratio"], row["vtr_success"], row["vtr_seconds"], flush=True)
