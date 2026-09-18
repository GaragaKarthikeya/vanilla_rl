"""C2 corrected: aspect-ratio sweep of the trimmed baseline with H-block tiles
placed on EACH aspect ratio's own grid.

Why: c2_ar_sweep.py computed tile coordinates once, on the square baseline
core grid, and reused them at every aspect ratio. VPR's auto_layout then had to
grow the grid until every fixed <single> tile fit and stretch the other side to
hold the ratio (mkDelayWorker32B 44x440 at AR 0.1), so that sweep understates a
fair non-learning baseline. See mismatches.md section 7.

Rule here, per circuit and aspect ratio AR:
  1. Find the smallest core grid (w, h) with w = max(1, round(AR * h)), for
     h = 1, 2, ..., such that
       a. the column pattern fits inside it: DSP columns x = 6, 14, 22, ... and
          BRAM columns x = 2, 10, 18, ..., tiles stacked from y = 1 at pitch 4
          (DSP) / 6 (BRAM), with x <= w and y + height - 1 <= h (the same
          pattern and fill order as c2_ar_sweep.py), and
       b. the remaining tiles hold the packed CLBs:
          w*h - 4*n_dsp - 6*n_bram >= n_clb.
  2. Place the blocks with that pattern on (w, h) and bake through the paper's
     template (auto_layout aspect_ratio=AR, <single> tiles), no atom-pinning
     constraints, default VPR seed -- identical to c2_ar_sweep.py otherwise.
  3. Record the grid VPR actually built (grid_w/grid_h include the IO ring) next
     to the planned core grid, so it is checkable whether VPR still stretched.

Memory: run under `ulimit -v` (see run_c2_perar.sh) so no single VTR job can
take the host down.

Usage (inside distrobox ubuntu-work):
    python c2_perar.py WORKERS
"""
import shutil
import sys
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed

from common import *  # noqa
from src.env.fpga_env import ASPECT_RATIOS
from src.evaluation.vtr_runner import VTRRunner
from src.layout.baker import bake_layout

HDR = ["circuit", "aspect_ratio", "planned_core_w", "planned_core_h", "grid_w", "grid_h",
       "area_mwta", "delay_ns", "power_w", "adp", "adp_reduction_pct_vs_baseline",
       "vtr_success", "vtr_seconds", "dsp_coords", "bram_coords", "note"]

DSP_H, BRAM_H = 4, 6


def fill(n, startx, h_blk, w, h):
    out, x = [], startx
    while len(out) < n and x <= w:
        y = 1
        while len(out) < n and y + h_blk - 1 <= h:
            out.append((x, y))
            y += h_blk
        x += 8
    return out if len(out) >= n else None


def plan(ar, n_clb, n_dsp, n_bram, h_max=2000):
    for h in range(1, h_max + 1):
        w = max(1, round(ar * h))
        if w * h - DSP_H * n_dsp - BRAM_H * n_bram < n_clb:
            continue
        d = fill(n_dsp, 6, DSP_H, w, h)
        b = fill(n_bram, 2, BRAM_H, w, h)
        if d is not None and b is not None:
            return w, h, d, b
    raise ValueError(f"no grid up to h={h_max} for AR {ar}")


def run_one(name, ar):
    m, r = baseline(name)
    req = r["requirements"]
    w, h, dsps, brams = plan(ar, req["clb"], req["dsp"], req["bram"])
    wid = uuid.uuid4().hex[:8]
    arch = REPO / f"temp_arch_c2p_{wid}.xml"
    rundir = REPO / "runs" / f"temp_run_c2p_{wid}"
    bake_layout(name, dsps, brams, width=w + 2, height=h + 2, output_path=str(arch), aspect_ratio=ar)
    t0 = now()
    rc = VTRRunner().run(REPO / "benchmarks" / f"{name}.v", arch, rundir, silent=True, timeout=1800)
    secs = now() - t0
    row = {"circuit": name, "aspect_ratio": ar, "planned_core_w": w, "planned_core_h": h,
           "vtr_seconds": round(secs, 2), "vtr_success": False,
           "dsp_coords": json.dumps(dsps), "bram_coords": json.dumps(brams)}
    vm = VTRRunner.parse_metrics(rundir / "vpr.out", rundir / "vpr.crit_path.out", rundir / f"{name}.power")
    if rc == 0 and vm.is_complete():
        res = VTRRunner.parse_resources(rundir / "vpr.out")
        a = adp(vm.routing_area, vm.delay_ns, vm.power_w)
        b = adp(m["routing_area"], m["delay_ns"], m["power_w"])
        row.update(grid_w=res.fpga_size[0] + 2, grid_h=res.fpga_size[1] + 2,
                   area_mwta=vm.routing_area, delay_ns=vm.delay_ns, power_w=vm.power_w, adp=a,
                   adp_reduction_pct_vs_baseline=(b - a) / b * 100, vtr_success=True)
    else:
        row["note"] = f"VTR flow rc={rc} or incomplete metrics (possibly the ulimit -v memory cap)"
    arch.unlink(missing_ok=True)
    shutil.rmtree(rundir, ignore_errors=True)
    return row


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    sink = CsvSink(OUT / "baseline_ar_sweep_perar.csv", HDR, ["circuit", "aspect_ratio"])
    jobs = [(n, ar) for n in TRAIN + HELDOUT for ar in ASPECT_RATIOS if not sink.has((n, ar))]
    print(f"C2 per-AR: {len(jobs)} jobs, {workers} workers", flush=True)
    with ThreadPoolExecutor(workers) as ex:
        futs = {ex.submit(run_one, n, ar): (n, ar) for n, ar in jobs}
        for f in as_completed(futs):
            n, ar = futs[f]
            try:
                row = f.result()
            except Exception as exc:
                row = {"circuit": n, "aspect_ratio": ar, "vtr_success": False, "vtr_seconds": "",
                       "note": f"driver error: {exc!r}"}
                print("ERR", n, ar, repr(exc), flush=True)
            sink.write(row)
            print("done", n, ar, row["vtr_success"], row.get("vtr_seconds"), flush=True)
