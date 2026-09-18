"""F4: the policy's floorplans without VPR instance pinning.

The GA leaves H-block instance assignment to VPR, while the policy writes a VPR
constraints file pinning each netlist atom to its tile
(src/layout/baker.py, --read_vpr_constraints). That confounds the comparison.

F4 re-evaluates the policy's own deterministic floorplans for the 6 held-out
circuits at all 3 seeds with the constraints file OMITTED and everything else
identical: same baked architecture (same tile coordinates, same aspect ratio),
same VTR build, same flow, default VPR seed.

The pinned rows are reused from data_export/results_per_circuit_seed.csv and are
not re-run.

Usage (inside distrobox ubuntu-work):
    python f4.py WORKERS
"""
import shutil
import subprocess
import sys
import uuid

from common3 import *  # noqa: F401,F403
import common3 as c3
from src.evaluation.vtr_runner import VTRRunner
from src.layout.baker import bake_layout

HDR = ["circuit", "seed", "variant", "area_mwta", "delay_ns", "power_w", "adp",
       "adp_reduction_pct", "vtr_success", "vtr_seconds", "cached", "aspect_ratio",
       "grid_w", "grid_h", "source"]


def unpinned(name, acts):
    """Bake the policy's floorplan and run VTR with NO constraints file."""
    wid = uuid.uuid4().hex[:8]
    rundir = REPO / "runs" / f"temp_run_f4_{wid}"     # noqa: F405
    arch = REPO / f"temp_arch_f4_{wid}.xml"           # noqa: F405
    ar, dsps, brams, _types, W, H = c3.decode_rollout(name, acts)
    bake_layout(name, dsps, brams, width=W + 2, height=H + 2, output_path=str(arch),
                aspect_ratio=ar, block_names=None, constraints_output_path=None)
    rundir.mkdir(parents=True, exist_ok=True)
    runner = VTRRunner()
    cmd = runner._build_cmd(REPO / "benchmarks" / f"{name}.v", arch, rundir, True, None)  # noqa: F405
    t0 = now()        # noqa: F405
    rc = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        timeout=3600).returncode
    secs = now() - t0  # noqa: F405
    row = {"vtr_seconds": round(secs, 2), "vtr_success": False, "cached": False,
           "aspect_ratio": ar}
    vm = VTRRunner.parse_metrics(rundir / "vpr.out", rundir / "vpr.crit_path.out",
                                 rundir / f"{name}.power")
    if rc == 0 and vm.is_complete():
        res = VTRRunner.parse_resources(rundir / "vpr.out")
        m, _ = baseline(name)     # noqa: F405
        a = adp(vm.routing_area, vm.delay_ns, vm.power_w)      # noqa: F405
        b = adp(m["routing_area"], m["delay_ns"], m["power_w"])  # noqa: F405
        row.update(area_mwta=vm.routing_area, delay_ns=vm.delay_ns, power_w=vm.power_w,
                   adp=a, adp_reduction_pct=(b - a) / b * 100, vtr_success=True,
                   grid_w=res.fpga_size[0] + 2, grid_h=res.fpga_size[1] + 2)
    arch.unlink(missing_ok=True)
    shutil.rmtree(rundir, ignore_errors=True)
    return row


def main(workers):
    det = c3.det_actions()
    ctrl = c3.control_rows()
    sink = CsvSink(OUT / "f4_no_pinning.csv", HDR, ["circuit", "seed", "variant"])  # noqa: F405

    for name in HELDOUT:        # noqa: F405
        for seed in SEEDS:      # noqa: F405
            if sink.has((name, seed, "pinned")):
                continue
            r = ctrl[(name, seed)]
            sink.write({
                "circuit": name, "seed": seed, "variant": "pinned",
                "area_mwta": r["policy_area_mwta"], "delay_ns": r["policy_delay_ns"],
                "power_w": r["policy_power_w"], "adp": r["policy_adp"],
                "adp_reduction_pct": r["adp_reduction_pct"], "vtr_success": True,
                "vtr_seconds": "", "cached": "", "aspect_ratio": r["policy_aspect_ratio"],
                "grid_w": r["policy_grid_w"], "grid_h": r["policy_grid_h"],
                "source": "data_export/results_per_circuit_seed.csv (paper zero-shot eval, "
                          "not re-run); vtr_seconds/cached null: not logged by that eval",
            })

    jobs = []
    for name in HELDOUT:        # noqa: F405
        for seed in SEEDS:      # noqa: F405
            if sink.has((name, seed, "unpinned")):
                continue
            jobs.append(({"circuit": name, "seed": seed, "variant": "unpinned",
                          "source": "data_export3/tools/f4.py; same baked arch as the policy's "
                                    "deterministic floorplan, VPR constraints file omitted"},
                         (name, det[(name, seed)])))
    run_pool(jobs, unpinned, sink, workers, "F4")    # noqa: F405


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 16)
