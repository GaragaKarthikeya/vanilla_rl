"""Shared setup for data_export Tier C drivers.

Does NOT modify any training/eval code. It only:
  * points VTRPaths at the relocated VTR checkout (the repo's .env still names
    the pre-move /home/digital-2/vtr-verilog-to-routing path, and
    load_env_file() would overwrite os.environ with it, so we never call it);
  * pins the universe dims to the values logged at training time
    (runs/train_multi11_seed*.log: MAX_WIDTH=48 MAX_HEIGHT=48 MAX_NODES=67
    MAX_EDGES=2694). compute_max_dims() can no longer be re-run because
    baselines/robot_rl_* is deleted in the working tree.
Must be run inside `distrobox enter ubuntu-work`.
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
os.chdir(REPO)

VTR = Path("/home/digital-2/workspace/vtr-verilog-to-routing")
os.environ["VTR_VENV_PATH"] = "/home/digital-2/.venv"
os.environ["VTR_FLOW_SCRIPT"] = str(VTR / "vtr_flow/scripts/run_vtr_flow.py")
os.environ["VTR_POWER_TECH_FILE"] = str(VTR / "vtr_flow/tech/PTM_45nm/45nm.xml")

MAX_W, MAX_H, MAX_NODES, MAX_EDGES = 48, 48, 67, 2694
TRAIN = ["fifo", "ch_intrinsics", "spree", "boundtop", "mmc_core", "diffeq1", "diffeq2",
         "raygentop", "mkSMAdapter4B", "or1200", "mkPktMerge"]
HELDOUT = ["custom_macbuf", "mkDelayWorker32B", "lightweight_cipher", "reduction_layer",
           "arm_core", "softmax"]
SEEDS = [7, 42, 123]
CKPT = {7: "runs/multi11_long_seed7.zip", 42: "runs/multi11_long_seed42_v2.zip",
        123: "runs/multi11_long_seed123.zip"}
# Isolated cache so Tier C never writes into the paper's cache DBs and every
# evaluation is a real VTR run with a measurable wall-clock time.
CACHE_SUFFIX = "_dataexport"
OUT = REPO / "data_export"


def baseline(name):
    m = json.loads((REPO / "baselines" / f"{name}_traditional_metric.txt").read_text())
    r = json.loads((REPO / "baselines" / f"{name}_traditional_resources.txt").read_text())
    return m, r


def adp(a, d, p):
    return a * d * p


class CsvSink:
    """Append-only CSV; supports resume by reading back existing keys."""

    def __init__(self, path, header, key_cols):
        self.path, self.header, self.key_cols = Path(path), header, key_cols
        self.done = set()
        if self.path.exists():
            with open(self.path) as fh:
                for row in csv.DictReader(fh):
                    self.done.add(tuple(str(row[k]) for k in key_cols))
        else:
            with open(self.path, "w", newline="") as fh:
                csv.writer(fh).writerow(header)

    def has(self, key):
        return tuple(str(k) for k in key) in self.done

    def write(self, row: dict):
        with open(self.path, "a", newline="") as fh:
            csv.writer(fh).writerow([row.get(h, "") for h in self.header])
        self.done.add(tuple(str(row[k]) for k in self.key_cols))


def now():
    return time.time()
