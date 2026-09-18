"""F6: training entropy / convergence curve. No new runs.

Source: the three training stdout logs, runs/train_multi11_seed{7,42_v2,123}.log.
These carry the same per-update values that were streamed to W&B -- the
"[Stability Report - update #N]" block is printed by CustomMaskablePPO
(src/training/ppo.py) once per PPO update, and the SB3 table immediately before
it gives total_timesteps and ep_rew_mean for the same update. The local
runs/wandb/*.wandb blobs are the transport for those same numbers, so the logs
are used directly and every row names its file and line.

Writes data_export3/f6_entropy.csv.
"""
import re

from common3 import *  # noqa: F401,F403

HDR = ["seed", "update_index", "timesteps", "policy_entropy", "explained_variance",
       "value_loss", "mean_episode_reward", "source"]

LOG = {7: "runs/train_multi11_seed7.log",
       42: "runs/train_multi11_seed42_v2.log",
       123: "runs/train_multi11_seed123.log"}

RE_TS = re.compile(r"\|\s*total_timesteps\s*\|\s*([0-9.e+-]+)\s*\|")
RE_REW = re.compile(r"\|\s*ep_rew_mean\s*\|\s*([-0-9.e+]+)\s*\|")
RE_UPD = re.compile(r"\[Stability Report\s*[—-]\s*update #(\d+)\]")
RE_KV = re.compile(r"^\s{2,}(policy_entropy|value_loss|explained_variance)\s*:\s*([-0-9.e+]+)")

sink = CsvSink(OUT / "f6_entropy.csv", HDR, ["seed", "update_index"])   # noqa: F405
for seed, rel in LOG.items():
    path = REPO / rel          # noqa: F405
    lines = path.read_text(errors="ignore").splitlines()
    ts = rew = None
    n = 0
    for i, ln in enumerate(lines):
        m = RE_TS.search(ln)
        if m:
            ts = int(float(m.group(1)))
            continue
        m = RE_REW.search(ln)
        if m:
            rew = float(m.group(1))
            continue
        m = RE_UPD.search(ln)
        if not m:
            continue
        upd = int(m.group(1))
        vals = {}
        for ln2 in lines[i + 1: i + 9]:       # the block is 7 metric lines
            k = RE_KV.match(ln2)
            if k:
                vals[k.group(1)] = float(k.group(2))
        if sink.has((seed, upd)) or "policy_entropy" not in vals:
            continue
        sink.write({"seed": seed, "update_index": upd, "timesteps": ts,
                    "policy_entropy": vals["policy_entropy"],
                    "explained_variance": vals.get("explained_variance"),
                    "value_loss": vals.get("value_loss"),
                    "mean_episode_reward": rew,
                    "source": f"{rel}:{i + 1} (Stability Report block); timesteps/ep_rew_mean "
                              "from the SB3 table immediately above"})
        n += 1
    print(f"seed {seed}: {n} updates from {rel}")
print("wrote f6_entropy.csv")
