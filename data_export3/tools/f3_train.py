"""F3: retrain seed 42 with the graph encoder removed.

src/ is NOT modified. Three names are rebound at runtime, inside this script only:

  1. trainer.GNNFeaturesExtractor -> NoGCNFeaturesExtractor (the point of F3;
     trainer.py:176 hard-codes the class, so there is no flag for it).
  2. trainer.compute_max_dims -> a function returning the pinned universe dims
     (48, 48, 67, 2694). The paper's universe list includes robot_rl, whose
     benchmark and baselines are deleted from the working tree, so the real
     compute_max_dims can no longer run. The pinned values are the ones logged
     by the paper's own seed-42 run (runs/train_multi11_seed42_v2.log), so the
     canvas and graph caps are identical to the paper's.
  3. trainer.CheckpointCallback -> a wrapper that redirects intermediate
     checkpoints to runs/f3_no_gcn_checkpoints_seed42/, away from the paper's
     runs/checkpoints_seed_42/ (same file names, 259 paper checkpoints there).

Every other hyperparameter is copied from the paper's seed-42 invocation
(wandb-metadata.json of run 7y4orlsm):

  --n_envs 24 --n_steps 48 --batch_size 144 --n_epochs 15 --ent_coef 0.01
  --ar_weight 1.0 --dl_weight 1.0 --pw_weight 1.0 --wl_weight 0.0
  --max_episodes 13000 --timesteps 100000 --vtr_timeout 300 --seed 42

W&B is off, the cache suffix is this request's isolated one, and the checkpoint
is written to runs/f3_no_gcn_seed42.zip so no paper artifact is overwritten.

Usage (inside distrobox ubuntu-work, from the repo root):
    python f3_train.py
"""
import time

from common3 import *  # noqa: F401,F403

import src.training.trainer as trainer
from no_gcn_extractor import NoGCNFeaturesExtractor

TRAIN11 = ["fifo", "ch_intrinsics", "spree", "boundtop", "mmc_core", "diffeq1", "diffeq2",
           "raygentop", "mkSMAdapter4B", "or1200", "mkPktMerge"]
# The paper's universe also lists robot_rl, which no longer exists on disk; the
# dims it determined are pinned below instead, so the canvas is unchanged.
UNIVERSE = TRAIN11 + ["softmax", "reduction_layer"]

PINNED = (MAX_W, MAX_H, MAX_NODES, MAX_EDGES)   # noqa: F405  (48, 48, 67, 2694)

trainer.GNNFeaturesExtractor = NoGCNFeaturesExtractor
trainer.compute_max_dims = lambda names: PINNED

# 3. trainer.py:143-147 saves intermediate checkpoints to
#    runs/checkpoints_seed_{seed}/multi_model_<N>_steps.zip WITHOUT the log
#    suffix -- the same directory and file names as the paper's own seed-42
#    run (259 checkpoints there). Redirect so no paper checkpoint is overwritten.
F3_CKPT_DIR = REPO / "runs" / "f3_no_gcn_checkpoints_seed42"   # noqa: F405
_OrigCheckpointCallback = trainer.CheckpointCallback


def _redirected_checkpoint_callback(*args, **kwargs):
    kwargs["save_path"] = str(F3_CKPT_DIR)
    return _OrigCheckpointCallback(*args, **kwargs)


trainer.CheckpointCallback = _redirected_checkpoint_callback

def main():
    cfg = trainer.TrainConfig(
        benchmark_names=TRAIN11,
        universe_benchmark_names=UNIVERSE,
        n_envs=24, n_steps=48, batch_size=144, n_epochs=15, ent_coef=0.01,
        ar_weight=1.0, dl_weight=1.0, pw_weight=1.0, wl_weight=0.0,
        max_episodes=13000, timesteps=100000, vtr_timeout=300, seed=42,
        save_path="runs/f3_no_gcn_seed42.zip",
        log_suffix="_f3_no_gcn_seed42",
        cache_suffix=CACHE_SUFFIX,        # noqa: F405
        use_wandb=False,
    )

    t0 = time.time()
    print(f"F3: training seed 42 WITHOUT the graph encoder. Pinned dims {PINNED}.", flush=True)
    trainer.train(cfg)
    hours = (time.time() - t0) / 3600.0
    (OUT / "logs" / "f3_train_wallclock.json").write_text(   # noqa: F405
        json.dumps({"training_wall_clock_hours": round(hours, 3),                 # noqa: F405
                    "checkpoint": "runs/f3_no_gcn_seed42.zip",
                    "source": "measured around trainer.train() in data_export3/tools/f3_train.py"},
                   indent=1))
    print(f"F3 training done in {hours:.2f} h", flush=True)


# SubprocVecEnv starts its 24 workers with forkserver, which re-imports this
# module in each worker; without this guard every worker would start training.
if __name__ == "__main__":
    main()
