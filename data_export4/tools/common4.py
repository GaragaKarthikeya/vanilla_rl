"""Shared setup for data_export4 drivers.

Reuses data_export3/tools/common3.py (and through it data_export/tools/common.py
and c345.py): VTR path fixes, pinned universe dims (48, 48, 67, 2694), CsvSink,
FPGAEnv construction, checkpoint loading, VTR replay. Only the output folder and
the cache namespace change: one isolated cache for this whole request,
runs/vtr_layout_cache_<circuit>_dataexport4.db.

Must be run inside `distrobox enter ubuntu-work`.
"""
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "data_export3" / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common3 import *  # noqa: F401,F403
import common3 as c3

# common3.make_env reads CACHE_SUFFIX from its own module globals at call time.
c3.CACHE_SUFFIX = "_dataexport4"
CACHE_SUFFIX = c3.CACHE_SUFFIX

OUT = _REPO / "data_export4"
(OUT / "logs").mkdir(parents=True, exist_ok=True)

ORIGINAL = ["custom_macbuf", "mkDelayWorker32B", "lightweight_cipher", "reduction_layer",
            "arm_core", "softmax"]
NEW = ["usb_uart_core", "uriscv_core", "lenet", "aes_inv_cipher", "8051", "stereovision1"]


def policy_after_forced_ar(model, env, ar_action):
    """Force step 0 to ar_action, then the policy places every H-block
    deterministically (same as data_export2/tools/e2.py:policy_blocks_after)."""
    obs, _ = env.reset()
    acts = [int(ar_action)]
    obs, _, term, _, _ = env.step(acts[0])
    assert not term
    while True:
        a, _ = model.predict(obs, action_masks=env.get_action_mask(), deterministic=True)
        acts.append(int(a))
        if env._current_step == env._total_blocks:
            return acts
        obs, _, term, _, _ = env.step(acts[-1])
        assert not term
