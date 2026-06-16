#!/usr/bin/env python3

import uuid
import shutil
from pathlib import Path
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.utils.cache import CacheRow, LayoutCache
from src.utils.config import VTRPaths
from src.evaluation.vtr_runner import VTRRunner

# Project root is two levels above this file (src/env/fpga_env.py → /)
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# DSP tile occupies 4 rows; BRAM tile occupies 6 rows
_BLOCK_HEIGHT = {1: 4, 2: 6}

# Predefined aspect ratios offered as the first action
ASPECT_RATIOS = [round(0.1 * i, 1) for i in range(1, 21)]  # 0.1 … 2.0


class FPGAEnv(gym.Env):
    """
    Gymnasium environment for heterogeneous FPGA block placement.

    Episode structure
    -----------------
    Step 0          : select an aspect ratio from ASPECT_RATIOS
    Steps 1 … N     : place each BRAM (sorted by nets) then each DSP (sorted by nets)
    Terminal step   : VTR evaluates the layout; reward is returned

    Observation space : Box(W, H, 4) float32
        ch0 — occupancy grid  (-1 = occupied tail, 0 = empty, 1 = DSP head, 2 = BRAM head)
        ch1 — block-type hint for the current step (3 during aspect-ratio step, 0 after done)
        ch2 — chosen aspect ratio value
        ch3 — net count for current block (normalized to [0, 1], -1.0 during aspect-ratio step)

    Action space : Discrete(W * H)
        Step 0 maps action index → ASPECT_RATIOS index.
        Other steps decode action as  x = 1 + action // H,  y = 1 + action % H  (1-indexed).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        benchmark_name: str = "diffeq1",
        width: int = 14,
        height: int = 14,
        req_dsp: int = 5,
        req_bram: int = 0,
        traditional_metrics: Optional[dict] = None,
        cache_db_path: Optional[str] = None,
        wl_weight: float = 0.10,
        pw_weight: float = 0.30,
        dl_weight: float = 0.60,
        ar_weight: float = 0.00,
        net_count_data: Optional[dict] = None,
        use_net_count_sort: bool = True,
        dsp_block_names: Optional[list[str]] = None,
        bram_block_names: Optional[list[str]] = None,
    ) -> None:
        super().__init__()

        self.benchmark_name = benchmark_name
        self.width = width
        self.height = height
        self.req_dsp = req_dsp
        self.req_bram = req_bram

        self.traditional_metrics: dict = traditional_metrics or {
            "delay_ns": 22.1348,
            "wirelength": 11437.0,
            "power_w": 0.007986,
            "routing_area": 835850.0,
        }

        db_path = cache_db_path or str(PROJECT_ROOT / "runs" / "vtr_layout_cache.db")
        self._cache = LayoutCache(db_path)

        self._vtr = VTRRunner(VTRPaths())

        self.wl_weight = wl_weight
        self.pw_weight = pw_weight
        self.dl_weight = dl_weight
        self.ar_weight = ar_weight

        # Named netlist block names for forced placement via VPR constraints.
        # Order must match the agent's placement sequence (DSPs first, then BRAMs).
        # When None, VPR's SA placer assigns blocks freely (original behaviour).
        self._dsp_block_names: list[str] = dsp_block_names or []
        self._bram_block_names: list[str] = bram_block_names or []

        # Store net count data for observation and debugging
        self._net_count_data = net_count_data or {}

        # Build placement sequence with optional net-count sorting
        # If net_count_data provided and use_net_count_sort=True:
        #   - BRAMs first, sorted by net count (highest first)
        #   - DSPs second, sorted by net count (highest first)
        # Otherwise: DSPs first, then BRAMs (original behavior)
        if use_net_count_sort and net_count_data:
            self._blocks_to_place, self._block_id_map = self._build_sorted_placement(
                req_dsp, req_bram, net_count_data
            )
        else:
            # Original order: DSPs first, then BRAMs. 1=DSP, 2=BRAM
            self._blocks_to_place = [1] * req_dsp + [2] * req_bram
            # Map: step_index -> actual instance_id (identity mapping)
            self._block_id_map = {i: i % req_dsp if i < req_dsp else i - req_dsp for i in range(req_dsp + req_bram)}

        self._total_blocks = len(self._blocks_to_place)

        self.action_space = spaces.Discrete(width * height)
        # Observation space now has 4 channels: occupancy, block_type, aspect_ratio, net_count
        self.observation_space = spaces.Box(
            low=-1.0, high=4.0, shape=(width, height, 4), dtype=np.float32
        )

        # Mutable episode state (initialised by reset)
        self._grid = np.zeros((width, height), dtype=np.int8)
        self._placed_dsps: list[tuple[int, int]] = []
        self._placed_brams: list[tuple[int, int]] = []
        self._current_step: int = 0
        self._chosen_aspect_ratio: float = 1.0

    # ------------------------------------------------------------------
    # Block placement ordering
    # ------------------------------------------------------------------

    def _build_sorted_placement(
        self, req_dsp: int, req_bram: int, net_count_data: dict
    ) -> tuple[list[int], dict[int, int]]:
        """
        Build placement sequence with blocks sorted by net count (highest first).

        Order: BRAMs (sorted by nets) → DSPs (sorted by nets)

        Args:
            req_dsp: Number of DSPs required
            req_bram: Number of BRAMs required
            net_count_data: Dict mapping ('TYPE', id) -> net_count

        Returns:
            (blocks_to_place, block_id_map) where:
            - blocks_to_place: list of block types (1=DSP, 2=BRAM) in order
            - block_id_map: maps step_index -> actual instance_id
        """
        # Extract and sort BRAMs by net count (descending)
        bram_blocks = []
        for i in range(req_bram):
            # Try both tuple and string key formats
            net_count = net_count_data.get(("BRAM", i), 0)
            if not net_count:
                net_count = net_count_data.get((f"('BRAM', {i})",), 0)
            bram_blocks.append((i, net_count))
        bram_blocks.sort(key=lambda x: x[1], reverse=True)

        # Extract and sort DSPs by net count (descending)
        dsp_blocks = []
        for i in range(req_dsp):
            # Try both tuple and string key formats
            net_count = net_count_data.get(("DSP", i), 0)
            if not net_count:
                net_count = net_count_data.get((f"('DSP', {i})",), 0)
            dsp_blocks.append((i, net_count))
        dsp_blocks.sort(key=lambda x: x[1], reverse=True)

        # Build placement sequence: BRAMs first, then DSPs
        blocks_to_place = [2] * req_bram + [1] * req_dsp
        block_id_map = {}

        # Map BRAM placement steps
        for step_idx, (actual_id, _) in enumerate(bram_blocks):
            block_id_map[step_idx] = actual_id

        # Map DSP placement steps
        for step_idx, (actual_id, _) in enumerate(dsp_blocks, start=req_bram):
            block_id_map[step_idx] = actual_id

        return blocks_to_place, block_id_map

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._grid[:] = 0
        self._placed_dsps = []
        self._placed_brams = []
        self._current_step = 0
        self._chosen_aspect_ratio = 1.0
        return self._obs(), {}

    def step(self, action: int):
        if self._current_step > self._total_blocks:
            return self._obs(), 0.0, True, False, self._terminal_info(success=False, status="already_completed")

        # --- Aspect-ratio selection (step 0) ---
        if self._current_step == 0:
            if not (0 <= action < len(ASPECT_RATIOS)):
                return self._obs(), -10.0, True, False, {"status": "invalid_aspect_ratio", **self._empty_eval_info()}
            self._chosen_aspect_ratio = ASPECT_RATIOS[action]
            self._current_step += 1
            return self._obs(), 0.0, False, False, {"status": "normal"}

        # --- Block placement (steps 1 … N) ---
        block_type = self._blocks_to_place[self._current_step - 1]
        bh = _BLOCK_HEIGHT[block_type]

        x = 1 + (action // self.height)
        y = 1 + (action % self.height)

        if not (1 <= x <= self.width) or not (1 <= y) or (y + bh - 1) > self.height:
            return self._obs(), -10.0, True, False, {"status": "out_of_bounds", **self._empty_eval_info()}

        if any(self._grid[x - 1, y - 1 + dy] != 0 for dy in range(bh)):
            return self._obs(), -10.0, True, False, {"status": "overlap", **self._empty_eval_info()}

        for dy in range(bh):
            self._grid[x - 1, y - 1 + dy] = -1
        self._grid[x - 1, y - 1] = block_type

        if block_type == 1:
            self._placed_dsps.append((x, y))
        else:
            self._placed_brams.append((x, y))

        self._current_step += 1

        if self._current_step > self._total_blocks:
            reward, info = self._evaluate_layout(self._chosen_aspect_ratio)
            return self._obs(), reward, True, False, info

        return self._obs(), 0.0, False, False, {"status": "normal"}

    # ------------------------------------------------------------------
    # Action mask (for MaskablePPO)
    # ------------------------------------------------------------------

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.width * self.height, dtype=np.int8)

        if self._current_step == 0:
            mask[: len(ASPECT_RATIOS)] = 1
        elif self._current_step <= self._total_blocks:
            block_type = self._blocks_to_place[self._current_step - 1]
            bh = _BLOCK_HEIGHT[block_type]
            for act in range(self.width * self.height):
                x = 1 + (act // self.height)
                y = 1 + (act % self.height)
                if not (1 <= x <= self.width):
                    continue
                if y < 1 or (y + bh - 1) > self.height:
                    continue
                if any(self._grid[x - 1, y - 1 + dy] != 0 for dy in range(bh)):
                    continue
                mask[act] = 1

        return mask

    def action_masks(self) -> np.ndarray:
        return self.get_action_mask().astype(bool)

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def _compute_reward(self, wl: float, pw: float, dl: float, routing_area: float) -> float:
        """Weighted log-ratio reward. Positive = better than baseline."""
        tm = self.traditional_metrics
        area_norm = max(routing_area, 1e-9) / max(tm.get("routing_area", 1.0), 1e-9)
        delay_norm = max(dl, 1e-9) / max(tm["delay_ns"], 1e-9)
        power_norm = max(pw, 1e-9) / max(tm["power_w"], 1e-9)
        wl_norm = max(wl, 1e-9) / max(tm.get("wirelength", 1.0), 1e-9)

        return float(
            -(
                self.ar_weight * np.log(area_norm)
                + self.dl_weight * np.log(delay_norm)
                + self.pw_weight * np.log(power_norm)
                + self.wl_weight * np.log(wl_norm)
            )
        )

    # ------------------------------------------------------------------
    # Layout evaluation (VTR)
    # ------------------------------------------------------------------

    def _evaluate_layout(self, aspect_ratio: float) -> tuple[float, dict]:
        info = self._base_eval_info(aspect_ratio)
        cache_key = self._cache_key(aspect_ratio)

        cached = self._cache.get(cache_key)
        if cached is not None:
            info["cached"] = True
            info["success"] = cached.success
            if cached.success:
                self._fill_success_info(info, cached)
                return self._compute_reward(cached.wirelength, cached.power_w, cached.delay_ns, cached.routing_area), info
            return -10.0, info

        # Bake + run VTR
        worker = uuid.uuid4().hex[:8]
        temp_arch = PROJECT_ROOT / f"temp_arch_{worker}.xml"
        temp_constraints = PROJECT_ROOT / f"temp_constraints_{worker}.xml"
        temp_run_dir = PROJECT_ROOT / "runs" / f"temp_run_{worker}"
        benchmark_file = PROJECT_ROOT / "benchmarks" / f"{self.benchmark_name}.v"

        all_block_names = self._dsp_block_names + self._bram_block_names

        try:
            from src.layout.baker import bake_layout
            result = bake_layout(
                benchmark_name=self.benchmark_name,
                dsps=self._placed_dsps,
                mems=self._placed_brams,
                width=self.width + 2,
                height=self.height + 2,
                output_path=str(temp_arch),
                aspect_ratio=aspect_ratio,
                block_names=all_block_names if all_block_names else None,
                constraints_output_path=str(temp_constraints) if all_block_names else None,
            )
            if result == -1:
                self._cache.put(cache_key, LayoutCache.failure_row())
                info["error"] = "bake_layout: invalid placement"
                return -10.0, info
        except Exception as exc:
            info["error"] = f"bake_layout error: {exc}"
            return -10.0, info

        if not self._vtr.paths.is_flow_available:
            info["error"] = "VTR flow script not found"
            return -10.0, info

        temp_run_dir.mkdir(parents=True, exist_ok=True)
        rc = self._vtr.run(
            benchmark_file, temp_arch, temp_run_dir,
            silent=True,
            constraints_file=temp_constraints if all_block_names else None,
        )

        vpr_out = temp_run_dir / "vpr.out"
        crit_path = temp_run_dir / "vpr.crit_path.out"
        power_file = temp_run_dir / f"{self.benchmark_name}.power"

        if rc == 0 and vpr_out.is_file():
            metrics = VTRRunner.parse_metrics(vpr_out, crit_path, power_file)
            resources = VTRRunner.parse_resources(vpr_out)

            if metrics.is_complete():
                grid_w = resources.fpga_size[0] + 2
                grid_h = resources.fpga_size[1] + 2
                row = CacheRow(
                    delay_ns=metrics.delay_ns,
                    wirelength=metrics.wirelength,
                    power_w=metrics.power_w,
                    routing_area=metrics.routing_area,
                    grid_w=grid_w,
                    grid_h=grid_h,
                    success=True,
                )
                self._cache.put(cache_key, row)
                self._fill_success_info(info, row)
                self._cleanup(temp_arch, temp_run_dir, temp_constraints)
                return self._compute_reward(metrics.wirelength, metrics.power_w, metrics.delay_ns, metrics.routing_area), info

        self._cache.put(cache_key, LayoutCache.failure_row())
        info["error"] = f"VTR failed (rc={rc})"
        self._cleanup(temp_arch, temp_run_dir, temp_constraints)
        return -10.0, info

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_current_block_net_count(self) -> float:
        """
        Get the net count for the block about to be placed (normalized to [0, 1]).

        Returns normalized net count in range [0, 1], or 0 if no data available.
        """
        if self._current_step == 0 or self._current_step > self._total_blocks:
            return 0.0

        step_idx = self._current_step - 1
        if step_idx >= len(self._blocks_to_place):
            return 0.0

        # Get the actual instance ID for this step
        actual_id = self._block_id_map.get(step_idx, step_idx)
        block_type = self._blocks_to_place[step_idx]
        block_type_str = "DSP" if block_type == 1 else "BRAM"

        # Try to find net count in the data dict
        # Try tuple format first (for parsed data)
        net_count = self._net_count_data.get((block_type_str, actual_id), 0)

        # If not found, try string key format
        if not net_count:
            for key, value in self._net_count_data.items():
                key_str = str(key)
                if f"{block_type_str}" in key_str and f", {actual_id})" in key_str:
                    net_count = value
                    break

        # Normalize to [0, 1] using 200 as ceiling
        return min(float(net_count) / 200.0, 1.0) if net_count else 0.0

    def _obs(self) -> np.ndarray:
        obs = np.zeros((self.width, self.height, 4), dtype=np.float32)
        obs[:, :, 0] = self._grid
        if self._current_step == 0:
            obs[:, :, 1] = 3  # aspect-ratio selection phase
            obs[:, :, 2] = -1.0
            obs[:, :, 3] = -1.0  # net count placeholder
        elif self._current_step <= self._total_blocks:
            obs[:, :, 1] = self._blocks_to_place[self._current_step - 1]
            obs[:, :, 2] = self._chosen_aspect_ratio
            obs[:, :, 3] = self._get_current_block_net_count()
        return obs

    def _cache_key(self, aspect_ratio: float) -> str:
        return (
            f"dsps:{sorted(self._placed_dsps)}"
            f"|brams:{sorted(self._placed_brams)}"
            f"|ratio:{aspect_ratio}"
        )

    def _base_eval_info(self, aspect_ratio: float) -> dict:
        return {
            "success": False,
            "cached": False,
            "placed_dsps": list(self._placed_dsps),
            "placed_brams": list(self._placed_brams),
            "aspect_ratio": aspect_ratio,
            "grid_W": "?",
            "grid_H": "?",
            "grid_clbs": "?",
            "routing_area": "?",
        }

    @staticmethod
    def _fill_success_info(info: dict, row: "CacheRow") -> None:
        info.update(
            {
                "success": True,
                "delay_ns": row.delay_ns,
                "wirelength": row.wirelength,
                "power_w": row.power_w,
                "routing_area": row.routing_area if row.routing_area > 0 else "?",
                "grid_W": row.grid_w if row.grid_w > 0 else "?",
                "grid_H": row.grid_h if row.grid_h > 0 else "?",
                "grid_clbs": (row.grid_w - 2) * (row.grid_h - 2) if row.grid_w > 0 else "?",
            }
        )

    def _terminal_info(self, success: bool, status: str) -> dict:
        return {
            "status": status,
            "placed_dsps": list(self._placed_dsps),
            "placed_brams": list(self._placed_brams),
            "success": success,
            "wirelength": float("inf"),
            "delay_ns": float("inf"),
            "power_w": float("inf"),
            "routing_area": float("inf"),
        }

    @staticmethod
    def _empty_eval_info() -> dict:
        return {
            "placed_dsps": [],
            "placed_brams": [],
            "success": False,
            "wirelength": float("inf"),
            "delay_ns": float("inf"),
            "power_w": float("inf"),
            "routing_area": float("inf"),
        }

    @staticmethod
    def _cleanup(arch_file: Path, run_dir: Path, constraints_file: Optional[Path] = None) -> None:
        try:
            if arch_file.is_file():
                arch_file.unlink()
            if constraints_file is not None and constraints_file.is_file():
                constraints_file.unlink()
            if run_dir.exists():
                shutil.rmtree(run_dir)
        except Exception:
            pass
