#!/usr/bin/env python3
"""
FPGA layout visualizer — reads VTR .place + arch XML and renders a PNG.

Usage (single layout):
    python3 src/visualization/plot_layout.py \
        --place runs/diffeq2_traditional/diffeq2.place \
        --arch  runs/diffeq2_traditional/k6_frac_N10_mem32K_40nm.xml \
        --out   visualizations/trad.png \
        --title "diffeq2 Traditional"

Usage (side-by-side comparison):
    python3 src/visualization/plot_layout.py \
        --compare \
        --place  runs/diffeq2_traditional/diffeq2.place \
        --arch   runs/diffeq2_traditional/k6_frac_N10_mem32K_40nm.xml \
        --place2 runs/best_run_diffeq2/diffeq2.place \
        --arch2  runs/best_run_diffeq2/best_baked_layout_diffeq2.xml \
        --title  "diffeq2 — Traditional" --title2 "diffeq2 — RL Best" \
        --out    visualizations/diffeq2_comparison.png
"""

import argparse
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontManager

# serif to match the other paper figures (Fig 1, Fig 3) and the acmart body
_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "stix",
    "axes.linewidth": 0,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
})

# ── Palette (light, professional) ───────────────────────────────────────────
C = {
    # Background fabric
    "void":        "#ffffff",
    "io_empty":    "#eef1f5",
    "clb_empty":   "#f6f8fa",
    "dsp_empty":   "#e3ecfa",   # pale blue  — unoccupied DSP slot
    "bram_empty":  "#fbeaea",   # pale red   — unoccupied BRAM slot

    # Placed / active tiles
    "io_used":     "#d97706",   # amber   — I/O pad in use
    "clb_used":    "#15803d",   # green   — logic block in use
    "dsp_used":    "#1d4ed8",   # blue    — DSP in use
    "bram_used":   "#b91c1c",   # red     — BRAM in use

    # Edge colors (darker shade of each fill, for crisp definition)
    "io_edge":     "#92400e",
    "clb_edge":    "#14532d",
    "dsp_edge":    "#1e3a8a",
    "bram_edge":   "#7f1d1d",

    # Annotation
    "grid_line":   "#d0d7de",
    "bbox":        "#7c3aed",   # purple  — active-logic bounding box
    "text":        "#1f2328",
    "subtext":     "#57606a",
}

DSP_HEIGHT  = 4
BRAM_HEIGHT = 6


# ── Helpers ───────────────────────────────────────────────────────────────────

def _block_kind(name: str) -> str:
    """Return 'dsp', 'bram', 'io', or 'clb' from a VPR block name."""
    n = name.lower()
    if n.startswith("$mul") or "mult_36" in n:
        return "dsp"
    if "bram" in n or "mem[" in n or n.startswith("$bram"):
        return "bram"
    # IO pads from VPR are on the perimeter; classify anything else as clb
    return "clb"


def _block_height(kind: str) -> int:
    if kind == "dsp":  return DSP_HEIGHT
    if kind == "bram": return BRAM_HEIGHT
    return 1


def parse_place(path: Path) -> tuple[int, int, dict]:
    """Return (grid_w, grid_h, {name: {x, y, kind}})."""
    blocks = {}
    gw = gh = 0
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("Array size:"):
                parts = line.split()
                gw, gh = int(parts[2]), int(parts[4])
            if not line or line.startswith("#") or line.startswith("Netlist") or line.startswith("Array"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                name, x, y = parts[0], int(parts[1]), int(parts[2])
                blocks[name] = {"x": x, "y": y, "kind": _block_kind(name)}
            except ValueError:
                pass
    return gw, gh, blocks


def parse_arch(path: Path, gw: int, gh: int) -> list[list[str]]:
    """
    Build a gw×gh grid of tile types from the VPR architecture XML.
    DSP <single> tags also fill the subsequent DSP_HEIGHT-1 rows so the
    background fabric shows the full tall-tile footprint.
    """
    grid = [["clb_empty"] * gh for _ in range(gw)]

    if not path or not path.is_file():
        # Fallback: IO perimeter, CLB interior
        for x in range(gw):
            for y in range(gh):
                if x in (0, gw-1) or y in (0, gh-1):
                    grid[x][y] = "io_empty" if not (x in (0,gw-1) and y in (0,gh-1)) else "void"
                else:
                    grid[x][y] = "clb_empty"
        return grid

    tree = ET.parse(path)
    root = tree.getroot()
    layout = root.find("layout/auto_layout")
    if layout is None:
        layout = root.find("layout/fixed_layout")
    if layout is None:
        return grid

    children = sorted(layout, key=lambda c: int(c.get("priority", "0")))

    TYPE_MAP = {
        "io":      "io_empty",
        "clb":     "clb_empty",
        "mult_36": "dsp_empty",
        "memory":  "bram_empty",
        "EMPTY":   "void",
    }

    def cell(t):
        return TYPE_MAP.get(t, "clb_empty")

    for child in children:
        t = child.get("type", "")
        ct = cell(t)

        if child.tag == "fill":
            for x in range(gw):
                for y in range(gh):
                    grid[x][y] = ct

        elif child.tag == "perimeter":
            for x in range(gw):
                grid[x][0]    = ct
                grid[x][gh-1] = ct
            for y in range(gh):
                grid[0][y]    = ct
                grid[gw-1][y] = ct

        elif child.tag == "corners":
            for x in (0, gw-1):
                for y in (0, gh-1):
                    grid[x][y] = ct

        elif child.tag == "col":
            sx = int(child.get("startx", 0))
            rx = int(child.get("repeatx", gw))
            sy = int(child.get("starty", 0))
            x  = sx
            while x < gw:
                for y in range(sy, gh):
                    grid[x][y] = ct
                if rx <= 0:
                    break
                x += rx

        elif child.tag == "single":
            x = int(child.get("x", 0))
            y = int(child.get("y", 0))
            h = DSP_HEIGHT if t == "mult_36" else (BRAM_HEIGHT if t == "memory" else 1)
            for dy in range(h):
                if x < gw and y + dy < gh:
                    grid[x][y + dy] = ct

    return grid


# ── Rendering ─────────────────────────────────────────────────────────────────

def draw_panel(ax, gw: int, gh: int, grid: list[list[str]],
               blocks: dict, title: str) -> None:

    ax.set_facecolor(C["void"])
    ax.set_xlim(0, gw)
    ax.set_ylim(0, gh)
    ax.set_aspect("equal")
    ax.set_title(title, color=C["text"], fontsize=13, fontweight="bold", pad=10)

    # ── Background fabric ────────────────────────────────────────────────────
    for x in range(gw):
        for y in range(gh):
            color = C.get(grid[x][y], C["clb_empty"])
            ax.add_patch(mpatches.Rectangle(
                (x, y), 1, 1,
                facecolor=color, edgecolor=C["grid_line"], linewidth=0.3, zorder=1
            ))

    # ── Placed blocks ────────────────────────────────────────────────────────
    bbox_xs, bbox_ys = [], []

    # Draw CLBs first so tall blocks render on top
    draw_order = sorted(blocks.items(), key=lambda kv: 0 if kv[1]["kind"] == "clb" else 1)

    for name, blk in draw_order:
        x, y, kind = blk["x"], blk["y"], blk["kind"]
        h = _block_height(kind)

        if kind == "dsp":
            fc, ec, lw = C["dsp_used"],  C["dsp_edge"],  1.5
        elif kind == "bram":
            fc, ec, lw = C["bram_used"], C["bram_edge"], 1.5
        elif grid[x][y] == "io_empty":
            fc, ec, lw = C["io_used"],   C["io_edge"],   1.0
        else:
            fc, ec, lw = C["clb_used"],  C["clb_edge"],  0.6

        pad = 0.12
        ax.add_patch(mpatches.FancyBboxPatch(
            (x + pad, y + pad), 1 - 2*pad, h - 2*pad,
            boxstyle="round,pad=0.04",
            facecolor=fc, edgecolor=ec, linewidth=lw, alpha=0.92, zorder=2
        ))

        # Label DSPs/BRAMs
        if kind in ("dsp", "bram"):
            label = "D" if kind == "dsp" else "B"
            ax.text(x + 0.5, y + h / 2, label,
                    color="white", ha="center", va="center",
                    fontsize=7, fontweight="bold", zorder=3)

        # Bounding box only over core logic (not IO)
        if kind not in ("io",) and grid[x][y] != "io_empty":
            bbox_xs += [x, x + 1]
            bbox_ys += [y, y + h]

    # ── Active-logic bounding box ────────────────────────────────────────────
    if bbox_xs and bbox_ys:
        bx0, bx1 = min(bbox_xs), max(bbox_xs)
        by0, by1 = min(bbox_ys), max(bbox_ys)
        pad = 0.25
        ax.add_patch(mpatches.Rectangle(
            (bx0 - pad, by0 - pad), (bx1 - bx0) + 2*pad, (by1 - by0) + 2*pad,
            facecolor="none", edgecolor=C["bbox"],
            linewidth=1.8, linestyle="--", zorder=4
        ))
        ax.text(bx0 - pad, by1 + pad + 0.15,
                f"Active box  {int(bx1-bx0)} × {int(by1-by0)}",
                color=C["bbox"], fontsize=8, fontweight="bold", zorder=4)

    # Grid coordinates (sparse)
    for x in range(0, gw, 2):
        ax.text(x + 0.5, -0.35, str(x), ha="center", va="top",
                color=C["subtext"], fontsize=6)
    for y in range(0, gh, 2):
        ax.text(-0.35, y + 0.5, str(y), ha="right", va="center",
                color=C["subtext"], fontsize=6)

    ax.axis("off")


def make_legend():
    return [
        mpatches.Patch(facecolor=C["io_used"],   label="I/O pad (used)"),
        mpatches.Patch(facecolor=C["clb_used"],  label="CLB (used)"),
        mpatches.Patch(facecolor=C["dsp_used"],  label="DSP — height 4 (used)"),
        mpatches.Patch(facecolor=C["bram_used"], label="BRAM — height 6 (used)"),
        mpatches.Patch(facecolor=C["io_empty"],  label="I/O tile (empty)"),
        mpatches.Patch(facecolor=C["clb_empty"], label="CLB tile (empty)"),
        mpatches.Patch(facecolor=C["dsp_empty"], label="DSP slot (empty)"),
        mpatches.Patch(facecolor=C["bram_empty"],label="BRAM slot (empty)"),
        mpatches.Patch(facecolor="none", edgecolor=C["bbox"],
                       linestyle="--", linewidth=1.5, label="Active-logic bounding box"),
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VTR layout visualizer")
    ap.add_argument("--place",   required=True,  help=".place file (layout 1)")
    ap.add_argument("--arch",    default=None,   help="arch XML  (layout 1)")
    ap.add_argument("--title",   default="FPGA Layout")
    ap.add_argument("--compare", action="store_true", help="Side-by-side comparison mode (2 or 3 panels)")
    ap.add_argument("--place2",  default=None,   help=".place file (layout 2)")
    ap.add_argument("--arch2",   default=None,   help="arch XML  (layout 2)")
    ap.add_argument("--title2",  default="FPGA Layout 2")
    ap.add_argument("--place3",  default=None,   help=".place file (layout 3, optional — enables 3-panel mode)")
    ap.add_argument("--arch3",   default=None,   help="arch XML  (layout 3)")
    ap.add_argument("--title3",  default="FPGA Layout 3")
    ap.add_argument("--out",     required=True,  help="Output PNG path")
    args = ap.parse_args()

    plt.style.use("default")

    panels = [(args.place, args.arch, args.title)]
    if args.compare:
        panels.append((args.place2, args.arch2, args.title2))
        if args.place3:
            panels.append((args.place3, args.arch3, args.title3))

    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(11 * n, 11), dpi=300)
    fig.patch.set_facecolor(C["void"])
    if n == 1:
        axes = [axes]

    for ax, (place, arch, title) in zip(axes, panels):
        gw, gh, blks = parse_place(Path(place))
        grid = parse_arch(Path(arch) if arch else None, gw, gh)
        draw_panel(ax, gw, gh, grid, blks, title)

    fig.legend(handles=make_legend(), loc="lower center", ncol=3 if n == 1 else 5,
               fontsize=9.5, frameon=True, facecolor=C["void"],
               edgecolor=C["grid_line"], labelcolor=C["text"],
               bbox_to_anchor=(0.5, 0.0 if n == 1 else 0.01))
    plt.tight_layout(rect=[0, 0.1 if n == 1 else 0.07, 1, 1])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, facecolor=C["void"], bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
