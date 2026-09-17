#!/usr/bin/env python3
"""fig_seed_variance.pdf regenerated for one ACM column (3.33 in) with every
text element >= 8 pt at that placed width.

Same data (copied verbatim from src/visualization/plot_seed_variance.py:30-82),
same layout (inline table Benchmark | Grid | (CLB, DSP, BRAM) left of a
dot/min-max/median strip chart, held-out group below in-pool group, rotated
group labels on the right, column header row, glyph legend, x axis 0..80).

The figure is drawn at exactly 3.33 in wide with NO bbox_inches="tight"
cropping, so 1 pt in the file == 1 pt on the page when placed at
width=\\columnwidth (3.33 in). Table columns are positioned in inches from the
figure's left edge (not in data units) so 8 pt text cannot collide.
"""
import sys
from pathlib import Path
from statistics import median as _median

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src" / "visualization"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from matplotlib.transforms import blended_transform_factory
import palette as PAL

BENCH_META = {
    "fifo":          ( "6×6",    4, 0,  1),
    "ch_intrinsics": ("10×10",  68, 0,  1),
    "diffeq2":       ("14×14",  27, 5,  0),
    "boundtop":      ("13×13",  46, 0,  1),
    "diffeq1":       ("14×14",  37, 5,  0),
    "spree":         ("12×12",  60, 1,  3),
    "mkSMAdapter4B": ("18×18", 156, 0,  5),
    "or1200":        ("25×25", 245, 1,  2),
    "mmc_core":      ("13×13", 128, 0,  1),
    "mkPktMerge":    ("26×26",  29, 0, 15),
    "raygentop":     ("17×17", 121, 6,  1),
    "softmax":           ("41×41", 1227, 8,  0),
    "reduction_layer":   ("42×42",  748, 0, 32),
    "cipher":            ("12×12",   40, 0,  3),
    "macbuf":            ("12×12",   14, 3,  1),
    "mkDelayWorker32B":  ("48×48",  464, 0, 43),
    "arm_core":          ("35×35",  868, 0, 24),
}
inpool = {
    "diffeq2": [68.34, 68.24, 69.08], "fifo": [60.20, 60.20, 56.29],
    "diffeq1": [48.45, 55.92, 53.01], "spree": [39.08, 38.89, 39.11],
    "mkPktMerge": [40.19, 37.78, 32.24], "ch_intrinsics": [36.25, 35.08, 36.25],
    "mkSMAdapter4B": [34.90, 35.51, 36.61], "boundtop": [33.84, 50.69, 21.33],
    "mmc_core": [24.42, 25.20, 25.44], "or1200": [5.75, 33.50, 12.48],
    "raygentop": [-2.14, -1.70, -2.20],
}
zeroshot = {
    "mkDelayWorker32B": [62.42, 54.88, 58.88], "macbuf": [47.76, 59.53, 59.09],
    "cipher": [30.22, 39.73, 30.51], "reduction_layer": [28.89, 21.51, 19.49],
    "arm_core": [6.99, 11.83, 8.52], "softmax": [5.65, 5.39, 10.77],
}

RL_BLUE, ZS_BLUE, REF_GRAY = PAL.TEAL, PAL.TERRA, PAL.ZERO
FS = 8.0  # minimum (and uniform) text size, pt, at 3.33 in

_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42, "font.size": FS,
})


def ordered(d):
    return sorted(d.items(), key=lambda kv: _median(kv[1]), reverse=True)


FIG_W, FIG_H = 3.33, 3.9
fig = plt.figure(figsize=(FIG_W, FIG_H))
_r = fig.canvas.get_renderer()


def w_in(txt, bold=False):
    t = fig.text(0, 0, txt, fontsize=FS, fontweight="bold" if bold else "normal")
    w = t.get_window_extent(_r).width / fig.dpi
    t.remove()
    return w


def disp(name):
    return name.replace("mkDelayWorker32B", "mkDelayW32B").replace("mkSMAdapter4B", "mkSMAdapt4B")


allrows = list(inpool) + list(zeroshot)
GAP = 0.07
X_NAME = 0.02
name_w = max(max(w_in(disp(n)) for n in allrows), w_in("Benchmark", True))
grid_w = max(max(w_in(BENCH_META[n][0]) for n in allrows), w_in("Grid", True))
X_GRID_L = X_NAME + name_w + GAP                     # grid column, left-aligned
clb_w = max(w_in(f"{BENCH_META[n][1]},") for n in allrows)
dsp_w = max(w_in(f"{BENCH_META[n][2]},") for n in allrows)
bram_w = max(w_in(f"{BENCH_META[n][3]})") for n in allrows)
par_w = w_in("(")
sp = w_in(" ")
X_PAR = X_GRID_L + grid_w + GAP
X_CLB_R = X_PAR + par_w + clb_w
X_DSP_R = X_CLB_R + sp + dsp_w
X_BRAM_R = X_DSP_R + sp + bram_w
trip_hdr_w = w_in("(CLB, DSP, BRAM)", True)
# header for the triple is left-aligned at X_PAR; table right edge is the max of both
TABLE_R = max(X_BRAM_R, X_PAR + trip_hdr_w)
XG = FIG_W - 0.10                                   # rotated group-label centre
AX_L, AX_R, AX_B, AX_T = TABLE_R + 0.10, XG - 0.14, 0.42, 3.55
print(f"columns(in): name={X_NAME:.2f}+{name_w:.2f} grid={X_GRID_L:.2f}+{grid_w:.2f} triple={X_PAR:.2f}..{TABLE_R:.2f} axes={AX_L:.2f}..{AX_R:.2f}")
ax = fig.add_axes([AX_L / FIG_W, AX_B / FIG_H, (AX_R - AX_L) / FIG_W, (AX_T - AX_B) / FIG_H])

gap = 1.6
y = 0
rows = []
for name, vals in reversed(ordered(zeroshot)):
    rows.append((y, name, vals, ZS_BLUE)); y += 1
zs_top = y - 1
y += gap
inpool_bot = y
for name, vals in reversed(ordered(inpool)):
    rows.append((y, name, vals, RL_BLUE)); y += 1
inpool_top = y - 1

ax.axvline(0, color=REF_GRAY, lw=0.7, ls=(0, (4, 3)), alpha=0.7, zorder=1)
for yy, name, vals, c in rows:
    lo, hi, med = min(vals), max(vals), _median(vals)
    ax.plot([lo, hi], [yy, yy], color=c, lw=0.8, alpha=0.45, zorder=2, solid_capstyle="round")
    ax.scatter(vals, [yy] * 3, s=9, color=c, alpha=0.85, zorder=3, edgecolor="white", linewidth=0.3)
    ax.plot([med, med], [yy - 0.28, yy + 0.28], color=c, lw=1.2, zorder=4, solid_capstyle="butt")
    if name == "raygentop":
        ax.plot([0, 0], [yy - 0.34, yy + 0.34], color=RL_BLUE, lw=1.3, zorder=5, solid_capstyle="butt", alpha=0.7)

ax.set_xlim(-6, 74)
ax.set_ylim(-0.7, inpool_top + 0.7)
ax.set_yticks([])
ax.set_xticks([0, 20, 40, 60])
ax.tick_params(axis="x", labelsize=FS, length=2.5, pad=1.5)
for sp_ in ("top", "right", "left"):
    ax.spines[sp_].set_visible(False)
# x label centred under the whole figure (axis is narrower than the label at 8 pt)
fig.text(FIG_W / 2, 0.03, "ADP reduction vs. baseline (%), per seed", fontsize=FS, ha="center", va="bottom",
         transform=fig.dpi_scale_trans)

T = blended_transform_factory(fig.dpi_scale_trans, ax.transData)
for yy, name, _v, _c in rows:
    grid, clb, dsp, bram = BENCH_META[name]
    kw = dict(fontsize=FS, va="center", transform=T, clip_on=False)
    fig.text(X_NAME, yy, disp(name), ha="left", **kw)
    fig.text(X_GRID_L, yy, grid, ha="left", **kw)
    fig.text(X_PAR, yy, "(", ha="left", **kw)
    fig.text(X_CLB_R, yy, f"{clb},", ha="right", **kw)
    fig.text(X_DSP_R, yy, f"{dsp},", ha="right", **kw)
    fig.text(X_BRAM_R, yy, f"{bram})", ha="right", **kw)

HDR_Y = inpool_top + 0.95
hk = dict(fontsize=FS, color=PAL.SUBINK, fontweight="bold", va="bottom", transform=T, clip_on=False)
fig.text(X_NAME, HDR_Y, "Benchmark", ha="left", **hk)
fig.text(X_GRID_L, HDR_Y, "Grid", ha="left", **hk)
fig.text(X_PAR, HDR_Y, "(CLB, DSP, BRAM)", ha="left", **hk)
fig.add_artist(plt.Line2D([X_NAME, FIG_W - 0.02], [inpool_top + 0.8] * 2, transform=T,
                          color=PAL.SUBINK, lw=0.5, alpha=0.5))
sep_y = (inpool_bot + zs_top) / 2
fig.add_artist(plt.Line2D([X_NAME, FIG_W - 0.02], [sep_y] * 2, transform=T,
                          color=PAL.SUBINK, lw=0.5, alpha=0.4, ls=(0, (4, 3))))

fig.text(XG, (inpool_bot + inpool_top) / 2, "In-pool (trained)", fontsize=FS, color=RL_BLUE, weight="bold",
         rotation=270, va="center", ha="center", transform=T)
fig.text(XG, zs_top / 2, "Held-out (zero-shot)", fontsize=FS, color=ZS_BLUE, weight="bold",
         rotation=270, va="center", ha="center", transform=T)

fig.text(FIG_W / 2, FIG_H - 0.03, "dots: 3 seeds (7/42/123)  |  tick: median  |  line: min–max",
         fontsize=FS, color=PAL.SUBINK, ha="center", va="top", transform=fig.dpi_scale_trans)

out = REPO / "data_export" / "fig_seed_variance.pdf"
fig.savefig(out)  # no tight bbox: page is exactly FIG_W x FIG_H inches
fig.savefig(REPO / "data_export" / "logs" / "fig_seed_variance_check.png", dpi=300)

# --- verification: every text >= 8 pt and fully inside the page
fig.canvas.draw()
r = fig.canvas.get_renderer()
bad = []
texts = [t for t in fig.findobj(matplotlib.text.Text) if t.get_text().strip()]
for t in texts:
    bb = t.get_window_extent(r)
    W, H = fig.bbox.width, fig.bbox.height
    if t.get_fontsize() < 8 or bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
        bad.append((t.get_text(), t.get_fontsize(), tuple(round(v, 1) for v in bb.extents)))
# pairwise overlap check among table/label texts
ext = [(t.get_text(), t.get_window_extent(r)) for t in texts]
ovl = [(a, b) for i, (a, ba) in enumerate(ext) for (b, bb) in ext[i + 1:] if ba.overlaps(bb)
       and ba.intersection(ba, bb).width > 0.5 and ba.intersection(ba, bb).height > 0.5]
print("texts", len(texts), "min_fontsize", min(t.get_fontsize() for t in texts), "page_in", fig.get_size_inches())
print("out_of_page_or_small", bad)
print("overlaps", ovl)
print("wrote", out)
