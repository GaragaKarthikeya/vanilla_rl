#!/usr/bin/env python3
"""
Figure 3 (results + reliability): per-seed ADP reduction, in-pool vs zero-shot.

This figure REPLACES the former per-benchmark result tables. With only three
seeds, mean +/- SD overstates what the data establish, so we show the raw
seed outcomes directly ("show your seeds"): three dots per benchmark, a line
spanning min-max, and a median tick. The eye reads reliability from the line
length: in-pool fitting is fragile on exactly two benchmarks (or1200,
boundtop, shaded), while every held-out zero-shot benchmark is tight.

In-pool values are recovered exactly from the three training logs via
reduction% = (1 - exp(-best_reward))*100 (reward = log(ADP0/ADP)); zero-shot
values are the deterministic per-seed evals (det_seed{7,42,123}_*.json).
Output: paper/fig_seed_variance.pdf (vector).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontManager
from statistics import median as _median

RL_BLUE = "#1f5fa8"   # in-pool (trained)
ZS_BLUE = "#5b9bd5"   # held-out (zero-shot)
REF_GRAY = "#9a9a9a"
HILITE   = "#f2e2c4"

_avail = {f.name for f in FontManager().ttflist}
for _f in ("Times New Roman", "Nimbus Roman", "STIX Two Text", "DejaVu Serif"):
    if _f in _avail:
        plt.rcParams["font.serif"] = [_f]
        break
plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "stix",
    "axes.linewidth": 0.7, "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "pdf.fonttype": 42,
})

# name -> [seed7, seed42, seed123]
inpool = {
    "diffeq2": [68.34, 68.24, 69.08], "fifo": [60.20, 60.20, 56.29],
    "diffeq1": [48.45, 55.92, 53.01], "spree": [39.08, 38.89, 39.11],
    "mkPktMerge": [40.19, 37.78, 32.24], "ch_intrinsics": [36.25, 35.08, 36.25],
    "mkSMAdapter4B": [34.90, 35.51, 36.61], "boundtop": [33.84, 50.69, 21.33],
    "mmc_core": [24.42, 25.20, 25.44], "or1200": [5.75, 33.50, 12.48],
    "raygentop": [-2.14, -1.70, -2.20],
}
zeroshot = {
    "mkDelayWorker32B": [62.42, 54.88, 58.88], "custom_macbuf": [47.76, 59.53, 59.09],
    "lightweight_cipher": [30.22, 39.73, 30.51], "reduction_layer": [28.89, 21.51, 19.49],
    "arm_core": [6.99, 11.83, 8.52], "softmax": [5.65, 5.39, 10.77],
}
VOLATILE = {"or1200", "boundtop"}

def ordered(d):  # by median descending
    return sorted(d.items(), key=lambda kv: _median(kv[1]), reverse=True)

fig, ax = plt.subplots(figsize=(3.35, 4.3))
gap = 1.4
y = 0
rows, ticks, labels = [], [], []
for name, vals in reversed(ordered(zeroshot)):
    rows.append((y, name, vals, ZS_BLUE)); ticks.append(y); labels.append(name); y += 1
zs_top = y - 1
y += gap
inpool_bot = y
for name, vals in reversed(ordered(inpool)):
    rows.append((y, name, vals, RL_BLUE)); ticks.append(y); labels.append(name); y += 1
inpool_top = y - 1

for yy, name, vals, c in rows:
    if name in VOLATILE:
        ax.axhspan(yy - 0.45, yy + 0.45, color=HILITE, zorder=0)

ax.axvline(0, color=REF_GRAY, lw=0.8, ls=(0, (4, 3)), alpha=0.7, zorder=1)

for yy, name, vals, c in rows:
    lo, hi, med = min(vals), max(vals), _median(vals)
    lwr = 1.6 if name in VOLATILE else 1.0
    ax.plot([lo, hi], [yy, yy], color=c, lw=lwr, alpha=0.55, zorder=2,
            solid_capstyle="round")
    ax.scatter(vals, [yy] * 3, s=20, color=c, alpha=0.9, zorder=3,
               edgecolor="white", linewidth=0.4)
    ax.scatter([med], [yy], marker="|", s=95, color=c, linewidth=1.5, zorder=4)

ax.set_yticks(ticks)
ax.set_yticklabels(labels, fontsize=7.5)
for lab in ax.get_yticklabels():
    if lab.get_text() in VOLATILE:
        lab.set_fontweight("bold")

xr = 89
ax.text(xr, (inpool_bot + inpool_top) / 2, "In-pool\n(trained)",
        fontsize=8, color=RL_BLUE, weight="bold", rotation=270, va="center", ha="center")
ax.text(xr, zs_top / 2, "Held-out\n(zero-shot)",
        fontsize=8, color=ZS_BLUE, weight="bold", rotation=270, va="center", ha="center")

# legend-in-place: what the glyphs mean (one row, top-left empty space)
ax.annotate("dots: 3 seeds (7/42/123)   |: median   line: min–max",
            xy=(0, 0), xytext=(-7, inpool_top + 0.6), fontsize=6.6,
            color="#444444", va="bottom", ha="left", annotation_clip=False)

# punchline arrow at the volatile bars
y_by = {name: yy for yy, name, vals, c in rows}
ax.annotate("two benchmarks carry\nall in-pool seed spread",
            xy=(max(inpool["boundtop"]), y_by["boundtop"]),
            xytext=(54, y_by["mmc_core"]),
            fontsize=7.2, color="#8a6d2b", va="center", ha="left",
            arrowprops=dict(arrowstyle="-", color="#8a6d2b", lw=0.7, alpha=0.7,
                            connectionstyle="arc3,rad=0.2"))

ax.set_xlim(-8, 94)
ax.set_ylim(-0.8, inpool_top + 1.4)
ax.set_xlabel("ADP reduction vs. baseline (%), per seed", fontsize=8.3)
ax.set_xticks([0, 20, 40, 60, 80])
ax.tick_params(axis="x", labelsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

fig.tight_layout(pad=0.3)
out = "paper/fig_seed_variance.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
