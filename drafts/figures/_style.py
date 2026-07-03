"""Shared style for P4 figures (publication quality, Pattern Recognition).

Palette = dataviz skill's validated brand-neutral default (colorblind-safe,
worst adjacent CVD dE = 24.2 on light surface). Categorical hues assigned in
fixed slot order; ink is primary/secondary/muted, never the series color.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---- Surfaces & ink (light mode, print) ----
SURFACE    = "#fcfcfb"
INK_PRIM   = "#0b0b0b"
INK_SEC    = "#52514e"
INK_MUTED  = "#898781"
GRIDLINE   = "#e1e0d9"
BASELINE   = "#c3c2b7"
RING       = "rgba(11,11,11,0.10)"  # not used directly in mpl

# ---- Categorical palette (fixed order; never cycle) ----
CAT = {
    "blue":    "#2a78d6",
    "aqua":    "#1baf7a",
    "yellow":  "#eda100",
    "green":   "#008300",
    "violet":  "#4a3aa7",
    "red":     "#e34948",
    "magenta": "#e87ba4",
    "orange":  "#eb6834",
}
CAT_ORDER = ["blue", "aqua", "yellow", "green", "violet", "red", "magenta", "orange"]

# Status (reserved)
STATUS = {"good": "#0ca30c", "warning": "#fab219",
          "serious": "#ec835a", "critical": "#d03b3b"}


def apply_rc(fontsize=9):
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": fontsize,
        "axes.titlesize": fontsize + 1,
        "axes.labelsize": fontsize,
        "xtick.labelsize": fontsize - 1,
        "ytick.labelsize": fontsize - 1,
        "legend.fontsize": fontsize - 1,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK_SEC,
        "xtick.color": INK_SEC,
        "ytick.color": INK_SEC,
        "text.color": INK_PRIM,
        "axes.titlecolor": INK_PRIM,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "grid.color": GRIDLINE,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "legend.borderaxespad": 0.0,
        "legend.labelspacing": 0.4,
        "legend.handletextpad": 0.5,
    })


def style_axes(ax, grid_axis="y"):
    """Recessive axes: thin baseline, hairline grid only on the meaningful axis."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
        ax.spines[s].set_linewidth(0.9)
    ax.tick_params(length=3, width=0.8, colors=INK_SEC)
    if grid_axis in ("y", "both"):
        ax.yaxis.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    if grid_axis in ("x", "both"):
        ax.xaxis.grid(True, color=GRIDLINE, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)


# single-column ~3.5", double-column ~7"
SINGLE_COL = 3.54
DOUBLE_COL = 7.09
