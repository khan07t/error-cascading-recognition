"""One chart style for every figure in the repo."""

import matplotlib.pyplot as plt

# categorical slots, always used in this order
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
MUTED = "#b9b8b3"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#ffffff"

def use_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": GRID, "axes.linewidth": 0.8, "axes.labelcolor": TEXT_2,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlecolor": TEXT,
        "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "xtick.major.size": 0, "ytick.major.size": 0,
        "legend.frameon": False, "legend.labelcolor": TEXT_2,
        "lines.linewidth": 2, "lines.markersize": 6,
        "savefig.dpi": 160, "savefig.bbox": "tight",
    })

def subtitle(ax, text):
    ax.set_title(ax.get_title(loc="left"), loc="left", pad=24)
    ax.text(0, 1.02, text, transform=ax.transAxes, color=TEXT_2, fontsize=9, va="bottom")

def label_bars(ax, bars, fmt="{:.2f}", inside=False):
    for b in bars:
        v = b.get_height()
        ax.annotate(fmt.format(v), (b.get_x() + b.get_width() / 2, v), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8.5, color=TEXT_2)
