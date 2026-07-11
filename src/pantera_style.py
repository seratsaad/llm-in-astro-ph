"""Shared plotting style matching the PANTERA letter figures:
DejaVu Serif fonts, warm palette (blue/green/orange), inward ticks on all four
sides with minor ticks, full box spines, light legend frame."""
import matplotlib as mpl

# Palette sampled from the PANTERA figures, mapped onto the keys the plot scripts use.
C = dict(
    blue="#0070B0",      # pantera blue
    green="#009070",     # pantera teal-green
    vermillion="#D05000",# pantera burnt orange (main highlight)
    orange="#E8A33D",    # amber (secondary warm)
    sky="#56A6D8",       # lighter blue
    purple="#8C5FA8",    # muted purple (4th/5th category)
    yellow="#F2C85B",    # soft band fill (ChatGPT-era shading)
    grey="#6F6F6F",
    black="#1A1A1A",
)

def apply():
    mpl.rcParams.update({
        "figure.dpi": 130, "savefig.dpi": 200,
        "font.family": "serif", "font.serif": ["DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "font.size": 12, "axes.labelsize": 12.5,
        "axes.titlesize": 13.5, "axes.titleweight": "bold",
        # full box, all four spines visible
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.linewidth": 1.1, "axes.edgecolor": "#1A1A1A",
        # inward ticks on all four sides, with minor ticks
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "xtick.major.size": 6, "ytick.major.size": 6,
        "xtick.minor.size": 3, "ytick.minor.size": 3,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.labelsize": 11, "ytick.labelsize": 11,
        # legend: light frame, serif
        "legend.frameon": True, "legend.framealpha": 0.9,
        "legend.edgecolor": "#BFBFBF", "legend.fancybox": False,
    })

def no_minor_x(ax):
    """Disable minor x ticks (for categorical / bar-label axes)."""
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())

def no_minor_y(ax):
    from matplotlib.ticker import NullLocator
    ax.yaxis.set_minor_locator(NullLocator())

apply()
