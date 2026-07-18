"""SCI-journal figure style, following xiao-yuling/sci-figure.

Times New Roman with Times-compatible math text; all four spines visible but
outward ticks only on the left and bottom; no minor ticks; thin 0.8 pt axes;
compact 8 pt base text; a restrained palette that survives grayscale review.
Text is kept editable in PDF/SVG exports (pdf.fonttype 42, svg.fonttype none).
The module name is kept for backward compatibility with the plot scripts.
"""
import matplotlib as mpl

# Restrained palette: a neutral ink base, one cool signal, one warm accent, and
# greys. Colors are spaced in lightness so they separate in grayscale. Red/green
# are used only where the meaning is directional (gain/loss, real/style).
C = dict(
    black="#1A1A1A",       # ink / primary
    blue="#2F5A87",        # cool signal
    vermillion="#B0492E",  # warm accent
    green="#4E7D5A",       # secondary (directional: "real astronomy")
    purple="#6E5A8A",
    sky="#7FA6C9",         # light cool
    orange="#CE8A3B",      # light warm
    yellow="#D8C56B",
    grey="#7A7A7A",
)

def apply():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400,
        "savefig.bbox": "tight",
        # Times New Roman, with Times math text
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "svg.fonttype": "none", "pdf.fonttype": 42,
        # compact type
        "font.size": 8.5, "axes.labelsize": 8.5,
        "axes.titlesize": 9, "axes.titleweight": "bold",
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        # full box, thin
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.linewidth": 0.8, "axes.edgecolor": "#1A1A1A",
        # outward ticks, left and bottom only, no minor ticks
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.top": False, "ytick.right": False,
        "xtick.minor.visible": False, "ytick.minor.visible": False,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        # restrained lines
        "lines.linewidth": 1.2, "lines.markersize": 3.5,
        "legend.frameon": False, "legend.handlelength": 1.6,
    })

def no_minor_x(ax):
    from matplotlib.ticker import NullLocator
    ax.xaxis.set_minor_locator(NullLocator())

def no_minor_y(ax):
    from matplotlib.ticker import NullLocator
    ax.yaxis.set_minor_locator(NullLocator())

apply()
