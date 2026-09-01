"""PANTERA paper figure style, ported from the PANTERA analysis module.

Okabe-Ito colorblind-safe palette, serif fonts with matching math text, thick
inward ticks on all four sides with minor ticks, frameless legends, no titles.
Base sizes are set for two-column composite panels rather than full-page
figures. Color key names are kept from the earlier module so the plotting
scripts run unchanged. Text stays editable in PDF exports (pdf.fonttype 42).
"""
import matplotlib as mpl

# Okabe-Ito palette under the existing key names
C = dict(
    black="#000000",
    blue="#0072B2",
    vermillion="#D55E00",
    green="#009E73",
    purple="#CC79A7",
    sky="#56B4E9",
    orange="#E69F00",
    yellow="#F0E442",
    grey="#999999",
)
CYCLE = [C["blue"], C["vermillion"], C["green"], C["purple"], C["orange"],
         C["sky"], C["black"]]


def apply(fs=8.5):
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 400, "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times", "serif"],
        "mathtext.fontset": "dejavuserif",
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": fs, "axes.labelsize": fs + 0.5,
        "axes.titlesize": fs, "axes.titleweight": "normal",
        "legend.fontsize": fs - 1, "legend.frameon": False,
        "xtick.labelsize": fs - 0.5, "ytick.labelsize": fs - 0.5,
        "axes.linewidth": 1.0,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.size": 4.5, "ytick.major.size": 4.5,
        "xtick.minor.size": 2.5, "ytick.minor.size": 2.5,
        "xtick.major.width": 1.0, "ytick.major.width": 1.0,
        "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.prop_cycle": mpl.cycler(color=CYCLE),
    })
