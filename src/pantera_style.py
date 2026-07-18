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
        # match the paper's 10 pt body text
        "font.size": 10, "axes.labelsize": 10,
        "axes.titlesize": 10.5, "axes.titleweight": "bold",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 9,
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


def place_labels(ax, xs, ys, names, colors=None, fontsize=6.5, priority=None,
                 avoid=None, logx=True, logy=True, obstacles=None):
    """Direct-label points Kobak-style: greedy candidate placement in normalized
    axes coordinates, leader lines when displaced, no label-label or
    label-point overlap. xs/ys in data coords; returns nothing (draws)."""
    import numpy as np
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    def nx(v): return (np.log10(v)-np.log10(x0))/(np.log10(x1)-np.log10(x0)) if logx else (v-x0)/(x1-x0)
    def ny(v): return (np.log10(v)-np.log10(y0))/(np.log10(y1)-np.log10(y0)) if logy else (v-y0)/(y1-y0)
    def ix(n): return 10**(np.log10(x0)+n*(np.log10(x1)-np.log10(x0))) if logx else x0+n*(x1-x0)
    def iy(n): return 10**(np.log10(y0)+n*(np.log10(y1)-np.log10(y0))) if logy else y0+n*(y1-y0)
    fw, fh = ax.figure.get_size_inches()
    bb = ax.get_position()
    axw_in, axh_in = fw*bb.width, fh*bb.height
    ch_w = fontsize*0.5/72/axw_in          # char width, axes fraction
    ln_h = fontsize*1.25/72/axh_in         # line height, axes fraction
    pts = [(nx(x), ny(y)) for x, y in zip(xs, ys)]
    if obstacles is not None:
        pts = pts + [(nx(x), ny(y)) for x, y in obstacles]
    order = range(len(names)) if priority is None else priority
    placed = list(avoid or [])             # boxes to avoid: (x0,y0,x1,y1) axes-frac
    ptr = 0.013                            # point clearance radius
    def box(nxl, nyl, w, ha):
        if ha == "left":  return (nxl, nyl-ln_h/2, nxl+w, nyl+ln_h/2)
        if ha == "right": return (nxl-w, nyl-ln_h/2, nxl, nyl+ln_h/2)
        return (nxl-w/2, nyl-ln_h/2, nxl+w/2, nyl+ln_h/2)
    def clash(b):
        if b[0] < 0.005 or b[2] > 0.995 or b[1] < 0.005 or b[3] > 0.995: return True
        for pb in placed:
            if not (b[2] < pb[0] or b[0] > pb[2] or b[3] < pb[1] or b[1] > pb[3]): return True
        for (px, py) in pts:
            if b[0]-ptr < px < b[2]+ptr and b[1]-ptr < py < b[3]+ptr: return True
        return False
    for i in order:
        p = pts[i]; w = len(names[i])*ch_w
        best = None
        for ext in (0.02, 0.04, 0.065, 0.095, 0.13):
            for dx, dy, ha in ((ext, 0, "left"), (-ext, 0, "right"), (0, ext, "center"),
                               (0, -ext, "center"), (ext, ext, "left"), (ext, -ext, "left"),
                               (-ext, ext, "right"), (-ext, -ext, "right")):
                b = box(p[0]+dx, p[1]+dy, w, ha)
                if not clash(b):
                    best = (p[0]+dx, p[1]+dy, ha, b, ext); break
            if best: break
        if best is None:
            continue                        # skip label rather than overlap
        lx, ly, ha, b, ext = best
        placed.append(b)
        col = (colors[i] if colors is not None else "#333333")
        if ext > 0.03:
            ax.plot([xs[i], ix(lx)], [ys[i], iy(ly)], lw=0.4, color="#999999", zorder=3)
        ax.annotate(names[i], (ix(lx), iy(ly)), fontsize=fontsize, ha=ha,
                    va="center", color=col, zorder=6)
