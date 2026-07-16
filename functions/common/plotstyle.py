"""Central and configurable plot style for all SCED figures.

A single :class:`PlotStyle` object carries ALL appearance settings (colors by role, legend
texts, title/axis templates, sizes, grid, band opacity). Each plotting function of
``SCED_plots`` / ``SCED_vaior`` / ``SCED_bayesian`` accepts ``style=None`` (= :data:`DEFAULT_STYLE`).
The reports (``report_sced_*``) and the templates expose ``STYLE`` to produce
publication-proof figures without touching the plotting code.

Usage (template) ::

    from functions.common.plotstyle import PlotStyle
    STYLE = PlotStyle(colors={"baseline": "#222", "treatment": "#d55"},
                      labels={"phase_a": "Baseline", "phase_b": "tDCS"},
                      ylabel="Motor score (FM-UE)", fontsize=12)
    # or, more simply, a dict:
    STYLE = {"colors": {"treatment": "#d55"}, "title": "My study"}

``style`` is accepted everywhere in 3 forms: ``None`` (defaults), ``dict`` (merged overrides),
or a ``PlotStyle`` instance. See :func:`resolve_style`.
"""
from dataclasses import dataclass, field, fields
import copy
import warnings

# Colors BY ROLE (stable semantics, independent of the plot type)
_DEFAULT_COLORS = {
    "baseline": "navy",              # segment/phase A (baseline)
    "treatment": "crimson",          # segment/phase B (intervention)
    "population": "#1f6f3f",         # pooled / population estimate (forest, bold)
    "responder": "#c0392b",          # responder case (forest)
    "nonresponder": "#888888",       # non-responder case (forest)
    "pooled_mean": "0.35",           # pooled mean +/- SE (event-time)
    "case": "0.3",                   # raw per-case series (stacked)
    "intervention_line": "crimson",  # vertical line marking intervention onset
    "zero_line": "crimson",          # zero line (forest / differences)
    "phase_mean": "steelblue",       # phase mean (stacked)
    # VAIOR - decision traffic light (points of the compared condition) + reference/band
    "vaior_pass": "#2ca02c",         # point beyond the band, favorable direction (green)
    "vaior_warn": "#e6b800",         # ambiguous (gold)
    "vaior_fail": "#d62728",         # unfavorable (red)
    "vaior_reference": "0.4",        # Theil-Sen trend + reference points
    # meta_forest (two-stage meta: raw vs shrunk)
    "meta_raw": "#9aa0a6",           # raw effect (stage 1)
    "meta_shrunk": "#1a73e8",        # credible shrunk effect (stage 2)
    "meta_shrunk_weak": "#7baaf7",   # non-credible shrunk effect
    "meta_diamond": "#0b6b3a",       # population diamond (mu)
    "meta_pi": "#cfd8dc",            # prediction interval band
}
_DEFAULT_LABELS = {
    "phase_a": "Phase A (baseline)",
    "phase_b": "Phase B (intervention)",
    "pooled_mean": "pooled mean +/- SE",
    "population": "Population",
}


@dataclass
class PlotStyle:
    """Appearance settings shared by all SCED figures. Each has a sensible default;
    only set what needs overriding. ``None`` on ``title``/``xlabel``/``ylabel``/
    ``figsize`` = keep the plot-specific default."""
    colors: dict = field(default_factory=lambda: dict(_DEFAULT_COLORS))   # by role (see _DEFAULT_COLORS)
    labels: dict = field(default_factory=lambda: dict(_DEFAULT_LABELS))   # legend texts
    title: str = None                 # title override (None = automatic plot title)
    xlabel: str = None                # X axis override (None = plot default)
    ylabel: str = None                # Y axis override (None = default, often the outcome name)
    fontsize: int = 11                # base size
    title_fontsize: int = None        # None = fontsize + 2
    legend_fontsize: int = None       # None = max(7, fontsize - 3)
    figsize: tuple = None             # None = default (varies by plot)
    dpi: int = 130                    # resolution (set 300/600 for submission)
    fmt: str = "png"                  # output format: "png" | "pdf" | "svg" | "eps" (vector = publication)
    transparent: bool = False         # transparent background (overlays / posters)
    grid: bool = True
    grid_alpha: float = 0.2
    band_alpha: float = 0.18          # opacity of HDI / 95% CI bands
    band_hatch: str = None            # band hatching (e.g. "////") for B&W printing (None = solid fill)
    integer_ticks: bool = False       # force integer ticks (both x AND y)
    decimal_comma: bool = False       # French decimal separator (comma) on the axes
    show_legend: bool = True
    legend_loc: str = "best"          # legend location (matplotlib loc)
    legend_ncol: int = 1
    legend_frameon: bool = True
    palette: str = "tab10"            # color cycle for multi-case series
    condition_colors: dict = None     # EXPLICIT mapping {condition/phase: color} (alternating/Brinley)
    linewidth: float = None           # width of model curves (None = plot default)
    markersize: float = None          # marker size (None = plot default)
    font_family: str = None           # font family (e.g. "serif", "Arial", "Helvetica")
    rcparams: dict = None             # matplotlib passthrough (wins over everything): font, widths, mathtext...
    post: object = None               # ESCAPE HATCH: callable(fig, ax) called just before saving
    #                                   (ax = main axis if unique, else None -> use fig.axes)

    def color(self, role, default=None):
        return self.colors.get(role, default)

    def label(self, key, default=None):
        return self.labels.get(key, default)

    def cond_color(self, cond, default=None):
        """Explicit color of a condition/phase if provided (otherwise ``default``)."""
        return (self.condition_colors or {}).get(cond, (self.condition_colors or {}).get(str(cond), default))

    def lw(self, default):
        return self.linewidth if self.linewidth is not None else default

    def ms(self, default):
        return self.markersize if self.markersize is not None else default

    @property
    def title_fs(self):
        return self.title_fontsize if self.title_fontsize is not None else self.fontsize + 2

    @property
    def legend_fs(self):
        return self.legend_fontsize if self.legend_fontsize is not None else max(7, self.fontsize - 3)

    def legend_kw(self, **extra):
        """kwargs for ``ax.legend(...)``: location, columns, frame, size (+ overrides)."""
        kw = dict(loc=self.legend_loc, ncol=self.legend_ncol, frameon=self.legend_frameon,
                  fontsize=self.legend_fs)
        kw.update({k: v for k, v in extra.items() if v is not None})
        return kw

    def band_kw(self, color):
        """``fill_between`` kwargs for an HDI/CI band of color ``color``: transparent solid fill OR hatching (B&W)."""
        if self.band_hatch:
            return dict(facecolor="none", edgecolor=color, hatch=self.band_hatch, lw=0.0, alpha=0.9)
        return dict(color=color, alpha=self.band_alpha, lw=0.0)

    def format_axes(self, ax, *, x=True, y=True):
        """Applies integer ticks and/or French decimal comma to ``ax`` (depending on the style)."""
        from matplotlib.ticker import MaxNLocator, FuncFormatter
        if self.integer_ticks:
            if x:
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if y:
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        if self.decimal_comma:
            fmt = FuncFormatter(lambda v, _pos: ("%g" % v).replace(".", ","))
            if x:
                ax.xaxis.set_major_formatter(fmt)
            if y:
                ax.yaxis.set_major_formatter(fmt)
        return ax

    def rc(self):
        """matplotlib rcParams to apply (base font + family + user passthrough)."""
        d = {"font.size": self.fontsize}
        if self.font_family:
            d["font.family"] = self.font_family
        if self.rcparams:
            d.update(self.rcparams)
        return d

    def apply_rc(self):
        """Applies :meth:`rc` to the global rcParams (consistent with the plots' existing style)."""
        import matplotlib.pyplot as plt
        plt.rcParams.update(self.rc())

    def finalize(self, fig, ax=None):
        """Applies the ``post(fig, ax)`` hook (matplotlib escape hatch) if defined.
        To be called just before :meth:`save`. ``ax`` = main axis (or None if multi-panel)."""
        if callable(self.post):
            self.post(fig, ax)

    def save(self, fig, save_path, base, ax=None):
        """``post`` hook then saves ``fig`` to ``<save_path>/<base>.<fmt>`` (dpi/transparency)."""
        import os
        self.finalize(fig, ax)
        os.makedirs(save_path, exist_ok=True)
        out = os.path.join(save_path, f"{base}.{self.fmt}")
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight", transparent=self.transparent)
        return out

    def merged(self, **over):
        """Deep copy with overrides; ``colors``/``labels``/``condition_colors`` are MERGED.
        Warns on any unknown field or any unknown key of ``colors``/``labels``
        (otherwise a silent typo); ``condition_colors``/``rcparams`` = free keys."""
        _fields = {f.name for f in fields(self)}
        new = copy.deepcopy(self)
        for k, v in over.items():
            if k not in _fields:
                warnings.warn(f"PlotStyle: unknown field '{k}' ignored (fields: {sorted(_fields)})",
                              stacklevel=2)
                continue
            if k in ("colors", "labels", "rcparams", "condition_colors") and isinstance(v, dict):
                if k == "colors":
                    _unknown = set(v) - set(_DEFAULT_COLORS)
                    if _unknown:
                        warnings.warn(f"PlotStyle.colors: unknown role(s) {sorted(_unknown)} "
                                      f"(known: {sorted(_DEFAULT_COLORS)})", stacklevel=2)
                elif k == "labels":
                    _unknown = set(v) - set(_DEFAULT_LABELS)
                    if _unknown:
                        warnings.warn(f"PlotStyle.labels: unknown key(s) {sorted(_unknown)} "
                                      f"(known: {sorted(_DEFAULT_LABELS)})", stacklevel=2)
                base = getattr(new, k)
                if base is None:
                    setattr(new, k, dict(v))
                else:
                    base.update(v)
            elif v is not None:
                setattr(new, k, v)
        return new

    @classmethod
    def preset(cls, name, **over):
        """Publication-proof preset style then overrides. Presets: ``"journal"`` (serif, PDF
        300 dpi, thin lines, frameless legend), ``"colorblind"`` (Wong palette + Dark2),
        ``"grayscale"`` (gray levels + Greys, PDF - B&W printing)."""
        if name not in _PRESETS:
            raise ValueError(f"unknown preset '{name}' (available: {sorted(_PRESETS)})")
        return DEFAULT_STYLE.merged(**_PRESETS[name]).merged(**over)


# Preset themes (see PlotStyle.preset). Wong 2011 = canonical colorblind-safe palette.
_PRESETS = {
    "journal": dict(font_family="serif", fmt="pdf", dpi=300, linewidth=1.4,
                    legend_frameon=False, grid_alpha=0.15,
                    rcparams={"axes.linewidth": 0.8, "axes.spines.top": False,
                              "axes.spines.right": False}),
    "colorblind": dict(palette="Dark2",
                       colors={"baseline": "#0072B2", "treatment": "#D55E00", "population": "#009E73",
                               "responder": "#CC79A7", "nonresponder": "#999999", "phase_mean": "#56B4E9",
                               "intervention_line": "#D55E00", "zero_line": "#D55E00",
                               "pooled_mean": "#000000", "case": "#56B4E9"}),
    "grayscale": dict(palette="Greys", fmt="pdf", dpi=300,
                      colors={"baseline": "#000000", "treatment": "#555555", "population": "#000000",
                              "responder": "#000000", "nonresponder": "#9a9a9a", "pooled_mean": "#222222",
                              "case": "#888888", "phase_mean": "#444444", "intervention_line": "#000000",
                              "zero_line": "#000000"}),
}

DEFAULT_STYLE = PlotStyle()


def resolve_style(style):
    """``None`` -> :data:`DEFAULT_STYLE`; ``str`` -> :meth:`PlotStyle.preset`; ``dict`` -> merged
    defaults; ``PlotStyle`` -> as is."""
    if style is None:
        return DEFAULT_STYLE
    if isinstance(style, PlotStyle):
        return style
    if isinstance(style, str):
        return PlotStyle.preset(style)
    if isinstance(style, dict):
        return DEFAULT_STYLE.merged(**style)
    raise TypeError(f"style must be None, str (preset), dict or PlotStyle (received {type(style).__name__})")
