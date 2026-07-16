"""Figures for the longitudinal multilevel (coupling) pipeline - styled via PlotStyle.

- :func:`plot_rmcorr` - per-cluster scatter plus within-cluster lines at the COMMON slope (rmcorr).
- :func:`plot_slope_scatter` - slope(x)~time vs slope(y)~time, one point per cluster.
The forest plot comparing the coupling estimates reuses ``SCED_plots.plot_case_forest``.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_rmcorr(df, *, x, y, subject, slope=None, save_path=None, fname=None,
                title=None, max_lines=30, style=None):
    """Scatter of ``y`` vs ``x`` coloured by ``subject`` plus one line per cluster at the
    **common slope** ``slope`` (rmcorr), anchored on the cluster mean. If there are too many
    clusters (> ``max_lines``), only the points plus the global common-slope line are drawn.
    ``style`` -> PlotStyle."""
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    d = df[[x, y, subject]].dropna()
    subs = list(pd.unique(d[subject]))
    fig, ax = plt.subplots(figsize=(sty.figsize or (8, 6)))
    palette = plt.get_cmap(sty.palette)(np.linspace(0, 1, max(10, min(len(subs), max_lines))))
    draw_lines = len(subs) <= max_lines
    for i, s in enumerate(subs):
        g = d[d[subject] == s]
        col = palette[i % len(palette)]
        ax.scatter(g[x], g[y], s=sty.ms(26), color=col, alpha=0.55, edgecolor="none", zorder=2)
        if draw_lines and slope is not None and len(g) >= 2:
            xm, ym = g[x].mean(), g[y].mean()
            xs = np.array([g[x].min(), g[x].max()])
            ax.plot(xs, ym + slope * (xs - xm), "-", color=col, lw=sty.lw(1.0), alpha=0.7, zorder=1)
    if slope is not None:                                    # global common-slope line (bold)
        xm, ym = d[x].mean(), d[y].mean()
        xs = np.array([d[x].min(), d[x].max()])
        ax.plot(xs, ym + slope * (xs - xm), "-", color=sty.color("population"), lw=sty.lw(2.6),
                zorder=4, label=f"common slope = {slope:.3f}")
        if sty.show_legend:
            ax.legend(**sty.legend_kw())
    ax.set_xlabel(sty.xlabel or x); ax.set_ylabel(sty.ylabel or y)
    ax.set_title(sty.title or title or f"rmcorr - {y} ~ {x} (within-{subject})", fontsize=sty.title_fs)
    ax.grid(sty.grid, alpha=sty.grid_alpha); sty.format_axes(ax)
    fig.tight_layout()
    if save_path is not None:
        sty.save(fig, save_path, fname or f"{y}_rmcorr", ax=ax)
    return fig


def plot_slope_scatter(slopes, *, save_path=None, fname=None, title=None, xlab="slope x ~ time",
                       ylab="slope y ~ time", style=None):
    """Scatter of slope(x) vs slope(y), one point per cluster (``slopes`` = DataFrame
    slope_x/slope_y) plus an OLS line and an annotated Pearson r. ``style`` -> PlotStyle."""
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    s = slopes.dropna(subset=["slope_x", "slope_y"])
    fig, ax = plt.subplots(figsize=(sty.figsize or (6.5, 6)))
    ax.scatter(s["slope_x"], s["slope_y"], s=sty.ms(50), color=sty.color("treatment"),
               alpha=0.8, edgecolor="white", linewidth=0.6, zorder=3)
    if len(s) >= 3 and s["slope_x"].var() > 0:
        b, a = np.polyfit(s["slope_x"], s["slope_y"], 1)
        xs = np.array([s["slope_x"].min(), s["slope_x"].max()])
        ax.plot(xs, a + b * xs, "-", color=sty.color("population"), lw=sty.lw(2.2), zorder=2)
        r = float(np.corrcoef(s["slope_x"], s["slope_y"])[0, 1])
        ax.text(0.02, 0.98, f"r = {r:.3f}  (n={len(s)})", transform=ax.transAxes, va="top",
                fontsize=sty.legend_fs, bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    ax.axhline(0, color="0.6", lw=0.8, ls=":"); ax.axvline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(sty.xlabel or xlab); ax.set_ylabel(sty.ylabel or ylab)
    ax.set_title(sty.title or title or "Slope-vs-slope (dynamics coupling)", fontsize=sty.title_fs)
    ax.grid(sty.grid, alpha=sty.grid_alpha); sty.format_axes(ax)
    fig.tight_layout()
    if save_path is not None:
        sty.save(fig, save_path, fname or "slope_vs_slope", ax=ax)
    return fig


def plot_trajectory(df, *, time, outcome, group, marginal=None, by=None, save_path=None,
                    fname=None, title=None, max_lines=60, style=None):
    """Per-subject spaghetti (light) plus the **marginal trajectory** (bold) with a 95% CI band
    if ``marginal`` (output of :func:`marginal_trajectory`) is given. ``by`` colours by group.
    ``style`` -> PlotStyle."""
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    fig, ax = plt.subplots(figsize=(sty.figsize or (8.5, 5.6)))
    by_levels = sorted(df[by].dropna().unique(), key=str) if by is not None else [None]
    pal = plt.get_cmap(sty.palette)(np.linspace(0, 1, max(2, len(by_levels))))
    cmap = {lev: pal[i] for i, lev in enumerate(by_levels)}
    subs = list(pd.unique(df[group]))
    for s in subs[:max_lines]:
        g = df[df[group] == s].sort_values(time)
        lev = g[by].iloc[0] if by is not None else None
        ax.plot(g[time], g[outcome], "-", color=(sty.cond_color(lev, cmap.get(lev)) if by else sty.color("case")),
                lw=sty.lw(0.7), alpha=0.30, zorder=1)
    if marginal is not None:
        for lev in by_levels:
            m = marginal if by is None else marginal[marginal[by] == lev]
            m = m.sort_values(time)
            col = sty.cond_color(lev, cmap.get(lev)) if by else sty.color("treatment")
            ax.fill_between(m[time], m["lo"], m["hi"], zorder=2, **sty.band_kw(col))
            ax.plot(m[time], m["pred"], "-", color=col, lw=sty.lw(2.8), zorder=4,
                    label=(str(lev) if by is not None else "marginal trajectory ± CI95"))
        if sty.show_legend:
            ax.legend(title=(by if by is not None else None), **sty.legend_kw())
    ax.set_xlabel(sty.xlabel or time); ax.set_ylabel(sty.ylabel or outcome)
    ax.set_title(sty.title or title or f"Trajectory - {outcome} ~ {time}", fontsize=sty.title_fs)
    ax.grid(sty.grid, alpha=sty.grid_alpha); sty.format_axes(ax)
    fig.tight_layout()
    if save_path is not None:
        sty.save(fig, save_path, fname or f"{outcome}_trajectory", ax=ax)
    return fig


def plot_blup_caterpillar(blup, *, save_path=None, fname=None, title=None, unit="", style=None):
    """Caterpillar of the BLUPs (per-subject random effects, sorted) - ``blup`` = output of
    :func:`blup_table` (columns group/blup). ``style`` -> PlotStyle."""
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    b = blup.sort_values("blup").reset_index(drop=True)
    n = len(b)
    fig, ax = plt.subplots(figsize=(sty.figsize or (7.5, 0.18 * n + 1.4)))
    ax.scatter(b["blup"], np.arange(n), s=sty.ms(24), color=sty.color("population"),
               edgecolor="black", linewidth=0.4, zorder=3)
    ax.axvline(0, color=sty.color("zero_line"), ls="--", lw=1, zorder=1)
    ax.set_yticks([])
    ax.set_xlabel(sty.xlabel or f"random effect (BLUP){(' ' + unit) if unit else ''}")
    ax.set_ylabel(f"subjects (n={n}, sorted)")
    ax.set_title(sty.title or title or "Caterpillar of random effects (BLUP)", fontsize=sty.title_fs)
    ax.grid(sty.grid, axis="x", alpha=sty.grid_alpha); sty.format_axes(ax, y=False)
    fig.tight_layout()
    if save_path is not None:
        sty.save(fig, save_path, fname or "blup_caterpillar", ax=ax)
    return fig
