"""
SCED - Visualisation (alternating treatments / N-of-1, single & group)
======================================================================

The figure IS part of the analysis in single-case research (visual analysis is a
primary standard, e.g. What Works Clearinghouse). ``plot_sced_alternating`` draws the
conventional view: outcome against session in time order, points coloured by
condition, with a dashed mean line per condition. For a group it draws one small panel
per unit. Returns the matplotlib Figure (and saves a PNG if ``save_path`` is given);
it never calls ``show`` so it is safe headless.
"""
import itertools
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter


# TODO(SCED, structured visual analysis): the current plots show raw points + per-phase
# mean lines only. Add the **Dual-Criteria (DC) and Conservative Dual-Criteria (CDC)**
# criterion lines (Fisher, Kelley & Lomas 2003, JABA): overlay the baseline split-middle
# trend line and the baseline-mean line (CDC raised by 0.25 SD), then count treatment
# points above both. CDC is the only method validated to control Type I error across all
# autocorrelation levels while keeping power for typical SCED phase lengths - it would
# turn the figure into an inferential visual-analysis tool, not just a descriptive plot.


def _panel(ax, g, session_col, condition_col, outcome_col, conditions, colors, title,
           *, condition_line=True):
    """Draw the classic SCED time-series panel on ``ax``: outcome against session in time
    order, points coloured by condition, a grey time-order line, and (if ``condition_line``)
    a dashed per-condition mean line. Mutates ``ax`` in place; returns nothing."""
    g = g.sort_values(session_col)
    s = pd.to_numeric(g[session_col], errors="coerce")
    y = pd.to_numeric(g[outcome_col], errors="coerce")
    ax.plot(s, y, color="0.7", lw=1.4, zorder=1)                     # time-order line
    for c in conditions:
        m = g[condition_col] == c
        ax.scatter(s[m], y[m], color=colors[c], s=45, zorder=5, label=str(c),
                   edgecolor="white", linewidth=0.8)
        if m.any() and condition_line:                              # per-condition mean line
            ax.axhline(y[m].mean(), color=colors[c], ls="--", lw=2, alpha=0.55, zorder=4)
    ax.set_title(title, fontsize=13)
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))           # sessions = integers


def _panel_dist(ax, g, condition_col, outcome_col, conditions, colors, title, *,
                kind, center, show_points, rng):
    """DISTRIBUTION view per condition (not a time series): box / violin / central point /
    point +/- SD, with central tendency mean or median, and an optional "every value" jittered
    cloud overlay."""
    data = [pd.to_numeric(g[g[condition_col] == c][outcome_col], errors="coerce").dropna().to_numpy()
            for c in conditions]
    x = np.arange(1, len(conditions) + 1)
    if kind == "box":
        bp = ax.boxplot(data, positions=x, widths=0.6, patch_artist=True,
                        showmeans=(center == "mean"),
                        meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black"),
                        medianprops=dict(color="black", lw=2))
        for patch, c in zip(bp["boxes"], conditions):
            patch.set_facecolor(colors[c]); patch.set_alpha(0.45)
    elif kind == "violin":
        vp = ax.violinplot(data, positions=x, showmeans=(center == "mean"),
                           showmedians=(center == "median"))
        for body, c in zip(vp["bodies"], conditions):
            body.set_facecolor(colors[c]); body.set_alpha(0.4)
    elif kind in ("mean", "mean_sd"):
        for xi, c, y in zip(x, conditions, data):
            if not len(y):
                continue
            ctr = float(np.mean(y) if center == "mean" else np.median(y))
            if kind == "mean_sd":
                ax.errorbar(xi, ctr, yerr=(float(np.std(y, ddof=1)) if len(y) > 1 else 0.0),
                            fmt="o", color=colors[c], ms=12, capsize=8, elinewidth=2)
            else:
                ax.plot(xi, ctr, "o", color=colors[c], ms=14)
    if show_points:                                   # "every value" as a cloud (jitter)
        for xi, c, y in zip(x, conditions, data):
            if len(y):
                ax.scatter(xi + rng.uniform(-0.12, 0.12, len(y)), y, color=colors[c], s=22,
                           alpha=0.5, zorder=6, edgecolor="white", linewidth=0.4)
    ax.set_xticks(x); ax.set_xticklabels([str(c) for c in conditions])
    ax.set_title(title, fontsize=13); ax.grid(True, axis="y", alpha=0.25)


def _panel_by_session(ax, g, session_col, condition_col, outcome_col, conditions, colors,
                      title, *, kind, center, show_points, rng, condition_line=True):
    """Box / violin / point-+/-SD **along the sessions** (standard time view): one box per
    session, coloured by the session's condition; >=2 values/session required."""
    g = g.sort_values(session_col)
    sv = pd.to_numeric(g[session_col], errors="coerce")
    sessions = sorted(sv.dropna().unique().tolist())
    data, cols = [], []
    for s in sessions:
        sub = g[sv == s]
        data.append(pd.to_numeric(sub[outcome_col], errors="coerce").dropna().to_numpy())
        cond = sub[condition_col].iloc[0] if len(sub) else None
        cols.append(colors.get(cond, "0.5"))
    pos = list(range(1, len(sessions) + 1))
    if kind == "box":
        bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True,
                        showmeans=(center == "mean"),
                        meanprops=dict(marker="D", markerfacecolor="black", markeredgecolor="black"),
                        medianprops=dict(color="black", lw=2))
        for patch, col in zip(bp["boxes"], cols):
            patch.set_facecolor(col); patch.set_alpha(0.45)
    elif kind == "violin":
        nz = [i for i, y in enumerate(data) if len(y) >= 2]
        if nz:
            vp = ax.violinplot([data[i] for i in nz], positions=[pos[i] for i in nz],
                               showmeans=(center == "mean"), showmedians=(center == "median"))
            for body, i in zip(vp["bodies"], nz):
                body.set_facecolor(cols[i]); body.set_alpha(0.4)
    elif kind in ("mean", "mean_sd"):
        ctr = [float(np.mean(y) if center == "mean" else np.median(y)) if len(y) else np.nan
               for y in data]
        ax.plot(pos, ctr, color="0.7", lw=1.5, zorder=1)              # time line
        for xi, y, col in zip(pos, data, cols):
            if not len(y):
                continue
            c0 = float(np.mean(y) if center == "mean" else np.median(y))
            if kind == "mean_sd":
                ax.errorbar(xi, c0, yerr=(float(np.std(y, ddof=1)) if len(y) > 1 else 0.0),
                            fmt="o", color=col, ms=10, capsize=5, elinewidth=2, zorder=4)
            else:
                ax.plot(xi, c0, "o", color=col, ms=12, zorder=4)
    # horizontal reference line PER CONDITION (global level), optional
    if condition_line:
        for c in conditions:
            yc = pd.to_numeric(g[g[condition_col] == c][outcome_col], errors="coerce").dropna()
            if len(yc):
                lvl = float(yc.mean() if center == "mean" else yc.median())
                ax.axhline(lvl, color=colors[c], ls="--", lw=2.5, alpha=0.5, zorder=2)
    if show_points:
        for xi, y, col in zip(pos, data, cols):
            if len(y):
                ax.scatter(xi + rng.uniform(-0.12, 0.12, len(y)), y, color=col, s=18,
                           alpha=0.5, zorder=6, edgecolor="white", linewidth=0.4)
    ax.set_xticks(pos)
    ax.set_xticklabels([str(int(s)) if float(s).is_integer() else str(s) for s in sessions])
    ax.set_title(title, fontsize=20); ax.grid(True, axis="y", alpha=0.25)
    ax.legend(handles=[plt.Line2D([0], [0], marker="s", ls="", color=colors[c], label=str(c))
                       for c in conditions], title="Condition", fontsize=8)


def plot_sced_alternating(df, *, session_col, condition_col, outcome_col, unit_col=None,
                          conditions=None, title=None, save_path=None, ncols=3,
                          kind="series", center="mean", show_points=True, by="session",
                          condition_line=True, bounds=None, style=None):
    """
    Plot an alternating-treatments / N-of-1 SCED design (single unit, or a grid of
    per-unit panels for a group). Returns the Figure.

    ``by`` selects the AXIS (and hence the view):
      - ``"session"`` (default): **standard time view** (x = session, time order) - keeps the
        classic SCED reading;
      - ``"condition"``: grouped **by condition** (x = condition, pooled distribution).

    ``kind`` selects the VIEW when there are several values:
      - ``"series"``: every value (cloud) - always temporal (ignores ``by``);
      - ``"box"`` / ``"violin"``: box / violin (per session if ``by="session"``, otherwise per
        condition);
      - ``"mean"``: central-tendency point; ``"mean_sd"``: point +/- standard deviation.
    ``center`` in {``"mean"``, ``"median"``}; ``show_points`` overlays **every value** (cloud).
    ``condition_line`` (default True) draws a **horizontal per-condition mean (or median)
    line**; set ``False`` to hide it.
    """
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style)
    if conditions is None:
        conditions = sorted(df[condition_col].dropna().unique().tolist(), key=str)
    palette = plt.get_cmap(sty.palette)(np.linspace(0, 1, max(10, len(conditions))))
    colors = {c: sty.cond_color(c, palette[i]) for i, c in enumerate(conditions)}   # explicit mapping possible
    sub = df[df[condition_col].isin(conditions)]

    # --- FONT: large characters (poster), DERIVED from fontsize (respects STYLE) ---
    _fs = sty.fontsize
    plt.rcParams.update({"axes.titlesize": _fs * 2, "axes.labelsize": _fs * 1.8,
                         "xtick.labelsize": _fs * 1.5, "ytick.labelsize": _fs * 1.5,
                         "legend.fontsize": _fs * 1.5})
    sty.apply_rc()                                          # font/family + user passthrough (wins)
    rng = np.random.RandomState(0)
    xlab = "Condition" if (kind != "series" and by == "condition") else "Session"

    def _draw(ax, gg, ttl):
        if kind == "series":
            _panel(ax, gg, session_col, condition_col, outcome_col, conditions, colors, ttl,
                   condition_line=condition_line)
        elif by == "condition":
            _panel_dist(ax, gg, condition_col, outcome_col, conditions, colors, ttl,
                        kind=kind, center=center, show_points=show_points, rng=rng)
        else:                                       # box/violin/mean ALONG THE SESSIONS
            _panel_by_session(ax, gg, session_col, condition_col, outcome_col, conditions,
                              colors, ttl, kind=kind, center=center, show_points=show_points,
                              rng=rng, condition_line=condition_line)

    if unit_col is None:
        fig, ax = plt.subplots(figsize=(sty.figsize or (12, 8)))
        _draw(ax, sub, sty.title or title or "Alternating SCED / N-of-1")
        ax.set_xlabel(sty.xlabel or xlab); ax.set_ylabel(sty.ylabel or outcome_col)
        if bounds is not None:
            ax.set_ylim(bounds[0], bounds[1])                         # real scale of the outcome
        if kind == "series" and sty.show_legend:
            ax.legend(title="Condition", framealpha=0.9, **sty.legend_kw())
    else:
        units = sorted(sub[unit_col].dropna().unique().tolist(), key=str)
        nrows = int(np.ceil(len(units) / ncols))
        fig, axes = plt.subplots(nrows, min(ncols, len(units)),
                                 figsize=(7 * min(ncols, len(units)), 5 * nrows),
                                 squeeze=False)
        for i, u in enumerate(units):
            ax = axes[i // ncols][i % ncols]
            _draw(ax, sub[sub[unit_col] == u], u)
            if kind == "series":
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if bounds is not None:
                ax.set_ylim(bounds[0], bounds[1])                     # real scale of the outcome
            else:
                ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
            if i == len(units) - 2:
                ax.set_xlabel(xlab); ax.set_ylabel(outcome_col)
            else:
                ax.set_xlabel(""); ax.set_ylabel("")

        for j in range(len(units), nrows * ncols):                   # blank extra axes
            axes[j // ncols][j % ncols].axis("off")
        handles = [plt.Line2D([0], [0], marker="o", ls="", color=colors[c], label=str(c))
                   for c in conditions]
        if sty.show_legend:
            fig.legend(handles=handles, title="Condition", loc="lower right", fontsize=20,
                       title_fontsize=20, framealpha=0.9)
        fig.suptitle(sty.title or title or "Alternating SCED - group (by unit)", fontsize=26)

    fig.tight_layout()
    if save_path is not None:
        sty.save(fig, save_path, f"{outcome_col}_sced_alternating_plot")
    return fig


def plot_sced_panels(df, *, session_col, condition_col, outcome_col, unit_col=None,
                     conditions=None, reference=None, improvement="increase",
                     kind="series", center="mean", show_points=True, by="session",
                     condition_line=True, vaior=True, brinley=True, brinley_mode="classic",
                     brinley_pairs="vs_reference", aggregate="mean", series_cols=None,
                     bounds=None, y_percent=False, row_height=2.2, figscale=1.0,
                     title=None, save_path=None, style=None):
    """
    SCED **dashboard**. Two layouts:

      - **Group + series** (``unit_col`` with >1 unit, ``kind="series"``) - "MBD-like" layout:
        the **time series are STACKED per patient** in the left column(s) (1 column for <=5
        patients -> each series full width, which favours the time axis), and the **condition /
        group plots** (VAIOR if enabled, Brinley, distribution by condition) occupy a **right
        column**. The time-order line therefore stays within-patient.
      - **Single-case / non-series** - in rows: series on top (full width), then a VAIOR row
        and a Brinley row (one panel per comparison).

    ``reference`` = VAIOR/Brinley reference condition (default: first of ``conditions``).
    ``series_cols`` forces the number of series columns (otherwise auto: 1 if <=5 patients).
    ``row_height`` sets the height of a series row - hence the total figure height
    (= row_height x number of rows); ``figscale`` enlarges (or shrinks) the WHOLE figure
    (width AND height) by a multiplicative factor. ``brinley_pairs``: ``"vs_reference"``
    (default, each condition vs ``reference``) or ``"all"`` (ALL pairs, e.g. C vs B).
    Y axis: ``bounds=(lo,hi)`` bounds the axis, ``bounds=None`` leaves it **AUTO**;
    ``y_percent=True`` shows the axis as a **percentage** (data 0-1 -> 40%...100%), Brinley
    on both axes.
    The dashboard VAIOR stays POOLED; the per-patient single-case VAIOR detail is provided by
    :func:`SCED_vaior.plot_vaior_grid`.

    ``aggregate`` (default ``"mean"``) - when there are **several measures per occasion**
    (runs/trials of the same visit), aggregates them into ONE point per (patient, session,
    condition) before plotting, so that each trajectory is smooth (one point per visit) instead
    of stacking the runs. ``"median"`` for the median, ``None`` to keep the raw measures.
    Without repeated measures (one row per session), this has no effect. Returns the Figure.
    """
    from functions.sced.plots.vaior import plot_vaior, plot_brinley   # late: avoids the circular import
    if conditions is None:
        conditions = sorted(df[condition_col].dropna().unique().tolist(), key=str)
    conditions = list(conditions)
    reference = reference if reference is not None else conditions[0]
    comps = [c for c in conditions if c != reference]
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    palette = plt.get_cmap(sty.palette)(np.linspace(0, 1, max(10, len(conditions))))
    colors = {c: sty.cond_color(c, palette[i]) for i, c in enumerate(conditions)}
    sub = df[df[condition_col].isin(conditions)].copy()
    # aggregate repeated measures (runs) into one value per occasion -> smooth trajectory
    if aggregate in ("mean", "median"):
        sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")
        keys = ([unit_col] if unit_col else []) + [session_col, condition_col]
        sub = (sub.groupby(keys, as_index=False, sort=False)[outcome_col]
                  .agg("mean" if aggregate == "mean" else "median"))
    elif aggregate is not None:
        raise ValueError("aggregate must be 'mean', 'median' or None")
    rng = np.random.RandomState(0)
    plt.rcParams.update({"axes.titlesize": 13, "axes.labelsize": 11,
                         "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8})

    def _yfmt(ax, both=False):
        """Bound the y axis (``bounds=(lo,hi)``) or leave it AUTO (``bounds=None``), and show it
        as a PERCENTAGE if ``y_percent`` (data 0-1 -> 40%...100%). ``both`` = also the x axis
        (Brinley: both axes are in the outcome unit)."""
        if bounds is not None:
            ax.set_ylim(bounds[0], bounds[1])
            if both:
                ax.set_xlim(bounds[0], bounds[1])
        if y_percent:
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
            if both:
                ax.xaxis.set_major_formatter(PercentFormatter(xmax=1.0))

    is_group_series = (unit_col is not None and kind == "series"
                       and sub[unit_col].nunique() > 1)

    # ===================================================================== #
    #  GROUP + series: "MBD-like" layout - time series STACKED (full width)
    #  in the left column(s), CONDITION / GROUP plots (VAIOR/Brinley/
    #  distribution) in a right column.
    # ===================================================================== #
    if is_group_series:
        units = sorted(sub[unit_col].dropna().unique().tolist(), key=str)
        nfac = len(units)
        scols = series_cols or _grid_cols(nfac)            # 1 if <=5 -> each series full width
        srows = int(np.ceil(nfac / scols))
        # Brinley comparisons: vs the reference (default) OR ALL pairs (e.g. C vs B)
        if str(brinley_pairs) == "all":
            bpairs = list(itertools.combinations(conditions, 2))    # (A,B),(A,C),(B,C)...
        else:
            bpairs = [(reference, c) for c in comps]
        # tuples (kind, cond_x, cond_y); dist has no pair
        summ = ([("vaior", reference, c) for c in comps] if vaior else []) \
            + ([("brinley", cx, cy) for (cx, cy) in bpairs] if brinley else []) \
            + [("dist", None, None)]                        # distribution by condition (group view)
        # Two independent subgrids (left = stacked series, right = summaries), each in EQUAL
        # blocks, with an EXPLICIT and GENEROUS hspace (constrained layout did not space them
        # enough). Explicit figure margins -> no overlap with titles/labels.
        nsumm = max(len(summ), 1)
        ncols = scols + 1
        block_h = (row_height * srows) / nsumm                      # height of one summary plot
        fig_w = (5.0 * scols + block_h + 1.6) * figscale            # narrow series + summary column
        fig_h = (row_height * srows + 1.4) * figscale
        fig = plt.figure(figsize=(fig_w, fig_h))
        outer = fig.add_gridspec(1, 2, width_ratios=[5.0 * scols, block_h + 0.8],
                                 left=0.07, right=0.975, top=0.90, bottom=0.07, wspace=0.18)
        left = outer[0, 0].subgridspec(srows, scols, hspace=0.65, wspace=0.30)
        right = outer[0, 1].subgridspec(nsumm, 1, hspace=0.55)
        # --- left: one wide series per patient, stacked (1 cell each) ---
        for idx, u in enumerate(units):
            r, cl = idx // scols, idx % scols
            ax = fig.add_subplot(left[r, cl])
            _panel(ax, sub[sub[unit_col] == u], session_col, condition_col, outcome_col,
                   conditions, colors, str(u), condition_line=condition_line)
            ax.set_title(str(u), fontsize=11)
            ax.set_ylabel(outcome_col if cl == 0 else "")
            _yfmt(ax)
            if idx + scols >= nfac:                         # last row of the column
                ax.set_xlabel("Session")
            else:
                ax.set_xticklabels([])
            if idx == 0:
                ax.legend(title="Condition", fontsize=7, loc="best")
        # --- right: condition / group plots, EQUAL blocks (1 cell each) ---
        legend_done = False                              # the Patient legend appears only once
        for si, (ks, cx, cy) in enumerate(summ):
            axr = fig.add_subplot(right[si])
            if ks == "vaior":
                plot_vaior(sub, session_col=session_col, condition_col=condition_col,
                           outcome_col=outcome_col, reference=cx, compared=cy,
                           improvement=improvement, ax=axr)
                _yfmt(axr)
            elif ks == "brinley":
                plot_brinley(sub, condition_col=condition_col, outcome_col=outcome_col,
                             cond_x=cx, cond_y=cy, session_col=session_col,
                             unit_col=unit_col, mode=brinley_mode, bounds=bounds, ax=axr)
                _yfmt(axr, both=True)                        # Brinley: x and y in the outcome unit
                axr.set_box_aspect(1)                        # y=x diagonal -> SQUARE box
                leg = axr.get_legend()                       # Patient legend: once only, compact
                if leg is not None:
                    if legend_done:
                        leg.remove()
                    else:
                        leg.set_title("Patient")
                        for t in leg.get_texts():
                            t.set_fontsize(6)
                        legend_done = True
            else:
                _panel_dist(axr, sub, condition_col, outcome_col, conditions, colors,
                            "Distribution by condition", kind="box", center=center,
                            show_points=show_points, rng=rng)
                axr.set_xlabel("Condition"); axr.set_ylabel(outcome_col)
                axr.set_title("Distribution by condition", fontsize=12)   # consistent size
                _yfmt(axr)
        fig.suptitle(title or f"SCED dashboard ({outcome_col})", fontsize=16)
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)
            fig.savefig(os.path.join(save_path, f"{outcome_col}_sced_panels.png"),
                        dpi=130, bbox_inches="tight")
        return fig

    # ===================================================================== #
    #  Single-case / non-series CASE: layout in ROWS (series on top).
    # ===================================================================== #
    ncol = max(len(comps), 1)
    nrow = 1 + int(bool(vaior)) + int(bool(brinley))
    fig = plt.figure(figsize=(6.2 * ncol, 5.0 * nrow))
    gs = fig.add_gridspec(nrow, ncol)
    ax0 = fig.add_subplot(gs[0, :])
    if kind == "series":
        _panel(ax0, sub, session_col, condition_col, outcome_col, conditions, colors,
               "Time series", condition_line=condition_line)
        ax0.legend(title="Condition", fontsize=8)
        ax0.set_xlabel("Session")
    elif by == "condition":
        _panel_dist(ax0, sub, condition_col, outcome_col, conditions, colors,
                    "Distribution by condition", kind=kind, center=center,
                    show_points=show_points, rng=rng)
        ax0.set_xlabel("Condition")
    else:
        _panel_by_session(ax0, sub, session_col, condition_col, outcome_col, conditions,
                          colors, "Time series", kind=kind, center=center,
                          show_points=show_points, rng=rng, condition_line=condition_line)
        ax0.set_xlabel("Session")
    ax0.set_ylabel(outcome_col)
    _yfmt(ax0)                                                   # bounds (or auto) + optional %

    row = 1
    if vaior:
        for j, comp in enumerate(comps):
            axv = fig.add_subplot(gs[row, j])
            plot_vaior(sub, session_col=session_col, condition_col=condition_col,
                       outcome_col=outcome_col, reference=reference, compared=comp,
                       improvement=improvement, ax=axv, style=style)
            _yfmt(axv)
        row += 1
    if brinley:
        for j, comp in enumerate(comps):
            axb = fig.add_subplot(gs[row, j])
            plot_brinley(sub, condition_col=condition_col, outcome_col=outcome_col,
                         cond_x=reference, cond_y=comp, session_col=session_col,
                         unit_col=unit_col, mode=brinley_mode, bounds=bounds, ax=axb, style=style)
            _yfmt(axb, both=True)

    fig.suptitle(sty.title or title or f"SCED dashboard ({outcome_col})", fontsize=sty.title_fs + 3)
    fig.tight_layout()
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        sty.save(fig, save_path, f"{outcome_col}_sced_panels")
    return fig


def _grid_cols(n):
    """Number of small-multiple columns: 1 (<=5), 2 (<=10), 3 (<=18), else 4."""
    return 1 if n <= 5 else (2 if n <= 10 else (3 if n <= 18 else 4))


def _batch(items, size):
    """Split into batches of <= ``size`` (a single batch if <= size)."""
    items = list(items)
    return [items] if len(items) <= size else [items[i:i + size] for i in range(0, len(items), size)]


def _draw_tier_panel(ax, g, t, *, session_col, outcome_col, start,
                     per_tier_row=None, improvement="increase", ylabel=None, last_in_col=True,
                     impute_col=None, bounds=None):
    """One rich multiple-baseline tier panel: raw series + A/B phase means + baseline
    (Theil-Sen, projected) and treatment trends + delta/Tau-U/NAP annotation. The y axis
    is SCALED to the tier's own data but CLAMPED within ``bounds`` (the outcome's real range,
    e.g. (0, 100)) so projected trend lines never blow it past the valid range. NO trend
    legend is drawn. ``impute_col`` (bool column) overlays imputed points as hollow
    crimson-edged markers. Shared by plot_mbd_panels and plot_multiple_baseline."""
    from functions.sced.plots.vaior import _theil_sen_fit
    from functions.sced.core import tau_u, nap
    g = g.sort_values(session_col)
    s = pd.to_numeric(g[session_col], errors="coerce").to_numpy()
    y = pd.to_numeric(g[outcome_col], errors="coerce").to_numpy()
    imp = (g[impute_col].to_numpy().astype(bool) if (impute_col and impute_col in g.columns)
           else np.zeros(len(g), dtype=bool))
    ok = np.isfinite(s) & np.isfinite(y); s, y, imp = s[ok], y[ok], imp[ok]
    ax.plot(s, y, "-o", color="0.35", ms=4, lw=1, zorder=3)
    if imp.any():                                          # imputed points : hollow crimson markers
        ax.scatter(s[imp], y[imp], s=80, facecolors="none", edgecolors="crimson",
                   linewidths=1.7, zorder=6)
    st = None if start is None else float(start)
    if st is not None and len(s):
        ax.axvspan(st - 0.5, s.max() + 0.5, color="#2ca02c", alpha=0.06, zorder=0)
        ax.axvline(st - 0.5, color="crimson", ls="--", lw=1.2, zorder=2)
        mA, mB = y[s < st], y[s >= st]
        for yy, x0, x1, col in [(mA, s.min(), st - 0.5, "steelblue"),
                                (mB, st - 0.5, s.max(), "#2ca02c")]:
            if len(yy):
                ax.hlines(yy.mean(), x0, x1, color=col, lw=1.8, alpha=0.85, zorder=4)
        if len(s[s < st]) >= 2:                            # baseline (Phase A) Theil-Sen, projected
            sl, ic = _theil_sen_fit(s[s < st], mA); xx = np.array([s.min(), s.max()])
            ax.plot(xx, sl * xx + ic, ls=":", color="navy", lw=1.3, alpha=0.8, zorder=2)
        if len(s[s >= st]) >= 2:                           # treatment (Phase B) Theil-Sen
            slB, icB = _theil_sen_fit(s[s >= st], mB); sB = s[s >= st]; xb = np.array([sB.min(), sB.max()])
            ax.plot(xb, slB * xb + icB, ls=":", color="#2ca02c", lw=1.7, alpha=0.95, zorder=2)
        if per_tier_row is not None:
            dd, tu, na = per_tier_row.get("diff"), per_tier_row.get("Tau-U"), per_tier_row.get("NAP")
        else:
            dd = (mB.mean() - mA.mean()) if len(mA) and len(mB) else np.nan
            tu = tau_u(mA, mB, improvement=improvement).get("tau") if len(mA) and len(mB) else np.nan
            na = nap(mA, mB, improvement=improvement) if len(mA) and len(mB) else np.nan
        txt = "  ".join(x for x in [f"Δ={dd:.1f}" if pd.notna(dd) else "",
                                    f"τ={tu:.2f}" if pd.notna(tu) else "",
                                    f"NAP={na:.2f}" if pd.notna(na) else ""] if x)
        if txt:
            ax.text(0.99, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    if len(y):                                            # y SCALED to the data but CLAMPED (bounds)
        span = float(y.max() - y.min()) or 1.0
        ylo, yhi = y.min() - 0.08 * span, y.max() + 0.08 * span
        if bounds is not None:
            ylo, yhi = max(ylo, float(bounds[0])), min(yhi, float(bounds[1]))
        if yhi > ylo:
            ax.set_ylim(ylo, yhi)
    ax.set_ylabel(ylabel if ylabel is not None else f"{t}", fontsize=9)
    ax.grid(True, alpha=0.2)
    if not last_in_col:
        ax.set_xticklabels([])


def _imputed_legend(fig, df, impute_col):
    """Add one figure-level 'imputed' legend (hollow crimson marker) if imputed points exist."""
    if not (impute_col and impute_col in getattr(df, "columns", []) and df[impute_col].any()):
        return
    from matplotlib.lines import Line2D
    proxy = Line2D([0], [0], marker="o", ls="", mfc="none", mec="crimson", mew=1.7,
                   ms=8, label="imputed")
    fig.legend(handles=[proxy], loc="upper right", fontsize=8, framealpha=0.9)


def plot_mbd_panels(df, *, tier_col, session_col, outcome_col, starts, phase_col=None,
                    baseline=None, treatment=None, improvement="increase", per_tier=None,
                    brinley_mode="classic", bounds=None, series_cols=None, max_per_fig=10,
                    name=None, title=None, save_path=None, style=None, impute_col=None):
    """
    **MBD dashboard**: annotated STAGGERED panels (one per tier - A/B means, **projected
    Theil-Sen baseline trend**, Delta / Tau-U / NAP inset), + **Brinley** and **distribution
    by phase** on the right. ``starts`` = {tier: B start}. ``per_tier`` (optional) = table
    Tier/diff/Tau-U/NAP.

    Adaptive layout: tiers are laid out over **several columns** as soon as > 5 (auto via
    ``series_cols``), and **split into several figures (batches)** if > ``max_per_fig`` (default
    10), written ``<name>_mbd_panels[_bN].png``. ``bounds=(lo,hi)`` bounds the y axis. Returns
    the LAST figure (all are saved if ``save_path``)."""
    from functions.sced.core import nap, tau_u
    from functions.sced.plots.vaior import _theil_sen_fit, plot_brinley
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    starts = {str(k): v for k, v in (starts or {}).items()}   # keys normalized to str (int OR str IDs)
    tiers = sorted(df[tier_col].dropna().unique().tolist(), key=str)
    if phase_col is not None:
        phs = df[phase_col].dropna().unique().tolist()
        baseline = baseline if baseline is not None else sorted(map(str, phs))[0]
        treatment = treatment if treatment is not None else [p for p in sorted(map(str, phs))
                                                             if str(p) != str(baseline)][-1]
    pt = None
    if per_tier is not None:
        pt = {str(r["Tier"]): r for _, r in per_tier.iterrows() if "Tier" in per_tier.columns}
    plt.rcParams.update({"axes.titlesize": 11, "axes.labelsize": 10,
                         "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})

    def _draw_tier(ax, t, last_in_col, show_legend=False):
        # individual panels AUTOSCALE their y axis (no shared bounds) and carry no trend legend
        _draw_tier_panel(ax, df[df[tier_col] == t], t, session_col=session_col,
                         outcome_col=outcome_col, start=starts.get(str(t)),
                         per_tier_row=(pt.get(str(t)) if pt is not None else None),
                         improvement=improvement, last_in_col=last_in_col, impute_col=impute_col,
                         bounds=bounds)
        if last_in_col:
            ax.set_xlabel("Session")

    base = name if name is not None else str(outcome_col)
    batches = _batch(tiers, max_per_fig)
    last = None
    for bi, tb in enumerate(batches):
        k = len(tb); scols = series_cols or _grid_cols(k)
        trows = int(np.ceil(k / scols))
        nrows = max(trows, 2)
        ncols = scols + (1 if phase_col is not None else 0)
        fig = plt.figure(figsize=(5.2 * ncols, max(2.4 * nrows, 6)))
        gs = fig.add_gridspec(nrows, ncols)
        for idx, t in enumerate(tb):
            row, col = idx // scols, idx % scols
            _draw_tier(fig.add_subplot(gs[row, col]), t, last_in_col=(idx + scols >= k),
                       show_legend=(idx == 0))
        if phase_col is not None:                              # right column: Brinley + distribution
            half = max(1, nrows // 2)
            try:
                plot_brinley(df, condition_col=phase_col, outcome_col=outcome_col, cond_x=baseline,
                             cond_y=treatment, session_col=session_col, unit_col=tier_col,
                             mode=brinley_mode, bounds=bounds, ax=fig.add_subplot(gs[0:half, scols]),
                             style=style)
            except Exception:
                pass
            axd = fig.add_subplot(gs[half:, scols])
            palette = plt.cm.tab10(np.linspace(0, 1, 10))
            _panel_dist(axd, df, phase_col, outcome_col, [baseline, treatment],
                        {baseline: palette[0], treatment: palette[2]}, "Distribution by phase",
                        kind="box", center="mean", show_points=True, rng=np.random.RandomState(0))
            axd.set_xlabel("Phase"); axd.set_ylabel(outcome_col)
            if bounds is not None:
                axd.set_ylim(bounds[0], bounds[1])
        lot = "" if len(batches) == 1 else f" (batch {bi + 1}/{len(batches)})"
        suff = "" if len(batches) == 1 else f"_b{bi + 1}"
        fig.suptitle((sty.title or title or f"MBD dashboard ({outcome_col})") + lot,
                     fontsize=sty.title_fs + 2)
        _imputed_legend(fig, df, impute_col)             # single "imputed" marker legend
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        if save_path is not None:
            sty.save(fig, save_path, f"{base}_mbd_panels{suff}")
        last = fig
    return last


def plot_multiple_baseline(df, *, tier_col, session_col, outcome_col, starts,
                           improvement="increase", per_tier=None, bounds=None,
                           title=None, save_path=None, style=None, impute_col=None, name=None):
    """
    Staggered multiple-baseline figure: one stacked RICH panel per tier (raw series +
    A/B phase means + baseline/treatment Theil-Sen trends + delta/Tau-U/NAP annotation),
    the same individual panels used in the dashboard. ``starts`` = ``{tier: start_session}``.
    Each panel SCALES its y axis to the tier's own data but CLAMPED within ``bounds``
    (outcome real range, e.g. (0,100)); shows no trend legend. ``impute_col`` (bool column)
    overlays imputed points as hollow crimson markers.
    ``style`` (PlotStyle/dict) drives the title. Returns the Figure (saves a PNG if given).
    """
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    starts = {str(k): v for k, v in (starts or {}).items()}   # str keys (robust to int OR str IDs)
    pt = {str(k): v for k, v in per_tier.items()} if isinstance(per_tier, dict) else None
    tiers = sorted(df[tier_col].dropna().unique().tolist(), key=str)
    fig, axes = plt.subplots(len(tiers), 1, figsize=(sty.figsize or (8, 2.1 * len(tiers))),
                             squeeze=False)
    for i, t in enumerate(tiers):
        _draw_tier_panel(axes[i][0], df[df[tier_col] == t], t, session_col=session_col,
                         outcome_col=outcome_col, start=starts.get(str(t)),
                         per_tier_row=(pt.get(str(t)) if pt is not None else None),
                         improvement=improvement, ylabel=f"{tier_col}={t}",
                         last_in_col=(i == len(tiers) - 1), impute_col=impute_col, bounds=bounds)
    axes[-1][0].set_xlabel(sty.xlabel or "Session")
    fig.suptitle(sty.title or title or "Multiple baseline (staggered intervention)", fontsize=sty.title_fs)
    _imputed_legend(fig, df, impute_col)             # single "imputed" marker legend
    fig.tight_layout()
    if save_path is not None:
        sty.save(fig, save_path, f"{name or outcome_col}_multiple_baseline_plot")
    return fig


def plot_mbd_pooled_fit(df, *, tier_col, session_col, outcome_col, starts, fit,
                        band_draws=None, per_case=True, min_frac=0.5, bounds=None,
                        y_percent=False, title=None, save_path=None, fname=None, ax=None, style=None):
    """**Pooled MBD in EVENT-TIME**: all cases realigned on THEIR intervention start (t = 0 at
    the 1st B point), with the **implicit trajectory of the multilevel model** (baseline trend
    b1, level change b2, slope change b3).

    Shows: the **per-patient spaghetti** (light) if ``per_case``; the **pooled mean +/- SE** per
    event-time; and the **population curve** in bold - segment A of slope ``b1`` then, after the
    intervention, segment B with the **b2 jump** and the **b1+b3 slope**. The curve is
    **anchored on the pooled baseline** (no need for the intercept), so it works identically for
    a **frequentist** fit (point estimates) or a **Bayesian** one (posterior means). ``fit`` =
    dict with ``baseline_trend_b1`` / ``level_change_b2`` / ``slope_change_b3`` (summary of
    :func:`SCED_mbd_multilevel.multilevel_mbd_model`, or posterior means of
    :func:`SCED_mbd_bayesian.bayes_hier_sced`).

    ``band_draws`` (**Bayesian** option) = dict ``{"b1":draws, "b2":draws, "b3":draws}`` of
    posterior draws -> draws a **95% credible band** (HDI) around segments A and B (uncertainty
    on trend/jump/slope). ``min_frac`` trims the event-time window where >= this fraction of
    cases still contributes. Reserved for the **MBD** (phase design), not the alternating one.
    ``bounds``/``y_percent`` as in the other plots. Returns the Figure."""
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    starts = {str(k): v for k, v in (starts or {}).items()}
    tiers = sorted(df[tier_col].dropna().unique().tolist(), key=str)
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(sty.figsize or (9.5, 5.4)))
    palette = plt.get_cmap(sty.palette)(np.linspace(0, 1, max(10, len(tiers))))
    ev = {}                                                   # tier -> (tau, y) in event-time
    for i, t in enumerate(tiers):
        g = df[df[tier_col].astype(str) == str(t)].sort_values(session_col)
        s = pd.to_numeric(g[session_col], errors="coerce").to_numpy()
        y = pd.to_numeric(g[outcome_col], errors="coerce").to_numpy()
        st = starts.get(str(t))
        ok = np.isfinite(s) & np.isfinite(y)
        s, y = s[ok], y[ok]
        if st is None or not len(s):
            continue
        tau = s - float(st)
        ev[str(t)] = (tau, y)
        if per_case:
            ax.plot(tau, y, "-o", color=palette[i], lw=1.0, ms=3, alpha=0.45, zorder=2,
                    label=None)
    k = max(len(ev), 1)
    # pooled mean +/- SE per integer event-time, trimmed to coverage >= min_frac
    t_lo, t_hi = -1, 1
    if ev:
        allt = np.concatenate([tau for tau, _ in ev.values()])
        lo, hi = int(np.floor(allt.min())), int(np.ceil(allt.max()))
        taus, means, ses = [], [], []
        for tt in range(lo, hi + 1):
            vals = [y[np.round(tau) == tt] for tau, y in ev.values()]
            ncase = sum(1 for v in vals if len(v))
            allv = np.concatenate([v for v in vals if len(v)]) if ncase else np.array([])
            if len(allv) and ncase / k >= min_frac:
                taus.append(tt); means.append(float(allv.mean()))
                ses.append(float(allv.std(ddof=1) / np.sqrt(len(allv))) if len(allv) > 1 else 0.0)
        if taus:
            taus = np.array(taus, float); means = np.array(means); ses = np.array(ses)
            ax.fill_between(taus, means - ses, means + ses, color="0.45", alpha=0.15, lw=0,
                            zorder=4)                                    # +/-SE as a light ribbon
            ax.plot(taus, means, "-o", color=sty.color("pooled_mean"), ms=4, lw=1.3, zorder=5,
                    label=sty.label("pooled_mean"))
            t_lo, t_hi = float(taus.min()), float(taus.max())
    # implicit model curve, ANCHORED on the pooled baseline (segment A: tau<0) -> no b0
    b1, b2 = _f(fit.get("baseline_trend_b1")), _f(fit.get("level_change_b2"))
    b3 = _f(fit.get("slope_change_b3"))
    A_y = np.concatenate([y[tau < 0] for tau, y in ev.values()]) if ev else np.array([])
    A_t = np.concatenate([tau[tau < 0] for tau, y in ev.values()]) if ev else np.array([])
    anchor = float(A_y.mean()) if len(A_y) else 0.0          # pooled baseline level
    tA_bar = float(A_t.mean()) if len(A_t) else 0.0          # mean baseline event-time

    def _curve(B1, B2, B3, ta, tb):                          # A then B lines (anchored to the baseline)
        A0 = anchor - B1 * tA_bar                            # value of line A at tau=0
        return anchor + B1 * (ta - tA_bar), A0 + B2 + (B1 + B3) * tb

    # Terms absent from the model (partial b1/+b2/+b3) -> 0: the model curve is drawn for
    # EVERY model (b1 alone, b1+b2, b1+b3, full), not only the full one.
    B1 = b1 if np.isfinite(b1) else 0.0
    B2 = b2 if np.isfinite(b2) else 0.0
    B3 = b3 if np.isfinite(b3) else 0.0
    if any(np.isfinite(x) for x in (b1, b2, b3)):
        ta = np.linspace(min(t_lo, 0), 0, 60)
        tb = np.linspace(0, max(t_hi, 0), 60)
        if band_draws is not None:                          # 95% credible band (Bayesian)
            _present = [len(np.atleast_1d(band_draws[k])) for k in ("b1", "b2", "b3")
                        if band_draws.get(k) is not None]
            nd = min(_present) if _present else 0

            def _bd(k):                                     # term draws, or zeros if absent from the model
                v = band_draws.get(k)
                return np.asarray(v, float)[:nd] if v is not None else np.zeros(nd)
            if nd >= 20:
                d1, d2, d3 = _bd("b1"), _bd("b2"), _bd("b3")
                YA = np.empty((nd, len(ta))); YB = np.empty((nd, len(tb)))
                for j in range(nd):
                    YA[j], YB[j] = _curve(d1[j], d2[j], d3[j], ta, tb)
                for tg, Y, col in [(ta, YA, sty.color("baseline")), (tb, YB, sty.color("treatment"))]:
                    qlo, qhi = np.percentile(Y, [2.5, 97.5], axis=0)
                    ax.fill_between(tg, qlo, qhi, zorder=3, **sty.band_kw(col))
        yA, yB = _curve(B1, B2, B3, ta, tb)
        ax.plot(ta, yA, "-", color=sty.color("baseline"), lw=sty.lw(2.6), zorder=6, label=sty.label("phase_a"))
        ax.plot(tb, yB, "-", color=sty.color("treatment"), lw=sty.lw(2.6), zorder=6, label=sty.label("phase_b"))
    ax.axvline(0, color=sty.color("intervention_line"), ls="--", lw=1.2, alpha=0.7, zorder=1)   # intervention
    ax.set_xlabel(sty.xlabel or "Time relative to intervention (sessions)")
    ax.set_ylabel(sty.ylabel or outcome_col)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if bounds is not None:
        ax.set_ylim(bounds[0], bounds[1])
    if y_percent:
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
    lab = "  ".join(x for x in [f"b1={b1:.2f}" if np.isfinite(b1) else "",
                                f"b2={b2:.2f}" if np.isfinite(b2) else "",
                                f"b3={b3:.2f}" if np.isfinite(b3) else ""] if x)
    if lab:
        ax.text(0.99, 0.02, lab, transform=ax.transAxes, ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.85))
    if sty.show_legend:
        ax.legend(**sty.legend_kw())
    ax.grid(sty.grid, alpha=sty.grid_alpha)
    sty.format_axes(ax)
    ax.set_title(sty.title or title or f"Pooled MBD (event-time): {outcome_col}", fontsize=sty.title_fs)
    if fig is not None and save_path is not None:
        sty.save(fig, save_path, fname or (outcome_col + "_mbd_pooled_fit"), ax=ax)
    return fig if fig is not None else ax.figure


def ml_band_draws(fit, *, n=2000, seed=0):
    """``{b1,b2,b3}`` draws from the **sampling distribution** of the frequentist fixed effects
    (multivariate Normal with mean (b1,b2,b3) and covariance ``fe_cov`` from
    :func:`SCED_mbd_multilevel.multilevel_mbd_model`) -> for a **95% CONFIDENCE band** on the
    inferential pooled plot, analogous to the Bayesian HDI band. Returns ``None`` if the
    covariance is absent. Warning: asymptotic SEs are unreliable with FEW cases (the primary
    SCED inference remains the randomization test) - the band is indicative.

    References: Gelman & Hill 2007 (simulation from the sampling distribution of fixed effects).
    R equivalent: arm::sim.
    """
    def _f(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    mean = [_f(fit.get("baseline_trend_b1")), _f(fit.get("level_change_b2")),
            _f(fit.get("slope_change_b3"))]
    cov = fit.get("fe_cov")
    if cov is None or any(not np.isfinite(m) for m in mean):
        return None
    try:
        draws = np.random.RandomState(seed).multivariate_normal(mean, np.asarray(cov, float), size=n)
    except Exception:
        return None
    return {"b1": draws[:, 0], "b2": draws[:, 1], "b3": draws[:, 2]}


def mbd_bayes_band_draws(idata, *, improvement="increase"):
    """Adapt an ``idata`` from :func:`SCED_mbd_bayesian.bayes_hier_sced` for
    ``plot_mbd_pooled_fit(fit=..., band_draws=...)``: returns ``(fit, band_draws)`` where
    ``band_draws`` = flattened posterior draws ``{b1,b2,b3}`` in POINTS (maps
    pop_trend_pts_ps->b1, pop_level->b2, pop_slope_pts_ps->b3, with fallbacks) and ``fit`` =
    their means. To pass to the pooled plot for the curve + 95% credible band.

    Warning: the ``pop_*`` quantities are ORIENTED (improvement>0). The pooled plot works on
    the RAW scale of the outcome -> reconvert by multiplying by ``sign`` (+1 increase,
    -1 decrease), otherwise the curve goes the wrong way for an outcome to be minimized."""
    post = idata.posterior if hasattr(idata, "posterior") else idata
    sign = 1.0 if improvement == "increase" else -1.0       # oriented -> raw scale

    def grab(*names):
        for n in names:
            if n in post:
                return np.asarray(post[n]).reshape(-1)
        return None
    draws = {}
    for key, names in (("b1", ("pop_trend_pts_ps", "pop_trend")),
                       ("b2", ("pop_level", "pop_level_pts")),
                       ("b3", ("pop_slope_pts_ps", "pop_slope"))):
        v = grab(*names)
        if v is not None:
            draws[key] = sign * v                            # reconvert oriented -> raw
    fit = {"baseline_trend_b1": float(np.mean(draws["b1"])) if "b1" in draws else np.nan,
           "level_change_b2": float(np.mean(draws["b2"])) if "b2" in draws else np.nan,
           "slope_change_b3": float(np.mean(draws["b3"])) if "b3" in draws else np.nan}
    return fit, draws


def _diag_var_names(post):
    """Parameters of interest present in the posterior (effects, not the per-case auxiliaries)."""
    pref = ("pop_level", "pop_slope", "pop_trend", "logit_b2", "mu_b2", "mu_b3", "mu_b1",
            "tau_level", "tau_slope", "phi", "rho", "nu", "es", "level_change", "delta")
    vn = [v for v in pref if v in post]
    return vn or [v for v in list(post.data_vars) if not v.startswith(("z_", "b0", "b1", "b2", "b3"))][:4]


def _trace_grid(idata, var_names, save_path, name, style=None):
    """MCMC trace in a BALANCED grid (columns/rows depending on the number of scalar parameters):
    one cell per parameter, all chains overlaid. Saves ``{name}_trace.<fmt>``.

    R equivalent: bayesplot::mcmc_trace.
    """
    import os
    import matplotlib.pyplot as plt
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style)
    post = idata.posterior
    scal = [v for v in var_names if v in post and set(post[v].dims) <= {"chain", "draw"}]
    scal = scal or [v for v in var_names if v in post][:6]
    k = max(len(scal), 1)
    ncols = 1 if k <= 3 else (2 if k <= 8 else 3)         # column/row balancing
    nrows = int(np.ceil(k / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.4 * ncols, 2.2 * nrows), squeeze=False)
    flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for i, v in enumerate(scal):
        ax = flat[i]; da = post[v]
        for ch in range(da.sizes["chain"]):
            ax.plot(da.isel(chain=ch).values, lw=0.6, alpha=0.8)
        ax.set_title(v, fontsize=10); ax.grid(True, alpha=0.2); ax.set_xlabel("draw", fontsize=8)
    for i in range(len(scal), len(flat)):
        flat[i].axis("off")
    fig.suptitle(f"MCMC trace: {name}", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fp = os.path.join(save_path, f"{name}_trace.{sty.fmt}")
    fig.savefig(fp, dpi=sty.dpi, bbox_inches="tight", transparent=sty.transparent); plt.close(fig)
    return fp


def plot_bayesian_diag(idata, *, save_path=".", name="model", var_names=None, rope=None, style=None):
    """DIAGNOSTIC plots of a fitted Bayesian model (ArviZ), ONE file per diagnostic in
    ``save_path``: **trace** (chains, compact), **rank** (ranks per chain), **forest** (effects +
    HDI + r_hat + ess), **posterior** (+HDI 95% / ROPE), **energy** (BFMI NUTS), **ess** (evolution),
    **autocorr**, **qq_loo_pit** (calibration ~ QQ; if log_likelihood + posterior_predictive) and
    **ppc** (if posterior_predictive). See ``bayesian_diag_legend()`` for the interpretation.
    Returns the list of written files. Silent per diagnostic.

    References: Vehtari, Gelman, Simpson, Carpenter & Burkner 2021 (rank-normalized R-hat / ESS diagnostics).
    R equivalent: bayesplot (mcmc_* / ppc_*) + posterior (rhat / ess).
    """
    import os
    import arviz as az
    import matplotlib.pyplot as plt
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    os.makedirs(save_path, exist_ok=True)
    post = getattr(idata, "posterior", None)
    if post is None:
        return []
    var_names = var_names or _diag_var_names(post)
    has_pp = hasattr(idata, "posterior_predictive") and hasattr(idata, "observed_data")
    has_ll = hasattr(idata, "log_likelihood")
    saved = []

    def _save(tag):
        fig = plt.gcf(); fp = os.path.join(save_path, f"{name}_{tag}.{sty.fmt}")
        fig.savefig(fp, dpi=sty.dpi, bbox_inches="tight", transparent=sty.transparent)
        plt.close(fig); saved.append(fp)

    try:                                                  # trace in a balanced grid (handles the param count)
        saved.append(_trace_grid(idata, var_names, save_path, name, style=sty))
    except Exception:
        plt.close("all")
    rope_d = rope if isinstance(rope, dict) else None     # az.plot_posterior expects {var:[(lo,hi)]}
    plots = [
        ("rank", lambda: az.plot_rank(idata, var_names=var_names)),
        ("forest", lambda: az.plot_forest(idata, var_names=var_names, combined=True,
                                          hdi_prob=0.95, r_hat=True, ess=True)),
        ("posterior", lambda: az.plot_posterior(idata, var_names=var_names, hdi_prob=0.95, rope=rope_d)),
        ("energy", lambda: az.plot_energy(idata)),
        ("ess", lambda: az.plot_ess(idata, var_names=var_names, kind="evolution")),
        ("autocorr", lambda: az.plot_autocorr(idata, var_names=var_names, combined=True)),
    ]
    if has_ll and has_pp:
        plots.append(("qq_loo_pit", lambda: az.plot_loo_pit(idata, y="y_obs", ecdf=True)))
    if has_pp:
        plots.append(("ppc", lambda: az.plot_ppc(idata, num_pp_samples=80)))
    for tag, fn in plots:
        try:
            fn(); _save(tag)
        except Exception:
            plt.close("all")
    return saved


def plot_bayesian_panel(idata, *, save_path=".", name="model", var_names=None, rope=None, style=None):
    """Diagnostic DASHBOARD: the most important views in ONE 2x2 figure - forest (effects +
    HDI + r_hat/ess), rank (convergence), energy (BFMI NUTS) and PPC (or posterior if no
    replicates). Saves ``{name}_panel.<fmt>``. Returns the path or None.

    R equivalent: bayesplot (mcmc_* / ppc_*) + posterior.
    """
    import os
    import arviz as az
    import matplotlib.pyplot as plt
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    post = getattr(idata, "posterior", None)
    if post is None:
        return None
    os.makedirs(save_path, exist_ok=True)
    vn = var_names or _diag_var_names(post)
    has_pp = hasattr(idata, "posterior_predictive") and hasattr(idata, "observed_data")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"MCMC diagnostics: {name}", fontsize=14)
    rope_d = rope if isinstance(rope, dict) else None

    def _try(ax, fn, title):
        try:
            fn(ax); ax.set_title(title, fontsize=11)
        except Exception:
            ax.clear(); ax.text(0.5, 0.5, f"{title}\n(unavailable)", ha="center", va="center"); ax.axis("off")
    # Note: r_hat/ess add axis columns -> incompatible with a single ax (cf. standalone forest)
    _try(axes[0, 0], lambda ax: az.plot_forest(idata, var_names=vn, combined=True, hdi_prob=0.95, ax=ax),
         "Forest (effects + 95% HDI)")
    _try(axes[0, 1], lambda ax: az.plot_rank(idata, var_names=vn[:1], ax=ax), "Rank (convergence)")
    _try(axes[1, 0], lambda ax: az.plot_energy(idata, ax=ax), "Energy (BFMI NUTS)")
    if has_pp:
        _try(axes[1, 1], lambda ax: az.plot_ppc(idata, num_pp_samples=60, ax=ax), "PPC (observed vs replicates)")
    else:
        _try(axes[1, 1], lambda ax: az.plot_posterior(idata, var_names=vn[:1], hdi_prob=0.95,
                                                      rope=rope_d, ax=ax), "Posterior + 95% HDI")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fp = os.path.join(save_path, f"{name}_panel.{sty.fmt}")
    fig.savefig(fp, dpi=sty.dpi, bbox_inches="tight", transparent=sty.transparent); plt.close(fig)
    return fp


def plot_ppc_grouped(idata, *, by, save_path=".", name="model", max_per_fig=10, n_rep=40, style=None):
    """GROUPED PPC: observed density (black) vs posterior replicates (blue) + predictive mean
    (orange), **per group level** (``by`` = "phase" or "case"). Much sharper than the pooled
    marginal PPC (which mixes the groups). Requires ``posterior_predictive`` + ``constant_data``
    ``obs_<by>`` (written by bayes_hier_sced). Small multiples (wrap), batches if > ``max_per_fig``.
    Returns the list of written .png files (empty if data absent).

    References: Gelman, Meng & Stern 1996 (posterior predictive check).
    R equivalent: bayesplot::ppc_dens_overlay_grouped.
    """
    import os
    import matplotlib.pyplot as plt
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    if not (hasattr(idata, "posterior_predictive") and hasattr(idata, "observed_data")):
        return []
    cd = getattr(idata, "constant_data", None)
    key = "obs_" + by
    if cd is None or key not in cd:
        return []
    from scipy.stats import gaussian_kde
    obs = np.asarray(idata.observed_data["y_obs"].values).ravel()
    pp = idata.posterior_predictive["y_obs"].stack(s=("chain", "draw")).values
    if pp.shape[0] != obs.shape[0]:
        pp = pp.T                                          # -> (n_obs, n_samples)
    grp = np.asarray(cd[key].values).ravel()
    levels = sorted(set(grp.tolist()))
    if by == "phase":
        lab = {0.0: "baseline (A)", 1.0: "treatment (B)"}
        names = [lab.get(float(l), str(l)) for l in levels]
    else:
        coords = list(idata.posterior.coords["case"].values) if "case" in idata.posterior.coords else None
        names = [str(coords[int(l)]) if coords is not None and int(l) < len(coords) else str(l) for l in levels]
    os.makedirs(save_path, exist_ok=True)
    rng = np.random.RandomState(0)
    lvb = _batch(levels, max_per_fig); nmb = _batch(names, max_per_fig)
    saved = []

    def _kde(x, xs):
        return gaussian_kde(x)(xs) if (x.size >= 3 and np.ptp(x) > 0) else None
    for bi, (lv, nm) in enumerate(zip(lvb, nmb)):
        k = len(lv); nc = 1 if k <= 2 else (2 if k <= 6 else 3); nr = int(np.ceil(k / nc))
        fig, axes = plt.subplots(nr, nc, figsize=(5.4 * nc, 3.4 * nr), squeeze=False)
        flat = [axes[r][c] for r in range(nr) for c in range(nc)]
        for i, (l, label) in enumerate(zip(lv, nm)):
            ax = flat[i]; mask = grp == l
            yo = obs[mask]; pr = pp[mask, :]
            if yo.size == 0:
                ax.axis("off"); continue
            lo = float(min(yo.min(), pr.min())); hi = float(max(yo.max(), pr.max()))
            xs = np.linspace(lo, hi, 200)
            cols = rng.choice(pr.shape[1], size=min(n_rep, pr.shape[1]), replace=False)
            for j in cols:
                dk = _kde(pr[:, j], xs)
                if dk is not None:
                    ax.plot(xs, dk, color="#4C9EE0", lw=0.5, alpha=0.25)
            dmean = _kde(pr.ravel(), xs)
            if dmean is not None:
                ax.plot(xs, dmean, color="orange", ls="--", lw=2, label="mean predictive")
            do = _kde(yo, xs)
            if do is not None:
                ax.plot(xs, do, color="black", lw=2, label="observed")
            ax.set_title(f"{label} (n={int(mask.sum())})", fontsize=10); ax.set_yticks([])
        for i in range(k, len(flat)):
            flat[i].axis("off")
        flat[0].legend(fontsize=7)
        lot = "" if len(lvb) == 1 else f" (batch {bi + 1}/{len(lvb)})"
        suff = "" if len(lvb) == 1 else f"_b{bi + 1}"
        fig.suptitle(f"PPC by {by}: {name}{lot}", fontsize=13)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fp = os.path.join(save_path, f"{name}_ppc_by_{by}{suff}.{sty.fmt}")
        fig.savefig(fp, dpi=sty.dpi, bbox_inches="tight", transparent=sty.transparent); plt.close(fig); saved.append(fp)
    return saved


def plot_meta_forest(result, *, save_path=None, title=None, unit="pts", style=None):
    """Forest plot of a SCED meta-analysis (output of ``SCED_mbd_meta.bayes_meta_analysis``).

    For each case: RAW effect (stage 1) +/- 95% CI (hollow circle) and SHRUNKEN effect
    (stage 2, median + HDI; filled point) - the shrinkage is VISIBLE. At the bottom: a diamond
    for mu (population) + a grey band = prediction interval (new case). Returns the Figure;
    saves a PNG if ``save_path``. Never shows (headless-safe).

    R equivalent: metafor::forest (forest plot of a meta-analysis).
    """
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    per = result["per_case"]; pop = result["population"]; het = result["heterogeneity"]
    k = len(per)
    fig, ax = plt.subplots(figsize=(sty.figsize or (8, 1.0 + 0.45 * (k + 2))))
    _raw, _shr = sty.color("meta_raw"), sty.color("meta_shrunk")
    ys = np.arange(k, 0, -1)                      # cases from top to bottom
    for y, pc in zip(ys, per):
        raw, se = pc["effect_raw"], pc["se"]; sh = pc["shrunk"]
        ax.plot([raw - 1.96 * se, raw + 1.96 * se], [y + 0.12, y + 0.12],
                color=_raw, lw=1.3, zorder=1)
        ax.scatter([raw], [y + 0.12], facecolors="none", edgecolors=_raw,
                   s=34, zorder=2, label="raw (stage 1)" if y == ys[0] else None)
        cred = sh["pd"] >= 0.95
        ax.plot([sh["hdi_low"], sh["hdi_high"]], [y - 0.12, y - 0.12],
                color=_shr, lw=sty.lw(2.2), zorder=3)
        ax.scatter([sh["median"]], [y - 0.12], color=_shr if cred else sty.color("meta_shrunk_weak"),
                   s=46, zorder=4, label="shrunken (stage 2)" if y == ys[0] else None)
    # population: diamond + prediction interval
    pi = het["prediction_interval"]
    ax.axhspan(-0.55, 0.45, xmin=0, xmax=1, color="#f1f3f4", zorder=0)
    ax.fill_betweenx([-0.05, 0.05], pi[0], pi[1], color=sty.color("meta_pi"), alpha=0.8,
                     zorder=1, label="prediction interval")
    mu = pop["median"]; dlo, dhi = pop["hdi_low"], pop["hdi_high"]
    ax.add_patch(plt.Polygon([[dlo, 0], [mu, 0.28], [dhi, 0], [mu, -0.28]],
                             closed=True, color=sty.color("meta_diamond"), zorder=5))
    ax.axvline(0, color="black", lw=0.8, ls=":")
    ax.set_yticks(list(ys) + [0])
    ax.set_yticklabels([pc["case"] for pc in per] +
                       [f"μ = {mu:.1f} (pd={pop['pd']:.2f}, I²={het['I2']:.0%})"])
    ax.set_ylim(-0.8, k + 0.8)
    ax.set_xlabel(sty.xlabel or f"Effect ({unit}), positive = improvement")
    ax.set_title(sty.title or title or "SCED meta-analysis (stage 2): raw vs shrunken", fontsize=sty.title_fs)
    if sty.show_legend:
        ax.legend(framealpha=0.9, **sty.legend_kw(loc="lower right"))
    sty.format_axes(ax, y=False)
    fig.tight_layout()
    if save_path:
        sty.finalize(fig, ax)
        fig.savefig(save_path, dpi=sty.dpi, bbox_inches="tight", transparent=sty.transparent); plt.close(fig)
    return fig


def plot_case_forest(labels, meds, los, his, *, pop=None, highlight=(), save_path=None,
                     fname=None, unit="pts", title=None, style=None):
    """GENERIC horizontal forest per case (+ population): median point + 95% HDI bar,
    dashed 0 line, responders highlighted. ``pop`` = (median, lo, hi) population (bold).
    ``highlight`` = indices of cases to colour (responders). Result output (-> plots/ folder).
    ``style`` (PlotStyle/dict): population/responder/non-responder colours, title, axes, sizes.

    R equivalent: metafor::forest / forestplot::forestplot.
    """
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from functions.common.plotstyle import resolve_style
    st = resolve_style(style); st.apply_rc()
    pop_label = st.label("population", "Population")
    labs = list(labels) + ([pop_label] if pop is not None else [])
    m = list(meds) + ([pop[0]] if pop is not None else [])
    lo = list(los) + ([pop[1]] if pop is not None else [])
    hi = list(his) + ([pop[2]] if pop is not None else [])
    fig, ax = plt.subplots(figsize=(st.figsize or (7.2, 0.55 * len(labs) + 1.4)))
    y = np.arange(len(labs))[::-1]
    for i in range(len(labs)):
        is_pop = (pop is not None and i == len(labs) - 1)
        is_r = (i in highlight) and not is_pop
        col = st.color("population") if is_pop else (st.color("responder") if is_r else st.color("nonresponder"))
        lw = 3.0 if is_pop else 2.0
        ax.plot([lo[i], hi[i]], [y[i], y[i]], color=col, lw=lw, solid_capstyle="round", zorder=2)
        ax.scatter([m[i]], [y[i]], s=(90 if is_pop else 60), color=col, zorder=3,
                   edgecolor="black", linewidth=0.6)
    ax.axvline(0, color=st.color("zero_line"), ls="--", lw=1, zorder=1)
    ax.set_yticks(y); ax.set_yticklabels(labs)
    for t, lab in zip(ax.get_yticklabels(), labs):
        if str(lab) == pop_label:
            t.set_fontweight("bold")
    ax.set_xlabel(st.xlabel or f"effect ({unit}, oriented)"); ax.margins(y=0.08)
    if st.title or title:
        ax.set_title(st.title or title, fontsize=st.title_fs)
    ax.grid(st.grid, axis="x", alpha=st.grid_alpha)
    st.format_axes(ax, y=False)                          # y = case labels, do not reformat
    fig.tight_layout()
    if save_path is not None:
        st.save(fig, save_path, fname or "forest", ax=ax)
    return fig
