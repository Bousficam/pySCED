"""
SCED - Manolov visual aids: VAIOR (2019) and modified Brinley plot (2021)
========================================================================

VAIOR - *Visual Aid Implying an Objective Rule* (Manolov & Vannest 2019, Behavior
Modification, doi:10.1177/0145445519854323). A robust **Theil-Sen** trend is fitted on the
REFERENCE condition, the **MAD** (median absolute deviation from the trend) is computed, a
**variability band** = trend +/- MAD is built and projected, then each point of the
COMPARED condition is classified:
  - **green**  : beyond the band (in the direction of improvement);
  - **gold**   : beyond the trend but within the band;
  - **red**    : neither.
Outputs: percentage of points beyond the band + a dichotomous decision (immediate = 100% of
the first 3; delayed/progressive = 100% of the last 3; overall = percentage >= min(100%,
2x the share of baseline points outside the band)). Reference tool:
https://manolov.shinyapps.io/TrendMAD

Modified Brinley plot (Manolov, Tanious & Onghena 2021, Perspectives on Behavior Science,
doi:10.1007/s40614-021-00289-9; after Blampied 2017): a scatter where each point pairs the
k-th measure of one condition with the k-th of the other; the y=x diagonal separates pairs
favouring one condition from those favouring the other.

Note: VAIOR is a VISUAL AID, not a significance test (no p-value; does not model
autocorrelation) - to be used IN COMPLEMENT to the randomization test and an effect size
(Manolov & Vannest 2019, p. 6, 25).
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _theil_sen_fit(x, y):
    """Theil-Sen slope (median of pairwise slopes) + median intercept (robust).

    References: Theil 1950; Sen 1968 (Theil-Sen robust slope estimator).
    R equivalent: mblm::mblm (Theil-Sen); Python scipy.stats.theilslopes.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    slopes = [(y[j] - y[i]) / (x[j] - x[i])
              for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] != x[i]]
    slope = float(np.median(slopes)) if slopes else 0.0
    intercept = float(np.median(y - slope * x))
    return slope, intercept


def vaior_compare(ref_x, ref_y, comp_x, comp_y, *, improvement="increase"):
    """
    Compute VAIOR for ONE comparison (reference vs compared). Returns a dict:
    ``{slope, intercept, mad, pct_beyond_band, baseline_out_prop, colors, decision_immediate,
    decision_delayed, decision_overall, n_compared}``. ``colors`` in {green, gold, red} per
    compared point (order of ``comp_x``). ``improvement`` orients the relevant bound.

    References: Manolov & Vannest 2019 (VAIOR - Visual Aid Implying an Objective Rule).
    R equivalent: no direct R equivalent (author's TrendMAD shiny app).
    """
    ref_x = np.asarray(ref_x, float); ref_y = np.asarray(ref_y, float)
    comp_x = np.asarray(comp_x, float); comp_y = np.asarray(comp_y, float)
    slope, intercept = _theil_sen_fit(ref_x, ref_y)
    pred_ref = slope * ref_x + intercept
    mad = float(np.median(np.abs(ref_y - pred_ref))) if len(ref_y) else np.nan
    pred_c = slope * comp_x + intercept
    up, lo = pred_c + mad, pred_c - mad
    if improvement == "increase":
        beyond_band = comp_y > up
        beyond_trend = (comp_y > pred_c) & ~beyond_band
    else:
        beyond_band = comp_y < lo
        beyond_trend = (comp_y < pred_c) & ~beyond_band
    colors = np.where(beyond_band, "green", np.where(beyond_trend, "gold", "red"))
    base_out = (float(np.mean((ref_y > pred_ref + mad) | (ref_y < pred_ref - mad)))
                if len(ref_y) and np.isfinite(mad) else np.nan)
    n = len(comp_y)
    pct = 100.0 * float(np.mean(beyond_band)) if n else np.nan
    thr = min(1.0, 2.0 * base_out) if np.isfinite(base_out) else 1.0
    return {
        "slope": round(slope, 4), "intercept": round(intercept, 4), "mad": round(mad, 4),
        "pct_beyond_band": round(pct, 1), "baseline_out_prop": round(base_out, 3),
        "colors": list(colors),
        "decision_immediate": bool(n >= 3 and beyond_band[:3].all()),
        "decision_delayed": bool(n >= 3 and beyond_band[-3:].all()),
        "decision_overall": bool(n and (beyond_band.all() or float(np.mean(beyond_band)) >= thr)),
        "n_compared": int(n)}


def plot_vaior(df, *, session_col, condition_col, outcome_col, reference, compared,
               improvement="increase", title=None, save_path=None, ax=None, fname_suffix="", style=None):
    """
    Draw the VAIOR visual aid: points of the ``reference`` condition + Theil-Sen trend
    + projected +/- MAD band; points of the ``compared`` condition coloured (green/gold/red).
    Annotates the percentage beyond the band and the overall decision. Returns ``(fig, result)``.

    VAIOR is canonically a SINGLE-CASE tool: ``df`` must describe ONE case (one patient, or
    already-aggregated data). For a group design, loop per patient via ``vaior_by_unit``
    rather than pooling (the +/- MAD band would mix within- and between-patient variability).
    ``fname_suffix`` distinguishes the .png files per patient.

    References: Manolov & Vannest 2019 (VAIOR - Visual Aid Implying an Objective Rule).
    R equivalent: no direct R equivalent (author's TrendMAD shiny app).
    """
    sub = df.copy()
    sub[session_col] = pd.to_numeric(sub[session_col], errors="coerce")
    sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")
    r = sub[(sub[condition_col] == reference)].dropna(subset=[session_col, outcome_col]).sort_values(session_col)
    c = sub[(sub[condition_col] == compared)].dropna(subset=[session_col, outcome_col]).sort_values(session_col)
    res = vaior_compare(r[session_col], r[outcome_col], c[session_col], c[outcome_col],
                        improvement=improvement)

    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(sty.figsize or (11, 7)))
    xs = np.linspace(sub[session_col].min(), sub[session_col].max(), 100)
    line = res["slope"] * xs + res["intercept"]
    _ref = sty.color("vaior_reference")
    ax.plot(xs, line, color=_ref, lw=sty.lw(2), label=f"Theil-Sen ({reference})")
    ax.fill_between(xs, line - res["mad"], line + res["mad"],
                    label="+/- MAD band", **sty.band_kw("0.6"))
    ax.scatter(r[session_col], r[outcome_col], facecolor="white", edgecolor=_ref, s=sty.ms(70),
               zorder=4, label=f"Reference ({reference})")
    cmap = {"green": sty.color("vaior_pass"), "gold": sty.color("vaior_warn"), "red": sty.color("vaior_fail")}
    ax.scatter(c[session_col], c[outcome_col],
               color=[cmap[col] for col in res["colors"]], s=sty.ms(90), zorder=5,
               edgecolor="black", linewidth=0.6, label=f"Compared ({compared})")
    ax.set_xlabel(sty.xlabel or "Session"); ax.set_ylabel(sty.ylabel or outcome_col)
    dec = "POSITIVE" if res["decision_overall"] else "negative"
    ax.set_title((sty.title or title or f"VAIOR: {compared} vs {reference}")
                 + f"\n{res['pct_beyond_band']}% beyond band, overall decision: {dec}",
                 fontsize=sty.title_fs)
    if sty.show_legend:
        ax.legend(framealpha=0.9, **sty.legend_kw())
    ax.grid(sty.grid, alpha=max(sty.grid_alpha, 0.25))
    sty.format_axes(ax)
    if fig is not None:
        fig.tight_layout()
        if save_path is not None:
            sty.save(fig, save_path, f"{outcome_col}_vaior_{compared}_vs_{reference}{fname_suffix}", ax=ax)
    return fig, res


def vaior_by_unit(df, *, unit_col, session_col, condition_col, outcome_col, reference, compared,
                  improvement="increase", save_path=None, close=True, style=None):
    """
    VAIOR **per patient** (canonical single-case mode applied to a group design): for each
    level of ``unit_col``, draws a VAIOR figure ``compared`` vs ``reference`` and collects the
    decision. Returns a ``DataFrame`` (one row per patient) with columns
    ``[unit, n_compared, pct_beyond_band, decision_immediate, decision_delayed,
    decision_overall, slope, mad]``. ``close=True`` closes the figures on the fly (avoids
    accumulating them in memory when there are many patients).

    References: Manolov & Vannest 2019 (VAIOR - Visual Aid Implying an Objective Rule).
    R equivalent: no direct R equivalent (author's TrendMAD shiny app).
    """
    rows = []
    for u in sorted(df[unit_col].dropna().unique().tolist(), key=str):
        sub = df[df[unit_col] == u]
        fig, res = plot_vaior(sub, session_col=session_col, condition_col=condition_col,
                              outcome_col=outcome_col, reference=reference, compared=compared,
                              improvement=improvement, save_path=save_path,
                              title=f"VAIOR: {compared} vs {reference} ({u})",
                              fname_suffix=f"_{u}", style=style)
        if close and fig is not None:
            plt.close(fig)
        rows.append({"unit": u, "n_compared": res["n_compared"],
                     "pct_beyond_band": res["pct_beyond_band"],
                     "decision_immediate": res["decision_immediate"],
                     "decision_delayed": res["decision_delayed"],
                     "decision_overall": res["decision_overall"],
                     "slope": res["slope"], "mad": res["mad"]})
    return pd.DataFrame(rows)


def plot_vaior_grid(df, *, unit_col, session_col, condition_col, outcome_col, reference,
                    compared=None, improvement="increase", conditions=None, bounds=None,
                    ncols=None, max_per_fig=10, name=None, title=None, save_path=None, style=None):
    """
    Grid of VAIOR small multiples (canonical single-case mode). ``compared`` = compared
    condition(s); None = all except ``reference``.

    Adaptive layout: with ONE compared condition, patients are laid out over several
    **columns** (auto via ``ncols``: 1 if <=5, 2 if <=10, 3 if <=18, else 4); with several
    comparisons -> patients(rows) x comparisons(columns). Beyond ``max_per_fig`` patients,
    several **figures (batches)** are written (``<name>_vaior_grid[_bN].png``).
    ``bounds=(lo,hi)`` bounds the y axis (real scale of the scores). Returns ``(last_fig, table)``.

    References: Manolov & Vannest 2019 (VAIOR - Visual Aid Implying an Objective Rule).
    R equivalent: no direct R equivalent (author's TrendMAD shiny app).
    """
    units = sorted(df[unit_col].dropna().unique().tolist(), key=str)
    if compared is None:
        conds = list(conditions) if conditions else sorted(
            df[condition_col].dropna().unique().tolist(), key=str)
        comps = [c for c in conds if c != reference]
    else:
        comps = [compared] if isinstance(compared, str) else list(compared)
    single = len(comps) == 1

    def _auto(n):
        return 1 if n <= 5 else (2 if n <= 10 else (3 if n <= 18 else 4))

    import copy as _copy
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    _cell_sty = _copy.deepcopy(sty); _cell_sty.title = None    # cells: keep the per-patient label

    def _draw(d, comp, ax, lbl):
        _, res = plot_vaior(d, session_col=session_col, condition_col=condition_col,
                            outcome_col=outcome_col, reference=reference, compared=comp,
                            improvement=improvement, ax=ax, title=lbl, style=_cell_sty)
        if bounds is not None:
            ax.set_ylim(bounds[0], bounds[1])
        return {"unit": d[unit_col].iloc[0] if len(d) else None, "compared": comp,
                "n_compared": res["n_compared"], "pct_beyond_band": res["pct_beyond_band"],
                "decision_immediate": res["decision_immediate"],
                "decision_delayed": res["decision_delayed"], "decision_overall": res["decision_overall"]}

    base = name if name is not None else str(outcome_col)
    batches = ([units] if len(units) <= max_per_fig
               else [units[i:i + max_per_fig] for i in range(0, len(units), max_per_fig)])
    rows, last = [], None
    for bi, ub in enumerate(batches):
        if single:                                            # one comparison: patients in a grid
            nc = ncols or _auto(len(ub)); nr = int(np.ceil(len(ub) / nc))
            fig, axes = plt.subplots(nr, nc, figsize=(5.6 * nc, 3.9 * nr), squeeze=False)
            flat = [axes[r][c] for r in range(nr) for c in range(nc)]
            for k, u in enumerate(ub):
                rows.append(_draw(df[df[unit_col] == u], comps[0], flat[k],
                                  f"{u}: {comps[0]} vs {reference}"))
            for k in range(len(ub), len(flat)):
                flat[k].axis("off")
        else:                                                 # several comparisons: patients x comps
            nr, ncc = len(ub), max(len(comps), 1)
            fig, axes = plt.subplots(nr, ncc, figsize=(5.6 * ncc, 3.9 * nr), squeeze=False)
            for i, u in enumerate(ub):
                for j, comp in enumerate(comps):
                    rows.append(_draw(df[df[unit_col] == u], comp, axes[i][j],
                                      f"{u}: {comp} vs {reference}"))
        lot = "" if len(batches) == 1 else f" (batch {bi + 1}/{len(batches)})"
        suff = "" if len(batches) == 1 else f"_b{bi + 1}"
        fig.suptitle((sty.title or title or f"VAIOR by patient: {outcome_col}") + lot, fontsize=sty.title_fs + 3)
        fig.tight_layout()
        if save_path is not None:
            sty.save(fig, save_path, f"{base}_vaior_grid{suff}")
        last = fig
    return last, pd.DataFrame(rows)


def plot_brinley(df, *, condition_col, outcome_col, cond_x, cond_y, session_col=None,
                 unit_col=None, mode="classic", bounds=None, title=None, save_path=None, ax=None,
                 style=None):
    """
    Brinley plot. Each point CROSSES a value of ``cond_x`` (abscissa) with one of ``cond_y``
    (ordinate); the y=x diagonal separates points favouring ``cond_y`` (above) from those
    favouring ``cond_x`` (below). With ``unit_col`` the patients are coloured. Two modes:

      - ``mode="classic"`` (DEFAULT; Blampied 2017) - **ONE point per patient**: x = MEAN of
        ``cond_x``, y = MEAN of ``cond_y``. Unambiguous reading (who responds = cloud above),
        aligned with between-patient heterogeneity; loses within-phase variability.
      - ``mode="paired"`` (Manolov, Tanious & Onghena 2021) - pairs the k-th measure of ``cond_x``
        with the k-th of ``cond_y`` (order of ``session_col``) -> several points/patient (shows the
        dispersion); the k-by-k pairing is POSITIONAL, hence purely descriptive.

    ``bounds=(lo, hi)`` bounds both axes to the REAL scale of the outcome (e.g. (0,100)) and draws
    the diagonal over that full extent (honest reading: no auto-zoom that exaggerates the effect).
    ``ax`` allows composing the figure as a panel (otherwise a dedicated figure is created). Returns the Figure.

    References: Blampied 2017 (Brinley plot); Manolov, Tanious & Onghena 2021 (modified Brinley plot).
    R equivalent: no direct R equivalent (draw manually with ggplot2 / base graphics).
    """
    if mode not in ("classic", "paired"):
        raise ValueError("mode: 'classic' (1 point/patient, means) or 'paired' (k-by-k paired).")
    from functions.common.plotstyle import resolve_style
    sty = resolve_style(style); sty.apply_rc()
    fig = None
    if ax is None:
        fig, ax = plt.subplots(figsize=(sty.figsize or (8, 8)))
    units = ([None] if unit_col is None else
             sorted(df[unit_col].dropna().unique().tolist(), key=str))
    palette = plt.get_cmap(sty.palette)(np.linspace(0, 1, max(10, len(units))))
    allv = []
    for i, u in enumerate(units):
        d = df if u is None else df[df[unit_col] == u]
        if session_col:
            d = d.sort_values(session_col)
        xa = pd.to_numeric(d[d[condition_col] == cond_x][outcome_col], errors="coerce").dropna().to_numpy()
        ya = pd.to_numeric(d[d[condition_col] == cond_y][outcome_col], errors="coerce").dropna().to_numpy()
        if len(xa) == 0 or len(ya) == 0:
            continue
        if mode == "classic":                       # 1 point = (mean A, mean B) of the patient
            xa, ya = np.array([xa.mean()]), np.array([ya.mean()])
        else:                                        # k-by-k paired (truncated to the shorter)
            m = min(len(xa), len(ya)); xa, ya = xa[:m], ya[:m]
        allv.extend([*xa, *ya])
        ax.scatter(xa, ya, color=palette[i], s=(110 if mode == "classic" else 70),
                   alpha=0.8, edgecolor="white", linewidth=0.6,
                   label=(None if u is None else str(u)), zorder=3)
    if bounds is not None:                                # real scale of the outcome (e.g. 0-100)
        lo, hi = float(bounds[0]), float(bounds[1])
    elif allv:
        lo, hi = min(allv), max(allv)
    else:
        lo = hi = None
    if lo is not None:
        ax.plot([lo, hi], [lo, hi], "--", color="0.3", lw=1.5, zorder=1)   # y=x diagonal
        if bounds is not None:
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(sty.xlabel or f"Baseline (Phase {cond_x})")
    ax.set_ylabel(sty.ylabel or f"Treatment (Phase {cond_y})")
    _lbl = "Brinley plot" if mode == "classic" else "Modified Brinley plot"
    ax.set_title(sty.title or title or _lbl, fontsize=sty.title_fs)
    if unit_col is not None and sty.show_legend:
        ax.legend(title="Patient", **sty.legend_kw())
    ax.grid(sty.grid, alpha=max(sty.grid_alpha, 0.25))
    sty.format_axes(ax)
    ax.set_aspect("equal", adjustable=("box" if bounds is not None else "datalim"))
    if fig is not None:
        fig.tight_layout()
        if save_path is not None:
            sty.save(fig, save_path, f"{outcome_col}_brinley_{mode}_{cond_y}_vs_{cond_x}", ax=ax)
    return fig
