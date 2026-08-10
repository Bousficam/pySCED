"""Canonical SCED CLUSTER analysis + reporting - the map-valued counterpart of
report_sced_alternating. DOMAIN-NEUTRAL : given an element map Y (n_obs, n_elements), an adjacency,
and a caller-supplied ``label_fn(idx, cluster_id) -> tag`` that names a cluster from its element
indices, it runs the DECLARED ANCOVA (run_ancova, role-adapted null per term),
extracts the significant clusters, and emits ONE grouped SCED report per axis (units = the axis's
clusters). The core knows nothing about Hz, electrodes, connectivity or any grid semantics - that is
entirely in ``label_fn`` (see spectral.py for the electrode x frequency labeller). Mass-univariate
vs network is ONLY the adjacency. So a domain template = build (Y, adjacency, design terms, label_fn)
and call this."""
import os

import numpy as np

from .design import run_ancova

__all__ = ["report_sced_cluster", "describe_clusters", "summary_rows",
           "grouped_axis_reports", "axis_family", "scalar_adjusted_effects",
           "cluster_effect_size_bounds", "apriori_effect_size"]


def axis_family(axis, res=None):
    """Group an axis into its test-type folder (report lands in <outcome>/<family>/). Prefers
    res['family'] stamped by run_ancova (the descriptor is the source of truth) ; falls back to the
    name convention only for results not produced by the descriptor."""
    if res is not None and res.get("family"):
        return res["family"]
    if axis.startswith("phase"):
        return "phase"
    return axis


def describe_clusters(res, alpha, axis, label_fn):
    """Significant clusters of ONE axis as [{name, p, idx}]. The NAME is domain-supplied :
    name = f"{axis}_{label_fn(idx, cluster_id)}" - the toolbox turns element indices into a tag only
    through the caller's label_fn (no Hz / electrode knowledge here). idx = flat element indices of the
    cluster."""
    out = []
    comp = res.get("comp_labels")
    if comp is None:
        return out
    for lab, p in res.get("comp_pvals", {}).items():
        if p >= alpha:
            continue
        idx = np.where(comp == int(lab))[0]                    # cluster elements (flat)
        out.append({"name": f"{axis}_{label_fn(idx, int(lab))}", "p": float(p), "idx": idx})
    return out


def cluster_effect_size_bounds(res, *, n_obs, alpha=0.05, grid_shape=None,
                               conversion="within", n_groups=None, df=None):
    """Cohen's d AND Hedges' g for each SIGNIFICANT cluster, as an explicit UPPER/LOWER BOUND PAIR.

    In : one axis result from ``cluster_run`` / ``spatial_*`` (needs ``stat``, ``comp_labels``,
    ``comp_pvals``, ``stat_kind``), the number of observations the statistic was computed on, and
    optionally the element lattice's ``grid_shape`` (n_rows, n_cols) so a flat cluster can be boxed.
    ``df`` overrides the degrees of freedom used by the small-sample correction; leave it out and
    the conversion's own convention applies (see below).
    Out : a list of dicts, one per significant cluster. No side effect, nothing mutated.

    d AND g, BOTH RETURNED. d is the raw standardized statistic; g is d times Hedges' (1981)
    small-sample factor ``J = 1 - 3/(4*df - 1)``, the same factor and the same algebra as
    ``core.effect_sizes.hedges_g_within`` and ``core.effect_sizes.hedges_g``, so a cluster effect
    size and a scalar one are on the same scale. d is UPWARD biased in small samples, which is the
    regime a single-case design lives in: at n_obs = 18 sessions J = 0.955, so d overstates by
    about 5 %, and at n_obs = 8 by about 11 %. REPORT g. d is kept alongside because it is what the
    cluster literature quotes and dropping it would make the numbers incomparable to Meyer et al.

    ``df`` defaults to the conversion's convention: ``n_obs - 1`` for "within" (one-sample /
    paired) and ``n1 + n2 - 2`` for "between". Those are the same defaults the scalar functions
    use. When the statistic comes from a GLM with k design columns the exact residual df is
    ``n_obs - k``, which is SMALLER, so J is smaller and g shrinks further; the caller knows k and
    the core does not, hence the parameter. The value actually used is returned in ``df_used`` so a
    report never has to guess which convention produced the number.

    WHY A PAIR AND NOT ONE NUMBER. A cluster has an irregular shape, so "the effect size of the
    cluster" is not a single well-defined quantity. Meyer, Lamers, Kayhan, Hunnius & Oostenveld
    (2021, Dev. Cogn. Neurosci. 52:101036, p. 6-8) compute three variants on one dataset and they
    disagree by roughly a factor of two (their infant ERP cluster : max 1.064, rectangle 0.557).
    Their recommendation, followed here, is to report the MAXIMUM within the cluster as the upper
    bound (it is positively biased - it is the element the cluster was selected for) and the
    RECTANGLE circumscribing the cluster as the lower bound (it dilutes the effect with the
    non-significant elements the box sweeps in), each with the exact extent it covers. The
    average-over-cluster variant is also returned but is DISCOURAGED for planning a future study,
    because the irregular support makes it poorly reproducible.

    This is descriptive, not inferential. The cluster p already carries the inference, and per
    R-CBP-9/10/11 the cluster's extent is not an estimate of the effect's location or timing - so
    ``extent_*`` fields below describe the box, they do not license a claim about where the effect is.

    FIDELITY CAVEAT on the aggregate bounds. ``d_max`` is exact - Meyer's Option 2 IS the largest
    per-element d. ``d_rect`` and ``d_mean_cluster`` are element-space APPROXIMATIONS: they average
    the per-element d map, whereas Meyer averages the DATA over the region and computes one d from
    it. Averaging data suppresses noise, so the faithful value is generally larger in magnitude than
    the element-averaged one. This function only receives the result, not ``Y``, so it cannot do the
    faithful computation; ``apriori_effect_size`` (which does receive ``Y``) does it correctly for a
    pre-specified region, and is the preferred route whenever a region was declared in advance.

    ``conversion`` maps the per-element statistic to d : "within" (paired / one-sample, d = t/sqrt(n),
    Meyer's own design) or "between" (two independent groups, d = t*sqrt(1/n1 + 1/n2), needs
    ``n_groups=(n1, n2)``).

    WHICH STATISTIC MAPS ARE ACCEPTED. Only a SIGNED single-column statistic converts, because only
    then is there one coefficient to standardise :

      - ``'t'`` / ``'rank'``       -> Cohen's d in the usual pooled-SD sense.
      - ``'W'`` / ``'G'``, ONE tested column -> accepted, but the SE in the ratio is White's HC0
        (W) or the variance-group-weighted one (G, Winkler et al. 2014), NOT the OLS pooled SD. The
        result is the ROBUST analogue of d and will differ from a pooled-SD d exactly where the
        errors are heteroscedastic - which is the reason the robust statistic was chosen in the
        first place. The returned ``d_kind`` field says so, so a report never labels it plainly
        "Cohen's d". G with a single variance group reduces to F / t, so the distinction is inert there.
      - ``'F'``, or ``'W'`` / ``'G'`` with SEVERAL tested columns (referenced to chi2) -> REJECTED.
        These test a block of columns jointly; there is no single coefficient, so no d exists. The
        function raises instead of returning a number.

    The signed-vs-unsigned distinction is read from ``res['signed']`` (stamped by ``cluster_run``),
    because a single-column and a multi-column W both arrive stamped ``stat_kind='W'`` and cannot be
    told apart from the map alone.
    """
    kind = str(res.get("stat_kind", "t"))
    stat = np.asarray(res.get("stat"), dtype=float)
    if stat.ndim != 1:
        raise ValueError("cluster_effect_size_bounds expects a 1-D per-element statistic map")
    # WHICH MAPS CARRY AN EFFECT SIZE. Only a statistic that IS a coefficient over its standard
    # error converts to a standardized mean difference. `signed` is read from the result because it
    # cannot be re-derived here: `_glm_statmap` returns a SIGNED robust ratio for a single-column
    # W / G and an UNSIGNED chi2 for a multi-column one, and both are stamped stat_kind='W'. Falling
    # back to "does the map contain a negative value" would misread an all-positive signed map as
    # unsigned, and (worse) an unsigned chi2 map as signed - that fallback previously let a
    # multi-column W through and produced a meaningless d.
    signed = res.get("signed")
    if signed is None:                          # result predates the `signed` field: infer, conservatively
        signed = kind in ("t", "rank") or (kind in ("W", "G") and bool(np.any(stat < 0)))
    if kind == "F" or not signed:
        raise ValueError(
            f"stat_kind={kind!r} with signed={signed!r} is an unsigned omnibus statistic (F, or a "
            "multi-column W / G referenced to chi2): it tests a block of columns jointly, so there "
            "is no single coefficient and no Cohen's d is defined for it. Re-run the axis with one "
            "tested column (a signed 't' or 'W') if a cluster effect size is wanted.")
    if kind in ("W", "G"):
        # A signed W / G IS beta over a standard error, so the t -> d algebra carries over, but the
        # SE is not the OLS one: W uses White's HC0 (heteroscedasticity-consistent) SE, and G uses a
        # variance-group-weighted SE (Winkler et al. 2014). The resulting d is therefore the ROBUST
        # analogue of Cohen's d, not Cohen's d computed on a pooled SD - it will differ, by design,
        # exactly where the errors are heteroscedastic. Reported in the row so the label never
        # overstates what was computed. (G with a single variance group reduces exactly to F / t,
        # so there the distinction is inert.)
        d_label = (f"robust d (from {kind}: coefficient over a "
                   + ("White HC0" if kind == "W" else "variance-group-weighted")
                   + " SE, not a pooled-SD Cohen's d)")
    else:
        d_label = "Cohen's d (from t, pooled-SD convention)"
    if conversion == "within":
        scale = 1.0 / np.sqrt(float(n_obs))
        df_def = float(n_obs) - 1.0
    elif conversion == "between":
        if not n_groups or len(n_groups) != 2:
            raise ValueError("conversion='between' requires n_groups=(n1, n2)")
        n1, n2 = (float(v) for v in n_groups)
        scale = np.sqrt(1.0 / n1 + 1.0 / n2)
        df_def = n1 + n2 - 2.0
    else:
        raise ValueError("conversion must be 'within' or 'between'")
    d_map = stat * scale                                    # per-element Cohen's d
    # HEDGES' SMALL-SAMPLE CORRECTION, the same factor as core.effect_sizes (Hedges 1981). It is a
    # single scalar, so it applies element-wise and every bound derived from the map is corrected
    # consistently - correcting only the maximum would leave the pair on two different scales.
    # Guarded below df = 1.25, where the factor turns negative and would FLIP the sign of g.
    df_used = float(df) if df is not None else df_def
    J = 1.0 - 3.0 / (4.0 * df_used - 1.0) if df_used > 1.25 else np.nan
    g_map = d_map * J

    comp = res.get("comp_labels")
    out = []
    if comp is None:
        return out
    comp = np.asarray(comp)
    for lab, p in (res.get("comp_pvals") or {}).items():
        if p >= alpha:
            continue
        m = comp == int(lab)
        idx = np.where(m)[0]
        if idx.size == 0:
            continue
        dc = d_map[m]
        # UPPER BOUND : the extreme element, taken in the cluster's own direction (a negative
        # cluster's bound is its most negative d, not its largest absolute value).
        i_max = idx[int(np.argmax(np.abs(dc)))]
        d_max = float(d_map[i_max])
        row = {"cluster": int(lab), "p": float(p), "n_elements": int(idx.size),
               "d_max": round(d_max, 4), "d_max_element": int(i_max),
               "d_mean_cluster": round(float(np.mean(dc)), 4),
               "g_max": round(float(g_map[i_max]), 4),
               "g_mean_cluster": round(float(np.mean(g_map[m])), 4),
               "hedges_J": round(float(J), 4) if np.isfinite(J) else np.nan,
               "df_used": df_used,
               "d_kind": d_label, "stat_kind": kind,
               "conversion": conversion, "n_obs": int(n_obs),
               "bound_note": ("d_max is a positively biased UPPER bound (the element the cluster was "
                              "selected for); d_rect is a LOWER bound. Report both. d_mean_cluster is "
                              "discouraged for planning future studies (irregular support)."),
               "g_note": ("g_* is d_* times Hedges' J = 1 - 3/(4*df - 1), the same small-sample "
                          "correction as core.effect_sizes. Report g: d is upward biased at these "
                          "sample sizes. d is kept for comparability with the cluster literature."),
               }
        # LOWER BOUND : the axis-aligned rectangle circumscribing the cluster on the element lattice.
        # Needs the grid geometry, which the domain-neutral core does not know - so it is optional and
        # its absence is REPORTED rather than silently skipped.
        if grid_shape is not None:
            rows_, cols_ = np.unravel_index(idx, tuple(grid_shape))
            r0, r1 = int(rows_.min()), int(rows_.max())
            c0, c1 = int(cols_.min()), int(cols_.max())
            box = np.zeros(tuple(grid_shape), dtype=bool)
            box[r0:r1 + 1, c0:c1 + 1] = True
            d_rect = float(np.mean(d_map.reshape(tuple(grid_shape))[box]))
            row.update({"d_rect": round(d_rect, 4),
                        "g_rect": round(float(np.mean(g_map.reshape(tuple(grid_shape))[box])), 4),
                        "extent_rows": (r0, r1), "extent_cols": (c0, c1),
                        "extent_n_box": int(box.sum()),
                        "box_fill": round(float(idx.size) / float(box.sum()), 3)})
        else:
            row.update({"d_rect": np.nan, "g_rect": np.nan,
                        "d_rect_note": "grid_shape not supplied: the circumscribing-rectangle LOWER "
                                       "bound needs the element lattice geometry (n_rows, n_cols). "
                                       "Only the upper bound is available."})
        out.append(row)
    return out


def apriori_effect_size(Y, mask, *, effect, nuisance=(), effect_kind="continuous",
                        nuisance_kind="continuous", stat="t", conversion="within",
                        n_groups=None, df=None, n_perm=0, seed=0):
    """Effect size on a PRE-SPECIFIED set of elements - no cluster, hence no bound pair.

    In : the element map ``Y`` (n_obs, n_elements), a boolean ``mask`` (or an index array) naming the
    elements the hypothesis was declared on, and the same design terms the cluster axes use. Out : a
    single dict. No side effect.

    WHEN TO PREFER THIS OVER ``cluster_effect_size_bounds``. The upper/lower bracket exists only
    because a data-driven cluster has an irregular shape discovered in the same data that is being
    summarised. When a channel / time-window / frequency-band hypothesis was specified in advance,
    that ambiguity does not arise and Meyer et al. (2021, p. 6) say to "skip cluster-based
    effect-size ambiguity and compute Cohen's d directly on that pre-defined parameter set". The
    number returned here is therefore a single point estimate, and a BETTER one: it pays neither the
    selection bias of the maximum-within-cluster bound nor the dilution of the circumscribing box.
    The mask must be pre-registered; deriving it from the same data reintroduces exactly the
    selection bias this route avoids, while dropping the bracket that made the bias visible.

    HOW IT DIFFERS FROM THE CLUSTER BOUNDS, AND WHY THAT MATTERS. Here the DATA are averaged over
    the mask first (one series per observation), then the model is fitted and the statistic
    converted to d - which is Meyer's own procedure for their Options 1 and 3. The cluster bounds
    instead average the per-element d MAP, because ``cluster_effect_size_bounds`` receives only the
    result, not ``Y``. The two are not identical: averaging the data suppresses noise, so the d of
    the averaged signal is generally LARGER than the average of the per-element d values. Treat the
    cluster ``d_rect`` / ``d_mean_cluster`` as element-space approximations and this function's
    value as the faithful computation.

    ``n_perm`` > 0 also returns a design-based p for the averaged series, via the scalar
    Freedman-Lane engine (``core.permutation.freedman_lane_test``), so the pre-specified test is
    inferential as well as descriptive. Returns ``{d, g, slope, stat, stat_kind, hedges_J, df_used,
    n_elements, p, n_perm, route}``.
    """
    from .core import _fl_design, _glm_statmap
    Y = np.asarray(Y, dtype=float)
    m = np.asarray(mask)
    if m.dtype != bool:                                  # accept an index array as well as a mask
        idx = m.astype(int)
        m = np.zeros(Y.shape[1], dtype=bool)
        m[idx] = True
    if not m.any():
        raise ValueError("apriori_effect_size: the mask selects no element")
    ym = Y[:, m].mean(axis=1, keepdims=True)             # AVERAGE THE DATA first (Meyer's procedure)
    nuis = list(nuisance) if nuisance is not None else []
    design, test_cols, nuis_cols, st = _fl_design(ym, effect, nuis, effect_kind, nuisance_kind, stat)
    statmap, effmap, Yv, _Z, _pZ, _thr, signed = _glm_statmap(
        ym, design, test_cols, nuis_cols, st, None, 0.5)
    if not signed:
        raise ValueError(
            f"stat={st!r} is unsigned here (several tested columns): no single coefficient, so no d. "
            "Test one column at a time for an a priori effect size.")
    tval = float(statmap(Yv)[0])
    n_obs = int(Y.shape[0])
    if conversion == "within":
        scale, df_def = 1.0 / np.sqrt(float(n_obs)), float(n_obs) - 1.0
    elif conversion == "between":
        if not n_groups or len(n_groups) != 2:
            raise ValueError("conversion='between' requires n_groups=(n1, n2)")
        n1, n2 = (float(v) for v in n_groups)
        scale, df_def = np.sqrt(1.0 / n1 + 1.0 / n2), n1 + n2 - 2.0
    else:
        raise ValueError("conversion must be 'within' or 'between'")
    df_used = float(df) if df is not None else df_def
    J = 1.0 - 3.0 / (4.0 * df_used - 1.0) if df_used > 1.25 else np.nan
    d = tval * scale
    out = {"d": round(float(d), 4), "g": round(float(d * J), 4) if np.isfinite(J) else np.nan,
           "slope": round(float(effmap(Yv)[0]), 6), "stat": round(tval, 4), "stat_kind": st,
           "hedges_J": round(float(J), 4) if np.isfinite(J) else np.nan, "df_used": df_used,
           "n_elements": int(m.sum()), "n_obs": n_obs, "conversion": conversion,
           "route": "a priori pre-specified mask (no cluster selection, no bound pair; "
                    "Meyer et al. 2021 p. 6). Report g, not d.",
           "p": np.nan, "n_perm": 0}
    if n_perm:
        from ..core import freedman_lane_test
        fl = freedman_lane_test(ym.ravel(), effect=effect, nuisance=(nuis or None),
                                effect_kind=effect_kind, nuisance_kind=nuisance_kind,
                                stat=("W" if st == "W" else "t"), n_perm=int(n_perm), seed=seed)
        out.update({"p": round(float(fl["p"]), 4), "n_perm": int(fl["n_perm"])})
    return out


def summary_rows(res_by_axis, eff_by_axis, alpha, label_fn, base, *, es=None):
    """One row PER AXIS for a consolidated report : the test, its cluster p, the number and names of
    significant clusters and the mean effect (slope/contrast) over the significant set. base : the caller's fixed fields (subject, level, outcome, tail...).

    ``es`` : optional kwargs forwarded to ``cluster_effect_size_bounds`` (``n_obs`` and, for the
    lower bound, ``grid_shape``). When given, each row gains the effect-size BRACKET of the axis's
    DOMINANT cluster - the one with the largest |upper bound| - as ``d_upper`` / ``d_lower`` and
    ``g_upper`` / ``g_lower``, identified by ``es_cluster`` and ``es_cluster_p``. Both members come
    from that one cluster, because a bracket that mixed clusters would pair the peak of one effect
    with the floor of another. ``g_lower_axis_weakest`` is reported SEPARATELY (not as part of the
    bracket): the weakest lower bound across all significant clusters, answering "what is the least
    this axis shows?". Prefer the g fields - d is upward biased at single-case sample sizes.
    Left out (``es=None``) the row is exactly what it was before, so existing reports are unchanged.
    An axis whose statistic admits no effect size (unsigned F / multi-column W or G) records the
    reason in ``d_note`` rather than dropping the column or emitting a bare NaN."""
    rows = []
    for axis, res in res_by_axis.items():
        descs = describe_clusters(res, alpha, axis, label_fn)
        sig = res["sig_elements"]; eff = eff_by_axis[axis]
        row = {**base, "axis": axis,
               "n_sig_elem": int(sig.sum()), "n_clusters": len(descs),
               "clusters": "; ".join(d["name"] for d in descs),
               "mean_slope": float(np.nanmean(np.where(sig, eff, np.nan))) if sig.any() else np.nan,
               "min_p": min(res["comp_pvals"].values()) if res["comp_pvals"] else 1.0}
        if es is not None:
            try:
                bounds = cluster_effect_size_bounds(res, alpha=alpha, **es)
                if bounds:
                    # THE PAIR MUST DESCRIBE ONE CLUSTER. Meyer et al. (2021) bracket a single
                    # effect between its upper and its lower bound, so both members are read off
                    # the SAME cluster - the axis's dominant one (largest |upper bound|). Taking
                    # the max upper across clusters and the min lower across clusters, as an
                    # earlier version did, pairs the peak of one effect with the floor of another:
                    # with clusters of opposite sign it produced d_upper = +1.25 / d_lower = -0.69,
                    # an "interval" straddling zero although neither cluster crosses zero.
                    top = max(bounds, key=lambda b: abs(b["d_max"]))
                    row.update({"es_cluster": top["cluster"], "es_cluster_p": top["p"],
                                "d_upper": top["d_max"], "d_lower": top.get("d_rect", np.nan),
                                "g_upper": top["g_max"], "g_lower": top.get("g_rect", np.nan),
                                "d_kind": top["d_kind"],
                                "hedges_J": top["hedges_J"], "df_used": top["df_used"],
                                "n_clusters_with_es": len(bounds)})
                    # Separate, axis-wide and deliberately NOT part of the pair: the weakest lower
                    # bound over all significant clusters, i.e. the most conservative reading the
                    # axis admits. Kept because it answers "what is the least this axis shows?",
                    # which the dominant cluster's own bracket does not.
                    grects = [b["g_rect"] for b in bounds if np.isfinite(b.get("g_rect", np.nan))]
                    row["g_lower_axis_weakest"] = min(grects, key=abs) if grects else np.nan
                    if not np.isfinite(row["d_lower"]):
                        row["d_note"] = "no grid_shape: lower bound unavailable, upper bound only"
            except ValueError as e:                 # unsigned statistic : say why, do not emit a number
                row["d_note"] = str(e).split(":")[0]
        # Fixed-covariate robust grid : primary = HJ+W (min_p above) ; surface the {FL,HJ}x{F,W}
        # sensitivity cells and the Breusch-Pagan homoscedasticity share so the report shows whether
        # F and W agree (homoscedastic) or diverge (keep the robust W).
        if res.get("sensitivity"):
            row["scheme"] = res.get("scheme", "")
            for c in res["sensitivity"]:
                row[f"p_{c['scheme'].split('-')[0]}_{c['stat']}"] = round(c["min_p"], 4)  # p_freedman_F ...
            bp = res.get("homoscedasticity") or {}
            row["frac_hetero"] = bp.get("frac_hetero", np.nan)
        rows.append(row)
    return rows


def grouped_axis_reports(res_by_axis, Y, phase, alpha, label_fn, report_cfg, report_dir):
    """One canonical SCED report per axis, filed by TEST-TYPE FAMILY : <report_dir>/<family>/<axis>/
    with family in {trend, phase} (phase groups every contrast). All that axis's significant
    clusters are stacked as UNITS (report_sced_alternating, unit_col='cluster' -> the 'Per Unit'
    table lists every cluster) ; a single cluster degrades to an n-of-1 report. report_cfg carries
    the SCED knobs (cond_order, dose, perm_phase, block_size, max_consecutive, n_perm)."""
    import pandas as pd
    from functions.sced.alternating.run import report_sced_alternating   # lazy : pulls statsmodels
    ph = [str(p) for p in phase]
    for axis, res in res_by_axis.items():
        cls = describe_clusters(res, alpha, axis, label_fn)
        if not cls:
            continue
        recs = []                                              # tidy : one row per (cluster, session)
        for cl in cls:
            val = Y[:, cl["idx"]].mean(axis=1)                 # cluster-mean value per session
            recs += [{"cluster": cl["name"], "session_idx": s, "phase": ph[s], "outcome": axis,
                      "value": float(val[s])} for s in range(len(val))]
        report_sced_alternating(
            pd.DataFrame(recs), unit_col=("cluster" if len(cls) > 1 else None),
            session_col="session_idx", condition_col="phase", outcome_name_col="outcome",
            value_col="value", outcomes=(axis,), conditions=report_cfg["cond_order"],
            condition_order=report_cfg["cond_order"], dose=report_cfg["dose"], time_covariate="auto",
            perm_method=report_cfg["perm_phase"], block_size=report_cfg.get("block_size"),
            max_consecutive=report_cfg.get("max_consecutive"), trend_test=True,
            n_perm=report_cfg["n_perm"], output_dir=os.path.join(report_dir, axis_family(axis, res)),
            plot_by="session", verbose=False)


def scalar_adjusted_effects(values, sessions, labels, cond_order, *, perm_phase="draper-stoneman",
                            block_size=None, max_consecutive=None, n_perm=2000, seed=0,
                            covariates=None, cov_names=None):
    """ANCOVA effect ESTIMATES + two-sided p for a single scalar series, one row per model term, each
    with the role-adapted null : the TIME slope (fixed covariate -> Freedman-Lane), each nuisance
    COVARIATE slope (e.g. fatigue ; fixed covariate -> Freedman-Lane) and every PHASE contrast (level
    - reference, the randomized dose -> Draper-Stoneman). This is the full ANCOVA
    ``outcome ~ time + phase + covariate(s)`` : every term is estimated adjusting for all the others.
    Reuses the map engine at n_elements=1 (a scalar is the trivial 1-element map).

    ``covariates`` : optional (n_sess,) or (n_sess, k) array of nuisance covariate(s) (e.g. fatigue) ;
    ``cov_names`` : their names. Rows with a NaN in the outcome / time / any covariate are dropped
    (complete-case) so the estimates stay defined.

    Returns a tidy DataFrame [Effect, Estimate, Unit, p (two-sided), Null, Interpretation]. Every p is
    a plain TWO-SIDED permutation p of the term's statistic (primary threshold set to 0, so the single
    element always 'clusters' and no thresholding gates the p) - NOT a cluster p ; a sub-threshold
    effect keeps its estimate instead of collapsing to p=1. The coefficient p-values are two-sided (no
    directional hypothesis on the estimate itself), unlike the omnibus condition F which is one-tailed
    by construction (upper tail of the F). Complements the omnibus randomization test and the
    nonoverlap ES, and fills the MISSING time-slope + covariate estimates of the alternating report."""
    import pandas as pd
    from scipy import sparse
    from .design import term, run_ancova
    from .spatial import spatial_contrast
    values = np.asarray(values, dtype=float)
    time = np.asarray(sessions, dtype=float)
    labels = np.asarray(labels)
    # covariate(s) matrix (n_sess, k)
    C = None
    if covariates is not None:
        C = np.asarray(covariates, dtype=float)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        names = list(cov_names) if cov_names is not None else [f"cov{j}" for j in range(C.shape[1])]
    else:
        names = []
    # complete-case mask : an estimate is undefined where the outcome, time or a covariate is missing
    ok = np.isfinite(values) & np.isfinite(time)
    if C is not None:
        ok &= np.isfinite(C).all(axis=1)
    values, time, labels = values[ok], time[ok], labels[ok]
    if C is not None:
        C = C[ok]
    Y = values.reshape(-1, 1)                                  # (n_sess, 1 element)
    adj = sparse.csr_matrix((1, 1))                            # one element, no neighbour
    kw = dict(primary_p=0.5, n_perm=n_perm, cluster_stat="intensity", tail="both", seed=seed)  # thr=0
    rows = []
    adj_note = "phase" + ("".join(f" + {n}" for n in names) if names else "")   # 'what time is adjusted for'
    # FULL ANCOVA in one router call : time + each covariate (fixed -> FL) + phase (randomized -> DS)
    fixed = [term("trend", time, role="fixed", kind="continuous")]
    for j, nm in enumerate(names):
        fixed.append(term(nm, C[:, j], role="fixed", kind="continuous"))
    allterms = fixed + [term("phase", labels, role="randomized", kind="factor", cond_order=cond_order)]
    rt, _clab, _pm = run_ancova(Y, adj, allterms, **kw)
    # time slope, adjusted for phase (+ covariates)
    tp = min(rt["trend"]["comp_pvals"].values()) if rt["trend"]["comp_pvals"] else 1.0
    rows.append({"Effect": "time slope", "Estimate": round(float(rt["trend"]["effect"][0]), 5),
                 "Unit": "/session", "p (two-sided)": round(float(tp), 4),
                 "Null": "Freedman-Lane (fixed covariate)",
                 "Interpretation": f"change in the outcome per session, adjusting for {adj_note}."})
    # covariate slopes (e.g. fatigue), adjusted for time + phase (+ the other covariates)
    for nm in names:
        cp = min(rt[nm]["comp_pvals"].values()) if rt[nm]["comp_pvals"] else 1.0
        other = "time + phase" + "".join(f" + {n}" for n in names if n != nm)
        rows.append({"Effect": f"{nm} slope", "Estimate": round(float(rt[nm]["effect"][0]), 5),
                     "Unit": f"/unit {nm}", "p (two-sided)": round(float(cp), 4),
                     "Null": "Freedman-Lane (fixed covariate)",
                     "Interpretation": f"change in the outcome per unit {nm}, adjusting for {other}."})
    # phase contrasts (randomized dose -> Draper-Stoneman), adjusted for time + covariates
    ref = str(cond_order[0])
    nuis = [time] + ([C[:, j] for j in range(C.shape[1])] if C is not None else [])
    nkind = ["continuous"] * len(nuis)
    for c in [str(x) for x in cond_order[1:]]:
        rc = spatial_contrast(Y, adj, factor=labels, contrast=c, cond_order=cond_order,
                              nuisance=nuis, nuisance_kind=nkind, perm_method=perm_phase,
                              block_size=block_size, max_consecutive=max_consecutive, **kw)
        pc = min(rc["comp_pvals"].values()) if rc["comp_pvals"] else 1.0
        rows.append({"Effect": f"{c} - {ref} contrast", "Estimate": round(float(rc["effect"][0]), 5),
                     "Unit": "outcome units", "p (two-sided)": round(float(pc), 4),
                     "Null": "Draper-Stoneman (randomized dose)",
                     "Interpretation": f"outcome difference {c} vs {ref} at matched {adj_note.replace('phase', 'time')}."})
    return pd.DataFrame(rows, columns=["Effect", "Estimate", "Unit", "p (two-sided)", "Null", "Interpretation"])


def group_adjusted_effects(values, sessions, labels, units, cond_order, *, covariates=None,
                           cov_names=None, n_perm=2000, seed=0):
    """GROUP (multi-unit) analogue of ``scalar_adjusted_effects`` : the full ANCOVA
    ``outcome ~ unit + time + phase + covariate(s)`` as a pooled OLS with UNIT FIXED EFFECTS
    (absorbs each patient's baseline level, matching the within-unit stratification), one row per
    term with a WITHIN-UNIT (stratified) two-sided permutation p. The time & covariate slopes get a
    within-unit Freedman-Lane null (fixed covariate : permute the reduced-model residuals within
    unit) ; the phase contrasts get a within-unit label-permutation null (randomized dose ->
    Draper-Stoneman, stratified). Complete-case on outcome / time / covariate(s).

    Returns the same tidy columns as ``scalar_adjusted_effects`` [Effect, Estimate, Unit,
    p (two-sided), Null, Interpretation] ; the Estimate is now the group fixed-effects coefficient
    (the common within-patient effect). The schedule constraints (max_consecutive / block_size) are
    NOT re-imposed on this complementary estimate - the PRIMARY stratified F already carries the exact
    design test ; here the within-unit label permutation is free."""
    import pandas as pd
    rng = np.random.RandomState(seed)
    y = np.asarray(values, float); t = np.asarray(sessions, float)
    lab = np.asarray(labels); u = np.asarray(units)
    C = None
    if covariates is not None:
        C = np.asarray(covariates, float)
        if C.ndim == 1:
            C = C.reshape(-1, 1)
        names = list(cov_names) if cov_names is not None else [f"cov{j}" for j in range(C.shape[1])]
    else:
        names = []
    ok = np.isfinite(y) & np.isfinite(t)
    if C is not None:
        ok &= np.isfinite(C).all(1)
    y, t, lab, u = y[ok], t[ok], lab[ok], u[ok]
    if C is not None:
        C = C[ok]
    uu = list(dict.fromkeys(u.tolist()))                         # unit order
    U = np.column_stack([(u == k).astype(float) for k in uu])    # unit fixed effects (no global intercept)
    order = [str(x) for x in cond_order]; ref = order[0]
    covcols = [C[:, j] for j in range(C.shape[1])] if C is not None else []

    def _build(l):                                               # X = [unit FE | time | cov... | phase...]
        cols = [U, t.reshape(-1, 1)] + [c.reshape(-1, 1) for c in covcols]
        cols += [np.column_stack([(np.asarray(l) == c).astype(float) for c in order[1:]])]
        return np.column_stack(cols)
    nU = U.shape[1]; i_time = nU
    i_cov = [nU + 1 + j for j in range(len(covcols))]
    i_phase = [nU + 1 + len(covcols) + j for j in range(len(order) - 1)]
    X = _build(lab)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    obs = {"time": beta[i_time]}
    for j, nm in enumerate(names):
        obs[nm] = beta[i_cov[j]]
    for j, c in enumerate(order[1:]):
        obs[f"{c}-{ref}"] = beta[i_phase[j]]
    idx_by_u = [np.where(u == k)[0] for k in uu]

    def _perm_within():                                         # a within-unit shuffle of row indices
        p = np.arange(len(y))
        for gi in idx_by_u:
            pp = gi.copy(); rng.shuffle(pp); p[gi] = pp
        return p

    def _fl_p(col_idx, key):                                    # fixed covariate : within-unit Freedman-Lane
        keep = [c for c in range(X.shape[1]) if c != col_idx]
        Xr = X[:, keep]; br, *_ = np.linalg.lstsq(Xr, y, rcond=None); fit = Xr @ br; resid = y - fit
        c = 1
        for _ in range(n_perm):
            yp = fit + resid[_perm_within()]
            bp, *_ = np.linalg.lstsq(X, yp, rcond=None)
            if abs(bp[col_idx]) >= abs(obs[key]) - 1e-12:
                c += 1
        return c / (n_perm + 1)

    ph_cnt = {f"{c}-{ref}": 1 for c in order[1:]}               # phase : within-unit label permutation
    for _ in range(n_perm):
        Xp = _build(lab[_perm_within()]); bp, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        for j, c in enumerate(order[1:]):
            if abs(bp[i_phase[j]]) >= abs(obs[f"{c}-{ref}"]) - 1e-12:
                ph_cnt[f"{c}-{ref}"] += 1

    adj_note = "phase" + ("".join(f" + {n}" for n in names) if names else "")
    rows = [{"Effect": "time slope", "Estimate": round(float(obs["time"]), 5), "Unit": "/session",
             "p (two-sided)": round(_fl_p(i_time, "time"), 4), "Null": "Freedman-Lane (within-unit)",
             "Interpretation": f"common within-patient change per session, adjusting for {adj_note} (unit FE)."}]
    for j, nm in enumerate(names):
        other = "time + phase" + "".join(f" + {n}" for n in names if n != nm)
        rows.append({"Effect": f"{nm} slope", "Estimate": round(float(obs[nm]), 5), "Unit": f"/unit {nm}",
                     "p (two-sided)": round(_fl_p(i_cov[j], nm), 4), "Null": "Freedman-Lane (within-unit)",
                     "Interpretation": f"common change per unit {nm}, adjusting for {other} (unit FE)."})
    for c in order[1:]:
        rows.append({"Effect": f"{c} - {ref} contrast", "Estimate": round(float(obs[f"{c}-{ref}"]), 5),
                     "Unit": "outcome units", "p (two-sided)": round(ph_cnt[f"{c}-{ref}"] / (n_perm + 1), 4),
                     "Null": "Draper-Stoneman (within-unit)",
                     "Interpretation": f"common {c} vs {ref} difference at matched time{' + ' + ' + '.join(names) if names else ''} (unit FE)."})
    return pd.DataFrame(rows, columns=["Effect", "Estimate", "Unit", "p (two-sided)", "Null", "Interpretation"])


def report_sced_cluster(Y, adjacency, terms, phase, *, label_fn, primary_p=0.01, n_perm=1000,
                        cluster_stat="intensity", tail="both", alpha=0.05, seed=0, base=None,
                        report_cfg=None, report_dir=None, stat=None, vg=None,
                        grid_shape=None, es_conversion="within", es_n_groups=None):
    """Run the declared ANCOVA axes on the element map Y with the given adjacency, and (if report_dir
    + report_cfg) write one grouped SCED report per axis. Domain-neutral : cluster names come from
    `label_fn(idx, cluster_id)` (see spectral.py for the electrode x freq labeller). Mass-univariate
    vs network is only `adjacency`. Returns {res, eff, clab, summary} ; `base` merges into each row.

    terms : the design declaration (functions.sced.cluster.design.term). The null is resolved PER
    TERM by run_ancova - fixed covariates -> Freedman-Lane, randomized factors -> Draper-Stoneman.

    stat : None (defaut) laisse chaque axe garder sa statistique historique ; une valeur explicite
           ('W' pour le Wald robuste de White, 'G' pour le G de Winkler) la remplace sur tous les
           axes. vg : etiquettes de groupe de variance par observation, longueur n_obs, utiles au
           seul 'G'. Les deux ne font que traverser jusqu'a run_ancova puis _glm_statmap.

    grid_shape : (n_rows, n_cols) of the element lattice, e.g. (n_channels, n_freqs). Supplying it
           adds the CLUSTER EFFECT-SIZE bound pair to every summary row (Meyer et al. 2021: report an
           upper and a lower bound, never one number - see cluster_effect_size_bounds). It is
           optional because this layer is domain-neutral and cannot infer the geometry from a flat
           element axis; without it the summary still gains the upper bound, and records that the
           lower bound is unavailable. n_obs is taken from Y, so it never has to be passed.
           es_conversion / es_n_groups select the statistic-to-d conversion ("within" by default,
           "between" needing es_n_groups=(n1, n2))."""
    res, clab, _pm = run_ancova(Y, adjacency, terms, primary_p=primary_p, n_perm=n_perm,
                                cluster_stat=cluster_stat, tail=tail, seed=seed,
                                stat=stat, vg=vg)
    eff = {a: res[a]["effect"] for a in res}               # run_ancova sets 'effect' for every axis
    es = {"n_obs": int(np.asarray(Y).shape[0]), "grid_shape": grid_shape,
          "conversion": es_conversion, "n_groups": es_n_groups}
    rows = summary_rows(res, eff, alpha, label_fn, base or {}, es=es)
    if report_dir is not None and report_cfg is not None:
        grouped_axis_reports(res, Y, phase, alpha, label_fn, report_cfg, report_dir)
    return {"res": res, "eff": eff, "clab": clab, "summary": rows}
