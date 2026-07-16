"""Canonical SCED CLUSTER analysis + reporting - the map-valued counterpart of
report_sced_alternating. DOMAIN-NEUTRAL : given an element map Y (n_obs, n_elements), an adjacency,
and a caller-supplied ``label_fn(idx, cluster_id) -> tag`` that names a cluster from its element
indices, it runs the DECLARED ANCOVA (run_ancova, role-adapted null per term) + optional relu onset,
extracts the significant clusters, and emits ONE grouped SCED report per axis (units = the axis's
clusters). The core knows nothing about Hz, electrodes, connectivity or any grid semantics - that is
entirely in ``label_fn`` (see spectral.py for the electrode x frequency labeller). Mass-univariate
vs network is ONLY the adjacency. So a domain template = build (Y, adjacency, design terms, label_fn)
and call this."""
import os

import numpy as np

from .design import run_ancova

__all__ = ["report_sced_cluster", "describe_clusters", "summary_rows",
           "grouped_axis_reports", "axis_family", "scalar_adjusted_effects"]


def axis_family(axis, res=None):
    """Group an axis into its test-type folder (report lands in <outcome>/<family>/). Prefers
    res['family'] stamped by run_ancova (the descriptor is the source of truth) ; falls back to the
    name convention only for results not produced by the descriptor."""
    if res is not None and res.get("family"):
        return res["family"]
    if axis == "onset":
        return "relu"
    if axis.startswith("phase"):
        return "phase"
    return axis


def describe_clusters(res, alpha, axis, label_fn):
    """Significant clusters of ONE axis as [{name, p, onset, idx}]. The NAME is domain-supplied :
    name = f"{axis}_{label_fn(idx, cluster_id)}" - the toolbox turns element indices into a tag only
    through the caller's label_fn (no Hz / electrode knowledge here). onset = median onset_hat over
    the cluster cells (NaN unless the axis is a hinge). idx = flat element indices of the cluster."""
    out = []
    comp = res.get("comp_labels")
    if comp is None:
        return out
    onset_hat = res.get("onset_hat")
    for lab, p in res.get("comp_pvals", {}).items():
        if p >= alpha:
            continue
        idx = np.where(comp == int(lab))[0]                    # cluster elements (flat)
        onset = (float(np.nanmedian(onset_hat[idx])) if onset_hat is not None
                 and np.isfinite(onset_hat[idx]).any() else np.nan)
        out.append({"name": f"{axis}_{label_fn(idx, int(lab))}", "p": float(p),
                    "onset": onset, "idx": idx})
    return out


def summary_rows(res_by_axis, eff_by_axis, alpha, label_fn, base):
    """One row PER AXIS for a consolidated report : the test, its cluster p, the number and names of
    significant clusters, the mean effect (slope/contrast) over the significant set, and (relu) the
    median onset. base : the caller's fixed fields (subject, level, outcome, tail...)."""
    rows = []
    for axis, res in res_by_axis.items():
        descs = describe_clusters(res, alpha, axis, label_fn)
        sig = res["sig_elements"]; eff = eff_by_axis[axis]
        onset = (float(np.nanmedian([d["onset"] for d in descs]))
                 if (res.get("kind") == "hinge" and descs) else np.nan)
        row = {**base, "axis": axis,
               "n_sig_elem": int(sig.sum()), "n_clusters": len(descs),
               "clusters": "; ".join(d["name"] for d in descs),
               "mean_slope": float(np.nanmean(np.where(sig, eff, np.nan))) if sig.any() else np.nan,
               "onset": onset,
               "min_p": min(res["comp_pvals"].values()) if res["comp_pvals"] else 1.0}
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
    with family in {trend, phase, relu} (phase groups every contrast). All that axis's significant
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
                            block_size=None, max_consecutive=None, n_perm=2000, seed=0):
    """ANCOVA effect ESTIMATES + p for a single scalar series, each with the role-adapted null : the
    time slope (fixed covariate -> Freedman-Lane) and every condition contrast (level - reference,
    the randomized dose -> Draper-Stoneman). Reuses the map engine at n_elements=1 (a scalar is the
    trivial 1-element map). Returns a tidy DataFrame [Effect, Estimate, p, Null]. This is the
    model-based estimate that complements the omnibus randomization test and the nonoverlap ES -
    notably it fills the MISSING time-slope estimate of the alternating report.

    p is a plain two-sided PERMUTATION p of the statistic (primary threshold set to 0, so the single
    element always 'clusters' and no thresholding gates the p) - NOT a cluster p ; a sub-threshold
    effect must not collapse to p=1 while its estimate is shown."""
    import pandas as pd
    from scipy import sparse
    from .design import term, run_ancova
    from .spatial import spatial_contrast
    Y = np.asarray(values, dtype=float).reshape(-1, 1)         # (n_sess, 1 element)
    adj = sparse.csr_matrix((1, 1))                            # one element, no neighbour
    time = np.asarray(sessions, dtype=float)
    kw = dict(primary_p=0.5, n_perm=n_perm, cluster_stat="intensity", tail="both", seed=seed)  # thr=0
    rows = []
    # time slope, adjusted for phase (fixed covariate -> Freedman-Lane)
    rt, _clab, _pm = run_ancova(Y, adj, [term("trend", time, role="fixed", kind="continuous"),
                                         term("phase", labels, role="randomized", kind="factor",
                                              cond_order=cond_order)], **kw)
    tp = min(rt["trend"]["comp_pvals"].values()) if rt["trend"]["comp_pvals"] else 1.0
    rows.append({"Effect": "time slope (/session)", "Estimate": round(float(rt["trend"]["effect"][0]), 5),
                 "p": round(float(tp), 4), "Null": "Freedman-Lane (fixed covariate)"})
    # condition contrasts (randomized dose -> Draper-Stoneman)
    ref = str(cond_order[0])
    for c in [str(x) for x in cond_order[1:]]:
        rc = spatial_contrast(Y, adj, factor=labels, contrast=c, cond_order=cond_order,
                              nuisance=[time], nuisance_kind=["continuous"], perm_method=perm_phase,
                              block_size=block_size, max_consecutive=max_consecutive, **kw)
        pc = min(rc["comp_pvals"].values()) if rc["comp_pvals"] else 1.0
        rows.append({"Effect": f"{c} - {ref} contrast (adj. time)",
                     "Estimate": round(float(rc["effect"][0]), 5), "p": round(float(pc), 4),
                     "Null": "Draper-Stoneman (randomized dose)"})
    return pd.DataFrame(rows, columns=["Effect", "Estimate", "p", "Null"])


def report_sced_cluster(Y, adjacency, terms, phase, *, label_fn, primary_p=0.01, n_perm=1000,
                        cluster_stat="intensity", tail="both", alpha=0.05, seed=0, base=None,
                        report_cfg=None, report_dir=None):
    """Run the declared ANCOVA axes on the element map Y with the given adjacency, and (if report_dir
    + report_cfg) write one grouped SCED report per axis. Domain-neutral : cluster names come from
    `label_fn(idx, cluster_id)` (see spectral.py for the electrode x freq labeller). Mass-univariate
    vs network is only `adjacency`. Returns {res, eff, clab, summary} ; `base` merges into each row.

    terms : the design declaration (functions.sced.cluster.design.term). The null is resolved PER
    TERM by run_ancova - fixed covariates -> Freedman-Lane, randomized factors -> Draper-Stoneman, a
    hinge term -> a relu change-point axis (declared, not hard-coded here)."""
    res, clab, _pm = run_ancova(Y, adjacency, terms, primary_p=primary_p, n_perm=n_perm,
                                cluster_stat=cluster_stat, tail=tail, seed=seed)
    eff = {a: res[a]["effect"] for a in res}               # run_ancova sets 'effect' for every axis
    rows = summary_rows(res, eff, alpha, label_fn, base or {})
    if report_dir is not None and report_cfg is not None:
        grouped_axis_reports(res, Y, phase, alpha, label_fn, report_cfg, report_dir)
    return {"res": res, "eff": eff, "clab": clab, "summary": rows}
