"""
Cluster-based permutation - shared engine (domain-neutral).

A per-element general linear model, thresholded, grouped into connected components on a
given adjacency, with family-wise error controlled at the CLUSTER level by a max-component
permutation null. Two adjacencies are built on top of this core :
  - network (graph edges, shared-node adjacency)   -> functions.sced.cluster.network (NBS).
  - spatial (element grid, e.g. electrode x freq)  -> functions.sced.cluster.spatial.

Nuisance covariates are handled by Freedman-Lane permutation of the reduced-model residuals
(Freedman & Lane 1983 ; Winkler et al. 2014) ; the per-element statistic is a t (single
tested column, signed) or a partial F (several tested columns), or a rank-transformed t.

References
----------
Zalesky, Fornito & Bullmore (2010) NeuroImage - Network-Based Statistic.
Maris & Oostenveld (2007) J. Neurosci. Methods - cluster-based permutation.
Freedman & Lane (1983) ; Winkler et al. (2014) - permutation with nuisance.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Adjacency labelers : edges (graph) and elements (grid)
# ---------------------------------------------------------------------------
def triu_edges(n_nodes):
    """Upper-triangle edge indices (iu, ju) of a complete graph on n_nodes nodes."""
    return np.triu_indices(n_nodes, k=1)


def freq_product_adjacency(A_within, n_freqs):
    """Element x frequency product adjacency, freq-major (index = f * n_within + i).

    Combines a within-frequency adjacency A_within (n_within x n_within : the edge graph for a
    frequency NBS, or an electrode graph for a sensor test) with a 1D frequency chain, so two
    cells (i, f) and (j, g) are neighbours iff (f == g and A_within[i, j]) OR (i == j and
    |f - g| == 1). Pure scipy analogue of mne.stats.combine_adjacency(n_freqs, A_within), so
    the toolbox stays mne-free. Flatten the data FREQ-MAJOR to match : moveaxis(freq before
    element) then reshape to (n_obs, n_freqs * n_within).
    """
    from scipy import sparse
    A = sparse.csr_matrix(A_within)
    m = A.shape[0]
    I_within = sparse.identity(m, format="csr")
    I_freq = sparse.identity(n_freqs, format="csr")
    if n_freqs > 1:                                    # tridiagonal chain over the freq bins
        off = np.ones(n_freqs - 1)
        C_freq = sparse.diags([off, off], [1, -1], format="csr")
    else:
        C_freq = sparse.csr_matrix((1, 1))
    return (sparse.kron(I_freq, A) + sparse.kron(C_freq, I_within)).tocsr()


def edge_components(supra_mask, iu, ju, n_nodes):
    """Connected components of the supra-threshold edges (shared-node graph adjacency).

    Returns (label_per_edge, sizes) : label 0 = edge not retained ; sizes counts edges
    per component.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    lbl = np.zeros(len(supra_mask), dtype=int)
    ei, ej = iu[supra_mask], ju[supra_mask]
    if ei.size == 0:
        return lbl, np.array([], dtype=int)
    A = coo_matrix((np.ones(ei.size), (ei, ej)), shape=(n_nodes, n_nodes))
    A = A + A.T
    _, node_lab = connected_components(A, directed=False)
    idx = np.where(supra_mask)[0]
    lbl[idx] = node_lab[ei] + 1                       # +1 so that 0 stays "no component"
    sizes = np.array([(lbl == c).sum() for c in np.unique(lbl[lbl > 0])], dtype=int)
    return lbl, sizes


def adjacency_components(supra_mask, adjacency):
    """Connected components of the supra-threshold ELEMENTS on a generic adjacency.

    adjacency : sparse (n_elements, n_elements) neighbour matrix (e.g. the electrode x
    frequency adjacency from mne.stats.combine_adjacency). Returns label_per_element
    (0 = element not retained). Sensor-space counterpart of edge_components.
    """
    from scipy.sparse.csgraph import connected_components
    lbl = np.zeros(len(supra_mask), dtype=int)
    idx = np.where(supra_mask)[0]
    if idx.size == 0:
        return lbl
    sub = adjacency[idx][:, idx]
    _, sub_lab = connected_components(sub, directed=False)
    lbl[idx] = sub_lab + 1                             # +1 so that 0 stays "no component"
    return lbl


# ---------------------------------------------------------------------------
# Cluster scoring, thresholding, trend helper
# ---------------------------------------------------------------------------
def _component_scores(comp_labels, stat, cluster_stat):
    """Score per component : element count (extent) or mass = sum |stat| (intensity).

    Extent favours large diffuse clusters ; intensity favours focal strong ones (few
    elements, high statistic) and gives a continuous null (finer p) when clusters are small.
    """
    scores = {}
    for lab in np.unique(comp_labels[comp_labels > 0]):
        m = comp_labels == lab
        scores[int(lab)] = (float(m.sum()) if cluster_stat == "size"
                            else float(np.abs(stat[m]).sum()))
    return scores


def _component_pvals(comp_labels, obs_scores, null_max, n_perm, alpha):
    """Per-component permutation p (score vs max-score null) and the union sig mask."""
    comp_pvals, sig = {}, np.zeros(len(comp_labels), dtype=bool)
    for lab, score in obs_scores.items():
        pv = (1 + np.sum(null_max >= score)) / (n_perm + 1)
        comp_pvals[int(lab)] = float(pv)
        if pv < alpha:
            sig |= (comp_labels == lab)
    return comp_pvals, sig


def _supra(stat, tail, thr, signed):
    """Supra-threshold mask. Unsigned (F) -> stat >= thr ; signed (t) -> tail-aware."""
    if not signed:
        return stat >= thr
    if tail == "both":
        return np.abs(stat) >= thr
    if tail == "pos":
        return stat >= thr
    return stat <= -thr


def _edge_trend_t(effects, x):
    """Pearson t (and r) of (x vs effect) per element. effects (n_sessions, n_elements).
    Kept for the trend wrapper's reporting (r alongside the GLM t)."""
    S = effects.shape[0]
    xc = x - x.mean()
    ec = effects - np.nanmean(effects, axis=0, keepdims=True)
    denom = np.sqrt((xc ** 2).sum()) * np.sqrt(np.nansum(ec ** 2, axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(denom > 0, (xc[:, None] * ec).sum(0) / denom, 0.0)
        t = r * np.sqrt((S - 2) / np.maximum(1 - r ** 2, 1e-12))
    return t, r


# ---------------------------------------------------------------------------
# GLM statistic map + cluster permutation
# ---------------------------------------------------------------------------
def _glm_statmap(Y, design, test_cols, nuisance_cols, stat, thresh, primary_p):
    """Build the per-element GLM statistic map + reduced model, shared by network and
    spatial engines. Returns (statmap, Y, Z, pZ, thresh, signed)."""
    if primary_p == "TFCE":                           # TODO: threshold-free cluster enhancement
        raise NotImplementedError("primary_p='TFCE' (threshold-free) not implemented yet - TODO")
    from scipy.stats import f as _fdist, t as _tdist, norm as _ndist, chi2 as _chi2, rankdata
    Y = np.asarray(Y, dtype=float)
    if stat == "rank":                                # monotone-robust : work on ranks
        Y = rankdata(Y, axis=0)
    n = Y.shape[0]
    D = np.asarray(design, dtype=float)
    test_cols = list(test_cols)
    if nuisance_cols is None:
        nuisance_cols = [c for c in range(D.shape[1]) if c not in test_cols]
    Z = D[:, list(nuisance_cols)] if len(nuisance_cols) else np.zeros((n, 0))
    p_x = len(test_cols)
    df2 = n - D.shape[1]
    if p_x == 0 or df2 <= 0:
        raise ValueError("need >= 1 tested column and enough observations")
    # A robust Wald W with a single tested column is a SIGNED statistic (sign of beta), so it
    # shares the two-tailed pos/neg cluster path of t ; a multi-column W is chi2 (unsigned like F).
    signed = stat in ("t", "rank") or (stat == "W" and p_x == 1)
    if stat in ("t", "rank") and p_x != 1:
        raise ValueError("stat 't' / 'rank' expects exactly one tested column")
    pD = np.linalg.pinv(D)
    pZ = np.linalg.pinv(Z) if Z.shape[1] else np.zeros((0, n))
    tcol = test_cols[0] if signed else None

    def statmap_F(Yv):
        rss_f = np.sum((Yv - D @ (pD @ Yv)) ** 2, axis=0)
        rss_r = (np.sum((Yv - Z @ (pZ @ Yv)) ** 2, axis=0) if Z.shape[1]
                 else np.sum(Yv ** 2, axis=0))
        F = ((rss_r - rss_f) / p_x) / np.maximum(rss_f / df2, 1e-12)
        if not signed:
            return F
        beta = (pD @ Yv)[tcol]                        # signed t = sign(beta) * sqrt(F)
        return np.sign(beta) * np.sqrt(np.maximum(F, 0.0))

    def statmap_W(Yv):
        """Robust Wald statistic with White's (1980) heteroscedasticity-consistent (HC0)
        covariance : W = beta' Omega^-1 beta, Omega = (D'D)^-1 D' diag(e^2) D (D'D)^-1 restricted
        to the tested block. For one tested column this collapses to the HC-robust t-ratio
        beta / sqrt(HC var) (signed) ; for several it is chi2(p_x)-referenced. Valid under
        heteroscedastic errors where the F permutation test is anti-conservative (Helwig 2019,
        DiCiccio & Romano 2017). pD == (D'D)^-1 D', so HC var(beta) = sum_i pD[c,i]^2 e_iv^2."""
        beta_all = pD @ Yv                            # (k, n_elem) all coefficients
        E2 = (Yv - D @ beta_all) ** 2                 # (n, n_elem) squared full-model residuals
        if p_x == 1:
            b = beta_all[tcol]                        # (n_elem,)
            var = (pD[tcol] ** 2) @ E2                # HC0 variance of the tested coefficient
            return b / np.sqrt(np.maximum(var, 1e-24))   # signed robust t = sign(beta)*sqrt(W)
        A = pD[test_cols, :]                          # (p_x, n) rows of (D'D)^-1 D' for tested cols
        W = np.empty(Yv.shape[1])
        for v in range(Yv.shape[1]):                  # per element : sandwich meat sum_i e^2 a_i a_i'
            Omega = (A * E2[:, v]) @ A.T              # (p_x, p_x)
            bv = beta_all[test_cols, v]
            W[v] = float(bv @ np.linalg.solve(Omega, bv))
        return W

    statmap = statmap_W if stat == "W" else statmap_F

    def effmap(Yv):
        """Per-element EFFECT ESTIMATE = coefficient of the tested column (signed t / rank / W
        single-column designs) ; NaN for a multi-column F / W design (no single coefficient). This
        is the interpretable effect (slope, contrast) reported and used for the colorbars."""
        if tcol is None:
            return np.full(Yv.shape[1], np.nan)
        return (pD @ Yv)[tcol]

    if thresh is None:
        if stat == "W":                               # signed W ~ N(0,1) ; multi-column W ~ chi2(p_x)
            thresh = (float(_ndist.ppf(1 - primary_p)) if p_x == 1
                      else float(_chi2.ppf(1 - primary_p, p_x)))
        elif not signed:
            thresh = float(_fdist.ppf(1 - primary_p, p_x, df2))
        else:
            thresh = float(_tdist.ppf(1 - primary_p, df2))
    return statmap, effmap, Y, Z, pZ, float(thresh), signed


def cluster_run(statmap, Y, Z, pZ, labeler, *, thresh, signed, tail,
                n_perm, alpha, seed, cluster_stat="size",
                scheme=None, ds_labels=None, restat=None):
    """Observed statistic -> primary threshold -> connected components (via `labeler`) ->
    permutation null of the largest component -> cluster-level FWER.

    `labeler(supra_mask) -> labels` abstracts the adjacency : edge_components (graph, NBS)
    or adjacency_components (sensor grid). Z / pZ define the reduced (nuisance) model.
    cluster_stat : 'size' | 'intensity'.

    Two permutation nulls, selected by `scheme` :
      - scheme is None (default) : Freedman-Lane residual permutation. Z / pZ residualise the
        data once, the permuted data keep the nuisance fit, so the null reflects "no tested
        effect given the nuisance". Exact for a count-balanced design with no schedule.
      - scheme is a callable : design-based randomization (Draper-Stoneman). `scheme(ds_labels,
        rng) -> permuted_labels` draws an admissible relabeling from the actual schedule (e.g.
        alternating_scheme / block_scheme from functions.sced.core), `restat(permuted_labels)
        -> statmap` rebuilds the effect design column, recomputed on the ORIGINAL data. Honours
        block / max-consecutive schedules FL cannot, and shares the scheme generators with the
        scalar randomization_test so both layers draw from the SAME reference set.
    cluster_stat : 'size' (extent) | 'intensity' (mass). None (no clustering) is an OFF switch
    handled by the caller (templates skip), not reached here.
    """
    stat_obs = statmap(Y)
    comp_labels = labeler(_supra(stat_obs, tail, thresh, signed))
    sizes_obs = np.array([int((comp_labels == L).sum())
                          for L in np.unique(comp_labels[comp_labels > 0])], dtype=int)
    obs_scores = _component_scores(comp_labels, stat_obs, cluster_stat)
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    if scheme is not None:
        # Design-based randomization (Draper-Stoneman): draw an admissible relabeling from the
        # schedule, rebuild the effect design column via `restat`, recompute the map on the
        # ORIGINAL data (the nuisance stays in the model, partialled out each refit). Honours
        # block / max-consecutive schedules that the FL residual permutation cannot. Same
        # reference set as the scalar randomization_test (shared scheme generators).
        labels = np.asarray(ds_labels)
        for k in range(n_perm):
            s = restat(scheme(labels, rng))(Y)
            sc = _component_scores(labeler(_supra(s, tail, thresh, signed)), s, cluster_stat)
            null_max[k] = max(sc.values()) if sc else 0.0
    else:
        # Freedman-Lane residual permutation (free exchangeability given the nuisance). Exact
        # reference set for a count-balanced design with no schedule constraint.
        fit = Z @ (pZ @ Y) if Z.shape[1] else np.zeros_like(Y)
        resid = Y - fit
        order = np.arange(Y.shape[0])
        for k in range(n_perm):
            rng.shuffle(order)
            s = statmap(fit + resid[order])
            sc = _component_scores(labeler(_supra(s, tail, thresh, signed)), s, cluster_stat)
            null_max[k] = max(sc.values()) if sc else 0.0
    comp_pvals, sig_mask = _component_pvals(comp_labels, obs_scores, null_max, n_perm, alpha)
    return {"stat": stat_obs, "thresh": float(thresh), "comp_labels": comp_labels,
            "comp_pvals": comp_pvals, "sig_edges": sig_mask, "sizes_obs": sizes_obs,
            "comp_scores": obs_scores, "cluster_stat": cluster_stat, "null_max": null_max}


def breusch_pagan_map(Y, design, *, sig_mask=None):
    """Per-element studentized Breusch-Pagan test (Koenker 1981 robust variant) of homoscedasticity
    of the full-model residuals against the design regressors. For each element it regresses the
    squared residuals on the design and forms LM = n * R^2 ~ chi2(k-1) ; a small p means that
    element's error variance depends on the regressors (heteroscedastic), so the F permutation test
    is anti-conservative there and the robust W statistic is preferred (Helwig 2019). This is the
    DIAGNOSTIC that motivates reporting F and W side by side - NOT a gate that switches the test
    (a data-dependent pre-test would itself distort the type I error).

    design : the FULL ANCOVA design [intercept, nuisance..., effect] (n, k). sig_mask : optional
    boolean over elements to summarise only the significant cluster. Returns {'lm', 'p' (per
    element), 'frac_hetero' (share with p<0.05 over the summarised set), 'median_p'}."""
    from scipy.stats import chi2 as _chi2
    Y = np.asarray(Y, dtype=float)
    D = np.asarray(design, dtype=float)
    n, k = D.shape
    pD = np.linalg.pinv(D)
    E = Y - D @ (pD @ Y)                               # (n, n_elem) full-model residuals
    g = E ** 2                                         # squared residuals = variance proxy
    ghat = D @ (pD @ g)                                # regress g on the design
    ss_res = np.sum((g - ghat) ** 2, axis=0)
    ss_tot = np.sum((g - g.mean(axis=0, keepdims=True)) ** 2, axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-24)
    lm = n * np.maximum(r2, 0.0)                        # Koenker LM statistic
    p = _chi2.sf(lm, max(k - 1, 1))
    m = (np.ones(Y.shape[1], bool) if sig_mask is None else np.asarray(sig_mask, bool))
    sel = p[m] if m.any() else p                       # summarise over the significant cluster if given
    return {"lm": lm, "p": p, "frac_hetero": float(np.mean(sel < 0.05)),
            "median_p": float(np.median(sel))}


# ---------------------------------------------------------------------------
# Change-point (relu / hinge) cluster test : appearance at an UNKNOWN onset.
# The general hinge math lives in functions.sced.cluster.changepoint (domain-neutral) ; here we
# only wrap it with the permutation null + adjacency.
# ---------------------------------------------------------------------------
def _default_hinge_Z(time, adjust_time):
    """Default reduced model when the caller passes no Z : [1] (flat -> ramp) or [1, time]
    (broken-line change of slope)."""
    n = np.asarray(time).size
    return (np.column_stack([np.ones(n), np.asarray(time, float)]) if adjust_time
            else np.ones((n, 1)))


def relu_run(Y, time, labeler, *, Z=None, onsets=None, adjust_time=False, t_thresh=None,
             primary_p=0.001, n_perm=1000, alpha=0.05, tail="both", seed=0,
             cluster_stat="size"):
    """Change-point (relu / hinge) cluster test on ELEMENTS with a given adjacency labeler.

    Per-element statistic = SUP over candidate onsets of the partial t of the hinge column
    relu_k(t) = max(0, time - onset_k) beyond the reduced model Z. Z defaults to [1] (or [1, time]
    if adjust_time) ; pass an explicit Z to adjust the change-point for ARBITRARY covariates (e.g.
    a phase factor + the linear time), which keeps the onset consistent with an ANCOVA. The
    breakpoint search is absorbed by the Freedman-Lane permutation : cluster_run recomputes the
    SAME sup on the permuted reduced-model residuals, so the cluster FWER stays valid.

    `labeler(supra_mask) -> component labels` abstracts the adjacency (edge_components for a graph,
    adjacency_components for an element grid). Returns the cluster_run keys plus 'onsets' (candidate
    grid), 'onset' / 'slope' (per-element best onset and hinge slope, full maps) and 'onset_hat' /
    'slope_hat' (the same, masked to NaN off the significant cluster)."""
    from scipy.stats import t as _tdist
    from .changepoint import hinge_fits, hinge_sup, hinge_betas
    Y = np.asarray(Y, dtype=float)
    if Z is None:
        Z = _default_hinge_Z(time, adjust_time)
    Zf, pZ, df2, fits, onsets = hinge_fits(time, onsets, Z)
    statmap, t_per_k = hinge_sup(Zf, pZ, df2, fits, tail)
    thresh = float(t_thresh) if t_thresh is not None else float(_tdist.ppf(1 - primary_p, df2))
    res = cluster_run(statmap, Y, Zf, pZ, labeler, thresh=thresh, signed=True, tail=tail,
                      n_perm=n_perm, alpha=alpha, seed=seed, cluster_stat=cluster_stat)
    ts_obs = np.stack([t_per_k(Y, D, pD, tcol) for D, pD, tcol in fits], axis=0)
    kidx = np.argmax(np.abs(ts_obs), axis=0)            # per-element best onset index
    best = onsets[kidx]
    slope = np.take_along_axis(hinge_betas(Y, fits), kidx[None, :], axis=0)[0]  # beta at best onset
    sig = res["sig_edges"]
    res["onsets"] = onsets                              # candidate grid
    res["onset"] = best                                 # per-element best onset (full map)
    res["onset_hat"] = np.where(sig, best, np.nan)      # masked to the significant cluster
    res["slope"] = slope                                # per-element hinge slope (full map)
    res["slope_hat"] = np.where(sig, slope, np.nan)     # masked to the significant cluster
    return res


def scalar_relu(y, time, *, onsets=None, adjust_time=False, t_thresh=None, primary_p=0.001,
                n_perm=1000, alpha=0.05, tail="both", seed=0):
    """Change-point (relu / hinge) test on a SINGLE scalar time series - no adjacency, no cluster.

    Scalar sibling of nbs_relu / spatial_relu : y (n_sessions,) is one outcome per session
    (density, laterality index, accuracy ...). Tests whether the outcome APPEARS or bends at an
    unknown onset ; with a single element the cluster machinery collapses to a plain permutation
    p of the sup-over-onsets statistic (Freedman-Lane residual permutation, which absorbs the
    breakpoint search). adjust_time keeps a linear time nuisance (broken-line change of slope).

    Returns : onset_hat (best onset), slope_hat (hinge slope = post-onset rate, or slope change
    if adjust_time), t (sup statistic), p (permutation p), thresh (primary t for reference),
    onsets (candidate grid)."""
    from scipy.stats import t as _tdist
    from .changepoint import hinge_fits, hinge_sup, hinge_betas
    y = np.asarray(y, dtype=float).reshape(-1, 1)       # (n_sessions, 1 element)
    Z, pZ, df2, fits, onsets = hinge_fits(time, onsets, _default_hinge_Z(time, adjust_time))
    statmap, t_per_k = hinge_sup(Z, pZ, df2, fits, tail)
    t_obs = float(statmap(y)[0])
    fit = Z @ (pZ @ y)                                  # reduced-model fit (nuisance kept)
    resid = y - fit
    rng = np.random.default_rng(seed)
    order = np.arange(y.shape[0])
    null = np.empty(n_perm)
    for k in range(n_perm):                             # Freedman-Lane null of the sup statistic
        rng.shuffle(order)
        null[k] = float(statmap(fit + resid[order])[0])
    if tail == "pos":
        p = (1 + np.sum(null >= t_obs)) / (n_perm + 1)
    elif tail == "neg":
        p = (1 + np.sum(null <= t_obs)) / (n_perm + 1)
    else:
        p = (1 + np.sum(np.abs(null) >= abs(t_obs))) / (n_perm + 1)
    ts = np.array([t_per_k(y, D, pD, tcol)[0] for D, pD, tcol in fits])
    kidx = int(np.argmax(np.abs(ts)))                   # best onset for the single series
    slope = float(hinge_betas(y, fits)[kidx, 0])
    thresh = float(t_thresh) if t_thresh is not None else float(_tdist.ppf(1 - primary_p, df2))
    return {"onset_hat": float(onsets[kidx]), "slope_hat": slope, "t": t_obs, "p": float(p),
            "onsets": onsets, "thresh": thresh, "null_max": null}


# ---------------------------------------------------------------------------
# Model terms (named effect / nuisance -> design columns)
# ---------------------------------------------------------------------------
def _term_matrix(values, kind):
    """Design columns for one model term. Returns (columns (n, k), is_discrete).

    The KIND (discrete factor vs continuous covariate) is set by ROLE, not guessed from the
    values, because auto cannot tell a 3-level integer factor from a 3-point continuous
    covariate. Pass kind explicitly from the template ; 'auto' is a best-effort dtype fallback.

      - discrete   : dummy-coded (reference level dropped), ANY dtype treated as labels.
      - continuous : coerced to float ; datetime64 -> days from the earliest date ; a
                     non-numeric label ('V1') raises (parse to a numeric index first).
      - auto       : object / str / bool -> discrete ; datetime -> continuous ; numeric ->
                     continuous. A numeric-coded factor needs kind='discrete'.
    """
    v = np.asarray(values)
    if kind == "auto":
        kind = "discrete" if v.dtype.kind in "OUSb" else "continuous"
    if kind == "discrete":
        levels = list(np.unique(v))[1:]                # drop the reference level
        X = (np.column_stack([(v == L).astype(float) for L in levels])
             if levels else np.zeros((len(v), 0)))
        return X, True
    if v.dtype.kind == "M":                            # datetime64 -> days from the min
        days = (v - v.min()) / np.timedelta64(1, "D")
        return np.asarray(days, dtype=float)[:, None], False
    try:
        col = v.astype(float)
    except (ValueError, TypeError):
        raise ValueError("continuous term is not numeric (e.g. a visit label 'V1' or a "
                         "non-datetime string) ; parse it to a numeric index first, or "
                         "pass kind='discrete' if it is really a factor")
    return col[:, None], False


def _fl_design(Y, effect, nuisance, effect_kind, nuisance_kind, stat):
    """Build the Freedman-Lane design [intercept, nuisance, effect] from named terms.
    Returns (design, test_cols, nuisance_cols, stat) with stat resolved ('auto' -> t/F)."""
    n = np.asarray(Y).shape[0]
    nuis_terms = list(nuisance) if isinstance(nuisance, (list, tuple)) else [nuisance]
    nuis_kinds = (list(nuisance_kind) if isinstance(nuisance_kind, (list, tuple))
                  else [nuisance_kind] * len(nuis_terms))
    Zcols = [np.ones((n, 1))]
    for term, k in zip(nuis_terms, nuis_kinds):
        Zcols.append(_term_matrix(term, k)[0])
    Z = np.hstack(Zcols)
    Xe, disc = _term_matrix(effect, effect_kind)
    p_e = Xe.shape[1]
    if p_e == 0:
        raise ValueError("effect has < 2 levels / no column")
    design = np.hstack([Z, Xe])
    test_cols = list(range(Z.shape[1], Z.shape[1] + p_e))
    nuis_cols = list(range(Z.shape[1]))
    if stat == "auto":
        stat = "F" if (disc and p_e > 1) else "t"
    if stat == "t" and p_e != 1:
        stat = "F"                                     # a multi-column effect cannot be a signed t
    return design, test_cols, nuis_cols, stat


def huh_jhun_whiten(Y, effect, nuisance, effect_kind, nuisance_kind, stat):
    """Huh-Jhun (2001) exact-exchangeability whitening for a FIXED nuisance-adjusted covariate -
    the SHARED core of every HJ test (cluster map via spatial_huh_jhun AND scalar via
    functions.sced.core.huh_jhun_test), so one implementation serves element maps and scalars alike.

    Projects onto an orthonormal basis Q of the orthogonal complement of the nuisance space Z
    (Q'Q = I, Q Q' = I - H_Z), giving a reduced (n - q)-dimensional whitened response w = Q'Y whose
    rows are EXACTLY iid / exchangeable under Gaussian iid errors - so a plain Freedman-Lane
    permutation on w has exact finite-sample type I even at small n. The whitened regressor Xw = Q'Xe
    has the SAME OLS coefficient as the ANCOVA-adjusted slope of the effect (Frisch-Waugh-Lovell), so
    the effect stays interpretable in original units. Returns (w, Xw, stat) with stat resolved by
    _fl_design ; the caller runs _glm_statmap(w, Xw, ..., nuisance=[]) since Z is already projected
    out. Numpy/scipy only (no statsmodels), so the numpy-only cluster import stays clean."""
    from scipy.linalg import null_space
    design, test_cols, nuis_cols, stat = _fl_design(Y, effect, nuisance, effect_kind,
                                                    nuisance_kind, stat)
    D = np.asarray(design, dtype=float)
    Z = D[:, nuis_cols]                                # nuisance incl intercept
    Xe = D[:, test_cols]                               # tested regressor(s)
    Q = null_space(Z.T)                                # (n, m) orthonormal, Q'Q = I, Q Q' = I - H_Z
    w = Q.T @ np.asarray(Y, dtype=float)               # (m, n_elem) exactly-exchangeable residuals
    Xw = Q.T @ Xe                                      # (m, p_x) whitened residualized regressor
    return w, Xw, stat


def build_scheme(Y, effect, nuisance, effect_kind, nuisance_kind, stat, test_cols, nuis_cols,
                 thresh, primary_p, perm_method, block_size, max_consecutive):
    """Design-based randomization schedule for the cluster null, or None to keep Freedman-Lane.

    Returns (scheme, ds_labels, restat) to hand to cluster_run. A schedule is used when the
    caller asks for a design-based method (``perm_method`` 'draper-stoneman' / 'randomization')
    or constrains the schedule (``block_size`` / ``max_consecutive``) - exactly the cases FL
    cannot honour ; otherwise returns (None, None, None) and the caller falls back to FL. The
    ``scheme`` comes from the SHARED scalar generators (alternating_scheme / block_scheme), so
    the cluster test and run_sced_alternating draw the same admissible relabelings. ``restat``
    rebuilds the effect design column from the permuted labels and returns a fresh statmap on
    the ORIGINAL data (Draper-Stoneman : permute labels, refit, keep the nuisance in the model).
    """
    design_based = (perm_method in ("draper-stoneman", "randomization")
                    or block_size is not None or max_consecutive is not None)
    if not design_based:
        return None, None, None
    # Import the shared scheme generators LAZILY : functions.sced.core pulls in statsmodels /
    # pandas, which the default Freedman-Lane path (and the numpy/scipy-only cluster import)
    # must not require. Only a design-based schedule needs them, and the envs that run one
    # (bci_session / Stats) have statsmodels.
    from ..core import alternating_scheme, block_scheme
    scheme = block_scheme(block_size) if block_size else alternating_scheme(max_consecutive)
    labels = np.asarray(effect)

    def restat(perm_labels):
        design, _tc, _nc, _st = _fl_design(Y, perm_labels, nuisance, effect_kind,
                                           nuisance_kind, stat)
        return _glm_statmap(Y, design, test_cols, nuis_cols, stat, thresh, primary_p)[0]

    return scheme, labels, restat


# ---------------------------------------------------------------------------
# ANCOVA contrast within a randomized factor : design + Draper-Stoneman schedule
# ---------------------------------------------------------------------------
def _contrast_design(Y, factor, contrast, cond_order, nuisance, nuisance_kind):
    """ANCOVA design for ONE factor contrast : [intercept, nuisance..., other-level dummies,
    contrast dummy]. reference = cond_order[0] ; the TESTED column is the `contrast` level dummy,
    whose coefficient is (contrast - reference) adjusted for the nuisance and the other levels.
    Returns (design, test_cols, nuisance_cols)."""
    n = np.asarray(Y).shape[0]
    fac = np.asarray([str(x) for x in factor])
    others = [str(c) for c in cond_order[1:]]              # non-reference levels
    nuis_terms = list(nuisance) if isinstance(nuisance, (list, tuple)) else [nuisance]
    nuis_kinds = (list(nuisance_kind) if isinstance(nuisance_kind, (list, tuple))
                  else [nuisance_kind] * len(nuis_terms))
    Zcols = [np.ones((n, 1))]
    for term, k in zip(nuis_terms, nuis_kinds):
        Zcols.append(_term_matrix(term, k)[0])
    for o in others:                                       # the OTHER contrasts stay in as nuisance
        if o != str(contrast):
            Zcols.append((fac == o).astype(float)[:, None])
    Z = np.hstack(Zcols)
    X = (fac == str(contrast)).astype(float)[:, None]      # tested column = the contrast dummy
    design = np.hstack([Z, X])
    return design, [design.shape[1] - 1], list(range(Z.shape[1]))


def build_contrast_scheme(Y, factor, contrast, cond_order, nuisance, nuisance_kind,
                          perm_method, block_size, max_consecutive, thresh, primary_p):
    """Draper-Stoneman randomization schedule for ONE ANCOVA contrast within a RANDOMIZED factor.

    Unlike build_scheme (which permutes the tested column), the scheme here permutes the WHOLE
    factor schedule as one unit (alternating_scheme / block_scheme preserve the level counts), and
    restat rebuilds the full ANCOVA from the permuted factor - so BOTH the contrast dummy and the
    other-level dummies move together and stay mutually exclusive. The null is thus the
    randomization distribution of the factor (Edgington), not a free permutation of a binary dummy.
    Returns (scheme, factor_labels, restat) to hand to cluster_run. FL is used when perm_method is
    'freedman-lane' and no schedule constraint is set (handled by the caller)."""
    from ..core import alternating_scheme, block_scheme
    scheme = block_scheme(block_size) if block_size else alternating_scheme(max_consecutive)
    factor = np.asarray([str(x) for x in factor])

    def restat(perm_factor):
        design, tc, nc = _contrast_design(Y, perm_factor, contrast, cond_order, nuisance, nuisance_kind)
        return _glm_statmap(Y, design, tc, nc, "t", thresh, primary_p)[0]

    return scheme, factor, restat
