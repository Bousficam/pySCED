"""
Spatial cluster-based permutation - on ELEMENTS with a generic adjacency.

Element space = any grid with a neighbour structure, typically electrode x frequency
(Maris & Oostenveld 2007). Input : a map (n_obs, n_elements) plus a sparse adjacency
(n_elements, n_elements). Build the adjacency once, e.g. with
EEG.Get_Data.build_spatiofreq_adjacency (mne find_ch_adjacency x frequency). Applies to any
element-valued map (node strength, ERD, PSD ...).

Element-space twins of the network (NBS) wrappers, over the same shared engine
(functions.sced.cluster.core) : spatial_trend, spatial_freedman_lane.
"""

import numpy as np

from .core import (adjacency_components, _glm_statmap, cluster_run, _fl_design, huh_jhun_whiten, build_scheme,
                   relu_run, _contrast_design, build_contrast_scheme, _term_matrix)

__all__ = ["spatial_glm", "spatial_trend", "spatial_freedman_lane", "spatial_huh_jhun",
           "spatial_relu", "spatial_contrast"]


def spatial_glm(Y, design, test_cols, adjacency, *, nuisance_cols=None, stat="F",
                thresh=None, primary_p=0.001, n_perm=1000, alpha=0.05, tail="both",
                seed=0, cluster_stat="size", scheme=None, ds_labels=None, restat=None):
    """Cluster-based-permutation GLM on ELEMENTS (e.g. electrode x frequency cells).

    Same per-element GLM statistic, threshold and permutation null as nbs_glm (Freedman-Lane
    by default, or a design-based randomization schedule via scheme / ds_labels / restat), but
    the clustering uses a generic sparse `adjacency` instead of the edge graph. Returns the
    cluster_run keys ; 'sig_edges' is the significant-element mask (aliased 'sig_elements').
    """
    statmap, effmap, Yv, Z, pZ, thr, signed = _glm_statmap(Y, design, test_cols, nuisance_cols,
                                                           stat, thresh, primary_p)
    # adjacency_components indexes the matrix (adjacency[idx][:, idx]), which needs a
    # subscriptable format ; combine_adjacency / find_ch_adjacency may hand back COO. Coerce
    # to CSR ONCE here (not per permutation) so any input format works.
    if hasattr(adjacency, "tocsr"):
        adjacency = adjacency.tocsr()
    res = cluster_run(statmap, Yv, Z, pZ, lambda m: adjacency_components(m, adjacency),
                      thresh=thr, signed=signed, tail=tail, n_perm=n_perm, alpha=alpha,
                      seed=seed, cluster_stat=cluster_stat,
                      scheme=scheme, ds_labels=ds_labels, restat=restat)
    res["sig_elements"] = res["sig_edges"]
    res["effect"] = effmap(Yv)                             # per-element coefficient (slope / contrast)
    return res


def spatial_trend(effects, adjacency, session_idx=None, t_thresh=2.5, primary_p=None, n_perm=1000,
                  alpha=0.05, tail="both", seed=0, method="pearson", cluster_stat="size"):
    """Sensor clusters whose per-element effect trends across sessions (design [1, time]).
    effects : (n_sessions, n_elements). method 'pearson' | 'spearman' (rank). Primary threshold :
    `primary_p` (per-element p, converted to the t threshold at the right df) takes precedence over
    the raw `t_thresh`."""
    effects = np.asarray(effects, dtype=float)
    S = effects.shape[0]
    x = np.arange(S, dtype=float) if session_idx is None else np.asarray(session_idx, float)
    design = np.column_stack([np.ones(S), x])
    res = spatial_glm(effects, design, [1], adjacency, nuisance_cols=[0],
                      stat="rank" if method == "spearman" else "t",
                      thresh=(None if primary_p is not None else t_thresh), primary_p=(primary_p or 0.001),
                      n_perm=n_perm, alpha=alpha, tail=tail, seed=seed, cluster_stat=cluster_stat)
    res["t"] = res["stat"]
    return res


def spatial_freedman_lane(Y, adjacency, *, effect, nuisance, effect_kind="auto",
                          nuisance_kind="auto", stat="auto", thresh=None, primary_p=0.001,
                          n_perm=1000, alpha=0.05, tail="both", seed=0, cluster_stat="size",
                          perm_method="freedman-lane", block_size=None, max_consecutive=None):
    """Sensor clusters with the partial effect of one term adjusted for nuisance terms.
    Element-space twin of nbs_freedman_lane : Y (n_obs, n_elements), same term semantics.

    Permutation null : Freedman-Lane (default) or a design-based randomization schedule when
    the effect is a randomized factor with a constrained schedule (``block_size`` /
    ``max_consecutive`` force the shared alternating_scheme / block_scheme, so the reference
    set matches the assignment). Same semantics as nbs_freedman_lane."""
    design, test_cols, nuis_cols, stat = _fl_design(Y, effect, nuisance, effect_kind,
                                                    nuisance_kind, stat)
    scheme, ds_labels, restat = build_scheme(Y, effect, nuisance, effect_kind, nuisance_kind,
                                             stat, test_cols, nuis_cols, thresh, primary_p,
                                             perm_method, block_size, max_consecutive)
    res = spatial_glm(Y, design, test_cols, adjacency, nuisance_cols=nuis_cols, stat=stat,
                      thresh=thresh, primary_p=primary_p, n_perm=n_perm, alpha=alpha,
                      tail=tail, seed=seed, cluster_stat=cluster_stat,
                      scheme=scheme, ds_labels=ds_labels, restat=restat)
    res["stat_kind"] = stat
    res["F" if stat == "F" else "t"] = res["stat"]
    if stat == "F":
        res["f_thresh"] = res["thresh"]
    return res


def spatial_huh_jhun(Y, adjacency, *, effect, nuisance, effect_kind="auto", nuisance_kind="auto",
                     stat="W", thresh=None, primary_p=0.001, n_perm=1000, alpha=0.05, tail="both",
                     seed=0, cluster_stat="size"):
    """Huh-Jhun (2001) whitened permutation test - the exact-exchangeability sibling of
    spatial_freedman_lane, for a FIXED nuisance-adjusted covariate. Projects the data onto an
    orthonormal basis Q of the orthogonal complement of the nuisance space (Q'Q = I, Q Q' = I - H_Z),
    giving a reduced (n - q - 1)-dimensional whitened response w = Q'Y whose components are EXACTLY
    iid / exchangeable under Gaussian iid errors - so the plain Freedman-Lane residual permutation on
    w has exact finite-sample type I even at small n, where FL / DS on the raw residuals are only
    approximate (Helwig 2019, best small-n control ; pairs with stat='W' for heteroscedasticity).

    The whitened regressor is Q'X (= Q' R_x X, since Q is orthogonal to Z), and its OLS coefficient
    in the whitened space equals the ANCOVA-adjusted slope of X (Frisch-Waugh-Lovell) - so 'effect'
    stays interpretable in the original units. Same term semantics as spatial_freedman_lane ; the
    default stat is the robust W (the recommended HJ+W combination). Not for a randomized factor -
    use spatial_contrast (Draper-Stoneman) there ; the randomization, not exchangeability, is the
    reference set for a randomized term."""
    w, Xw, stat = huh_jhun_whiten(Y, effect, nuisance, effect_kind, nuisance_kind, stat)  # shared HJ core
    if hasattr(adjacency, "tocsr"):
        adjacency = adjacency.tocsr()
    # Whitened space has NO intercept / nuisance (projected out) : model w = Xw beta + e, so the
    # cluster_run FL branch (scheme=None, empty Z) permutes the m whitened rows = the HJ null.
    statmap, effmap, wv, Zw, pZw, thr, signed = _glm_statmap(
        w, Xw, list(range(Xw.shape[1])), [], stat, thresh, primary_p)
    res = cluster_run(statmap, wv, Zw, pZw, lambda m_: adjacency_components(m_, adjacency),
                      thresh=thr, signed=signed, tail=tail, n_perm=n_perm, alpha=alpha,
                      seed=seed, cluster_stat=cluster_stat)
    res["sig_elements"] = res["sig_edges"]
    res["effect"] = effmap(wv)                         # = ANCOVA-adjusted slope (FWL), interpretable
    res["stat_kind"] = stat
    res["scheme"] = "huh-jhun"
    res["W" if stat == "W" else "t"] = res["stat"]     # alias by statistic, like the FL wrapper
    return res


def spatial_contrast(Y, adjacency, *, factor, contrast, cond_order, nuisance, nuisance_kind="continuous",
                     perm_method="draper-stoneman", block_size=None, max_consecutive=None,
                     thresh=None, primary_p=0.001, n_perm=1000, alpha=0.05, tail="both", seed=0,
                     cluster_stat="size"):
    """Cluster test of ONE ANCOVA contrast (``contrast`` level minus the reference cond_order[0])
    within a RANDOMIZED factor, adjusting for ``nuisance`` (e.g. time) and the other levels.

    Null = Draper-Stoneman randomization of the FACTOR schedule (permute the whole assignment,
    rebuild the ANCOVA, recompute the contrast) - the valid reference set for a randomized factor,
    unlike a free permutation of the binary contrast dummy. Falls back to Freedman-Lane residual
    permutation of the same design when ``perm_method='freedman-lane'`` and no schedule constraint
    is set. Element space (generic adjacency). Returns the cluster_run keys + 'effect' (the signed
    contrast estimate) and 'sig_elements'."""
    design, test_cols, nuis_cols = _contrast_design(Y, factor, contrast, cond_order,
                                                    nuisance, nuisance_kind)
    design_based = (perm_method in ("draper-stoneman", "randomization")
                    or block_size is not None or max_consecutive is not None)
    if design_based:
        scheme, ds_labels, restat = build_contrast_scheme(Y, factor, contrast, cond_order, nuisance,
            nuisance_kind, perm_method, block_size, max_consecutive, thresh, primary_p)
    else:
        scheme = ds_labels = restat = None
    res = spatial_glm(Y, design, test_cols, adjacency, nuisance_cols=nuis_cols, stat="t",
                      thresh=thresh, primary_p=primary_p, n_perm=n_perm, alpha=alpha, tail=tail,
                      seed=seed, cluster_stat=cluster_stat, scheme=scheme, ds_labels=ds_labels,
                      restat=restat)
    res["t"] = res["stat"]
    return res


def spatial_relu(Y, adjacency, time, *, nuisance=None, nuisance_kind="auto", onsets=None,
                 adjust_time=False, t_thresh=None, primary_p=0.001, n_perm=1000, alpha=0.05,
                 tail="both", seed=0, cluster_stat="size"):
    """Sensor / element clusters that APPEAR / bend at an UNKNOWN onset (relu / hinge change-point).

    Element-space twin of nbs_relu : Y (n_obs, n_elements) + a generic `adjacency`. Per-element
    statistic = sup over candidate onsets of the hinge partial t ; cluster FWER via Freedman-Lane
    (the breakpoint search is absorbed by the permutation). The reduced model is [1, (time if
    adjust_time), nuisance...]: pass `nuisance` (terms + nuisance_kind, like spatial_freedman_lane)
    to adjust the onset for ARBITRARY covariates (e.g. a phase factor), keeping it consistent with
    an ANCOVA - the hinge then tests a temporal bend not explained by the covariates. Returns the
    cluster_run keys + 'onsets', 'onset'/'slope', 'onset_hat'/'slope_hat', 'sig_elements'."""
    n = np.asarray(Y).shape[0]
    cols = [np.ones((n, 1))]
    if adjust_time:
        cols.append(np.asarray(time, dtype=float)[:, None])
    if nuisance is not None:                            # arbitrary covariates in the reduced model
        terms = list(nuisance) if isinstance(nuisance, (list, tuple)) else [nuisance]
        kinds = (list(nuisance_kind) if isinstance(nuisance_kind, (list, tuple))
                 else [nuisance_kind] * len(terms))
        for tm, k in zip(terms, kinds):
            cols.append(_term_matrix(tm, k)[0])
    Z = np.hstack(cols)
    if hasattr(adjacency, "tocsr"):                    # adjacency_components needs a CSR (indexable)
        adjacency = adjacency.tocsr()
    res = relu_run(Y, time, lambda m: adjacency_components(m, adjacency), Z=Z,
                   onsets=onsets, adjust_time=adjust_time, t_thresh=t_thresh, primary_p=primary_p,
                   n_perm=n_perm, alpha=alpha, tail=tail, seed=seed, cluster_stat=cluster_stat)
    res["t"] = res["stat"]
    res["sig_elements"] = res["sig_edges"]
    return res
