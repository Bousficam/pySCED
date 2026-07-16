"""
Network-Based Statistic (NBS) - cluster-based permutation on GRAPH EDGES.

Edge adjacency = shared-node graph (two edges cluster if they share a node ; Zalesky et al.
2010). Input : an edge array (n_obs, n_edges) plus the upper-triangle indices iu, ju over
n_nodes nodes. Domain-neutral (functional connectivity, structural connectome, any graph).

Wrappers over the shared engine (functions.sced.cluster.core) : preconfigured designs.
  - nbs_trend          : per-edge slope of the effect vs the session index (time).
  - nbs_freedman_lane  : partial effect of one term adjusted for nuisance terms.
  - nbs_step           : level change at a known onset session.
  - nbs_relu           : appearance at an unknown onset (hinge / broken-line, sup over breaks).
"""

import numpy as np

from .core import (triu_edges, edge_components, _edge_trend_t, _glm_statmap, cluster_run,
                   _fl_design, build_scheme, freq_product_adjacency, relu_run)
from .spatial import spatial_trend, spatial_freedman_lane, spatial_relu

__all__ = ["triu_edges", "edge_components", "edge_adjacency", "nbs_glm", "nbs_trend",
           "nbs_freedman_lane", "nbs_step", "nbs_relu",
           "nbs_freq_trend", "nbs_freq_freedman_lane", "nbs_freq_relu"]


def edge_adjacency(iu, ju, n_nodes):
    """Sparse (n_edges x n_edges) NBS edge graph : two edges neighbour iff they share a node.

    Explicit matrix form of the shared-node adjacency that edge_components builds implicitly.
    Needed to combine with a frequency chain (freq_product_adjacency) for the edge x frequency
    NBS. Built from the node -> incident-edges incidence : the edges incident to a node form a
    clique.
    """
    from scipy import sparse
    iu = np.asarray(iu); ju = np.asarray(ju)
    n_edges = len(iu)
    inc = {}                                           # node -> list of incident edge indices
    for e in range(n_edges):
        inc.setdefault(int(iu[e]), []).append(e)
        inc.setdefault(int(ju[e]), []).append(e)
    rows, cols = [], []
    for edges in inc.values():                         # each node's incident edges = a clique
        for x in range(len(edges)):
            for y in range(x + 1, len(edges)):
                rows += [edges[x], edges[y]]
                cols += [edges[y], edges[x]]
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n_edges, n_edges))
    return (A > 0).astype(float).tocsr()


def nbs_glm(Y, design, test_cols, iu, ju, n_nodes, *, nuisance_cols=None,
            stat="F", perm="freedman-lane", thresh=None, primary_p=0.001,
            n_perm=1000, alpha=0.05, tail="both", seed=0, cluster_stat="size",
            scheme=None, ds_labels=None, restat=None):
    """Per-edge GLM, thresholded and clustered on the edge graph, subnetwork-level FWER.

    stat {'t','F','rank'} ; nuisance handled by Freedman-Lane (default) or a design-based
    randomization schedule (scheme / ds_labels / restat, see cluster_run) ; cluster_stat
    'size' (extent) or 'intensity' (mass). Returns iu, ju, stat, thresh, comp_labels,
    comp_pvals, sig_edges, sizes_obs, comp_scores, null_max.
    """
    statmap, effmap, Yv, Z, pZ, thr, signed = _glm_statmap(Y, design, test_cols, nuisance_cols,
                                                           stat, thresh, primary_p)
    res = cluster_run(statmap, Yv, Z, pZ, lambda m: edge_components(m, iu, ju, n_nodes)[0],
                      thresh=thr, signed=signed, tail=tail, n_perm=n_perm, alpha=alpha,
                      seed=seed, cluster_stat=cluster_stat,
                      scheme=scheme, ds_labels=ds_labels, restat=restat)
    res["iu"] = iu; res["ju"] = ju
    res["effect"] = effmap(Yv)                             # per-element coefficient (slope / contrast)
    return res


def nbs_trend(effects, iu, ju, n_nodes, session_idx=None, t_thresh=2.5, primary_p=None,
              n_perm=1000, alpha=0.05, tail="both", seed=0, method="pearson",
              cluster_stat="size"):
    """Subnetworks whose per-edge effect trends across sessions (design [1, time]).
    method 'pearson' (linear slope) or 'spearman' (rank). `primary_p` (per-element p -> t threshold
    at the right df) takes precedence over the raw `t_thresh`."""
    effects = np.asarray(effects, dtype=float)
    S = effects.shape[0]
    x = np.arange(S, dtype=float) if session_idx is None else np.asarray(session_idx, float)
    design = np.column_stack([np.ones(S), x])
    res = nbs_glm(effects, design, [1], iu, ju, n_nodes, nuisance_cols=[0],
                  stat="rank" if method == "spearman" else "t",
                  thresh=(None if primary_p is not None else t_thresh), primary_p=(primary_p or 0.001),
                  n_perm=n_perm, alpha=alpha, tail=tail, seed=seed, cluster_stat=cluster_stat)
    _, r_obs = _edge_trend_t(effects, x)
    res["t"] = res["stat"]; res["r"] = r_obs           # legacy aliases
    return res


def nbs_freedman_lane(Y, iu, ju, n_nodes, *, effect, nuisance,
                      effect_kind="auto", nuisance_kind="auto", stat="auto",
                      thresh=None, primary_p=0.001, n_perm=1000, alpha=0.05,
                      tail="both", seed=0, cluster_stat="size",
                      perm_method="freedman-lane", block_size=None, max_consecutive=None):
    """Subnetworks with the partial effect of ONE term adjusted for the NUISANCE terms.

    Design [intercept, nuisance, effect] ; each term continuous (one column) or discrete
    (dummy-coded), auto-detected by dtype or forced by *_kind. Covers both symmetric
    questions of an alternating design (dose|time, time|dose) and any covariate nuisance
    (fatigue ...). stat 'auto' -> t for a continuous / 2-level effect, F for a >= 3-level
    factor. Returns the nbs_glm keys + a 't' or 'F' alias (and 'f_thresh' when F), 'stat_kind'.

    Permutation null (mirrors run_sced_alternating) : ``perm_method='freedman-lane'`` (default,
    residual permutation, exact for a count-balanced design) or a design-based randomization
    schedule when the effect is a randomized factor with a constrained schedule -
    ``block_size`` (randomized block : one of each per block) / ``max_consecutive`` (no more
    than k in a row) force the shared alternating_scheme / block_scheme so the reference set
    matches how conditions were actually assigned. With neither, FL is used.
    """
    design, test_cols, nuis_cols, stat = _fl_design(Y, effect, nuisance, effect_kind,
                                                    nuisance_kind, stat)
    scheme, ds_labels, restat = build_scheme(Y, effect, nuisance, effect_kind, nuisance_kind,
                                             stat, test_cols, nuis_cols, thresh, primary_p,
                                             perm_method, block_size, max_consecutive)
    res = nbs_glm(Y, design, test_cols, iu, ju, n_nodes, nuisance_cols=nuis_cols,
                  stat=stat, thresh=thresh, primary_p=primary_p, n_perm=n_perm,
                  alpha=alpha, tail=tail, seed=seed, cluster_stat=cluster_stat,
                  scheme=scheme, ds_labels=ds_labels, restat=restat)
    res["stat_kind"] = stat
    res["F" if stat == "F" else "t"] = res["stat"]
    if stat == "F":
        res["f_thresh"] = res["thresh"]
    return res


def nbs_step(Y, onset, time, iu, ju, n_nodes, *, adjust_time=True, t_thresh=None,
             primary_p=0.001, n_perm=1000, alpha=0.05, tail="both", seed=0,
             cluster_stat="size"):
    """Subnetworks with a level change at a KNOWN onset session (SCED intervention).

    Design [1, (time), step] with step = 1 for session index >= onset. adjust_time keeps a
    linear time nuisance so the step is the level change on top of the ongoing trend.
    """
    tvec = np.asarray(time, float)
    n = np.asarray(Y).shape[0]
    step = (tvec >= onset).astype(float)
    cols = [np.ones(n), tvec, step] if adjust_time else [np.ones(n), step]
    design = np.column_stack(cols)
    tcol = design.shape[1] - 1
    nuis = list(range(tcol))
    res = nbs_glm(Y, design, [tcol], iu, ju, n_nodes, nuisance_cols=nuis, stat="t",
                  thresh=t_thresh, primary_p=primary_p, n_perm=n_perm, alpha=alpha,
                  tail=tail, seed=seed, cluster_stat=cluster_stat)
    res["t"] = res["stat"]
    return res


def nbs_relu(Y, time, iu, ju, n_nodes, *, onsets=None, adjust_time=False, t_thresh=None,
             primary_p=0.001, n_perm=1000, alpha=0.05, tail="both", seed=0,
             cluster_stat="size"):
    """Subnetworks that APPEAR at an UNKNOWN onset (hinge / relu change-point).

    Per-edge statistic = SUP over candidate onsets of the partial t of the hinge column
    relu_k(t) = max(0, time - k) beyond the nuisance [1] (or [1, time] if adjust_time). The
    breakpoint search is absorbed by the permutation, which recomputes the SAME sup on the
    permuted reduced-model residuals (Freedman-Lane), so the subnetwork FWER stays valid.
    Returns the nbs_glm keys plus 'onsets' and 'onset_hat' (per-edge best onset, NaN off sig).

    Thin wrapper over core.relu_run (shared hinge engine) with the NBS edge labeler ; the sup
    statistic and its Freedman-Lane null live in core, so the sensor (spatial_relu) and edge x
    freq (nbs_freq_relu) variants reuse the exact same change-point test.
    """
    res = relu_run(Y, time, lambda m: edge_components(m, iu, ju, n_nodes)[0],
                   onsets=onsets, adjust_time=adjust_time, t_thresh=t_thresh, primary_p=primary_p,
                   n_perm=n_perm, alpha=alpha, tail=tail, seed=seed, cluster_stat=cluster_stat)
    res["iu"] = iu; res["ju"] = ju
    res["t"] = res["stat"]
    return res


# --------------------------------------------------------------------------- #
# Edge x FREQUENCY NBS : subnetworks that extend over the edge graph AND contiguous freq bins
# --------------------------------------------------------------------------- #
# The element space is (edge, freq bin) : clustering uses the product of the shared-node edge
# graph (within a bin) and a 1D frequency chain (same edge, adjacent bins), so a subnetwork can
# span a data-driven frequency range instead of a pre-defined band. Same engine as the sensor
# spatial test (generic adjacency), so these wrappers just build the product adjacency and
# delegate to spatial_trend / spatial_freedman_lane. Input Y is flattened FREQ-MAJOR
# (index f * n_edges + e), matching freq_product_adjacency. Heavier than the per-band NBS
# (n_edges x n_freqs elements), and the significant set is an edge x freq mask, not a matrix.

def nbs_freq_trend(effects, iu, ju, n_nodes, n_freqs, *, t_thresh=2.5, primary_p=None, n_perm=1000,
                   alpha=0.05, tail="both", seed=0, method="pearson", cluster_stat="size",
                   adjacency=None):
    """Edge x frequency subnetworks whose per-cell effect trends across sessions.
    effects : (n_sessions, n_freqs * n_edges) flattened freq-major. `primary_p` overrides
    `t_thresh`. Pass a prebuilt `adjacency` to skip the product-adjacency construction."""
    ADJ = adjacency if adjacency is not None else freq_product_adjacency(
        edge_adjacency(iu, ju, n_nodes), n_freqs)
    res = spatial_trend(effects, ADJ, t_thresh=t_thresh, primary_p=primary_p, n_perm=n_perm,
                        alpha=alpha, tail=tail, seed=seed, method=method, cluster_stat=cluster_stat)
    res["iu"] = iu; res["ju"] = ju; res["n_freqs"] = n_freqs; res["adjacency"] = ADJ
    return res


def nbs_freq_freedman_lane(Y, iu, ju, n_nodes, n_freqs, *, effect, nuisance,
                           effect_kind="auto", nuisance_kind="auto", stat="auto", thresh=None,
                           primary_p=0.001, n_perm=1000, alpha=0.05, tail="both", seed=0,
                           cluster_stat="size", perm_method="freedman-lane", block_size=None,
                           max_consecutive=None, adjacency=None):
    """Edge x frequency subnetworks with the partial effect of ONE term (dose, or MI-vs-Rest
    condition) adjusted for nuisance terms. Y : (n_obs, n_freqs * n_edges) flattened freq-major.
    Same term / permutation semantics as nbs_freedman_lane ; nuisance=[] gives a pure
    (intercept-only) two-group test. Pass a prebuilt `adjacency` to reuse it across calls."""
    ADJ = adjacency if adjacency is not None else freq_product_adjacency(
        edge_adjacency(iu, ju, n_nodes), n_freqs)
    res = spatial_freedman_lane(Y, ADJ, effect=effect, nuisance=nuisance,
                                effect_kind=effect_kind, nuisance_kind=nuisance_kind, stat=stat,
                                thresh=thresh, primary_p=primary_p, n_perm=n_perm, alpha=alpha,
                                tail=tail, seed=seed, cluster_stat=cluster_stat,
                                perm_method=perm_method, block_size=block_size,
                                max_consecutive=max_consecutive)
    res["iu"] = iu; res["ju"] = ju; res["n_freqs"] = n_freqs; res["adjacency"] = ADJ
    return res


def nbs_freq_relu(Y, iu, ju, n_nodes, n_freqs, time, *, onsets=None, adjust_time=False,
                  t_thresh=None, primary_p=0.001, n_perm=1000, alpha=0.05, tail="both", seed=0,
                  cluster_stat="size", adjacency=None):
    """Edge x frequency subnetworks that APPEAR at an UNKNOWN onset (relu / hinge change-point).
    Y : (n_sessions, n_freqs * n_edges) flattened freq-major. Delegates to spatial_relu on the
    product adjacency, exactly as nbs_freq_trend delegates to spatial_trend. Pass a prebuilt
    `adjacency` to reuse it across calls."""
    ADJ = adjacency if adjacency is not None else freq_product_adjacency(
        edge_adjacency(iu, ju, n_nodes), n_freqs)
    res = spatial_relu(Y, ADJ, time, onsets=onsets, adjust_time=adjust_time, t_thresh=t_thresh,
                       primary_p=primary_p, n_perm=n_perm, alpha=alpha, tail=tail, seed=seed,
                       cluster_stat=cluster_stat)
    res["iu"] = iu; res["ju"] = ju; res["n_freqs"] = n_freqs; res["adjacency"] = ADJ
    return res
