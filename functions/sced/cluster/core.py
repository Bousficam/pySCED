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

import os

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


def check_adjacency(adjacency, *, names=None):
    """Pre-flight check of the element NEIGHBOURHOOD, which is what the clustering step operates on.

    In : the adjacency (dense or sparse). Out : a dict of findings; never raises, never mutates. An
    asymmetric or inhomogeneous neighbour definition silently distorts which clusters can form at all,
    so Meyer et al. (2021, p. 5-6) prescribe verifying symmetry and per-element homogeneity - and
    plotting the neighbourhood - BEFORE running the test. ``degree`` here excludes self-adjacency.
    ``isolated`` elements can never join a cluster; a large ``degree_ratio`` means some elements can
    recruit far more neighbours than others, which biases where clusters form.
    """
    A = adjacency
    try:
        M = np.asarray(A.todense(), dtype=float) if hasattr(A, "todense") else np.asarray(A, float)
    except Exception:
        M = np.asarray(A, dtype=float)
    n = M.shape[0]
    deg = (M != 0).sum(axis=1) - (np.diag(M) != 0).astype(int)
    asym = int(np.sum((M != 0) != (M != 0).T) // 2)
    dmin, dmax = (int(deg.min()), int(deg.max())) if n else (0, 0)
    iso = np.where(deg == 0)[0]
    return {"n_elements": int(n), "symmetric": asym == 0, "n_asymmetric_pairs": asym,
            "degree_min": dmin, "degree_max": dmax,
            "degree_mean": round(float(deg.mean()), 2) if n else 0.0,
            "degree_ratio": (round(dmax / dmin, 2) if dmin else float("inf")),
            "n_isolated": int(iso.size),
            "isolated": ([str(names[i]) for i in iso] if names is not None else iso.tolist())[:20],
            "verdict": ("OK" if (asym == 0 and iso.size == 0 and dmin and dmax / dmin <= 3)
                        else "INSPECT: " + "; ".join(filter(None, [
                            f"{asym} asymmetric pair(s)" if asym else "",
                            f"{iso.size} isolated element(s) (can never cluster)" if iso.size else "",
                            (f"degree range {dmin}-{dmax} is inhomogeneous" if dmin and dmax / dmin > 3
                             else "")]))),
            "reference": "Meyer et al. 2021 (Dev. Cogn. Neurosci. 52:101036) p. 5-6: define "
                         "neighbourhoods bidirectionally and plot them to verify symmetry and "
                         "homogeneity for the electrode layout in use."}


def adjacency_components(supra_mask, adjacency):
    """Connected components of the supra-threshold ELEMENTS on a generic adjacency.

    adjacency : sparse (n_elements, n_elements) neighbour matrix (e.g. the electrode x
    frequency adjacency from mne.stats.combine_adjacency). Returns label_per_element
    (0 = element not retained). Sensor-space counterpart of edge_components.

    MEASURED, do not "optimise" blindly. A TFCE run calls this 50 times per permutation, so it
    dominates that path (84 per cent of the run at 272 elements). The obvious suspect, the double
    fancy-index below, is NOT the cost : replacing it with a pre-extracted edge list masked per
    call was measured 8 per cent SLOWER end to end, because rebuilding a COO per call costs more
    than the CSR slice saves. The remaining cost is connected_components itself, once per height,
    and the only real gain left is algorithmic - a union-find over elements sorted by height would
    compute all heights in ONE pass, the TFCE masks being nested by construction.
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
# Exchangeability blocks (Winkler et al. 2014, 2015)
# ---------------------------------------------------------------------------
# A permutation test is valid only for rearrangements the data's dependence structure admits. A
# FREE permutation of the n rows asserts one error term and one variance : it is the right group
# when the rows are the repeated measures of a single unit, and the WRONG one as soon as Y stacks
# several units, where it destroys the within-unit dependence the observed statistic still carries
# and narrows the null (anti-conservative). Winkler et al. (2014, p. 384-386) replace the free
# group by EXCHANGEABILITY BLOCKS, rearranged either internally (within-block) or as a whole
# (whole-block) ; Winkler et al. (2015) nest the two. Measured here on a run x session design with
# a session random effect, a session-level term tested by free permutation reaches a type I error
# of 0.275 at alpha = 0.05, against 0.040 for the whole-block regime.
def block_index(blocks):
    """Row indices of each block, in order of first appearance.

    In : a label per row. Out : {label: index array}. No side effect.
    """
    b = np.asarray(blocks)
    return {lab: np.where(b == lab)[0] for lab in dict.fromkeys(b.tolist())}


def block_permuter(blocks=None, *, exchange="within", units=None):
    """Build the row-permutation generator for one exchangeability regime.

    In : the block label per row (None = free permutation, the historical default), the regime
         ('within' | 'whole' | 'nested'), and, for the block-moving regimes, the label of the
         SUPER-block each row belongs to (blocks are exchanged only inside their super-block ;
         None means every block is exchangeable with every other).
    Out : a callable ``perm(rng, n) -> index vector`` where ``out[i]`` is the SOURCE row of
          position i. Side effect : consumes randomness from the generator it is handed.

      'within' - rows move inside their block, the blocks stay put. The regime for a term that
                 VARIES inside the block. A term constant within block is untestable here : the
                 permuted statistic equals the observed one and p tends to 1.
      'whole'  - whole blocks are exchanged inside their super-block, contents intact. The regime
                 for a term constant within block. Requires equal block sizes.
      'nested' - both at once (blocks exchanged, then rows shuffled inside), i.e. two levels of
                 exchangeability (Winkler et al. 2015).

    ``blocks=None`` reproduces the previous free shuffle EXACTLY, including its consumption of the
    generator, so a call that does not ask for blocks is unchanged down to the last draw.
    """
    if blocks is None:                                  # free permutation : the historical path,
        order = np.empty(0, dtype=int)                  # same in-place shuffle of a kept array

        def perm_free(rng, n):
            nonlocal order
            if order.size != n:
                order = np.arange(n)
            rng.shuffle(order)
            return order
        return perm_free
    idx = block_index(blocks)
    if exchange == "within":
        def perm_within(rng, n):
            out = np.arange(n)
            for rows in idx.values():
                out[rows] = rng.permutation(rows)
            return out
        return perm_within
    if exchange not in ("whole", "nested"):
        raise ValueError(f"exchange must be 'within', 'whole' or 'nested', not {exchange!r}")
    # whole / nested : group the blocks by super-block, then check they are interchangeable at all.
    if units is None:
        by_super = {None: list(idx)}
    else:
        u = np.asarray(units)
        by_super = {}
        for lab, rows in idx.items():
            vals = set(u[rows].tolist())
            if len(vals) != 1:
                raise ValueError(f"block_permuter : block {lab!r} spans several units ({vals}). "
                                 "Exchangeability blocks must nest inside their unit.")
            by_super.setdefault(vals.pop(), []).append(lab)
    for sup, members in by_super.items():
        sizes = {len(idx[b]) for b in members}
        if len(sizes) > 1:
            raise ValueError(
                f"block_permuter : inside unit {sup!r} the blocks do not all have the same size "
                f"({sorted(sizes)}), so exchanging whole blocks is not admissible - it would "
                "change the sample size attached to the tested term rather than relabel the same "
                "data. Restrict to complete blocks, or aggregate and use the 'within' regime.")
    shuffle_within = exchange == "nested"

    def perm_whole(rng, n):
        out = np.arange(n)
        for members in by_super.values():
            order = rng.permutation(len(members))
            for pos, b in enumerate(members):
                src = idx[members[order[pos]]]
                out[idx[b]] = rng.permutation(src) if shuffle_within else src
        return out
    return perm_whole


def scheme_by_block(scheme, labels, blocks=None, *, exchange="within", units=None):
    """Wrap a randomization schedule so it draws inside the exchangeability blocks.

    In : a ``scheme(labels, rng) -> relabeled`` generator (alternating_scheme / block_scheme), the
         observed label vector, the blocks, the regime and the super-block labels.
    Out : a callable ``draw(rng) -> relabeled vector``. Side effect : consumes randomness.

    Design-based (Draper-Stoneman) randomization has the same exchangeability problem as the
    residual permutation : redrawing the assignment ACROSS units invents schedules the trial never
    could have produced. Under 'within' the schedule is redrawn separately inside each block, so
    each block keeps its own condition counts ; under 'whole' / 'nested' the block's label sequence
    travels with the block. ``blocks=None`` calls the scheme on the whole vector, as before.
    """
    labels = np.asarray(labels)
    if blocks is None:
        return lambda rng: scheme(labels, rng)
    if exchange == "within":
        idx = block_index(blocks)

        def draw_within(rng):
            out = np.array(labels, copy=True)
            for rows in idx.values():
                out[rows] = scheme(labels[rows], rng)
            return out
        return draw_within
    perm = block_permuter(blocks, exchange=exchange, units=units)
    return lambda rng: labels[perm(rng, labels.shape[0])]


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


def _merge_directions(rp, rn):
    """Combine the positive-direction and negative-direction cluster results of a two-tailed signed
    run into ONE res dict (same keys as a single pass), with sign-separated clusters. Labels of the
    negative pass are offset so pos/neg never collide ; their component supports are disjoint (a cell
    is either pos-supra or neg-supra, never both), so comp_labels sum cleanly. Each direction keeps
    its OWN same-sign max-cluster null (Zalesky NBS convention), so a strong cluster of one sign does
    not raise the bar for the other. null_max is concatenated for diagnostics only."""
    off = int(rp["comp_labels"].max())
    neg_lab = np.where(rn["comp_labels"] > 0, rn["comp_labels"] + off, 0)
    comp_pvals = dict(rp["comp_pvals"]); comp_pvals.update({k + off: v for k, v in rn["comp_pvals"].items()})
    comp_scores = dict(rp["comp_scores"]); comp_scores.update({k + off: v for k, v in rn["comp_scores"].items()})
    # TWO-TAILED CONVENTION, stated explicitly rather than left implicit. Each direction is judged
    # against its OWN same-sign max-cluster null at the same alpha, i.e. two one-sided tests, so the
    # per-direction p is NOT already two-tailed. Meyer et al. (2021, p. 5) require the convention to
    # be declared and corrected one of two equivalent ways: halve alpha, or double the Monte-Carlo p.
    # Neither is applied here (that would change every p this project has already reported); the
    # convention is REPORTED so the reader can apply it, and so a bare p is never ambiguous by 2x.
    merged = {"tail_convention": "two one-sided passes, each vs its own same-sign null at alpha; "
                                 "p is NOT doubled - halve alpha or double p for a two-tailed claim "
                                 "(Meyer et al. 2021 p. 5; Zalesky NBS sign-separated convention)",
              "signed": rp.get("signed", True),   # a two-tailed pass only exists for a signed stat
              "stat": rp["stat"], "thresh": rp["thresh"], "comp_labels": rp["comp_labels"] + neg_lab,
              "comp_pvals": comp_pvals, "sig_edges": rp["sig_edges"] | rn["sig_edges"],
              "sizes_obs": np.concatenate([rp["sizes_obs"], rn["sizes_obs"]]),
              "comp_scores": comp_scores, "cluster_stat": rp["cluster_stat"],
              "null_max": np.concatenate([rp["null_max"], rn["null_max"]])}
    if rp.get("tfce"):                                 # carry the threshold-free per-element maps
        merged["tfce"] = True                          # pos/neg supports are disjoint -> sum the maps,
        merged["tfce_obs"] = rp["tfce_obs"] + rn["tfce_obs"]   # take the per-element min p across dirs
        merged["p_elem"] = np.minimum(rp["p_elem"], rn["p_elem"])
    return merged


# ---------------------------------------------------------------------------
# TFCE - Threshold-Free Cluster Enhancement (Smith & Nichols 2009, NeuroImage 44:83-98)
# ---------------------------------------------------------------------------
# Instead of one arbitrary cluster-forming threshold, TFCE integrates each element's cluster support
# over ALL heights h : TFCE(i) = integral_0^{h_i} extent(h, i)^E * h^H dh, where extent(h, i) is the
# size of the supra-threshold cluster containing i at height h (the SAME adjacency labeler as the
# fixed-threshold path). The TFCE map is the statistic ; inference is the max-TFCE permutation null,
# giving a per-ELEMENT FWER-corrected p (not a per-cluster p). Signed t : done per direction (the
# pos/neg split in cluster_run), so a positive and a negative cluster never merge.
#
# The HEIGHT exponent H = 2 in every dimension (Smith & Nichols 2009 ; Mensen & Khatami 2013 keep
# H = 2 for EEG). The EXTENT exponent E depends on the DIMENSIONALITY of the element lattice, because
# "extent" (a count of connected neighbours) scales with the lattice dimension and E rebalances it
# against the height term (Smith & Nichols 2009, sec. on weighting parameters ; Mensen & Khatami 2013
# p. 113-114). We set E per dimension :
#
#   1D  (linear / chain lattice - e.g. a FREQUENCY profile, a TIME course, a TBSS-like skeleton) :
#         E = 1.0  -- FSL randomise's 1D/TBSS convention (`--T2`, H=2 E=1) and Mensen & Khatami 2013
#         "TFCE-B" (E=1, H=2), the empirically best-performing EEG setting (rank 1 on MCC, p. 114).
#   2D  (surface / lattice - e.g. EEG channels x frequency, a cortical surface, a spatio-spectral map):
#         E = 2/3  -- Mensen & Khatami 2013 "TFCE-A" (E=2/3, H=2), the theoretically-derived
#         (Smith & Nichols) value and the most CONSISTENT across sources / SNR (p. 113-114).
#   3D  (volume - e.g. an fMRI / VBM voxel grid) :
#         E = 0.5  -- Smith & Nichols 2009 fMRI default (FSL randomise default, H=2 E=0.5).
#
# Our pipeline uses only 1D (density L3 = frequency chain) and 2D (node L2 = spatial x frequency) ;
# 3D is documented for completeness and reachable only by explicit override (TFCE_DIM=3 or dim=3),
# since element degree alone cannot separate a dense 2D lattice from a 3D volume.
TFCE_H = 2.0                                            # height exponent (all dimensions)
TFCE_E_1D = 1.0                                         # extent exponent, linear/chain lattice (1D)
TFCE_E_2D = 2.0 / 3.0                                   # extent exponent, surface/lattice (2D)
TFCE_E_3D = 0.5                                         # extent exponent, volume (3D)
TFCE_E = TFCE_E_2D                                      # neutral fallback when dimensionality unknown
TFCE_STEPS = int(os.environ.get("TFCE_STEPS", 50))     # height discretisation steps (cost knob ; env-tunable
#   for heavy sweeps - the Riemann sum converges, so 25-50 is ample, lower = faster/coarser)


def tfce_e_for(adjacency, *, dim=None):
    """TFCE extent exponent E for the element lattice DIMENSIONALITY (H = TFCE_H always). `dim`
    (1|2|3) or the env TFCE_DIM force the choice ; otherwise the dimensionality is inferred from the
    adjacency : a path/chain (max off-diagonal degree <= 2) is 1D -> E=1 ; anything denser is treated
    as a 2D surface/lattice -> E=2/3. A true 3D volume (E=0.5) must be requested explicitly, as degree
    alone cannot tell a dense 2D lattice from a 3D one. Refs : Smith & Nichols (2009) ; Mensen &
    Khatami (2013, p. 113-114 : TFCE-A E=2/3, TFCE-B E=1) ; FSL randomise (`--T2` E=1 for TBSS)."""
    if dim is None:
        dim = int(os.environ.get("TFCE_DIM", 0)) or None
    if dim == 1:
        return TFCE_E_1D
    if dim == 2:
        return TFCE_E_2D
    if dim == 3:
        return TFCE_E_3D
    A = adjacency
    try:
        deg = np.asarray(A.sum(axis=1)).ravel()
        diag = np.asarray(A.diagonal()).ravel() if hasattr(A, "diagonal") else np.diag(np.asarray(A))
    except Exception:
        A = np.asarray(A, dtype=float); deg = A.sum(1); diag = np.diag(A)
    dmax = float(np.max(deg - diag)) if deg.size else 0.0
    return TFCE_E_1D if dmax <= 2.0 + 1e-9 else TFCE_E_2D   # chain -> 1D, else 2D (our case)


def _tfce_map(stat, labeler, tail, signed, *, E=TFCE_E, H=TFCE_H, n_steps=TFCE_STEPS):
    """TFCE-enhanced map for one direction. `stat` is the per-element statistic ; `labeler(mask) ->
    component labels` supplies the adjacency. For a signed statistic the height is oriented to the
    tested tail (pos -> stat, neg -> -stat) ; unsigned (F) uses stat directly. Only positive heights
    contribute. Returns the enhanced map (>= 0), same shape as `stat`."""
    if signed:
        h_elem = stat if tail == "pos" else (-stat if tail == "neg" else np.abs(stat))
    else:
        h_elem = stat
    h_elem = np.maximum(np.asarray(h_elem, float), 0.0)
    hmax = float(h_elem.max()) if h_elem.size else 0.0
    out = np.zeros_like(h_elem)
    if hmax <= 0:
        return out
    dh = hmax / n_steps
    for step in range(1, n_steps + 1):                 # heights dh, 2dh, ..., hmax (midpoint-free Riemann sum)
        h = step * dh
        supra = h_elem >= h
        if not supra.any():
            continue
        labels = labeler(supra)
        # extent^E * h^H * dh, added to every member of each component. Vectorised over components
        # via bincount rather than looped per label : `sizes[labels]` gives each element the extent
        # of ITS component in one pass. Arithmetically identical to the per-label loop, and the
        # loop was measurable - this map is evaluated n_steps times per permutation.
        sizes = np.bincount(labels)
        if sizes.size > 1:                             # at least one component above this height
            out += np.where(labels > 0, (sizes[labels] ** E) * (h ** H) * dh, 0.0)
    return out


def _tfce_run(stat_obs, statmap, Y, Z, pZ, labeler, *, signed, tail, n_perm, alpha, seed,
              scheme, ds_labels, restat, tfce_e=None, perm=None, relabel=None):
    """Threshold-free counterpart of the fixed-threshold body of cluster_run : enhance the observed
    map, build the max-TFCE permutation null (same FL / Draper-Stoneman schemes), then a per-element
    FWER p = fraction of permuted max-TFCE >= the element's TFCE. Significant elements are grouped
    into clusters for REPORTING only (each cluster's p = its min element p). Returns the cluster_run
    keys (+ 'tfce', 'tfce_obs', 'p_elem'). `tfce_e` = extent exponent E (dimensionality-dependent,
    resolved by the caller via tfce_e_for) ; None falls back to the neutral TFCE_E."""
    E = TFCE_E if tfce_e is None else float(tfce_e)
    tfce_obs = _tfce_map(stat_obs, labeler, tail, signed, E=E)
    rng = np.random.default_rng(seed)
    null_max = np.empty(n_perm)
    perm = block_permuter() if perm is None else perm
    if scheme is not None:                             # design-based randomization (Draper-Stoneman)
        relabel = (lambda r: scheme(np.asarray(ds_labels), r)) if relabel is None else relabel
        for k in range(n_perm):
            s = restat(relabel(rng))(Y)
            null_max[k] = _tfce_map(s, labeler, tail, signed, E=E).max()
    else:                                              # Freedman-Lane residual permutation
        fit = Z @ (pZ @ Y) if Z.shape[1] else np.zeros_like(Y)
        resid = Y - fit
        for k in range(n_perm):
            s = statmap(fit + resid[perm(rng, Y.shape[0])])
            null_max[k] = _tfce_map(s, labeler, tail, signed, E=E).max()
    # per-element FWER-corrected p (element's TFCE vs the max-TFCE null)
    p_elem = (1 + (null_max[None, :] >= tfce_obs[:, None]).sum(axis=1)) / (n_perm + 1)
    sig = p_elem < alpha
    comp_labels = labeler(sig)                         # cluster the significant elements (reporting)
    comp_scores, comp_pvals = {}, {}
    for lab in np.unique(comp_labels[comp_labels > 0]):
        m = comp_labels == lab
        comp_scores[int(lab)] = float(tfce_obs[m].sum())    # cluster TFCE mass
        comp_pvals[int(lab)] = float(p_elem[m].min())       # cluster p = min element p
    sizes_obs = np.array([int((comp_labels == L).sum())
                          for L in np.unique(comp_labels[comp_labels > 0])], dtype=int)
    return {"stat": stat_obs, "thresh": float("nan"), "tfce": True, "tfce_obs": tfce_obs,
            "p_elem": p_elem, "comp_labels": comp_labels, "comp_pvals": comp_pvals,
            "sig_edges": sig, "sizes_obs": sizes_obs, "comp_scores": comp_scores,
            "cluster_stat": "tfce", "null_max": null_max, "signed": bool(signed)}


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
def _glm_statmap(Y, design, test_cols, nuisance_cols, stat, thresh, primary_p, vg=None):
    """Build the per-element GLM statistic map + reduced model, shared by network and
    spatial engines. Returns (statmap, Y, Z, pZ, thresh, signed).

    stat : 'F'  classic F / signed t   - assumes ONE variance for all observations.
           'W'  robust Wald, White (1980) HC0 - one variance per OBSERVATION, no structure.
           'G'  Winkler et al. (2014) Eq. 6-7 - one variance per VARIANCE GROUP `vg`. Pivotal
                under heteroscedasticity where F is not. With a single VG it reduces exactly to F
                (and to t^2 when one column is tested), so 'G' with vg=None is not an error, it is
                simply F computed the long way.
           'rank' F on ranks (monotone-robust).
    vg   : variance-group label per observation, length n. Required by 'G' to be informative.
           Winkler et al. (2015) constrains it : any two observations that the exchangeability
           blocks allow to be swapped MUST share a variance group, so vg is normally the
           permutation block itself (here : the subject, or the subject x session).
    """
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
    signed = stat in ("t", "rank") or (stat in ("W", "G") and p_x == 1)
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

    # ---- G : Winkler et al. 2014, Eq. 6-7 -------------------------------------------------
    # G = [b' (C (M'WM)^-1 C')^-1 b] / (Lambda * rank(C)), W diagonale = 1/variance du groupe.
    # M'WM se factorise en somme ponderee de matrices FIXES : M'WM = sum_g w_g * (M_g' M_g), donc
    # les A_g se precalculent une fois et seule la ponderation depend de l'element.
    if vg is not None:
        _vg = np.asarray(vg)
        _groups = list(dict.fromkeys(_vg.tolist()))
        _rows = [np.where(_vg == g)[0] for g in _groups]
        _A = np.stack([D[r].T @ D[r] for r in _rows])                  # (n_g, k, k)
        # Degres de liberte EFFECTIFS du groupe : somme de la diagonale de la matrice residuelle
        # R = I - M(M'M)^-1 M' restreinte au groupe. Les residus proviennent de l'ajustement GLOBAL
        # sur les n observations, pas d'un ajustement par groupe : `n_g - rang(M_g)` sur-estimerait
        # alors les ddl et sous-estimerait la variance du groupe, ce qui detruit la pivotalite.
        _hii = np.einsum("ij,ji->i", D, pD)                            # levier de chaque ligne
        _dof = np.array([max((1.0 - _hii[r]).sum(), 1e-6) for r in _rows], float)
    else:
        _rows, _A, _dof = None, None, None

    def statmap_G(Yv):
        """G de Winkler : F generalise a plusieurs groupes de variance, pivotal sous
        heteroscedasticite. Un seul groupe -> Lambda = 1 et G = F exactement."""
        if _rows is None:                       # aucun groupe declare : G est F
            return statmap_F(Yv)
        beta_ols = pD @ Yv                                               # (k, n_elem)
        E2 = (Yv - D @ beta_ols) ** 2                                    # (n, n_elem)
        # variance par groupe et par element, puis poids w_g = 1 / sigma2_g
        s2 = np.stack([E2[r].sum(axis=0) / d for r, d in zip(_rows, _dof)])   # (n_g, n_elem)
        w = 1.0 / np.maximum(s2, 1e-24)                                  # (n_g, n_elem)
        MWM = np.einsum("gv,gij->vij", w, _A)                            # (n_elem, k, k)
        inv = np.linalg.pinv(MWM)
        # psi_hat est l'estimateur PONDERE (M'WM)^-1 M'W Y, pas celui des moindres carres
        # ordinaires : la covariance du denominateur est celle du modele pondere, apparier les deux
        # est ce qui rend G pivotal. Avec un seul groupe les deux estimateurs coincident, si bien
        # qu'un test a un seul VG ne revele pas la confusion.
        MWY = np.einsum("gv,gkv->kv", w,
                        np.stack([D[r].T @ Yv[r] for r in _rows]))       # (k, n_elem)
        beta = np.einsum("vij,jv->iv", inv, MWY)                         # (k, n_elem)
        sub = inv[:, test_cols][:, :, test_cols]                         # (n_elem, p_x, p_x)
        b = beta[test_cols].T                                            # (n_elem, p_x)
        if p_x == 1:
            v = np.maximum(sub[:, 0, 0], 1e-24)
            g = (b[:, 0] ** 2) / v
        else:
            g = np.einsum("vi,vij,vj->v", b, np.linalg.pinv(sub), b)
        # Correction de Welch. Elle s'annule si p_x == 1 ou si un seul groupe porte tout le poids.
        s_rank = float(p_x)
        if p_x > 1 and len(_rows) > 1:
            n_g = np.array([len(r) for r in _rows], float)[:, None]
            tr = (w * n_g).sum(axis=0)                                   # trace(W) par element
            frac = (w * n_g) / np.maximum(tr, 1e-24)
            lam = 1.0 + (2 * (s_rank - 1) / (s_rank * (s_rank + 2))) * \
                  ((1.0 - frac) ** 2 / _dof[:, None]).sum(axis=0)
        else:
            lam = 1.0
        g = g / (lam * s_rank)
        return np.sign(b[:, 0]) * np.sqrt(np.maximum(g, 0.0)) if p_x == 1 else g

    statmap = {"W": statmap_W, "G": statmap_G}.get(stat, statmap_F)

    def effmap(Yv):
        """Per-element EFFECT ESTIMATE = coefficient of the tested column (signed t / rank / W
        single-column designs) ; NaN for a multi-column F / W design (no single coefficient). This
        is the interpretable effect (slope, contrast) reported and used for the colorbars."""
        if tcol is None:
            return np.full(Yv.shape[1], np.nan)
        return (pD @ Yv)[tcol]

    if primary_p == "TFCE":                           # threshold-free : no cluster-forming threshold ;
        thresh = "TFCE"                               # sentinel -> cluster_run takes the _tfce_run path
    elif thresh is None:
        if stat == "W":                               # signed W ~ N(0,1) ; multi-column W ~ chi2(p_x)
            thresh = (float(_ndist.ppf(1 - primary_p)) if p_x == 1
                      else float(_chi2.ppf(1 - primary_p, p_x)))
        elif stat == "G":                             # G est normalise par rank(C) : chi2/p_x ;
            thresh = (float(_ndist.ppf(1 - primary_p)) if p_x == 1   # signe -> N(0,1) comme W
                      else float(_chi2.ppf(1 - primary_p, p_x)) / p_x)
        elif not signed:
            thresh = float(_fdist.ppf(1 - primary_p, p_x, df2))
        else:
            thresh = float(_tdist.ppf(1 - primary_p, df2))
    ret_thresh = thresh if isinstance(thresh, str) else float(thresh)
    return statmap, effmap, Y, Z, pZ, ret_thresh, signed


def cluster_run(statmap, Y, Z, pZ, labeler, *, thresh, signed, tail,
                n_perm, alpha, seed, cluster_stat="size",
                scheme=None, ds_labels=None, restat=None, tfce_e=None,
                blocks=None, exchange="within", units=None):
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

    Two-tailed signed test (tail='both') : run the POSITIVE and NEGATIVE directions as two separate
    one-sided passes, each clustered on its own supra map and judged against its OWN same-sign
    max-cluster null (Zalesky NBS / sign-separated Maris-Oostenveld), then merged. Never fuses an
    adjacent pos and neg element into one blob nor lets one sign's strong cluster inflate the other.

    EXCHANGEABILITY (`blocks`, `exchange`, `units`). Both nulls above rearrange the n ROWS of Y.
    Left free - the default, and the only behaviour available before - that rearrangement asserts
    that the rows are exchangeable with one another, which holds for the repeated measures of ONE
    unit and fails as soon as Y stacks several : the null then loses the within-unit dependence the
    observed statistic keeps, and the test turns anti-conservative. Pass `blocks` (one label per
    row) to restrict the rearrangement to Winkler's exchangeability blocks, `exchange` to pick the
    regime, and `units` to say inside what the blocks may be exchanged. See `block_permuter`.
    Defaults keep the free permutation, so every existing call is unchanged.
    """
    if signed and tail == "both":
        one = dict(thresh=thresh, signed=signed, n_perm=n_perm, alpha=alpha, seed=seed,
                   cluster_stat=cluster_stat, scheme=scheme, ds_labels=ds_labels, restat=restat,
                   tfce_e=tfce_e, blocks=blocks, exchange=exchange, units=units)
        rp = cluster_run(statmap, Y, Z, pZ, labeler, tail="pos", **one)
        rn = cluster_run(statmap, Y, Z, pZ, labeler, tail="neg", **one)
        return _merge_directions(rp, rn)
    stat_obs = statmap(Y)
    # One generator per run, built before either branch, so the fixed-threshold and the
    # threshold-free paths draw from the SAME reference set for the same arguments.
    perm = block_permuter(blocks, exchange=exchange, units=units)
    relabel = (scheme_by_block(scheme, ds_labels, blocks, exchange=exchange, units=units)
               if scheme is not None else None)
    if isinstance(thresh, str) and thresh == "TFCE":  # threshold-free : integrate over all heights
        return _tfce_run(stat_obs, statmap, Y, Z, pZ, labeler, signed=signed, tail=tail,
                         n_perm=n_perm, alpha=alpha, seed=seed,
                         scheme=scheme, ds_labels=ds_labels, restat=restat, tfce_e=tfce_e,
                         perm=perm, relabel=relabel)
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
        for k in range(n_perm):
            s = restat(relabel(rng))(Y)
            sc = _component_scores(labeler(_supra(s, tail, thresh, signed)), s, cluster_stat)
            null_max[k] = max(sc.values()) if sc else 0.0
    else:
        # Freedman-Lane residual permutation, restricted to the exchangeability blocks when
        # `blocks` is given (free otherwise). Exact reference set for a count-balanced design with
        # no schedule constraint.
        fit = Z @ (pZ @ Y) if Z.shape[1] else np.zeros_like(Y)
        resid = Y - fit
        for k in range(n_perm):
            s = statmap(fit + resid[perm(rng, Y.shape[0])])
            sc = _component_scores(labeler(_supra(s, tail, thresh, signed)), s, cluster_stat)
            null_max[k] = max(sc.values()) if sc else 0.0
    comp_pvals, sig_mask = _component_pvals(comp_labels, obs_scores, null_max, n_perm, alpha)
    # `signed` is carried into the result because downstream consumers cannot re-derive it: a
    # single-column W / G is a SIGNED robust ratio, a multi-column one is an unsigned chi2, and both
    # arrive here as stat_kind='W'. Only the former has a coefficient, hence an effect size.
    return {"stat": stat_obs, "thresh": float(thresh), "comp_labels": comp_labels,
            "comp_pvals": comp_pvals, "sig_edges": sig_mask, "sizes_obs": sizes_obs,
            "comp_scores": obs_scores, "cluster_stat": cluster_stat, "null_max": null_max,
            "signed": bool(signed)}


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
                 thresh, primary_p, perm_method, block_size, max_consecutive, vg=None):
    """Design-based randomization schedule for the cluster null, or None to keep Freedman-Lane.

    Returns (scheme, ds_labels, restat) to hand to cluster_run. A schedule is used when the
    caller asks for a design-based method (``perm_method`` 'draper-stoneman' / 'randomization')
    or constrains the schedule (``block_size`` / ``max_consecutive``) - exactly the cases FL
    cannot honour ; otherwise returns (None, None, None) and the caller falls back to FL. The
    ``scheme`` comes from the SHARED scalar generators (alternating_scheme / block_scheme), so
    the cluster test and run_sced_alternating draw the same admissible relabelings. ``restat``
    rebuilds the effect design column from the permuted labels and returns a fresh statmap on
    the ORIGINAL data (Draper-Stoneman : permute labels, refit, keep the nuisance in the model).

    ``stat`` et ``vg`` sont ceux de _glm_statmap et traversent tels quels : la carte recalculee a
    chaque relabelisation emploie donc la MEME statistique (et les memes groupes de variance) que la
    carte observee, ce qui est la condition pour que le nul soit celui de la statistique rapportee.
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
        return _glm_statmap(Y, design, test_cols, nuis_cols, stat, thresh, primary_p, vg)[0]

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
                          perm_method, block_size, max_consecutive, thresh, primary_p,
                          stat="t", vg=None):
    """Draper-Stoneman randomization schedule for ONE ANCOVA contrast within a RANDOMIZED factor.

    Unlike build_scheme (which permutes the tested column), the scheme here permutes the WHOLE
    factor schedule as one unit (alternating_scheme / block_scheme preserve the level counts), and
    restat rebuilds the full ANCOVA from the permuted factor - so BOTH the contrast dummy and the
    other-level dummies move together and stay mutually exclusive. The null is thus the
    randomization distribution of the factor (Edgington), not a free permutation of a binary dummy.
    Returns (scheme, factor_labels, restat) to hand to cluster_run. FL is used when perm_method is
    'freedman-lane' and no schedule constraint is set (handled by the caller).

    ``stat`` (defaut 't', le t signe du contraste) et ``vg`` sont ceux de _glm_statmap : la carte
    recalculee a chaque relabelisation emploie la meme statistique et les memes groupes de variance
    que la carte observee."""
    from ..core import alternating_scheme, block_scheme
    scheme = block_scheme(block_size) if block_size else alternating_scheme(max_consecutive)
    factor = np.asarray([str(x) for x in factor])

    def restat(perm_factor):
        design, tc, nc = _contrast_design(Y, perm_factor, contrast, cond_order, nuisance, nuisance_kind)
        return _glm_statmap(Y, design, tc, nc, stat, thresh, primary_p, vg)[0]

    return scheme, factor, restat
