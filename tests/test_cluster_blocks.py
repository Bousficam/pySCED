"""Exchangeability blocks in the cluster layer (Winkler et al. 2014, 2015).

The cluster engine used to permute the n rows of Y freely, which asserts a single error term and a
single variance. That is right for the repeated measures of ONE unit and wrong as soon as Y stacks
several, where the free rearrangement destroys the within-unit dependence the observed statistic
still carries. These tests pin down four things:

  - the default path is unchanged, down to the exact draws consumed from the generator ;
  - each regime rearranges what it claims to rearrange, and nothing else ;
  - the admissibility guards fire instead of silently producing an invalid reference set ;
  - the regimes behave as advertised on simulated data with a known dependence structure - free
    permutation is anti-conservative for a term constant within block, the whole-block regime is
    not, and the within-block regime cannot test such a term at all.

The last group is a Monte-Carlo measurement, so its thresholds are deliberately loose: they are set
to catch a broken reference set, not to certify an exact error rate.
"""
import numpy as np
import pytest
from scipy import sparse

from functions.sced.cluster.core import (block_index, block_permuter, scheme_by_block,
                                         cluster_run, adjacency_components, _glm_statmap)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _chain(n_elem):
    """Adjacency of a 1-D chain lattice (element i neighbours i-1 and i+1)."""
    return sparse.diags([np.ones(n_elem - 1), np.ones(n_elem - 1)], [1, -1], format="csr")


def _nested_design(n_sub=3, n_sess=6, n_run=3):
    """Row labels of a run nested in session nested in subject design.

    Out : (subject, session, n) with session unique across subjects. No side effect.
    """
    sub = np.repeat(np.arange(n_sub), n_sess * n_run)
    sess = np.array([f"{s}|{k}" for s, k in
                     zip(sub, np.tile(np.repeat(np.arange(n_sess), n_run), n_sub))])
    return sub, sess, len(sub)


# --------------------------------------------------------------------------- #
# 1. The default path is untouched
# --------------------------------------------------------------------------- #
def test_free_permuter_reproduces_the_previous_shuffle():
    """blocks=None must consume the generator exactly as the previous in-place shuffle did.

    The old loop kept ONE index array and re-shuffled it in place; a reimplementation that rebuilt
    the array each draw would consume the same numbers but produce a different sequence. This is
    what licenses the claim that existing results are unchanged.
    """
    perm = block_permuter()
    rng = np.random.default_rng(0)
    got = [perm(rng, 12).copy() for _ in range(5)]          # ONE generator, five successive draws
    ref_rng = np.random.default_rng(0)
    order = np.arange(12)
    want = []
    for _ in range(5):
        ref_rng.shuffle(order)
        want.append(order.copy())
    assert all(np.array_equal(g, w) for g, w in zip(got, want))


def test_cluster_run_default_matches_a_free_shuffle_reference():
    """cluster_run without blocks must give the same null as an explicit free-shuffle loop."""
    rng = np.random.default_rng(3)
    n, n_elem = 20, 8
    Y = rng.normal(size=(n, n_elem))
    x = np.arange(n, dtype=float)
    design = np.column_stack([np.ones(n), x])
    A = _chain(n_elem)
    statmap, _eff, Yv, Z, pZ, thr, signed = _glm_statmap(Y, design, [1], [0], "t", None, 0.05)
    lab = lambda m: adjacency_components(m, A)
    res = cluster_run(statmap, Yv, Z, pZ, lab, thresh=thr, signed=signed, tail="pos",
                      n_perm=50, alpha=0.05, seed=11)
    # reference : the loop as it was written before the blocks were introduced
    g = np.random.default_rng(11)
    fit = Z @ (pZ @ Yv)
    resid = Yv - fit
    order = np.arange(n)
    ref = []
    for _ in range(50):
        g.shuffle(order)
        s = statmap(fit + resid[order])
        comp = lab(s >= thr)
        sc = [float((comp == c).sum()) for c in np.unique(comp[comp > 0])]
        ref.append(max(sc) if sc else 0.0)
    assert np.array_equal(res["null_max"], np.array(ref))


# --------------------------------------------------------------------------- #
# 2. Each regime rearranges what it claims to
# --------------------------------------------------------------------------- #
def test_within_regime_keeps_every_row_in_its_block():
    sub, sess, n = _nested_design()
    perm = block_permuter(sess, exchange="within")
    out = perm(np.random.default_rng(0), n)
    assert sorted(out.tolist()) == list(range(n))          # still a permutation
    assert np.array_equal(sess[out], sess)                 # no row left its block


def test_whole_regime_moves_blocks_as_units_inside_the_unit():
    sub, sess, n = _nested_design()
    perm = block_permuter(sess, exchange="whole", units=sub)
    out = perm(np.random.default_rng(0), n)
    assert sorted(out.tolist()) == list(range(n))
    assert np.array_equal(sub[out], sub)                   # no row left its SUBJECT
    # every destination block received exactly one source block, contents intact and in order
    for rows in block_index(sess).values():
        src = out[rows]
        assert len(set(sess[src].tolist())) == 1           # one source block only
        assert np.array_equal(src, np.sort(src))           # internal order preserved
    assert not np.array_equal(sess[out], sess)             # and blocks did move


def test_nested_regime_moves_blocks_and_shuffles_inside_them():
    sub, sess, n = _nested_design()
    perm = block_permuter(sess, exchange="nested", units=sub)
    outs = [perm(np.random.default_rng(k), n) for k in range(5)]
    assert all(np.array_equal(sub[o], sub) for o in outs)
    # at least one draw must break the internal order, otherwise this is just the whole regime
    assert any(any(not np.array_equal(o[rows], np.sort(o[rows]))
                   for rows in block_index(sess).values()) for o in outs)


def test_free_regime_moves_rows_across_units():
    sub, sess, n = _nested_design()
    out = block_permuter()(np.random.default_rng(0), n)
    assert not np.array_equal(sub[out], sub)               # this is the defect being fixed


# --------------------------------------------------------------------------- #
# 3. The guards fire
# --------------------------------------------------------------------------- #
def test_unequal_block_sizes_are_refused_for_whole_block():
    sub, sess, n = _nested_design()
    boiteux = sess.copy()
    boiteux[boiteux == "0|1"] = "0|0"                      # one block of 6, the others of 3
    with pytest.raises(ValueError, match="same size"):
        block_permuter(boiteux, exchange="whole", units=sub)


def test_block_spanning_two_units_is_refused():
    sub, sess, n = _nested_design()
    achev = np.array([f"B{i % 4}" for i in range(n)])       # blocks cut across subjects
    with pytest.raises(ValueError, match="spans several units"):
        block_permuter(achev, exchange="whole", units=sub)


def test_unknown_regime_is_refused():
    sub, sess, n = _nested_design()
    with pytest.raises(ValueError, match="within"):
        block_permuter(sess, exchange="wholesale", units=sub)


# --------------------------------------------------------------------------- #
# 4. The randomization schedule is stratified too
# --------------------------------------------------------------------------- #
def test_scheme_by_block_preserves_each_block_condition_counts():
    """Draper-Stoneman must redraw the schedule INSIDE a block, never across blocks: redrawing
    across units invents assignments the trial could not have produced."""
    sub, sess, n = _nested_design(n_sub=2, n_sess=4, n_run=3)
    labels = np.array(["A", "B", "C"] * (n // 3))
    draw = scheme_by_block(lambda L, rng: rng.permutation(L), labels, sess)
    out = draw(np.random.default_rng(0))
    for rows in block_index(sess).values():
        assert sorted(out[rows].tolist()) == sorted(labels[rows].tolist())


def test_scheme_by_block_without_blocks_is_the_bare_scheme():
    labels = np.array(["A", "B"] * 6)
    draw = scheme_by_block(lambda L, rng: rng.permutation(L), labels, None)
    got = draw(np.random.default_rng(4))
    want = np.random.default_rng(4).permutation(labels)
    assert np.array_equal(got, want)


# --------------------------------------------------------------------------- #
# 5. What the regimes do to the error rate
# --------------------------------------------------------------------------- #
def _type_I(exchange, blocks, n_sim=40, n_perm=100, tau=1.0, seed0=0):
    """Share of simulations with at least one significant cluster, under H0.

    In : the regime, the block labels (None = free), the simulation and permutation counts, and the
         size of the session random effect. Out : the rate. Side effect : none.

    The data carry a session-level random effect shared by the runs of a session, and the tested
    covariate is CONSTANT within session - the shape that free permutation gets wrong.
    """
    sub, sess, n = _nested_design()
    A = _chain(10)
    lab = lambda m: adjacency_components(m, A)
    sess_ids = list(dict.fromkeys(sess.tolist()))
    hits = 0
    for s in range(n_sim):
        rng = np.random.default_rng(seed0 + s)
        s_eff = {k: rng.normal(0, tau, 10) for k in sess_ids}
        Y = np.stack([rng.normal(0, 1.0, 10) + s_eff[k] for k in sess])
        Y += rng.normal(0, 2.0, (3, 10))[sub]                        # subject level
        xv = {k: rng.normal() for k in sess_ids}
        x = np.array([xv[k] for k in sess])                          # constant within session
        design = np.column_stack([np.ones(n), x])
        statmap, _e, Yv, Z, pZ, thr, signed = _glm_statmap(Y, design, [1], [0], "t", None, 0.05)
        res = cluster_run(statmap, Yv, Z, pZ, lab, thresh=thr, signed=signed, tail="both",
                          n_perm=n_perm, alpha=0.05, seed=s,
                          blocks=blocks(sess), exchange=exchange, units=sub)
        hits += any(p < 0.05 for p in res["comp_pvals"].values())
    return hits / n_sim


def test_free_permutation_is_anticonservative_for_a_block_constant_term():
    """The defect, measured: with a session random effect and no true effect, free permutation
    declares clusters far above the nominal 5 per cent."""
    rate = _type_I("within", lambda s: None)
    assert rate > 0.15, f"expected a clearly inflated rate, got {rate}"


def test_whole_block_restores_the_nominal_rate():
    """Same data, same statistic, blocks exchanged as units: the rate comes back to nominal."""
    rate = _type_I("whole", lambda s: s)
    assert rate <= 0.12, f"expected a near-nominal rate, got {rate}"


def test_within_block_cannot_test_a_block_constant_term():
    """A term constant within block does not move under within-block permutation, so the permuted
    statistic equals the observed one and the test is empty. This is why the whole-block regime
    exists, and why a session-level term must otherwise be tested on an aggregated frame."""
    rate = _type_I("within", lambda s: s)
    assert rate == 0.0, f"expected an empty test, got {rate}"
