"""Confidence intervals by randomization test inversion (Michiels et al. 2017).

The source paper prints its worked example in full (Table 1), so the estimate, the p-value and the
size of the reference set can be checked against published numbers. Its printed INTERVAL cannot: an
exhaustive enumeration of the 12870 assignments rejects both published endpoints, so the bounds are
checked against that exact reference distribution instead (see the test for the detail). The rest
pin the properties that make an inversion interval an interval at all - duality with the test,
self-consistency at the boundaries, the design floor, and the frozen standardizing SD without which
the yardstick moves with the candidate being tested.
"""
import itertools

import numpy as np
import pytest

from functions.sced.core.permutation import (randomization_interval, randomization_test,
                                             frozen_sd_statistic, alternating_scheme)


# Table 1 of Michiels et al. (2017, p. 365): quality-of-life scores, 8 control vs 8 treated.
A = [7, 5, 7, 4, 5, 3, 4, 2]
B = [9, 8, 7, 9, 8, 6, 9, 8]
VALUES = np.array(A + B, dtype=float)
TREATED = np.array([False] * 8 + [True] * 8)


def _setup(standardize=True, n_perm=4000):
    stat = frozen_sd_statistic(VALUES, TREATED, standardize=standardize)
    return dict(labels=TREATED, values=VALUES, statistic=stat, scheme=alternating_scheme(),
                treated=TREATED, n_perm=n_perm, random_state=0)


# --------------------------------------------------------------------------- #
# 1. The published example
# --------------------------------------------------------------------------- #
def _exhaustive_p(delta, sd):
    """Exact two-sided p over ALL 12870 assignments of 8 treated among 16, for a candidate shift.

    Ground truth for the tests below: no Monte-Carlo error, so a boundary located with it is the
    boundary, not an estimate of it.
    """
    y = VALUES - delta * TREATED
    combos = np.array(list(itertools.combinations(range(16), 8)))
    M = np.zeros((len(combos), 16), bool)
    np.put_along_axis(M, combos, True, axis=1)
    s_t = (y[None, :] * M).sum(1)
    st = (s_t / 8 - (y.sum() - s_t) / 8) / sd
    obs = (y[TREATED].mean() - y[~TREATED].mean()) / sd
    return float(np.mean(np.abs(st) >= abs(obs) - 1e-12))


def test_reproduces_the_published_estimate_and_p_value():
    """Michiels et al. (2017, p. 365, 368): theta_obs = 3.375, d = 1.5050, two-sided p = .0014,
    and 12870 admissible assignments. All four reproduce from the printed Table 1 data."""
    r = randomization_interval(**_setup(), confidence=0.95, precision=1e-3)
    assert r["estimate"] == pytest.approx(3.375, abs=1e-9)          # the shift, in data units
    assert r["estimate_statistic"] == pytest.approx(1.5050, abs=5e-4)
    assert r["p_value"] == pytest.approx(0.0014, abs=0.004)
    assert r["m_t"] == 12870


def test_the_bounds_sit_where_the_EXACT_p_crosses_alpha():
    """The published interval for this example, [0.56, 2.45] in d units (p. 368), does NOT
    reproduce, and the discrepancy is not Monte-Carlo error: enumerating all 12870 assignments
    gives an exact two-sided p of 0.0118 at d = 0.56 and 0.0140 at d = 2.45, both far below .05,
    so neither endpoint can belong to a 95 per cent interval on these data. The exact p crosses
    .05 between d = 0.78 and d = 0.85 on the low side and between 2.20 and 2.30 on the high side,
    which is where this implementation puts the bounds.

    This test therefore checks the implementation against the EXACT reference distribution rather
    than against the printed interval. If a future reading of the paper explains the printed
    values (a different two-sided convention, a different search rule), this is the test to
    revisit.
    """
    cfg = _setup(n_perm=6000)
    r = randomization_interval(**cfg, confidence=0.95, precision=1e-3)
    sd = cfg["statistic"].frozen_sd
    assert _exhaustive_p(r["ci_high"] - 0.15, sd) >= 0.05      # just inside: retained
    assert _exhaustive_p(r["ci_high"] + 0.15, sd) < 0.05       # just outside: rejected
    assert _exhaustive_p(r["ci_low"] + 0.15, sd) >= 0.05
    assert _exhaustive_p(r["ci_low"] - 0.15, sd) < 0.05
    # and the published endpoints are rejected outright
    assert _exhaustive_p(3.375 + (0.56 - 1.50496) * sd, sd) < 0.05
    assert _exhaustive_p(3.375 + (2.45 - 1.50496) * sd, sd) < 0.05


def test_agrees_with_garthwaites_published_interval():
    """Independent external check, on a second published worked example.

    Garthwaite (1996, Biometrics 52:1387-1393, p. 1390) inverts the same test on lizard running
    distances, 15 malaria-infected against 15 uninfected, by a Robbins-Monro search rather than by
    the bisection used here, and reports theta_hat = 5.36 with a 95% CI of (-0.30, 10.69). Two
    independent search procedures on the same printed data must land within their Monte-Carlo
    error of each other; the point estimate must match exactly.
    """
    X = [16.4, 29.4, 37.1, 23.0, 24.1, 24.5, 16.4, 29.1, 36.7, 28.7, 30.2, 21.8, 37.1, 20.3, 28.3]
    Y = [22.2, 34.8, 42.1, 32.9, 26.4, 30.6, 32.9, 37.5, 18.4, 27.5, 45.5, 34.0, 45.5, 24.5, 28.7]
    v = np.array(X + Y, dtype=float)
    t = np.array([False] * 15 + [True] * 15)          # treated = uninfected, theta = mu_Y - mu_X
    r = randomization_interval(t, v, statistic=frozen_sd_statistic(v, t, standardize=False),
                               scheme=alternating_scheme(), treated=t, n_perm=4000,
                               random_state=0, confidence=0.95, precision=1e-3)
    assert r["estimate"] == pytest.approx(5.36, abs=0.005)
    assert r["ci_low"] == pytest.approx(-0.30, abs=0.4)
    assert r["ci_high"] == pytest.approx(10.69, abs=0.4)
    assert r["ci_low"] <= 0.0 <= r["ci_high"] and r["p_value"] > 0.05      # duality, barely


def test_unstandardized_estimate_matches_the_paper():
    """Same data, raw mean difference: theta_obs = 3.375 (p. 365)."""
    r = randomization_interval(**_setup(standardize=False), confidence=0.95)
    assert r["estimate"] == pytest.approx(3.375, abs=1e-9)
    assert r["p_value"] < 0.05 and r["ci_low"] > 0


# --------------------------------------------------------------------------- #
# 2. Duality: the interval IS the set of non-rejected values
# --------------------------------------------------------------------------- #
def test_zero_is_inside_the_interval_exactly_when_the_test_does_not_reject():
    """The defining property. Checked in both directions: a real effect (rejected, 0 outside) and
    data with no effect (not rejected, 0 inside)."""
    r = randomization_interval(**_setup(), confidence=0.95)
    assert r["p_value"] < 0.05 and not (r["ci_low"] <= 0.0 <= r["ci_high"])

    rng = np.random.default_rng(3)
    flat = rng.normal(0, 1.0, 16)                      # no group difference
    stat = frozen_sd_statistic(flat, TREATED)
    r0 = randomization_interval(TREATED, flat, statistic=stat, scheme=alternating_scheme(),
                                treated=TREATED, n_perm=2000, random_state=0, confidence=0.95)
    assert r0["p_value"] > 0.05 and r0["ci_low"] <= 0.0 <= r0["ci_high"]


def test_boundaries_are_where_the_test_flips():
    """Just inside the interval the test must retain, just outside it must reject."""
    cfg = _setup()
    r = randomization_interval(**cfg, confidence=0.95, precision=1e-3)
    stat, scheme = cfg["statistic"], cfg["scheme"]

    def p_at(delta):
        return randomization_test(TREATED, VALUES - delta * TREATED, statistic=stat,
                                  scheme=scheme, n_perm=cfg["n_perm"], random_state=0)["p_value"]

    eps = 0.05
    assert p_at(r["ci_high"] - eps) >= 0.05 and p_at(r["ci_high"] + eps) < 0.05
    assert p_at(r["ci_low"] + eps) >= 0.05 and p_at(r["ci_low"] - eps) < 0.05


def test_a_wider_confidence_level_gives_a_wider_interval():
    r95 = randomization_interval(**_setup(), confidence=0.95)
    r80 = randomization_interval(**_setup(), confidence=0.80)
    assert r80["ci_low"] > r95["ci_low"] and r80["ci_high"] < r95["ci_high"]


# --------------------------------------------------------------------------- #
# 3. The design floor
# --------------------------------------------------------------------------- #
def test_a_design_too_small_for_the_confidence_level_is_refused():
    """Four observations, two per condition: 6 assignments, so the smallest attainable p is 1/7 =
    0.143. No 95% interval exists and the function must say so rather than bracket nothing."""
    v = np.array([1.0, 2.0, 8.0, 9.0])
    t = np.array([False, False, True, True])
    with pytest.raises(ValueError, match="No 95% interval exists|smallest attainable"):
        randomization_interval(t, v, statistic=frozen_sd_statistic(v, t),
                               scheme=alternating_scheme(), treated=t, n_perm=200,
                               random_state=0, confidence=0.95)


def test_the_same_design_admits_a_lower_confidence_level():
    """The floor bounds the confidence level, it does not forbid every interval."""
    v = np.array([1.0, 2.0, 8.0, 9.0])
    t = np.array([False, False, True, True])
    r = randomization_interval(t, v, statistic=frozen_sd_statistic(v, t),
                               scheme=alternating_scheme(), treated=t, n_perm=500,
                               random_state=0, confidence=0.80)
    assert r["m_t"] == 6 and r["p_floor"] == pytest.approx(1 / 7, abs=1e-9)


# --------------------------------------------------------------------------- #
# 4. The frozen standardizing SD
# --------------------------------------------------------------------------- #
def test_the_standardizing_sd_does_not_move_with_the_candidate():
    """The SD is computed once on the observed data, so shifting the treated scores by a candidate
    leaves it untouched - Michiels et al. (2017, p. 368)."""
    stat = frozen_sd_statistic(VALUES, TREATED)
    before = stat.frozen_sd
    for delta in (0.0, 1.0, 5.0, 20.0):
        shifted = VALUES - delta * TREATED
        assert stat(TREATED, shifted) == pytest.approx(
            (shifted[TREATED].mean() - shifted[~TREATED].mean()) / before, abs=1e-12)
    assert stat.frozen_sd == before


def test_recomputing_the_sd_would_move_the_yardstick():
    """The trap the frozen SD avoids: an SD recomputed on the shifted data MOVES with the candidate,
    so successive candidates would be judged on different scales. Demonstrated rather than asserted
    in a comment, because the failure is silent.

    The direction of the move depends on whether the shift pulls the two groups together or apart
    (here it pulls them together first, so the naive SD shrinks); what invalidates the inversion is
    that it moves at all."""
    naive = [float(np.std(VALUES - d * TREATED, ddof=1)) for d in (0.0, 2.0, 6.0)]
    assert abs(naive[2] - naive[0]) > 0.1, "a recomputed SD moves with the candidate"
    assert frozen_sd_statistic(VALUES, TREATED).frozen_sd == pytest.approx(naive[0], abs=1e-12)


# --------------------------------------------------------------------------- #
# 5. Reproducibility
# --------------------------------------------------------------------------- #
def test_the_interval_is_reproducible():
    """Every candidate is tested with the same generator seed, so two runs agree exactly."""
    a = randomization_interval(**_setup(n_perm=1000), confidence=0.95)
    b = randomization_interval(**_setup(n_perm=1000), confidence=0.95)
    assert (a["ci_low"], a["ci_high"]) == (b["ci_low"], b["ci_high"])
