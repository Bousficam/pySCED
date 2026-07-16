"""Didactic figure (2x2): the Bayesian PRINCIPLE (sampling/diagnostics have their own
figures - mcmc.py, convergence_diagnostics.py). (1) posterior ~ prior x likelihood.
(2) sequential updating: the posterior narrows as data accumulate. (3) HDI / ROPE / pd
decision. (4) the role of the prior: sceptical vs weak vs flat give different posteriors
for the same data. English labels, ASCII only."""
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE, ORANGE, GREY, POS, RED = "#4C72B0", "#DD8452", "#777777", "#2e7d32", "#C44E52"


def post_normal(m0, v0, mL, vL):
    prec = 1 / v0 + 1 / vL
    return (m0 / v0 + mL / vL) / prec, (1 / prec)**0.5


fig, ax = plt.subplots(2, 2, figsize=(14, 9))
fig.suptitle("The Bayesian principle", fontsize=14, fontweight="bold")

# ---- 1. posterior ~ prior x likelihood ----
a = ax[0, 0]
th = np.linspace(-3, 5, 500)
mp, sp = post_normal(0.0, 1.6**2, 2.2, 0.7**2)
a.plot(th, stats.norm.pdf(th, 0, 1.6), color=GREY, lw=2, label="prior")
a.plot(th, stats.norm.pdf(th, 2.2, 0.7), color=ORANGE, lw=2, label="likelihood (data)")
a.plot(th, stats.norm.pdf(th, mp, sp), color=BLUE, lw=2.5, label="posterior")
a.fill_between(th, stats.norm.pdf(th, mp, sp), color=BLUE, alpha=0.15)
a.set_title("1. posterior ~ prior x likelihood", fontweight="bold", fontsize=10.5)
a.set_xlabel("effect"); a.set_ylabel("density"); a.legend(fontsize=8, loc="upper right")
a.text(0.02, 0.97, "the data UPDATE the prior belief", transform=a.transAxes, va="top", fontsize=8, color=GREY)

# ---- 2. sequential updating ----
a = ax[0, 1]
th = np.linspace(-1, 4, 500); truth = 1.5
for n, c, alpha in [(1, "#c9d6e6", 1), (5, "#7fa8d0", 1), (30, BLUE, 1)]:
    mn, sn = post_normal(0.0, 3.0**2, truth, 1.0 / n)          # n obs of var 1, mean ~ truth
    a.plot(th, stats.norm.pdf(th, mn, sn), color=c, lw=2.2, label=f"after n = {n}")
a.axvline(truth, color=RED, ls="--", lw=1, label="true effect")
a.set_title("2. Sequential updating: the posterior NARROWS", fontweight="bold", fontsize=10)
a.set_xlabel("effect"); a.set_ylabel("posterior density"); a.legend(fontsize=8, loc="upper left")
a.text(0.98, 0.97, "more data -> less uncertainty ;\nthe posterior converges to the true effect",
       transform=a.transAxes, ha="right", va="top", fontsize=8, color=GREY)

# ---- 3. HDI / ROPE / pd ----
a = ax[1, 0]
m, s = 1.2, 0.6
xx = np.linspace(-2, 4, 500); pdf = stats.norm.pdf(xx, m, s)
a.plot(xx, pdf, color=BLUE, lw=2)
lo, hi = m - 1.96 * s, m + 1.96 * s
mask = (xx >= lo) & (xx <= hi)
a.fill_between(xx[mask], pdf[mask], color=BLUE, alpha=0.25, label=f"HDI 95% [{lo:.2f}, {hi:.2f}]")
rope = 0.3
a.axvspan(-rope, rope, color=GREY, alpha=0.3, label=f"ROPE [-{rope}, {rope}]")
a.axvline(0, color="black", lw=1)
pd_val = 1 - stats.norm.cdf(0, m, s)
a.fill_between(xx[xx > 0], stats.norm.pdf(xx[xx > 0], m, s), color=POS, alpha=0.10)
a.set_ylim(0, 0.80)
a.text(2.35, 0.36, f"pd = P(effect>0)\n= {pd_val:.2f}", ha="left", color=POS, fontsize=8.5, fontweight="bold")
a.set_title("3. Decision: HDI vs ROPE, and pd", fontweight="bold", fontsize=10.5)
a.set_xlabel("effect"); a.set_ylabel("posterior density"); a.legend(fontsize=8, loc="upper left")
a.text(0.98, 0.54, "HDI = 95% credible interval\nROPE = zone of negligible effect\nHDI OUTSIDE ROPE -> credible effect\npd -> strength of the direction",
       transform=a.transAxes, ha="right", va="top", fontsize=7.8, color=GREY,
       bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.85))

# ---- 4. role of the prior ----
a = ax[1, 1]
th = np.linspace(-2, 4, 500); mL, vL = 1.8, 0.6**2
a.plot(th, stats.norm.pdf(th, mL, vL**0.5), color=ORANGE, lw=1.8, ls="--", label="likelihood (data)")
for v0, c, lab in [(0.4**2, RED, "SCEPTICAL prior (narrow on 0)"), (1.2**2, BLUE, "weak prior"), (8.0**2, POS, "FLAT prior")]:
    mpp, spp = post_normal(0.0, v0, mL, vL)
    a.plot(th, stats.norm.pdf(th, mpp, spp), color=c, lw=2, label=f"{lab} -> post. {mpp:.2f}")
a.set_title("4. The role of the PRIOR (same data)", fontweight="bold", fontsize=10.5)
a.set_xlabel("effect"); a.set_ylabel("density"); a.legend(fontsize=7.8, loc="upper right")
a.text(0.02, 0.97, "a sceptical prior PULLS the posterior toward 0\n(especially with little data) ;\na flat prior lets it follow the data",
       transform=a.transAxes, va="top", fontsize=8, color=GREY)

fig.text(0.5, 0.005, "Kruschke 2018 (HDI/ROPE)  ·  Makowski 2019 (pd)  ·  Gelman et al. 2013 (workflow)  ·  MCMC & diagnostics: dedicated figures",
         ha="center", fontsize=7, color="#9a9a9a", style="italic")
fig.tight_layout(rect=[0, 0.02, 1, 0.95])
out = "/Users/camile.bousfiha/PycharmProjects/pysced/docs/figures/bayesian_principle.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
