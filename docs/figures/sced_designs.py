"""Didactic figure: the main SINGLE-CASE (SCED) design families - phase/reversal
(AB, ABAB), alternating-treatments (ATD, incl. block randomization and N-of-1
crossover), multiple-baseline (MBD), and changing-criterion (CCD). Each panel shows a
mini schematic + the core feature, the effect condition it requires, what is randomized,
and when to use it. English labels, ASCII only."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BLUE, ORANGE, GREEN, RED, GREY = "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#777777"
INFO, WARN, POS = "#1f4e79", "#b5651d", "#2e7d32"
SUBJ = [BLUE, GREEN, RED]


def frame(ax, bg="#fbfbfb"):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.add_patch(FancyBboxPatch((0.01, 0.02), 0.98, 0.96, boxstyle="round,pad=0.01,rounding_size=0.03",
                                fc=bg, ec="#dcdcdc", lw=1, transform=ax.transAxes, zorder=0))


def block(ax, title, items, top=0.44, bot=0.10):
    ax.text(0.5, 0.955, title, fontsize=11, fontweight="bold", va="top", ha="center")
    n = len(items)
    ys = [(top + bot) / 2] if n == 1 else [top - i * (top - bot) / (n - 1) for i in range(n)]
    for (txt, col), y in zip(items, ys):
        ax.text(0.5, y, txt, fontsize=8.5, color=col, va="top", ha="center")


def ref(ax, t):
    ax.text(0.975, 0.045, t, ha="right", va="bottom", fontsize=6.5, color="#9a9a9a", style="italic")


def sch_line(ax, x0=0.06, x1=0.62, y=0.75):
    ax.plot([x0, x1], [y, y], color=GREY, lw=0.8, alpha=0.3, zorder=1)


fig, ax = plt.subplots(2, 2, figsize=(14.5, 9))
fig.suptitle("The main single-case (SCED) design families\n"
             "same reading scheme : feature, required effect, what is randomized, when to use it",
             fontsize=13, fontweight="bold")

# ---- 1. Phase / reversal (AB, ABAB) ----
a = ax[0, 0]; frame(a)
xs = np.linspace(0.06, 0.64, 20)
phase = np.array([0]*5 + [1]*5 + [0]*5 + [1]*5)       # A B A B
yv = 0.66 + 0.14 * phase + np.array([0.01, -0.01]*10) * 0.5
for i in range(20):
    a.scatter(xs[i], yv[i], color=(BLUE if phase[i] == 0 else ORANGE), s=22, zorder=3)
a.plot(xs, yv, color=GREY, lw=0.8, alpha=0.4)
for xb in [xs[5], xs[10], xs[15]]:
    a.axvline(xb, ymin=0.55, ymax=0.82, color=GREY, ls="--", lw=0.7)
for lab, xc in [("A", xs[2]), ("B", xs[7]), ("A", xs[12]), ("B", xs[17])]:
    a.text(xc, 0.60, lab, ha="center", fontsize=8, fontweight="bold",
           color=(BLUE if lab == "A" else ORANGE))
block(a, "1. Phase / reversal  (AB, ABAB)", [
    ("the treatment is INTRODUCED then WITHDRAWN (A-B-A-B)", "black"),
    ("requires a REVERSIBLE effect (behavior returns in A)", WARN),
    ("randomization : the phase CHANGE POINT(S)", INFO),
    ("when : reversible behavior, 1 subject ; >=5 obs/phase (WWC)", GREY)])
ref(a, "Onghena 1992 ; Kratochwill 2012")

# ---- 2. ATD / alternating (+ N-of-1) ----
a = ax[0, 1]; frame(a); sch_line(a, y=0.76)
seq = [0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0]
for k, lab in enumerate(seq):
    x = 0.08 + k * 0.048
    y = 0.70 + 0.14 * lab
    a.scatter(x, y, color=(BLUE if lab == 0 else ORANGE), s=40, edgecolor="black", lw=0.4, zorder=3)
a.text(0.66, 0.83, "B", fontsize=8, color=ORANGE, fontweight="bold")
a.text(0.66, 0.70, "A", fontsize=8, color=BLUE, fontweight="bold")
block(a, "2. Alternating (ATD)  +  N-of-1", [
    ("2+ conditions ALTERNATED rapidly in 1 subject", "black"),
    ("requires an ON/OFF effect : no carryover / remanence", WARN),
    ("randomization : A/B LABELS (blocks / alternation)", INFO),
    ("N-of-1 = randomized multi-cycle crossover (n=1 trial)", GREY)])
ref(a, "Onghena 1992 ; Kravitz 2014")

# ---- 3. Multiple-baseline (MBD) ----
a = ax[1, 0]; frame(a)
starts = [0.22, 0.34, 0.46]
step = 0.045
for i in range(3):
    y = 0.82 - i * 0.075
    s = starts[i]
    a.plot([0.06, s], [y, y], color=SUBJ[i], lw=1.3)                       # baseline
    a.plot([s, s], [y, y + step], color=SUBJ[i], lw=1.3)                   # riser at onset
    a.plot([s, 0.64], [y + step, y + step], color=SUBJ[i], lw=1.3)        # treatment
    a.axvline(s, ymin=0.58, ymax=0.95, color=SUBJ[i], ls="--", lw=0.7)
    a.scatter(s, y + step, s=90, facecolor="none", edgecolor=SUBJ[i], lw=1.6, zorder=5)  # ring on the elbow
block(a, "3. Multiple-baseline  (MBD)", [
    ("STAGGERED introduction of the intervention over >=3 series", "black"),
    ("for a PERSISTENT effect (NON reversible ; acquisition)", WARN),
    ("randomization : the START POINTS (MB / WW / KL / ...)", INFO),
    ("control via CONCURRENT baselines ; >=5 obs/phase", GREY)])
ref(a, "Kratochwill 2012 ; Levin 2016")

# ---- 4. Changing-criterion (CCD) ----
a = ax[1, 1]; frame(a)
steps_x = [0.06, 0.21, 0.36, 0.51, 0.64]
crit = [0.70, 0.83, 0.74, 0.87]                            # BIDIRECTIONAL (up, down, up) = randomized
for j in range(4):
    a.plot([steps_x[j], steps_x[j+1]], [crit[j], crit[j]], color=GREY, ls="--", lw=1)   # criterion
    if j:                                                   # vertical connector between criterion steps
        a.plot([steps_x[j], steps_x[j]], [crit[j-1], crit[j]], color=GREY, ls="--", lw=0.7)
    xx = np.linspace(steps_x[j] + 0.01, steps_x[j+1] - 0.01, 4)
    a.scatter(xx, crit[j] + np.array([-0.012, 0.006, -0.006, 0.01]), color=GREEN, s=22, zorder=3)  # behavior tracks
a.text(0.67, crit[-1], "criterion", fontsize=7.5, color=GREY, va="center")
block(a, "4. Changing-criterion  (CCD)", [
    ("the CRITERION changes in steps ; the behavior FOLLOWS it", "black"),
    ("requires a GRADABLE effect (dose-response, shaping)", WARN),
    ("randomization : the CRITERION SEQUENCE (values/order, poss. bidirectional)", INFO),
    ("inference : does behavior follow the RANDOMLY DRAWN criterion ? (association)", GREY)])
ref(a, "Manolov 2024 ; Kratochwill 2012")

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "/Users/camile.bousfiha/PycharmProjects/pysced/docs/figures/sced_designs.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
