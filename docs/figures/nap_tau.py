"""Didactic figure: NAP and Tau (Tau-U) as NONOVERLAP effect sizes, with an explicit
all-pairs arrow diagram so the overlap counting is visible. Uses the toolbox functions.
English labels, ASCII only. (The trend correction has its own figure: tau_correction.py)"""
import sys
sys.path.insert(0, "/Users/camile.bousfiha/PycharmProjects/Stats")
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functions.sced.core import nap, tau_u

BLUE, ORANGE, GREY = "#4C72B0", "#DD8452", "#888888"
GREEN, RED = "#4C9F70", "#C44E52"

def ref(a, t, loc="br"):
    """Discreet source note (1st author, year) in a panel corner."""
    x, ha = (0.985, "right") if loc == "br" else (0.015, "left")
    a.text(x, 0.02, t, transform=a.transAxes, ha=ha, va="bottom",
           fontsize=6.5, color="#9a9a9a", style="italic")

A = np.array([8, 11, 9, 12, 10]); B = np.array([13, 10, 14, 12, 15])
nap1 = nap(A, B)
tau1 = tau_u(A, B, correct_baseline_trend=False)["tau"]
M = np.sign(B[:, None] - A[None, :])                       # rows B, cols A: +1 B>A, -1 B<A, 0 tie
n_up = int(np.sum(M > 0)); n_tie = int(np.sum(M == 0)); n_dn = int(np.sum(M < 0)); nAB = A.size * B.size

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
fig.suptitle("NAP and Tau (Tau-U): NONOVERLAP effect sizes (all A-B pairs)",
             fontsize=13, fontweight="bold")
sA = np.arange(1, A.size + 1); sB = np.arange(A.size + 1, A.size + B.size + 1)

# Panel 1: time series
a = ax[0, 0]
a.scatter(sA, A, c=BLUE, s=75, label=f"Baseline A (n={A.size})", zorder=3)
a.scatter(sB, B, c=ORANGE, s=75, label=f"Treatment B (n={B.size})", zorder=3)
a.axvline(A.size + 0.5, color=GREY, ls="--")
a.set_title("1. Observed series: A vs B", fontweight="bold", fontsize=10)
a.set_xlabel("Session"); a.set_ylabel("Score"); a.legend(fontsize=8)
ref(a, "Parker 2009", "br")

# Panel 2: all-pairs ARROW diagram (dominance). A on left column, B on right column.
# One arrow A->B per pair: green if B>A (improvement), red if B<A (overlap), grey tie.
d = ax[0, 1]
xA, xB = 0.0, 1.0
for j, av in enumerate(A):
    d.scatter(xA, av, c=BLUE, s=70, zorder=4)
for i, bv in enumerate(B):
    d.scatter(xB, bv, c=ORANGE, s=70, zorder=4)
for i, bv in enumerate(B):
    for j, av in enumerate(A):
        col = GREEN if bv > av else (RED if bv < av else GREY)
        d.annotate("", xy=(xB - 0.03, bv), xytext=(xA + 0.03, av),
                   arrowprops=dict(arrowstyle="->", color=col, lw=1.3, alpha=0.55))
d.set_xlim(-0.35, 1.35); d.set_xticks([xA, xB]); d.set_xticklabels(["A points", "B points"])
d.set_ylabel("Score")
d.set_title(f"2. All A-B pairs ({nAB} arrows)\n"
            f"green (B>A) = {n_up}  |  red (B<A, overlap) = {n_dn}  |  grey (tie) = {n_tie}",
            fontweight="bold", fontsize=9.5)
d.text(0.5, 0.055, "each arrow compares 1 A point to 1 B point ; overlap = the red arrows",
       transform=d.transAxes, ha="center", fontsize=7.5, color=GREY)
ref(d, "Parker 2009", "br")

# Panel 3: NAP tally matrix
m = ax[1, 0]
cmap = {1: GREEN, -1: RED, 0: GREY}
for i in range(B.size):
    for j in range(A.size):
        m.add_patch(plt.Rectangle((j, B.size - 1 - i), 1, 1, color=cmap[M[i, j]], alpha=0.8))
        m.text(j + 0.5, B.size - 1 - i + 0.5, "+" if M[i, j] > 0 else ("-" if M[i, j] < 0 else "="),
               ha="center", va="center", color="white", fontweight="bold")
m.set_xlim(0, A.size); m.set_ylim(0, B.size)
m.set_xticks(np.arange(A.size) + 0.5); m.set_xticklabels(A)
m.set_yticks(np.arange(B.size) + 0.5); m.set_yticklabels(B[::-1])
m.set_xlabel("A point"); m.set_ylabel("B point")
m.set_title(f"3. NAP tally (same pairs, as a table)\n"
            f"NAP = (#green + 0.5*#grey)/total = ({n_up} + 0.5*{n_tie})/{nAB} = {nap1:.2f}",
            fontweight="bold", fontsize=9.5)
m.text(1.0, -0.16, "Parker 2009", transform=m.transAxes, ha="right", va="top",
       fontsize=6.5, color="#9a9a9a", style="italic")

# Panel 4: NAP <-> Tau scales
g = ax[1, 1]; g.axis("off"); g.set_xlim(0, 1); g.set_ylim(0, 1)
g.set_title("4. From NAP to Tau: same information, two scales\nTau (A vs B) = 2 x NAP - 1",
            fontweight="bold", fontsize=10)
g.annotate("", xy=(0.95, 0.60), xytext=(0.05, 0.60), arrowprops=dict(arrowstyle="-", lw=1.5))
for v, lab in [(-1, "-1"), (0, "0"), (1, "+1")]:
    x = 0.05 + (v + 1) / 2 * 0.9
    g.plot([x, x], [0.58, 0.62], color="black"); g.text(x, 0.51, lab, ha="center", fontsize=9)
xt = 0.05 + (tau1 + 1) / 2 * 0.9
g.scatter([xt], [0.60], color=ORANGE, s=120, zorder=3)
g.text(xt, 0.67, f"Tau = {tau1:+.2f}", ha="center", color=ORANGE, fontweight="bold")
g.text(0.5, 0.42, "Tau: -1 = all worse, 0 = chance, +1 = total separation", ha="center", fontsize=8)
g.annotate("", xy=(0.95, 0.24), xytext=(0.05, 0.24), arrowprops=dict(arrowstyle="-", lw=1.5))
for v, lab in [(0.0, "0"), (0.5, "0.5\n(chance)"), (1.0, "1")]:
    x = 0.05 + v * 0.9
    g.plot([x, x], [0.22, 0.26], color="black"); g.text(x, 0.14, lab, ha="center", fontsize=8)
xn = 0.05 + nap1 * 0.9
g.scatter([xn], [0.24], color=BLUE, s=120, zorder=3)
g.text(xn, 0.31, f"NAP = {nap1:.2f}", ha="center", color=BLUE, fontweight="bold")
ref(g, "Parker 2011", "br")

fig.tight_layout(rect=[0, 0, 1, 0.95])
out = "/Users/camile.bousfiha/PycharmProjects/pysced/docs/figures/nap_tau.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out, f"| NAP={nap1:.3f} Tau={tau1:+.3f} up={n_up} dn={n_dn} tie={n_tie}")
