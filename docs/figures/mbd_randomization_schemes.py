"""Didactic figure: randomization test in a MULTIPLE-BASELINE design (MBD) and the
different randomization SCHEMES for the intervention start point. 3 tiers with a
staggered onset. Schemes (per functions/sced/mbd): MB (Marascuilo-Busk 1988,
independent per-tier windows, prod k_i), WW (Wampold-Worsham 1986, permute a fixed
pool of N onsets across tiers, N!), MB-R (distinct/without replacement), KL / Revusky
(case x start-point). English labels, ASCII only."""
import numpy as np
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

rng = np.random.default_rng(11)
BLUE, ORANGE, GREY = "#4C72B0", "#DD8452", "#888888"
TIER_C = ["#4C72B0", "#55A868", "#C44E52"]

T = 18
t = np.arange(1, T + 1)
windows = [[4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]   # admissible starts per tier (staggered)
obs_starts = [5, 9, 13]                                       # observed onsets
jump = 4.0                                                    # true immediate level change
offset = 14                                                   # vertical spacing between tiers

Y = []
for i, s in enumerate(obs_starts):
    base = 10 + i * offset + rng.normal(0, 0.9, T)
    base[t >= s] += jump
    Y.append(base)
Y = np.array(Y)

def level_stat(starts):
    """Mean across tiers of (mean after start - mean before start)."""
    vals = []
    for i, s in enumerate(starts):
        before = Y[i][t < s]; after = Y[i][t >= s]
        if len(before) and len(after):
            vals.append(after.mean() - before.mean())
    return float(np.mean(vals))

obs = level_stat(obs_starts)

# MB (independent): every combination of one start per tier from its window (with repetition)
mb_combos = list(itertools.product(*windows))                # 4*4*4 = 64
null_mb = np.array([level_stat(c) for c in mb_combos])
p_mb = np.mean(null_mb >= obs - 1e-9)

# WW: fixed pool of onsets assigned (permuted) to tiers
pool = [5, 9, 13]
ww_perms = list(itertools.permutations(pool))                # 3! = 6
ww_alt = (13, 5, 9)                                          # one permuted assignment (not the observed)

# ---------------------------------------------------------------- plotting
fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
fig.suptitle("Randomization test in MULTIPLE-BASELINE (MBD): the intervention start point is drawn at random\n"
             "the stat = mean level change at onset ; the p comes from the set of admissible start points",
             fontsize=11.5, fontweight="bold")

# Panel 1: observed MBD, staggered onsets
a = ax[0, 0]
for i in range(3):
    s = obs_starts[i]
    a.plot(t, Y[i], "-o", color=TIER_C[i], ms=4, lw=1.2)
    a.axvspan(0.5, s - 0.5, ymin=0, ymax=1, alpha=0)          # keep axis
    a.axvline(s, color=TIER_C[i], ls="--", lw=1)
    a.scatter(s, Y[i][s - 1], s=300, facecolor="none", edgecolor=TIER_C[i], lw=3, zorder=6)  # onset ring
    a.annotate("start", (s, Y[i][s - 1]), textcoords="offset points", xytext=(6, 10),
               fontsize=7.5, color=TIER_C[i], fontweight="bold")
    a.text(1, Y[i].max() + 1, f"Subject {i+1} (onset session {s})", color=TIER_C[i], fontsize=8, fontweight="bold")
    a.axvspan(s - 0.5, T + 0.5, color=TIER_C[i], alpha=0.06)
a.set_title(f"1. Observed MBD: STAGGERED intervention\nobserved stat = {obs:+.2f}", fontweight="bold", fontsize=10)
a.set_xlabel("Session"); a.set_ylabel("Score (subjects offset vertically)")
a.xaxis.set_major_locator(MaxNLocator(integer=True))     # sessions are integers
a.yaxis.set_major_locator(MaxNLocator(integer=True))

# Panel 2: MB (independent windows)
b = ax[0, 1]
for i in range(3):
    yb = 3 - i
    b.plot([windows[i][0], windows[i][-1]], [yb, yb], color=TIER_C[i], lw=6, alpha=0.25,
           solid_capstyle="round")
    for st in windows[i]:
        b.scatter(st, yb, color=TIER_C[i], s=45, zorder=3)
    b.scatter(obs_starts[i], yb, facecolor="none", edgecolor="crimson", s=170, lw=2, zorder=4)
    b.text(2.5, yb, f"Subject {i+1}", color=TIER_C[i], fontsize=8, ha="right", va="center", fontweight="bold")
b.set_title("2. MB scheme (Marascuilo-Busk 1988): INDEPENDENT windows\n"
            f"each subject draws a start in its window -> prod k_i = {len(mb_combos)} combinations",
            fontweight="bold", fontsize=9.5)
b.set_xlabel("Candidate start session"); b.set_yticks([]); b.set_xlim(2, T)
b.text(0.98, 0.02, "red circle = observed onset", transform=b.transAxes, ha="right",
       fontsize=7, color="crimson")

# Panel 3: WW = the SAME fixed pool of onsets {5,9,13} reassigned to tiers, N! ways.
# Grid: rows = tiers, columns = the 6 assignments; each cell = the onset that tier gets.
c = ax[1, 0]
onset_col = {5: "#a6cee3", 9: "#b2df8a", 13: "#fb9a99"}        # one color per onset VALUE
for col, perm in enumerate(ww_perms):                          # ww_perms = permutations of (5,9,13)
    for row in range(3):
        val = perm[row]
        c.add_patch(Rectangle((col, 2 - row), 1, 1, facecolor=onset_col[val], edgecolor="white"))
        c.text(col + 0.5, 2 - row + 0.5, str(val), ha="center", va="center", fontsize=11, fontweight="bold")
obs_col = ww_perms.index(tuple(pool))                          # observed assignment (5,9,13)
c.add_patch(Rectangle((obs_col, 0), 1, 3, facecolor="none", edgecolor="crimson", lw=2.8, zorder=5))
c.text(obs_col + 0.5, 3.08, "observed", ha="center", color="crimson", fontsize=8, fontweight="bold")
c.set_xlim(0, len(ww_perms)); c.set_ylim(0, 3.2)
c.set_xticks(np.arange(len(ww_perms)) + 0.5); c.set_xticklabels([f"#{k+1}" for k in range(len(ww_perms))])
c.set_yticks(np.arange(3) + 0.5); c.set_yticklabels(["Subject 3", "Subject 2", "Subject 1"])
c.set_xlabel(f"Assignment ({len(ww_perms)} permutations of the pool)")
c.set_title("3. WW scheme (Wampold-Worsham 1986): the FIXED pool of onsets {5,9,13}\n"
            f"is REASSIGNED to subjects in every way -> N! = {len(ww_perms)} ; color = onset value",
            fontweight="bold", fontsize=9.5)

# Panel 4: null distribution (MB) + observed + p
d = ax[1, 1]
d.hist(null_mb, bins=20, color=GREY, alpha=0.6, edgecolor="white")
d.axvline(obs, color="crimson", lw=2.5, label=f"observed = {obs:+.2f}")
d.set_title(f"4. EXACT null distribution - MB scheme ({len(mb_combos)} combinations)\n"
            f"p (one-sided, improvement) = {p_mb:.4f}", fontweight="bold", fontsize=9.5)
d.set_xlabel("level change stat over all admissible starts")
d.set_ylabel("Number of combinations"); d.legend(fontsize=8)
d.text(0.98, 0.80, "Other schemes:\nMB-R = without replacement (distinct starts)\n"
       "KL / Revusky = case x start (N! x prod k_i)", transform=d.transAxes,
       ha="right", va="top", fontsize=7, color=GREY,
       bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.7))

for _ax, _t, _lc in [(ax[0, 0], "Levin 2016", "br"), (ax[0, 1], "Marascuilo 1988", "bl"),
                     (ax[1, 0], "Wampold 1986", "br"), (ax[1, 1], "Levin 2016", "bl")]:
    _x, _ha = (0.985, "right") if _lc == "br" else (0.015, "left")
    _ax.text(_x, 0.02, _t, transform=_ax.transAxes, ha=_ha, va="bottom",
             fontsize=6.5, color="#9a9a9a", style="italic")
fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "/Users/camile.bousfiha/PycharmProjects/pysced/docs/figures/mbd_randomization_schemes.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out, "| obs", round(obs, 3), "p_MB", round(p_mb, 4))
