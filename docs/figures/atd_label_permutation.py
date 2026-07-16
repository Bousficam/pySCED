"""Didactic figure: classic LABEL permutation randomization test in an ATD
(alternating treatments design). Contrast with Freedman-Lane: here the DATA stay
fixed and the A/B LABELS are reshuffled, within the design's admissible set.
Randomized-block ATD: 6 blocks of 2 sessions, one A + one B per block (randomized).
Reference set = 2^6 = 64 admissible label schedules -> exact null. English, ASCII."""
import numpy as np
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

rng = np.random.default_rng(3)
BLUE, ORANGE, GREY = "#4C72B0", "#DD8452", "#888888"

# --- randomized-block ATD: 6 blocks x 2 sessions ---
n_block = 6
sess = np.arange(1, 2 * n_block + 1)
# observed labels: within each block, a random order of (A, B)
obs_second_is_B = rng.integers(0, 2, n_block).astype(bool)   # True: block = A,B ; False: B,A
labels = np.empty(2 * n_block, dtype="<U1")
for k in range(n_block):
    labels[2 * k], labels[2 * k + 1] = ("A", "B") if obs_second_is_B[k] else ("B", "A")

# --- data (FIXED): baseline level + real B effect + noise ; NO time trend here (ATD) ---
effect = 2.6
y = 10 + effect * (labels == "B") + rng.normal(0, 1.2, 2 * n_block)

def stat(lab):
    return y[lab == "B"].mean() - y[lab == "A"].mean()

obs = stat(labels)

# --- exact reference set: each block independently A-B or B-A -> 2^6 = 64 schedules ---
schedules = []
for combo in itertools.product([True, False], repeat=n_block):     # True: A,B ; False: B,A
    lab = np.empty(2 * n_block, dtype="<U1")
    for k, ab in enumerate(combo):
        lab[2 * k], lab[2 * k + 1] = ("A", "B") if ab else ("B", "A")
    schedules.append(lab)
null = np.array([stat(l) for l in schedules])
p_val = np.mean(np.abs(null) >= abs(obs) - 1e-9)

# one alternative admissible relabeling (flip half the blocks)
alt = labels.copy()
flip = [0, 2, 4]
for k in flip:
    alt[2 * k], alt[2 * k + 1] = alt[2 * k + 1], alt[2 * k]

# ---------------------------------------------------------------- plotting
POS, RED = "#2e7d32", "#C44E52"
fig = plt.figure(figsize=(13, 11))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.82], hspace=0.42, wspace=0.2)
ax = np.array([[fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])],
               [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]], dtype=object)
foot = fig.add_subplot(gs[2, :])
fig.suptitle("Randomization test by permutation of the LABELS (block ATD):\n"
             "the data stay fixed, we reshuffle the admissible A/B labels",
             fontsize=12.5, fontweight="bold")

def draw_series(a, lab, title, showstat):
    for k in range(n_block):
        a.add_patch(Rectangle((2 * k + 0.5, a.get_ylim()[0]), 2, 100,
                              color=(GREY if k % 2 else "#dddddd"), alpha=0.15, zorder=0))
    for i in range(2 * n_block):
        col = ORANGE if lab[i] == "B" else BLUE
        a.scatter(sess[i], y[i], c=col, s=70, zorder=3, edgecolor="white")
        a.annotate(lab[i], (sess[i], y[i]), textcoords="offset points", xytext=(0, 8),
                   ha="center", fontsize=8, fontweight="bold", color=col)
    a.set_title(title + (f"\nstat B-A = {showstat:+.2f}"), fontweight="bold", fontsize=10)
    a.set_xlabel("Session (pairs = blocks)"); a.set_ylabel("Score")
    a.set_xticks(sess)

# Panel 1: observed
ax[0, 0].set_ylim(y.min() - 2, y.max() + 2)
draw_series(ax[0, 0], labels, "1. Observed data: the A/B labels carry the condition", obs)

# Panel 2: one alternative admissible relabeling
ax[0, 1].set_ylim(y.min() - 2, y.max() + 2)
draw_series(ax[0, 1], alt, "2. One admissible reshuffle: same points, blocks "
                          + ",".join(str(k + 1) for k in flip) + " swapped", stat(alt))

# Panel 3: scheme schematic (each block: 2 choices -> 2^6 = 64)
s = ax[1, 0]; s.axis("off")
s.set_title("3. Block randomization scheme\neach block: A-B or B-A (2 choices) "
            f"-> reference set = 2^{n_block} = {2**n_block} schedules",
            fontweight="bold", fontsize=10)
for k in range(n_block):
    yb = 0.85 - k * 0.14
    s.text(0.02, yb, f"Block {k+1}", fontsize=9, va="center")
    for j, (l1, l2, col) in enumerate([("A", "B", BLUE), ("B", "A", ORANGE)]):
        x0 = 0.30 + j * 0.30
        s.add_patch(Rectangle((x0, yb - 0.05), 0.10, 0.10, transform=s.transAxes,
                              color=(BLUE if l1 == "A" else ORANGE), alpha=0.5))
        s.add_patch(Rectangle((x0 + 0.10, yb - 0.05), 0.10, 0.10, transform=s.transAxes,
                              color=(BLUE if l2 == "A" else ORANGE), alpha=0.5))
        s.text(x0 + 0.10, yb, f"{l1} {l2}", ha="center", va="center", fontsize=8, transform=s.transAxes)
s.text(0.30, 0.95, "choice 1", fontsize=8, ha="center"); s.text(0.60, 0.95, "choice 2", fontsize=8, ha="center")

# Panel 4: exact null + observed + p
d = ax[1, 1]
d.hist(null, bins=25, color=GREY, alpha=0.6, edgecolor="white")
d.axvline(obs, color="crimson", lw=2.5, label=f"observed = {obs:+.2f}")
d.axvline(-obs, color="crimson", lw=1, ls=":")
d.set_title(f"4. EXACT null distribution ({2**n_block} schedules)\n"
            f"p (two-sided) = {p_val:.4f}", fontweight="bold", fontsize=10)
d.set_xlabel("stat B-A over all admissible schedules"); d.set_ylabel("Nb of schedules")
d.legend(fontsize=8)

for _ax, _t in [(ax[0, 0], "Edgington 2007"), (ax[0, 1], "Onghena 1992"),
                (ax[1, 0], "Edgington 2007"), (ax[1, 1], "Levin 2021")]:
    _ax.text(0.985, 0.02, _t, transform=_ax.transAxes, ha="right", va="bottom",
             fontsize=6.5, color="#9a9a9a", style="italic")

# ---- footer : name in the taxonomy + link to Draper-Stoneman / Freedman-Lane ----
foot.axis("off"); foot.set_xlim(0, 1); foot.set_ylim(0, 1)
foot.add_patch(Rectangle((0.005, 0.02), 0.99, 0.96, transform=foot.transAxes,
               fc="#f6f7f9", ec="#c9ced6", lw=1))
foot.text(0.5, 0.93, "What it is called - and the link to Draper-Stoneman",
          ha="center", va="top", fontsize=11, fontweight="bold")
# left column : the two design-based names
L = [
    ("ATD rando = DESIGN-BASED (exact, drawn from the plan):", "black"),
    ("- CRD: free labels, $n_A/n_B$ fixed -> C($N,n_A$)", BLUE),
    ("- RBD: blocks (pairs), A-B order drawn per block -> $2^k$", BLUE),
    ("     (this figure = RBD)", GREY),
    ("  nuisance (time) handled by the DESIGN (blocks)", GREY),
]
y = 0.80
for txt, c in L:
    foot.text(0.03, y, txt, fontsize=8.8, color=c, va="top"); y -= 0.135
# right column : the model-based equivalence
Rr = [
    ("Written as regression  $y=b_0+b_1\\cdot$label ($+Z$):", "black"),
    ("- permuting the label column = DRAPER-STONEMAN", ORANGE),
    ("- WITHOUT time effect: DS $\\equiv$ this label test", POS),
    ("- WITH time effect: DS does not residualize Z", RED),
    ("     -> bias if label~time -> FREEDMAN-LANE", RED),
]
y = 0.80
for txt, c in Rr:
    foot.text(0.52, y, txt, fontsize=8.8, color=c, va="top"); y -= 0.135
foot.text(0.5, 0.05, "Rule: no time effect -> label rando / Draper-Stoneman   |   time effect -> Freedman-Lane",
          ha="center", va="bottom", fontsize=8.8, fontweight="bold", color="#333333")

out = "/Users/camile.bousfiha/PycharmProjects/pysced/docs/figures/atd_label_permutation.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out, "| obs", round(obs, 3), "p", round(p_val, 4))
