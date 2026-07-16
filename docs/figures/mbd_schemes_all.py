"""Didactic figure: the SIX MBD randomization schemes, one per ROW. Left = compact
subjects x session schematic; right = what is randomized, reference-set size, smallest
attainable p (1/size), statistic, and a bulleted pro/con list (green +, red -).
Reference-set sizes computed on concrete windows. English labels, ASCII only.
Schemes (functions/sced/mbd): MB, MB-R, WW (baseline.py); KL, Revusky, Rev-M
(procedures.py). Refs: Marascuilo & Busk 1988; Wampold & Worsham 1986; Koehler & Levin
1998; Revusky 1967; Levin, Ferron & Gafurov 2016."""
import numpy as np
import itertools
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BLUE, GREEN, RED, GREY = "#4C72B0", "#55A868", "#C44E52", "#888888"
POS_G, NEG_R = "#2e7d32", "#c62828"
SUBJ = [BLUE, GREEN, RED]

W = [[5, 6, 7, 8], [6, 7, 8, 9], [7, 8, 9, 10]]                 # start-point family windows (overlapping)
mb_size = int(np.prod([len(w) for w in W]))
mbr_size = sum(1 for c in itertools.product(*W) if len(set(c)) == 3)
ord_size = sum(1 for c in itertools.product(*W) if c[0] < c[1] < c[2])
POOL = [6, 8, 10]                                              # case-permutation family pool
ww_size = math.factorial(3)
SLOT = [[4, 5, 6], [7, 8, 9], [10, 11, 12]]                    # case x start family slots
kl_size = ww_size * int(np.prod([len(s) for s in SLOT]))


def pmin(n):
    return f"1/{n} = {1.0/n:.3f}"


def frame(ax, labels=("Subject 1", "Subject 2", "Subject 3")):
    ax.set_ylim(0.3, 3.7); ax.set_xlim(3.5, 12.5)
    ax.set_yticks([3, 2, 1]); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Onset session", fontsize=8); ax.tick_params(labelsize=7)


def windows_family(ax, example):
    for i in range(3):
        y = 3 - i
        ax.plot([W[i][0], W[i][-1]], [y, y], color=SUBJ[i], lw=7, alpha=0.25, solid_capstyle="round")
        for st in W[i]:
            ax.scatter(st, y, color=SUBJ[i], s=35, zorder=3)
    for i, st in enumerate(example):
        ax.scatter(st, 3 - i, color=SUBJ[i], s=130, edgecolor="black", zorder=5)
    frame(ax)


def draw_sch(ax, key):
    if key == "MB":
        windows_family(ax, [6, 6, 9]); ax.axvline(6, color="black", ls=":", lw=0.8)
    elif key == "MBR":
        windows_family(ax, [6, 8, 10])
    elif key == "WW":
        for p in POOL:
            ax.axvline(p, color=GREY, ls="--", lw=1)
        for i, p in enumerate(POOL):
            ax.scatter(p, 3 - i, color=SUBJ[i], s=130, edgecolor="black", zorder=5)
        frame(ax)
    elif key == "KL":
        for i in range(3):
            y = 3 - i
            ax.plot([SLOT[i][0], SLOT[i][-1]], [y, y], color="black", lw=7, alpha=0.12, solid_capstyle="round")
            for st in SLOT[i]:
                ax.scatter(st, y, color="black", s=20, alpha=0.4, zorder=2)
        for i, (slot_idx, st) in enumerate([(0, 5), (1, 8), (2, 11)]):
            ax.scatter(st, 3 - slot_idx, color=SUBJ[i], s=130, edgecolor="black", zorder=5)
        frame(ax, labels=("Slot 3 (late)", "Slot 2", "Slot 1 (early)"))
    elif key == "REV":
        for i in range(3):
            y = 3 - i
            ax.plot([3.7, 12.3], [y, y], color=SUBJ[i], lw=1.2, alpha=0.35, zorder=1)
            ax.scatter(POOL[i], y, color=SUBJ[i], s=130, edgecolor="black", zorder=5)
        for j in range(3):
            x = POOL[j]; peers = [3 - k for k in range(j + 1, 3)]
            ax.axvline(x, color=GREY, ls=":", lw=0.7)
            if peers:
                ax.annotate("", xy=(x, min(peers)), xytext=(x, 3 - j),
                            arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.7, alpha=0.85))
                ax.text(x + 0.18, (3 - j + min(peers)) / 2, f"step {j+1}", color="crimson",
                        fontsize=6.5, rotation=90, va="center")
        frame(ax)
    elif key == "REVM":
        chosen = [5, 8, 11]
        for i in range(3):
            y = 3 - i
            ax.plot([SLOT[i][0], SLOT[i][-1]], [y, y], color=SUBJ[i], lw=7, alpha=0.20, solid_capstyle="round")
            for st in SLOT[i]:
                ax.scatter(st, y, color=SUBJ[i], s=22, alpha=0.5, zorder=2)
            ax.plot([3.7, 12.3], [y, y], color=SUBJ[i], lw=0.8, alpha=0.25, zorder=1)
            ax.scatter(chosen[i], y, color=SUBJ[i], s=130, edgecolor="black", zorder=5)
        for j in range(3):
            x = chosen[j]; peers = [3 - k for k in range(j + 1, 3)]
            if peers:
                ax.annotate("", xy=(x, min(peers)), xytext=(x, 3 - j),
                            arrowprops=dict(arrowstyle="<->", color="crimson", lw=1.7, alpha=0.85))
        frame(ax)


def text_panel(ax, s):
    ax.axis("off"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(0.0, 0.98, s["title"], fontsize=11, fontweight="bold", va="top")
    ax.text(0.0, 0.82, f"Randomized: {s['rnd']}", fontsize=8, color=GREY, va="top")
    ax.text(0.0, 0.72, f"Size = {s['size']}    |    p_min = {s['pmin']}    |    Stat: {s['stat']}",
            fontsize=8, va="top")
    y = 0.56
    for p in s["pros"]:
        ax.text(0.0, y, f"+   {p}", fontsize=8.3, color=POS_G, va="top"); y -= 0.115
    for c in s["cons"]:
        ax.text(0.0, y, f"-   {c}", fontsize=8.3, color=NEG_R, va="top"); y -= 0.115
    ax.text(0.995, 0.02, s["ref"], ha="right", va="bottom", fontsize=6.5, color="#9a9a9a", style="italic")


schemes = [
    dict(key="MB", title="1. MB (Marascuilo-Busk)", ref="Marascuilo 1988",
         rnd="the onset per subject (independent windows, with replacement)",
         size=f"prod k_i = {mb_size}", pmin=pmin(mb_size), stat="level / slope (within-subject)",
         pros=["low p_min for N=3 (large set) -> usable in small samples",
               "each subject = its own control -> between-subject level differences OK"],
         cons=["LEVEL stat confounded with the TIME effect if baseline rises (prefer slope)",
               "onsets may coincide (offset not guaranteed); assumes equal series"]),
    dict(key="MBR", title="2. MB-R (restricted MB)", ref="Levin 2016",
         rnd="the onset per subject, but DISTINCT (without replacement)",
         size=str(mbr_size), pmin=pmin(mbr_size), stat="level / slope (within-subject)",
         pros=["strict offset guaranteed (distinct onsets); p_min still low",
               "tolerates unequal series / cases ending early",
               "TIME effect -> trend-robust stat (detrended level / slope / ITEI)"],
         cons=["slightly smaller set than MB (distinct onsets imposed)",
               "per-subject windows to pre-specify"]),
    dict(key="WW", title="3. WW (Wampold-Worsham)", ref="Wampold 1986",
         rnd="WHICH subject gets WHICH onset (fixed pool permuted)",
         size=f"N! = {ww_size}", pmin=pmin(ww_size), stat="level / slope (within-subject)",
         pros=["simple; strict offset; no window to define",
               "subject = its own control (between-subject level differences OK)"],
         cons=["p_min = 1/N!: UNUSABLE if N < 4 (0.167 at N=3)",
               "level stat sensitive to the TIME effect; assumes equal series"]),
    dict(key="KL", title="4. KL (Koehler-Levin)", ref="Koehler 1998",
         rnd="BOTH: subject -> slot (N!) AND onset within slot (prod k_i)",
         size=f"N! x prod k_i = {kl_size}", pmin=pmin(kl_size), stat="level / slope (within-subject)",
         pros=["largest set -> power / resolution even at small N",
               "subject = its own control (between-subject levels OK)"],
         cons=["level stat sensitive to the TIME effect if baseline not flat",
               "heavy (windows + assignment); assumes equal series"]),
    dict(key="REV", title="5. Revusky (stepwise)", ref="Revusky 1967",
         rnd="the ORDER of subjects (fixed pool, like WW); cross-sectional stepwise test",
         size=f"N! = {ww_size}", pmin=pmin(ww_size), stat="stepwise RANKS (entering vs baseline)",
         pros=["built-in control: the concurrent baseline absorbs the common TIME effect",
               "RANKS -> robust to OUTLIERS (Parker 2011)"],
         cons=["p_min = 1/N! (unusable if N < 4)",
               "assumes an IMMEDIATE effect; ranks = loss of info if effect is gradual",
               "Levin 2016 recommends ABANDONING it (prefer Rev-M / KL)"]),
    dict(key="REVM", title="6. Rev-M (modified Revusky)", ref="Levin 2016",
         rnd="like Revusky, but the onset is ALSO drawn (case x onset)",
         size=f"N! x prod k_i = {kl_size}", pmin=pmin(kl_size), stat="stepwise MEANS B-A",
         pros=["built-in TIME control (like Revusky) + large set -> power at small N",
               "MEANS = more powerful than ranks if effect is clear and homogeneous"],
         cons=["means sensitive to OUTLIERS (vs robust ranks)",
               "immediate effect assumed; heavy to specify"]),
]

fig = plt.figure(figsize=(13, 17))
fig.suptitle("The 6 randomization schemes in MULTIPLE-BASELINE\n"
             "scheme (left); what is randomized, set size, p_min = 1/size, statistic and pros/cons (right)",
             fontsize=13, fontweight="bold")
gs = gridspec.GridSpec(6, 2, width_ratios=[1.0, 1.3], hspace=0.6, wspace=0.06,
                       left=0.07, right=0.98, top=0.93, bottom=0.03)
for r, s in enumerate(schemes):
    draw_sch(fig.add_subplot(gs[r, 0]), s["key"])
    text_panel(fig.add_subplot(gs[r, 1]), s)

out = "/Users/camile.bousfiha/PycharmProjects/pysced/docs/figures/mbd_schemes_all.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
print(f"sizes: MB={mb_size} MB-R={mbr_size} ordered={ord_size} WW={ww_size} KL/RevM={kl_size} Rev={ww_size}")
