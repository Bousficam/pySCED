"""Didactic figure: HOW baseline-corrected Tau works, step by step. A rising baseline
that simply continues into B inflates the raw nonoverlap. The correction removes the
baseline monotonic trend (Theil-Sen) before scoring the overlap. Toolbox functions +
faithful Tarlow 2016 / Brossard 2018 logic. English labels, ASCII only."""
import sys
sys.path.insert(0, "/Users/camile.bousfiha/PycharmProjects/Stats")
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functions.sced.core import nap, tau_u

BLUE, ORANGE, GREY = "#4C72B0", "#DD8452", "#888888"

def ref(a, t, loc="br"):
    """Discreet source note (1st author, year) in a panel corner."""
    xy = {"br": (0.985, 0.02, "right", "bottom"), "tl": (0.015, 0.98, "left", "top")}[loc]
    a.text(xy[0], xy[1], t, transform=a.transAxes, ha=xy[2], va=xy[3],
           fontsize=6.5, color="#9a9a9a", style="italic")

# Rising baseline that CONTINUES into B (no extra treatment jump), with noise.
_rng = np.random.default_rng(5)
sA = np.arange(1, 9); sB = np.arange(9, 14)
A = np.round(3 + 1.0 * sA + _rng.normal(0, 0.8, sA.size), 1)
B = np.round(3 + 1.0 * sB + _rng.normal(0, 0.8, sB.size), 1)
sess = np.concatenate([sA, sB]); y = np.concatenate([A, B])

raw = tau_u(A, B, correct_baseline_trend=False)
cor = tau_u(A, B, correct_baseline_trend=True)
nap_raw = nap(A, B)

# baseline Theil-Sen slope (fit on A only) -> detrend the whole series by it (Tarlow)
ts = stats.theilslopes(A, sA)                       # (slope, intercept, lo, hi)
slope, icpt = ts[0], ts[1]
trend = icpt + slope * sess
resid = y - trend                                   # detrended residuals
rA, rB = resid[:A.size], resid[A.size:]
nap_res = nap(rA, rB)                                # nonoverlap AFTER removing the trend

fig, ax = plt.subplots(2, 2, figsize=(12.5, 9))
fig.suptitle("Tau correction: remove the baseline trend BEFORE scoring the nonoverlap\n"
             "(Tarlow 2016 baseline-corrected Tau / Brossard 2018)", fontsize=12, fontweight="bold")

# Step 1: raw view
a = ax[0, 0]
a.scatter(sA, A, c=BLUE, s=70, label="Baseline A", zorder=3)
a.scatter(sB, B, c=ORANGE, s=70, label="Treatment B", zorder=3)
a.axvline(A.size + 0.5, color=GREY, ls=":")
a.set_title(f"1. RAW view: B sits above A -> almost total nonoverlap\n"
            f"raw NAP = {nap_raw:.2f} ; raw Tau p = {raw['p_value']:.3f} (looks significant)",
            fontweight="bold", fontsize=9.5)
a.set_xlabel("Session"); a.set_ylabel("Score"); a.legend(fontsize=8, loc="upper left")
ref(a, "Parker 2011", "br")

# Step 2: fit baseline trend + extrapolate
b = ax[0, 1]
b.scatter(sA, A, c=BLUE, s=70, zorder=3); b.scatter(sB, B, c=ORANGE, s=70, zorder=3)
b.plot(sA, icpt + slope * sA, color=BLUE, lw=2, label=f"baseline trend (Theil-Sen, slope={slope:.2f})")
b.plot(sB, icpt + slope * sB, color=BLUE, lw=2, ls="--", label="extrapolated onto phase B")
b.axvline(A.size + 0.5, color=GREY, ls=":")
b.set_title(f"2. The baseline is ALREADY rising (tau_A={raw['baseline_trend_tau']:+.2f}, p={raw['baseline_trend_p']:.3f})\n"
            "extrapolated, the trend already predicts the level of B", fontweight="bold", fontsize=9.5)
b.set_xlabel("Session"); b.set_ylabel("Score"); b.legend(fontsize=7.5, loc="upper left")
ref(b, "Sen 1968", "br")

# Step 3: detrend -> residuals
c = ax[1, 0]
c.axhline(0, color=GREY, lw=1.2)
c.vlines(sA, 0, rA, color=BLUE, lw=2); c.scatter(sA, rA, c=BLUE, s=60, zorder=3, label="residual A")
c.vlines(sB, 0, rB, color=ORANGE, lw=2); c.scatter(sB, rB, c=ORANGE, s=60, zorder=3, label="residual B")
c.axvline(A.size + 0.5, color=GREY, ls=":")
c.set_title("3. DETREND: residual = data - trend\n"
            "once the rise is removed, A and B overlap (no more jump)",
            fontweight="bold", fontsize=9.5)
c.set_xlabel("Session"); c.set_ylabel("Residual (score - trend)"); c.legend(fontsize=8)
ref(c, "Tarlow 2016", "br")

# Step 4: score the overlap on residuals -> corrected result
d = ax[1, 1]
labels = ["Raw Tau\n(on data)", "Corrected Tau\n(on residuals)"]
pvals = [raw["p_value"], cor["p_value"]]
cols = ["#C44E52" if p < 0.05 else "#4C9F70" for p in pvals]
bars = d.bar(labels, pvals, color=cols, alpha=0.8, width=0.55)
d.axhline(0.05, color="black", ls="--", lw=1); d.text(1.45, 0.06, "0.05 threshold", fontsize=8, ha="right")
for bar, p in zip(bars, pvals):
    d.text(bar.get_x() + bar.get_width() / 2, p + 0.02, f"p={p:.3f}", ha="center", fontsize=9, fontweight="bold")
d.set_ylim(0, 1); d.set_ylabel("p-value")
d.set_title(f"4. Score the nonoverlap of the RESIDUALS\n"
            f"NAP on residuals = {nap_res:.2f} (falls back toward 0.5) ; effect NOT significant after correction",
            fontweight="bold", fontsize=9.5)
d.text(0.5, 0.80, "The raw nonoverlap came from the\npre-existing drift, not the treatment.",
       transform=d.transAxes, ha="center", fontsize=8,
       bbox=dict(boxstyle="round", fc="white", ec=GREY, alpha=0.85))
ref(d, "Tarlow 2016 ; Brossard 2018", "tl")

fig.tight_layout(rect=[0, 0, 1, 0.93])
out = "/Users/camile.bousfiha/PycharmProjects/pysced/docs/figures/tau_correction.png"
fig.savefig(out, dpi=130, bbox_inches="tight")
print("saved", out)
print(f"raw p={raw['p_value']:.3f} NAP_raw={nap_raw:.2f} | corrected p={cor['p_value']:.3f} NAP_resid={nap_res:.2f} slope={slope:.2f}")
