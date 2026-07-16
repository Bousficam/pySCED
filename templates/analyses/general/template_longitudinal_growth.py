"""
GUIDE D'ANALYSE — GROWTH-CURVE multiniveau (forme de la trajectoire dans le temps)
==================================================================================

Question type : comment un outcome évolue-t-il DANS LE TEMPS (forme : linéaire ? courbe ?),
les sujets diffèrent-ils dans leur changement (pentes aléatoires), un prédicteur module-t-il
la trajectoire (``by`` × temps), une covariable variant dans le temps est-elle couplée
(within/between) ?

Pipeline (fréquentiste, Phase 2) :
  1. SÉLECTION DE FORME : linear vs poly2 vs spline (AIC/BIC + LRT emboîté) -> meilleure forme.
  2. AJUSTEMENT : LMM ``outcome ~ f(temps) [+ by×f(temps)] [+ predictors] + (random | sujet)``,
     random = intercept | int_slope (pente aléatoire) ; 3 niveaux possible (group_l3 = cluster).
  3. SORTIES : effets fixes + ICC + R² (Nakagawa) ; TRAJECTOIRE MARGINALE (emmeans-like) ± IC ;
     CATERPILLAR des effets aléatoires (BLUP) ; option within/between d'une covariable temporelle.

Toute la logique vit dans functions/Longitudinal_report.py (report_longitudinal_growth).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.report import report_longitudinal_growth

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES ET COLONNES (format long : 1 ligne par (sujet, temps))
# =========================================================================== #
CSV_PATH   = None           # ▸ .csv/.xlsx ; None = jeu de démo (trajectoire quadratique + by)
OUTCOME    = "y"            # ▸ outcome continu
TIME       = "t"           # ▸ temps continu (numérique)
GROUP      = "subj"         # ▸ sujet (niveau-2, effets aléatoires)
GROUP_L3   = None           # ▸ cluster (niveau-3, intercept aléatoire niché) ; None = 2 niveaux
BY         = "arm"          # ▸ prédicteur invariant -> trajectoires différentielles (×temps) ; None = aucun
PREDICTORS = None           # ▸ covariables invariantes additionnelles (effets principaux) ; liste de termes patsy
WITHIN_BETWEEN = None       # ▸ covariable VARIANT dans le temps à décomposer within/between ; None = aucune

# =========================================================================== #
#  ▸▸▸ 2. MODÈLE
# =========================================================================== #
SHAPE          = "auto"     # ▸ forme du temps : "auto" (meilleur AIC) | "linear" | "poly2" | "spline" | "discrete"
#                              | "pspline" (GAMM bayésien : base spline + pénalité de lissage ; côté BAYES)
KNOTS          = 4          # ▸ df de la spline (si SHAPE="spline")
RANDOM         = "int_slope"  # ▸ "intercept" | "int_slope" (pente aléatoire sur le temps)
COMPARE_SHAPES = ("linear", "poly2", "spline")  # ▸ formes mises en concurrence (AIC/BIC/LRT)
KR             = False      # ▸ df Satterthwaite via R (lmerTest) en plus du Wald

# --- arm BAYÉSIEN (Phase 3 ; même forme retenue) ---
BAYES        = False        # ▸ True = ajoute le growth-curve bayésien (PyMC) : HDI/ROPE/pd, LOO/WAIC, trajectoire postérieure
BAYES_FAMILY = "gaussian"   # ▸ famille bayésienne : gaussian | poisson | binomial | ordinal (GLMM growth)
N_TRIALS     = None         # ▸ binomial : nb d'essais (entier ou nom de colonne)
ROPE         = "auto"       # ▸ ROPE sur les termes de temps (unité d'origine). "auto" = 0.1·SD(outcome)
DRAWS = 2000; TUNE = 2000; CHAINS = 4; SEED = 42
CACHE_DIR    = None         # ▸ cache fit-or-load des modèles bayésiens (.nc)
DIAGNOSTICS  = False        # ▸ diagnostics MCMC dans <outcome>/Analyse/Bayes/diagnostics/

# =========================================================================== #
#  ▸▸▸ 3. SORTIE
# =========================================================================== #
OUTPUT_DIR = None           # ▸ racine UNIQUE : <outcome>/Analyse/Inferentielle/ + <outcome>/Plot/
SAVE_PATH  = None           # ▸ (avancé) dossier unique
STYLE      = None           # ▸ style des figures (PlotStyle/dict) ; None = défauts


def load_data():
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(0); rows = []          # démo : trajectoire quadratique, bras module la pente
    for s in range(40):
        a = rng.normal(50, 6); b = rng.normal(4, 1.5); arm = s % 2
        for t in range(6):
            y = a + b * t - 0.4 * t ** 2 + (2.0 * arm) * t + rng.normal(0, 2.5)
            rows.append({"subj": f"S{s}", "t": t, "y": round(y, 2), "arm": f"arm{arm}"})
    return pd.DataFrame(rows)


def main():
    df = load_data()
    report_longitudinal_growth(
        df, outcome=OUTCOME, time=TIME, group=GROUP, shape=SHAPE, knots=KNOTS, by=BY,
        predictors=PREDICTORS, group_l3=GROUP_L3, random=RANDOM, compare_shapes=COMPARE_SHAPES,
        within_between=WITHIN_BETWEEN, kr=KR, bayes=BAYES, bayes_family=BAYES_FAMILY, n_trials=N_TRIALS,
        rope=ROPE, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED, cache_dir=CACHE_DIR,
        diagnostics=DIAGNOSTICS, output_dir=OUTPUT_DIR, save_path=SAVE_PATH, style=STYLE)


if __name__ == "__main__":
    main()
