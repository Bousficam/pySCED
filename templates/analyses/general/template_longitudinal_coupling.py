"""
GUIDE D'ANALYSE — COUPLAGE longitudinal multiniveau (covariable focale variant dans le temps)
==============================================================================================

Question type : un outcome (``acc``) décline-t-il quand une covariable variant dans le temps
(``eva`` = fatigue) monte, À TEMPS ÉGAL (confond temporel ``bloc`` mis en covariable) ?
Cas phare : Fatigue × Performance BCI (bras pilote BCINET) — 5 patients, ~18 séances, 4 paires
(eva, acc) par séance.

QUATRE analyses, du simple au rigoureux (toutes orientées ESTIMATION, pas NHST de groupe) :
  - rmcorr (primaire)        : corrélation à mesures répétées intra-séance (Bakdash 2017).
  - pente-vs-pente           : pentes eva~temps et acc~temps par séance, puis corrélées.
  - mixte fréquentiste       : acc ~ eva + C(bloc) [+ C(patient) fixe si <6] + (1|session),
                               df Satterthwaite/KR via R (lmerTest) si dispo.
  - mixte bayésien           : acc ~ eva + C(bloc) + (1|patient/session), gaussian + beta,
                               prior SD = student_t(3,0, 0.1·SD(acc)) -> HDI + ROPE + pd.

CONTRÔLE : VIF du confond (eva ~ bloc). Si EVA ~ monotone avec bloc -> VIF élevé : le terme eva
ne se sépare du temps que si EVA est NON-monotone (ex. pause). Le VIF est rapporté.

Toute la logique vit dans functions/Longitudinal_report.py (report_longitudinal_coupling) ; ce
template ne fait que charger les données et appeler la fonction.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.report import report_longitudinal_coupling

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES ET COLONNES (format long : 1 ligne par (séance, bloc))
# =========================================================================== #
CSV_PATH    = None          # ▸ .csv/.xlsx ; None = jeu de démo conforme (Fatigue×Perf)
OUTCOME     = "acc"         # ▸ outcome (ex. accuracy MI)
X           = "eva"         # ▸ covariable focale variant dans le temps (ex. fatigue EVA)
TIME        = "bloc"        # ▸ position temporelle intra-séance (1..4) = confond temporel
GROUP       = "session"     # ▸ intercept aléatoire de niveau-2 (séance) + cluster du rmcorr
GROUP_L3    = "patient"     # ▸ niveau-3 (patient) : FIXE si <6 niveaux (fréq), aléatoire (bayés). None = 2 niveaux
CAT         = ["bloc"]      # ▸ covariables catégorielles (codage référence). bloc en FACTEUR (non linéaire)

# =========================================================================== #
#  ▸▸▸ 2. INFÉRENCE
# =========================================================================== #
ROPE          = "auto"      # ▸ ROPE sur b_x (gaussian, unité d'origine). "auto" = 0.1·SD(acc)/SD(eva). Nombre OU None
RUN_BAYES     = True        # ▸ ajuster aussi les modèles mixtes bayésiens
BAYES_FAMILIES = ("gaussian", "beta")  # ▸ gaussian (principal) + beta (sensibilité, acc bornée)
BAYES_BOUNDS  = None        # ▸ bornes beta (lo, hi) ; None = min/max observés
DRAWS = 2000; TUNE = 2000; CHAINS = 4; SEED = 42
CACHE_DIR     = None        # ▸ cache fit-or-load des modèles bayésiens (.nc) ; None = pas de cache

# =========================================================================== #
#  ▸▸▸ 3. SORTIE
# =========================================================================== #
OUTPUT_DIR  = None          # ▸ racine UNIQUE : <outcome>/Analyse/{Inferentielle,Bayes}/ + <outcome>/Plot/
SAVE_PATH   = None          # ▸ (avancé) dossier unique
STYLE       = None          # ▸ style des figures (PlotStyle/dict) ; None = défauts


def load_data():
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(1); rows = []        # démo : eva↑ et acc↓ avec couplage intra-séance
    for pi in range(5):
        base = 70 + rng.normal(0, 5)
        for s in range(18):
            sess = f"P{pi}_S{s}"; lvl = rng.normal(0, 2)
            for b in range(1, 5):
                eva = 30 + 7 * b + rng.normal(0, 5)      # fatigue monte avec le bloc (+ bruit -> non strictement monotone)
                acc = base + lvl - 0.35 * eva + rng.normal(0, 3)
                rows.append({"patient": f"P{pi}", "session": sess, "bloc": b,
                             "eva": round(eva, 1), "acc": round(acc, 1)})
    return pd.DataFrame(rows)


def main():
    df = load_data()
    report_longitudinal_coupling(
        df, outcome=OUTCOME, x=X, time=TIME, group=GROUP, group_l3=GROUP_L3, cat=CAT,
        rope=ROPE, run_bayes=RUN_BAYES, bayes_families=BAYES_FAMILIES, bayes_bounds=BAYES_BOUNDS,
        draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED, cache_dir=CACHE_DIR,
        output_dir=OUTPUT_DIR, save_path=SAVE_PATH, style=STYLE)


if __name__ == "__main__":
    main()
