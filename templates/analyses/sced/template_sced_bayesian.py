"""
GUIDE D'ANALYSE — SCED bayésien : effet de condition + Bayes factor
===================================================================

Objet : estimer l'effet d'une condition (alternant / N-of-1, ou MULTINIVEAU) par une
approche BAYÉSIENNE, en complément du test de randomisation (inférence primaire). On
obtient : l'effet a posteriori (taille d'effet adaptée à la famille), son HDI (intervalle
de plus haute densité), P(effet bénéfique), un BAYES FACTOR BF10 (évidence POUR vs CONTRE
un effet ; BF10<1 = appui à H0), une décision ROPE optionnelle, et — sur demande — les
PLOTS DE DIAGNOSTIC standard (trace, rank, forest+R̂/ESS, posterior+HDI/ROPE, energy, PPC).

Données — format LONG, une ligne par séance :
    session (entier) | condition (A, B, …) | outcome   [+ colonnes de regroupement / covariables]

Réglages (en miroir du template alternant) : type d'outcome, priors informatifs en
moyenne/SD bruts, niveaux hiérarchiques, covariable de temps, covariables, HDI/ROPE.
NB : nécessite ``pymc`` + ``arviz`` (cf. requirements.txt).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.sced.bayes.condition import report_sced_bayesian_condition

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES ET COLONNES
# =========================================================================== #
CSV_PATH = None              # ▸ chemin du fichier .csv ou .xlsx. None = jeu de démo intégré.
# COL = NOM EXACT de chaque colonne dans ton fichier (pas la valeur) :
COL = {"session": "session",      # ▸ colonne de l'ORDRE TEMPOREL (entier)
       "condition": "condition",  # ▸ colonne de CONDITION (valeurs dans CONDITIONS ci-dessous)
       "outcome": "score"}        # ▸ colonne du SCORE numérique à analyser

# =========================================================================== #
#  ▸▸▸ 2. CONTRASTE ET FAMILLE
# =========================================================================== #
CONDITIONS   = ("A", "B")    # ▸ conditions à inclure (référence en premier)
REFERENCE    = None          # ▸ condition de référence (None = première de CONDITIONS)
TARGET       = None          # ▸ condition cible du contraste (None = dernière)
OUTCOME_TYPE = "continuous"  # ▸ "continuous" | "robust" (Student-t) | "binary" (0/1) | "count"
#                              continuous/robust → d de Cohen ; binary → odds ratio ; count → rate ratio
IMPROVEMENT  = "increase"    # ▸ "increase" si un score plus haut = mieux ; sinon "decrease"

# =========================================================================== #
#  ▸▸▸ 3. PRIORS  (par défaut faiblement informatif ; sinon préciser moyenne/SD BRUTS)
# =========================================================================== #
#   • non informatif (défaut)            → laisser PRIOR_MEAN/PRIOR_SD = None (JZS Cauchy)
#   • INFORMATIF en moyenne/SD BRUTS     → PRIOR_MEAN / PRIOR_SD dans les UNITÉS NATURELLES :
#       - continuous/robust : unités brutes de l'outcome (ex. +5 points, ±2) → traduit en d
#       - binary/count      : odds ratio / rate ratio attendu (ex. 1.5, ±0.3) → traduit en log
#     (sert aussi à injecter le postérieur d'une étude précédente = mise à jour séquentielle)
PRIOR_MEAN  = None           # ▸ moyenne a priori de l'effet (unités naturelles) ; None = non informatif
PRIOR_SD    = None           # ▸ écart-type a priori (incertitude) ; à fournir SI PRIOR_MEAN l'est
PRIOR_SCALE = None           # ▸ (avancé) échelle du Cauchy non informatif (défaut 0.707)

# =========================================================================== #
#  ▸▸▸ 4. NIVEAUX HIÉRARCHIQUES & NUISANCES
# =========================================================================== #
GROUP_COLS     = None        # ▸ None = cas unique ; ["patient"] = 1 niveau ; ["site","patient"] = 2 niveaux (intercept aléatoire chacun)
RANDOM_SLOPE   = False       # ▸ True = effet du traitement ALÉATOIRE sur le 1er niveau (l'effet varie par patient)
TIME_COVARIATE = "none"      # ▸ "none" | "linear" | "log" (ajoute le temps comme covariable de nuisance)
COVARIATES     = None        # ▸ ajustement ANCOVA : liste de colonnes numériques (supposées NON affectées par la condition)

# =========================================================================== #
#  ▸▸▸ 5. DÉCISION : HDI & ROPE
# =========================================================================== #
HDI_PROB = 0.95              # ▸ masse de l'intervalle de plus haute densité (0.95, ou 0.89…)
ROPE     = None              # ▸ région d'équivalence pratique au nul, en UNITÉS D'EFFET :
#                              continuous/robust → en d (ex. (-0.1, 0.1)) ; binary/count → ratio (ex. (0.9, 1.1)) ; None = pas de ROPE

# =========================================================================== #
#  ▸▸▸ 6. ÉCHANTILLONNAGE MCMC & PLOTS
# =========================================================================== #
DRAWS     = 2000             # ▸ tirages post warm-up
TUNE      = 1000             # ▸ warm-up
CHAINS    = 4                # ▸ chaînes (≥4 recommandé pour des diagnostics fiables)
OUTPUT_DIR = None            # ▸ racine UNIQUE, arbo PAR OUTCOME : <outcome>/Analyse/bayes/condition/ (xlsx + diagnostics/).
SAVE_PATH  = None            # ▸ (avancé) dossier rapport xlsx seul
DIAGNOSTICS = False          # ▸ True → diagnostics MCMC dans <sortie>/diagnostics/ (à côté du rapport)


def load_data():
    """Chargement : CSV_PATH si défini, sinon jeu de démo (à remplacer)."""
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(0); rows = []
    for b in range(10):
        for c in rng.permutation(["A", "B"]):
            s = b * 2 + (1 if c == "A" else 2)
            rows.append({"session": s, "condition": c, "patient": "P1",
                         "score": (6 if c == "B" else 0) + rng.normal(20, 2)})
    return pd.DataFrame(rows)


def main():
    """Charge les données et délègue analyse + rapport + diagnostics à
    functions/SCED_bayesian.py (report_sced_bayesian_condition)."""
    df = load_data()
    report_sced_bayesian_condition(
        df, session_col=COL["session"], condition_col=COL["condition"],
        outcome_col=COL["outcome"], conditions=CONDITIONS, reference=REFERENCE,
        target=TARGET, outcome_type=OUTCOME_TYPE, improvement=IMPROVEMENT,
        prior_mean=PRIOR_MEAN, prior_sd=PRIOR_SD, prior_scale=PRIOR_SCALE,
        group_cols=GROUP_COLS, random_slope=RANDOM_SLOPE, time_covariate=TIME_COVARIATE,
        covariate_cols=COVARIATES, hdi_prob=HDI_PROB, rope=ROPE,
        draws=DRAWS, tune=TUNE, chains=CHAINS, random_state=0,
        output_dir=OUTPUT_DIR, save_path=SAVE_PATH, diagnostics=DIAGNOSTICS)


if __name__ == "__main__":
    main()
