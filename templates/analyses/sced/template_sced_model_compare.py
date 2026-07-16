"""TEMPLATE — COMPARAISON DE MODÈLES SCED (bayésien) par comparateurs canoniques.

Compare des STRUCTURES DE MOYENNE emboîtées (trend / +level / +slope / full) par PSIS-LOO
ET WAIC (poids de stacking + pseudo-BMA+), par (cohorte x outcome). Ne compare QUE la moyenne ;
l'erreur (AR) et le POOLING sont fixés (axes de sensibilité). Rapport : 1re feuille = données &
design, puis modèles comparés, LOO, WAIC, décisions, guide ; modèles bayésiens sauvés en netCDF.

Toute la logique d'analyse + d'écriture du rapport consolidé stylé vit dans
functions/SCED_bayes_report.py (report_sced_model_compare) ; ce template ne fait que charger
les données et appeler la fonction avec les paramètres ci-dessous.

Usage : éditer le bloc PARAMÈTRES puis `python template_sced_model_compare.py` (ou piloter via
un script qui surcharge les globals et appelle main()).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.sced.bayes.report import report_sced_model_compare

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES, COLONNES
# =========================================================================== #
# Chaque knob *_COL = le NOM EXACT de la colonne dans ton fichier (pas la valeur).
CSV_PATH     = None              # ▸ chemin du fichier .csv ou .xlsx. None = jeu de démo intégré.
TIER_COL     = "tier"            # ▸ colonne du CAS/patient (1 série par cas). Ex: "Participant".
SESSION_COL  = "session"         # ▸ colonne de l'ORDRE TEMPOREL (entier). Calculée si tu renseignes DATE_COL.
PHASE_COL    = "phase"           # ▸ colonne de PHASE (valeurs de PHASES, défaut "A"=baseline / "B"=traitement).
# --- Format des outcomes : TIDY (OUTCOME_NAME_COL+VALUE_COL) OU LARGE (1 colonne/outcome) ---
OUTCOME_NAME_COL = None          # ▸ (TIDY) colonne qui NOMME l'outcome. Ex: "Epreuve". None = format large.
VALUE_COL        = None          # ▸ (TIDY) colonne du SCORE numérique. Ex: "Score". None = format large.
OUTCOMES     = ["accuracy"]      # ▸ LARGE: liste des colonnes-mesures. TIDY: None = tous, ou liste. Chacun à part.
CASE_INSENSITIVE = True          # ▸ True/False : tolère casse/espaces sur colonnes + phases.
DATE_COL     = None              # ▸ nom d'une colonne DATE -> session 1..n par cas. None = pas de dates.
DATE_DAYFIRST = False            # ▸ True/False : True si dates JJ/MM/AAAA.
GROUP_COL    = None              # ▸ colonne de COHORTE (ex: "profil") analysée séparément (+ « (tous) »). None = une seule.

# =========================================================================== #
#  ▸▸▸ 2. PHASES, FAMILLES, MODÈLES À COMPARER
# =========================================================================== #
PHASES          = ("A", "B")
BASELINE_PHASE  = None
TREATMENT_PHASE = None
IMPROVEMENT  = "increase"
FAMILY       = "gaussian"        # ▸ str OU dict {outcome: famille} (gaussian/student/beta/binomial)
BOUNDS       = (0, 100)          # ▸ bornes beta/binomial (tuple OU dict par outcome)
N_TRIALS     = None              # ▸ binomial : N items (int OU dict)
MODELS       = None              # ▸ None = 4 canoniques (M0 trend / Mi +level / Mg +slope / Mf full)
#                                  OU dict {nom: set de termes parmi 'trend','level','slope'}.
POOLING      = "partial"         # ▸ 'partial' (RI+RS) | 'random_intercept' (RI) | 'correlated'
#                                  (RI+RS corrélés, covariance LKJ : capte le lien niveau<->effet,
#                                  p.ex. plafond) | 'none' (pas de pooling). correlated = continu seulement.
AR           = False             # ▸ AR1 : False recommandé (LOO/WAIC supposent indép. conditionnelle)
ROPE         = 0.5               # ▸ ROPE = Region Of Practical Equivalence (Kruschke 2018) :
#                                  intervalle autour du nul (0) dans lequel un effet est jugé
#                                  PRATIQUEMENT négligeable. On rapporte P(effet > ROPE) = proba que
#                                  l'effet depasse le seuil minimal cliniquement important. La valeur
#                                  (ici sur le BC-SMD : 0.5 = effet "moyen") DOIT refleter ce seuil.
SE_MULT      = 2.0               # ▸ seuil de décision : décisif si Δelpd > SE_MULT × dse

# =========================================================================== #
#  ▸▸▸ 3. MCMC + SORTIES
# =========================================================================== #
DRAWS = 2000; TUNE = 2000; CHAINS = 4; SEED = 42; TARGET_ACCEPT = 0.99
OUTPUT_DIR   = None              # ▸ racine UNIQUE, arbo PAR OUTCOME : <outcome>/Analyse/bayes/<hier|meta>/
#                                  model_compare_summary.xlsx (comparaison) + <bN…>/ (rapports par modèle) +
#                                  models/ (=cache) + diagnostics/ ; <outcome>/Plot/forest/<bN…>/. Supersede SAVE_PATH.
SAVE_PATH    = None              # ▸ (avancé) dossier rapport .xlsx + models/*.nc (None = pas de fichier)
DIAGNOSTICS  = False             # ▸ True → diagnostics MCMC dans <Analyse/Bayes>/diagnostics/ (à côté de models/).
CACHE_DIR    = None             # ▸ dossier CACHE PARTAGE des modeles (.nc) : fit-or-load. Défaut sous
#                                  OUTPUT_DIR = Analyse/Bayes/models.
#                                  Si un modele identique (spec+donnees) existe -> recharge (pas de MCMC).
FORCE_REFIT  = False            # ▸ True = ignore le cache et reajuste tout
PER_MODEL_REPORTS = True        # ▸ True = wrapper : sort AUSSI le rapport bayes de CHAQUE modèle
#                                  (sous-dossier b1/b1b2/b1b3/... ; réutilise le cache)
POOLED_TITLE = None             # ▸ titre des plots poolés bayésiens (article-proof) ; None = titre auto
STYLE = None                    # ▸ style GLOBAL des figures (légende/couleurs/axes). None = défauts ; dict
#                                 de surcharges OU functions.common.plotstyle.PlotStyle.


def load_data():
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(1); rows = []
    for ci in range(5):
        st = 5 + ci; base = 45 + rng.normal(0, 6)
        for s in range(1, 16):
            ph = "B" if s >= st else "A"
            rows.append({"tier": f"P{ci}", "session": s, "phase": ph,
                         "accuracy": round(base + 2.0 * s + (8 if ph == "B" else 0)
                                           + rng.normal(0, 5), 1)})
    return pd.DataFrame(rows)


def main():
    """Charge les données et délègue TOUTE l'analyse + l'écriture du rapport à
    functions/SCED_bayes_report.py (report_sced_model_compare)."""
    df = load_data()
    report_sced_model_compare(
        df, tier_col=TIER_COL, session_col=SESSION_COL, phase_col=PHASE_COL,
        outcome_name_col=OUTCOME_NAME_COL, value_col=VALUE_COL, outcomes=OUTCOMES,
        case_insensitive=CASE_INSENSITIVE, date_col=DATE_COL, date_dayfirst=DATE_DAYFIRST,
        group_col=GROUP_COL, phases=PHASES, baseline_phase=BASELINE_PHASE,
        treatment_phase=TREATMENT_PHASE, improvement=IMPROVEMENT, family=FAMILY, bounds=BOUNDS,
        n_trials=N_TRIALS, models=MODELS, pooling=POOLING, ar=AR, rope=ROPE, se_mult=SE_MULT,
        draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED, target_accept=TARGET_ACCEPT,
        output_dir=OUTPUT_DIR, save_path=SAVE_PATH, diagnostics=DIAGNOSTICS, cache_dir=CACHE_DIR,
        force_refit=FORCE_REFIT, per_model_reports=PER_MODEL_REPORTS, pooled_title=POOLED_TITLE,
        style=STYLE)


if __name__ == "__main__":
    main()
