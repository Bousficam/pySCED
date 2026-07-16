"""TEMPLATE — BAYÉSIEN PAR PATIENT + SÉLECTION DE MODÈLE par patient (PSIS-LOO).

Pour CHAQUE patient, ajuste les modèles candidats (régressions beta INDÉPENDANTES par cas,
pooling='none'), choisit le PLUS PLAUSIBLE par PSIS-LOO calculé PAR PATIENT, et écrit un
rapport qui affiche, par patient, son meilleur modèle + diagnostics, et une feuille d'estimands
(comme les autres rapports). Réutilise la pipeline existante (bayes_hier_sced + comparateurs LOO)
via functions.sced.bayes.report.report_sced_bayesian_percase.

Feuilles : Données & design - Meilleur modèle - Estimands - Comparaison modèles - Glossaire -
Guide d'interprétation - Références. Fichier : bayesian_percase_summary.xlsx

Usage : éditer le bloc PARAMÈTRES puis `python template_sced_bayesian_percase.py`.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.sced.bayes.report import report_sced_bayesian_percase

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

# =========================================================================== #
#  ▸▸▸ 2. PHASES, FAMILLE, MODÈLES CANDIDATS
# =========================================================================== #
PHASES          = ("A", "B")
BASELINE_PHASE  = None
TREATMENT_PHASE = None
IMPROVEMENT  = "increase"
FAMILY       = "beta"            # ▸ str OU dict {outcome: famille}. beta = outcome borné (lien logit).
BOUNDS       = (0, 100)          # ▸ bornes beta/binomial (tuple OU dict par outcome)
N_TRIALS     = None              # ▸ binomial : N items (int OU dict)
MODELS       = None              # ▸ None = 4 candidats (M0 trend / Mi +level / Mg +slope / Mf full)
#                                  OU dict {nom: set de termes parmi 'trend','level','slope'}.
ROPE         = "auto"            # ▸ ROPE (Kruschke 2018) : seuil d'équivalence pratique (MCID) en POINTS.
#                                  "auto" = 0.1 x SD(outcome) (défaut). Appliquée à l'EFFET FIN seulement.
SE_MULT      = 2.0               # ▸ décisif PAR PATIENT si Δelpd > SE_MULT × dse (sinon indistinguable)

# =========================================================================== #
#  ▸▸▸ 3. MCMC + SORTIES
# =========================================================================== #
DRAWS = 2000; TUNE = 2000; CHAINS = 4; SEED = 42; TARGET_ACCEPT = 0.99
OUTPUT_DIR   = None              # ▸ racine arbo PAR OUTCOME : <outcome>/Analyse/bayes/percase/
#                                  bayesian_percase_summary.xlsx + models/ (cache). Supersede SAVE_PATH.
SAVE_PATH    = None              # ▸ (avancé) dossier UNIQUE du rapport .xlsx (None = pas de fichier)
CACHE_DIR    = None              # ▸ cache fit-or-load des modèles (.nc). Défaut sous OUTPUT_DIR.
FORCE_REFIT  = False             # ▸ True = ignore le cache et réajuste
DIAGNOSTICS  = False             # ▸ True → diagnostics MCMC par modèle dans <sortie>/diagnostics/
STYLE = None                     # ▸ style des figures (PlotStyle/dict) ; None = défauts


def load_data():
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(1); rows = []          # démo : 5 cas, effets hétérogènes (saut/pente)
    for ci in range(5):
        st = 5 + ci; base = 45 + rng.normal(0, 6)
        for s in range(1, 16):
            ph = "B" if s >= st else "A"
            jump = (10 if ci % 2 == 0 else 0); slope = (1.5 if ci % 2 else 0) * max(0, s - st + 1)
            rows.append({"tier": f"P{ci}", "session": s, "phase": ph,
                         "accuracy": round(float(np.clip(base + (jump if ph == "B" else 0) + slope
                                                         + rng.normal(0, 5), 0, 100)), 1)})
    return pd.DataFrame(rows)


def main():
    """Charge les données et délègue TOUTE l'analyse + l'écriture du rapport à
    functions/SCED_bayes_report.py (report_sced_bayesian_percase)."""
    df = load_data()
    report_sced_bayesian_percase(
        df, tier_col=TIER_COL, session_col=SESSION_COL, phase_col=PHASE_COL,
        outcome_name_col=OUTCOME_NAME_COL, value_col=VALUE_COL, outcomes=OUTCOMES,
        case_insensitive=CASE_INSENSITIVE, date_col=DATE_COL, date_dayfirst=DATE_DAYFIRST,
        phases=PHASES, baseline_phase=BASELINE_PHASE, treatment_phase=TREATMENT_PHASE,
        improvement=IMPROVEMENT, family=FAMILY, bounds=BOUNDS, n_trials=N_TRIALS, models=MODELS,
        rope=ROPE, se_mult=SE_MULT, draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED,
        target_accept=TARGET_ACCEPT, output_dir=OUTPUT_DIR, save_path=SAVE_PATH,
        cache_dir=CACHE_DIR, force_refit=FORCE_REFIT, diagnostics=DIAGNOSTICS, style=STYLE)


if __name__ == "__main__":
    main()
