"""
GUIDE D'ANALYSE — SCED bayésien pour designs en PHASES (A->B) — UNIFIÉ
=====================================================================

Un seul template bayésien de phases, piloté par TROIS AXES orthogonaux (le moteur en est dérivé) :

    POOLING : partage entre cas — partial (RI+RS) | random_intercept | correlated (LKJ) |
              none (par cas indépendant) | meta (two-stage : μ + τ + I² + intervalle de prédiction).
    ONSET   : "known" = bascule = étiquette de phase (modèle hiérarchique, termes b1/b2/b3) ;
              "unknown" = point de bascule ESTIMÉ (BUCP = immédiateté) — NON poolable -> force none.
    AR      : bruit AR1 intra-cas (= l'ancien "BITS") ; gaussian/student ; IGNORÉ en beta/binomial.

(Rindskopf 2014 ; Van den Noortgate & Onghena ; Moeyaert ; Natesan Batley et al. 2020.)
Choix : population/partage -> POOLING ; immédiateté / bascule inconnue -> ONSET="unknown".
Distinct de template_sced_bayesian.py (effet de CONDITION d'un design ALTERNANT + Bayes factor).

Inférence (les deux modes) : HDI 95% + pd + ROPE. Pas de « significatif » : pd>=0.95 = direction.
Bayésien = lent : prévoir du temps si beaucoup de cas.

Toute la logique d'analyse + d'écriture des rapports consolidés stylés vit dans
functions/SCED_bayes_report.py (report_sced_bayesian_mbd) ; ce template ne fait que charger
les données et appeler la fonction avec les paramètres ci-dessous.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.sced.bayes.report import report_sced_bayesian_mbd

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES ET COLONNES  (communs aux deux modes)
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
#  ▸▸▸ 2. TROIS AXES ORTHOGONAUX : POOLING · ONSET · AR  (+ phases / inférence)
# =========================================================================== #
#  Le moteur est DÉRIVÉ de ces axes (plus de MODE) :
#    ONSET="known"   -> modèle hiérarchique (bascule connue), partage réglé par POOLING ;
#    ONSET="unknown" -> change-point BUCP (le moment de l'effet est ESTIMÉ) — NON poolable,
#                       donc FORCE POOLING="none" (analyse single-case).
POOLING      = "partial"         # ▸ PARTAGE entre cas : partial (RI+RS) | random_intercept (RI) |
#                                  correlated (LKJ) | none (PAR CAS indépendant) | meta (two-stage)
ONSET        = "known"           # ▸ "known" (bascule = étiquette de phase) | "unknown" (BUCP, estime le
#                                  moment de l'effet ; force POOLING="none")
AR           = True              # ▸ bruit AR1 intra-cas (= l'ancien "BITS") ; dispo partout SAUF beta/binomial
#                                  (logit) ; bool OU dict {outcome: bool}
PHASES          = ("A", "B")     # ▸ baseline d'abord
BASELINE_PHASE  = None           # ▸ défaut = PHASES[0]
TREATMENT_PHASE = None           # ▸ défaut = PHASES[-1]
IMPROVEMENT  = "increase"        # ▸ "increase" si plus haut = mieux ; sinon "decrease"
ROPE         = 0.5               # ▸ ROPE (Kruschke 2018), en UNITÉ D'ORIGINE (MCID). Nombre OU "auto"
#                                  (=0.1·SD). Rapporte %in ROPE + décision HDI vs ROPE (effet/équiv./indécis).

# --- 2a. structure de moyenne + familles (ONSET="known") ------------------
HYPOTHESIS   = "immediate_flat"  # ▸ immediate_flat | cumulative_flat | immediate_trend | cumulative_trend
TERMS        = None              # ▸ override explicite des termes : sous-ensemble de {'trend','level',
#                                  'slope'} (prime sur HYPOTHESIS). None = dérivé de HYPOTHESIS.
FAMILY       = "gaussian"        # ▸ gaussian | student | beta | binomial — str OU dict {outcome: famille}
BOUNDS       = (0, 100)          # ▸ bornes beta/binomial (échelle de l'outcome) — tuple OU dict par outcome
N_TRIALS     = None              # ▸ binomial : N items — entier OU dict par outcome
TARGET_ACCEPT = 0.95             # ▸ monter à 0.99 si divergences (séries courtes, trend+AR1)

# --- 2b. paramètres ONSET="unknown" (change-point BUCP, single-case) -------
TREND        = False             # ▸ effet cumulatif/tardif : ajoute une pente en B → es_end + b3
BASELINE_TREND = False           # ▸ pente baseline PRÉ-SPÉCIFIÉE (True/False) : True = effet net de la
#                                  tendance baseline. (Le test Tarlow reste rapporté en diagnostic.)
MIN_PER_PHASE = 3                # ▸ cas ignoré si une phase a moins de points

# --- MCMC (communs) -------------------------------------------------------
DRAWS = 2000; TUNE = 2000; CHAINS = 4; SEED = 42
OUTPUT_DIR   = None              # ▸ racine UNIQUE, arbo PAR OUTCOME : <outcome>/Analyse/bayes/<hier|meta>/
#                                  <bN…>/ (xlsx) + models/ (=cache) + diagnostics/ ; <outcome>/Plot/forest/<bN…>/ ;
#                                  <outcome>/Plot/poolé/ (poolé bayésien systématique). Supersede SAVE_PATH/CACHE_DIR.
SAVE_PATH    = None              # ▸ (avancé) dossier rapport .xlsx + models/*.nc (None = pas de fichier)
DIAGNOSTICS  = False             # ▸ True → diagnostics MCMC dans <Analyse/Bayes>/diagnostics/ (à côté de
#                                  models/) ; sinon via template_sced_visualize.py séparément.
CACHE_DIR    = None              # ▸ dossier CACHE PARTAGÉ des modèles (.nc) : fit-or-load. Si un modèle
#                                  identique (spec + données) existe → recharge (pas de MCMC). Défaut sous
#                                  OUTPUT_DIR = Analyse/Bayes/models. Pointer ailleurs pour un cache inter-études.
FORCE_REFIT  = False             # ▸ True = ignore le cache et réajuste tout
MCID         = None              # ▸ seuil clinique pour « Répondeur » (par cas, sur l'effet fin-B) :
#                                  répondeur = direction crédible (pd>=0.95) ET |effet|>=MCID. scalaire OU dict.
POOLED_TITLE = None              # ▸ titre des plots poolés bayésiens (article-proof) ; None = titre auto
STYLE = None                     # ▸ style GLOBAL des figures (légende/couleurs/axes/tailles). None = défauts.
#                                  dict de surcharges, ex. {"colors": {"treatment": "#d55"},
#                                  "labels": {"phase_b": "tDCS"}, "ylabel": "FM-UE", "fontsize": 12}
#                                  OU une instance functions.common.plotstyle.PlotStyle (cf. ce module).
FOREST_ESTIMAND = "auto"         # ▸ estimand(s) du/des forest(s) (Plot/forest/) : "auto" = CALIBRÉ sur les termes
#                                  (level+slope -> 3 forests b2 + b3 + fin-B ; b1+b3 -> b3 ; b1+b2 -> b2)
#                                  OU un seul explicite : "effect_end" | "slope" | "level"


def load_data():
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(1); rows = []
    for ci in range(6):
        e = 0.0; base = 55 + rng.normal(0, 8)
        for s in range(1, 16):
            ph = 1 if s >= 6 else 0; e = 0.3 * e + rng.normal(0, 4)
            rows.append({"tier": f"T{ci+1}", "session": s, "phase": "B" if ph else "A",
                         "accuracy": base + (10 if ph else 0) + e})
    return pd.DataFrame(rows)


def main():
    """Charge les données et délègue TOUTE l'analyse + l'écriture des rapports à
    functions/SCED_bayes_report.py (report_sced_bayesian_mbd)."""
    df = load_data()
    report_sced_bayesian_mbd(
        df, pooling=POOLING, onset=ONSET, ar=AR,                     # 3 axes (remplace MODE/MODEL)
        tier_col=TIER_COL, session_col=SESSION_COL, phase_col=PHASE_COL,
        outcome_name_col=OUTCOME_NAME_COL, value_col=VALUE_COL, outcomes=OUTCOMES,
        case_insensitive=CASE_INSENSITIVE, date_col=DATE_COL, date_dayfirst=DATE_DAYFIRST,
        group_col=GROUP_COL, phases=PHASES, baseline_phase=BASELINE_PHASE,
        treatment_phase=TREATMENT_PHASE, improvement=IMPROVEMENT, rope=ROPE,
        hypothesis=HYPOTHESIS, terms=TERMS, family=FAMILY, bounds=BOUNDS, n_trials=N_TRIALS,
        target_accept=TARGET_ACCEPT,
        trend=TREND, baseline_trend=BASELINE_TREND, min_per_phase=MIN_PER_PHASE,
        draws=DRAWS, tune=TUNE, chains=CHAINS, seed=SEED,
        output_dir=OUTPUT_DIR, save_path=SAVE_PATH, diagnostics=DIAGNOSTICS, cache_dir=CACHE_DIR,
        force_refit=FORCE_REFIT, mcid=MCID, forest_estimand=FOREST_ESTIMAND, pooled_title=POOLED_TITLE,
        style=STYLE)


if __name__ == "__main__":
    main()
