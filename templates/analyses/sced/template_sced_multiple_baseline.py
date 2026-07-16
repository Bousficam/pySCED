"""
GUIDE D'ANALYSE — SCED Ligne de base multiple (multiple baseline design)
========================================================================

Objet : plan où l'intervention est introduite à des moments DÉCALÉS selon les paliers
(patients / comportements / contextes). Un effet est démontré si chaque palier change
À SON moment d'introduction, pendant que les paliers encore en baseline ne bougent pas.

Inférence : test de randomisation sur le MOMENT d'introduction (Edgington & Onghena ;
Marascuilo & Busk ; procédures Levin et al. 2017). C'est le moteur VALIDE pour un design
A→B en phases (≠ test alternant, qui suppose l'affectation randomisée par séance).

Données attendues — format LONG, une ligne par (palier, session) :
    tier (palier) | session (ordre entier) | phase (A/B) | outcome
(la phase B commence au moment d'introduction propre au palier.)

Confort d'usage (aligné sur le template alternant) :
  • CASE_INSENSITIVE  — colonnes/conditions insensibles à la casse ;
  • DATE_COL          — convertit une DATE en session ordonnée 1..n (par palier) ;
  • format TIDY       — OUTCOME_NAME_COL/VALUE_COL : chaque outcome analysé SÉPARÉMENT ;
  • GROUP_COL         — cohortes (ex. aigu/chronique) analysées séparément ;
  • VAIOR par palier  — aide visuelle de Manolov (baseline vs intervention).

Toute la logique d'analyse + d'écriture du rapport consolidé stylé (MBD_summary.xlsx)
vit dans functions/SCED_mbd_report.py ; ce template ne fait que charger les données et
appeler report_sced_multiple_baseline avec les paramètres ci-dessous.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.sced.mbd.report import report_sced_multiple_baseline

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES, COLONNES   (mets ici les NOMS EXACTS de tes colonnes)
# =========================================================================== #
# Chaque knob *_COL = le nom de la colonne CORRESPONDANTE dans ton fichier (pas la valeur).
CSV_PATH     = None              # ▸ chemin du fichier .csv ou .xlsx. None = jeu de démo intégré.
TIER_COL     = "tier"            # ▸ colonne du PALIER (= 1 série temporelle : patient / comportement /
#                                  contexte). Ex: "Participant", "patient_id". Texte ou nombre.
SESSION_COL  = "session"         # ▸ colonne de l'ORDRE TEMPOREL (entier 1,2,3…). Laisse tel quel et
#                                  renseigne DATE_COL si tu n'as que des dates (la session est alors calculée).
PHASE_COL    = "phase"           # ▸ colonne de PHASE. Valeurs attendues = celles de PHASES ci-dessous
#                                  (défaut "A"=baseline, "B"=traitement). Mets None si tu utilises INTERVENTION_STARTS.
# --- Format des outcomes : choisis UN des deux ---
#   (a) TIDY/empilé : une colonne nomme l'outcome, une colonne porte le score (1 ligne/score) ->
#       renseigne OUTCOME_NAME_COL + VALUE_COL, et OUTCOMES = None (tous) ou liste.
#   (b) LARGE : 1 colonne PAR outcome -> laisse OUTCOME_NAME_COL/VALUE_COL = None, et OUTCOMES = [noms].
OUTCOME_NAME_COL = None          # ▸ (TIDY) colonne qui NOMME l'outcome. Ex: "Epreuve". None = format large.
VALUE_COL        = None          # ▸ (TIDY) colonne du SCORE numérique. Ex: "Score". None = format large.
OUTCOMES     = ["accuracy"]      # ▸ LARGE: liste des colonnes-mesures. TIDY: None = TOUS les outcomes empilés,
#                                  ou liste = sous-ensemble. CHAQUE outcome est analysé SÉPARÉMENT.
CASE_INSENSITIVE = True          # ▸ True/False. True = tolère casse/espaces sur les noms de colonnes ET les phases.
DATE_COL     = None              # ▸ nom d'une colonne DATE -> convertie en session 1..n par palier. None = pas de dates.
DATE_DAYFIRST = False            # ▸ True/False. True si tes dates sont au format JJ/MM/AAAA.
GROUP_COL    = None              # ▸ colonne de COHORTE (ex: "profil" aigu/chronique) analysée SÉPARÉMENT. None = une seule cohorte.
INCLUDE_POOLED = True            # ▸ True/False (utile seulement si GROUP_COL). True = analyse AUSSI tous les
#                                  paliers ensemble (« tous ») en plus de chaque cohorte (gain de puissance).

# =========================================================================== #
#  ▸▸▸ 2. PHASES ET CONTRASTE
# =========================================================================== #
PHASES          = ("A", "B")     # ▸ étiquettes canoniques de phase (baseline d'abord)
BASELINE_PHASE  = None           # ▸ ≥3 phases : phase de contrôle du contraste (ex. "B" sham). None = A/B
TREATMENT_PHASE = None           # ▸ ≥3 phases : phase active (ex. "C" TMS réelle). A devient un rodage exclu
IMPROVEMENT  = "increase"        # ▸ "increase" si plus haut = mieux ; sinon "decrease"
STATISTIC    = "level"           # ▸ "level" (saut) | "slope" (progressif) | "combined" (ITS) |
#                                  "itei" (transition trend-robuste) | "tau_u" | "nap"
#                                  (tau_u/nap : p de randomisation DE la statistique de chevauchement, conv. SCDA)

# =========================================================================== #
#  ▸▸▸ 3. PROCÉDURE DE RANDOMISATION (par ses FEATURES ; Levin et al. 2017)
# =========================================================================== #
#   case=T,start=F,within  -> WW    | case=F,start=T,remise=T -> MB   | case=F,start=T,remise=F -> MB-R
#   case=T,start=T,within  -> KL    | case=T,start=F,between  -> Rev  | case=T,start=T,between  -> Rev-M
CASE_RANDOMIZATION        = False  # ▸ permuter les CAS entre positions (WW, KL, Rev, Rev-M)
START_POINT_RANDOMIZATION = True   # ▸ randomiser le MOMENT dans une fenêtre (MB, MB-R, KL, Rev-M)
REPLACEMENT               = False  # ▸ start-points avec remise (MB) ou sans (MB-R)
COMPARISON                = "within"  # ▸ "within" | "between" (Revusky stepwise)
# ⚠ DEUX FORMULATIONS de la fenêtre de randomisation — n'en renseigner QU'UNE :
START_WINDOW = None              # ▸ (A) par le MOMENT D'INTRODUCTION (n° de session du 1er point B),
#                                  INCLUS : (début_min, début_max), ex. (5, 10) = B commence dans {5..10}.
BASELINE_WINDOW = None           # ▸ (B) par le NOMBRE DE POINTS DE BASELINE A, INCLUS : (n_min, n_max),
#                                  ex. (4, 9) = la phase A a 4 à 9 points → B commence en session {5..10}.
#                                  Conversion : début = nb_points_A + 1, donc BASELINE_WINDOW=(4,9)
#                                  ÉQUIVAUT à START_WINDOW=(5,10). (Si les deux sont donnés : erreur.)
#                                  None / None = fenêtres AUTO (déduites de min_baseline/min_treatment).
WINDOWS      = None              # ▸ (avancé) fenêtres explicites : dict {palier:[…]} (MB), listes par position (KL/Rev-M), liste plate = pool (MB-R). Ignoré si START_WINDOW/BASELINE_WINDOW fourni.
INTERVENTION_STARTS = None       # ▸ ou dict {palier: session de début} si pas de PHASE_COL
N_PERM       = 5000

# =========================================================================== #
#  ▸▸▸ 4. SORTIES, MULTINIVEAU, VISUELS
# =========================================================================== #
OUTPUT_DIR   = None              # ▸ racine UNIQUE, arbo PAR OUTCOME : <outcome>/Analyse/permutation_test/
#                                  (permutation_test_summary.xlsx) ; <outcome>/Plot/ (descriptifs) ;
#                                  <outcome>/Plot/poolé/ (poolé inférentiel). Supersede SAVE_PATH/PLOT_PATH.
SAVE_PATH    = None              # ▸ (avancé) dossier rapport .xlsx seul (None = pas de fichier)
PLOT_PATH    = "."               # ▸ (avancé) dossier figures .png (None = pas de figure)
POOLED_TITLE = None              # ▸ titre des plots poolés (article-proof) ; None = titre auto
STYLE = None                     # ▸ style GLOBAL des figures (légende/couleurs/axes/tailles). None = défauts ;
#                                  dict de surcharges OU functions.common.plotstyle.PlotStyle.
MULTILEVEL            = True      # ▸ True → modèle piecewise multiniveau (taille d'effet b2 niveau,
#                                  b3 pente, ICC) — inférence MODÈLE, indépendante de la randomisation
TREATMENT_TRAJECTORY  = "continuous"  # ▸ "continuous" | "two_piece" (montée puis plateau, Cheng 2025)
RAMP                  = 3         # ▸ nb d'obs de montée avant plateau (two_piece)
MODEL_SELECTION       = False     # ▸ True → feuille Model Selection (AIC/BIC/LRT ; Manolov & Moeyaert 2025)
BC_SMD                = True       # ▸ taille d'effet BC-SMD (d design-comparable, Hedges-Pustejovsky-
#                                  Shadish) au récapitulatif — comparable à un Cohen's d de RCT
BC_SMD_TREND          = True       # ▸ AUSSI le BC-SMD corrigé de la tendance baseline (lme y~time+phase) :
#                                  un g qui s'effondre sous correction = effet largement porté par la tendance
PLOT_VAIOR   = True              # ▸ VAIOR par palier (Manolov & Vannest 2019) : baseline vs intervention
VAIOR_POOLED = False             # ▸ True = un VAIOR poolé ; False = une grille VAIOR par palier
PLOT_PANELS  = True              # ▸ tableau de bord MBD : figure étagée annotée (tendance baseline +
#                                  saut + Tau-U/NAP par palier) + Brinley + distribution par phase
BRINLEY_MODE = "classic"         # ▸ "classic" (1 point/patient = moyennes A/B) | "paired" (apparié k-à-k)
BOUNDS       = None              # ▸ (lo, hi) : borne l'axe outcome à l'échelle RÉELLE (ex. (0,100)) ;
#                                  tuple OU dict {outcome: (lo,hi)} ; None = auto-échelle
IMPUTE_COL   = None              # ▸ colonne booléenne marquant les points IMPUTÉS (cercles creux + légende
#                                  sur les figures étagées / multiple-baseline) ; insensible à la casse ; None = aucun
# NB : ce template est DESIGN-BASED (test de permutation). L'INFÉRENCE BAYÉSIENNE (poolée,
# par cas, comparaison de modèles) se fait dans les TEMPLATES BAYÉSIENS dédiés, sur le vrai
# postérieur inféré : template_sced_bayesian_mbd / template_sced_model_compare /
# template_sced_bayesian_percase. (Le poolé inférentiel ML b1/b2/b3 reste produit ici si MULTILEVEL=True.)
MCID         = None              # ▸ seuil clinique (MCID) pour la colonne « Répondeur » : un cas est
#                                  répondeur si direction crédible robuste à la tendance ET |diff|>=MCID.
#                                  scalaire OU dict {outcome: mcid} ; None = magnitude non requise.


def load_data():
    """Chargement : lecture de CSV_PATH si défini, sinon jeu de démo (à remplacer)."""
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(1); rows = []
    for ti, st in enumerate((5, 9, 13, 17)):             # 4 paliers, intervention décalée
        for s in range(1, 21):
            ph = "B" if s >= st else "A"
            rows.append({"tier": f"T{ti+1}", "session": s, "phase": ph,
                         "accuracy": round(float(np.clip(60 + (12 if ph == "B" else 0)
                                                         + 0.4 * s + rng.normal(0, 4), 0, 100)), 1)})
    return pd.DataFrame(rows)


def main():
    """Charge les données et délègue TOUTE l'analyse + l'écriture du rapport à
    functions/SCED_mbd_report.py (report_sced_multiple_baseline)."""
    df = load_data()
    report_sced_multiple_baseline(
        df, tier_col=TIER_COL, session_col=SESSION_COL, phase_col=PHASE_COL,
        outcome_name_col=OUTCOME_NAME_COL, value_col=VALUE_COL, outcomes=OUTCOMES,
        case_insensitive=CASE_INSENSITIVE, date_col=DATE_COL, date_dayfirst=DATE_DAYFIRST,
        group_col=GROUP_COL, include_pooled=INCLUDE_POOLED,
        phases=PHASES, baseline_phase=BASELINE_PHASE, treatment_phase=TREATMENT_PHASE,
        improvement=IMPROVEMENT, statistic=STATISTIC,
        case_randomization=CASE_RANDOMIZATION, start_point_randomization=START_POINT_RANDOMIZATION,
        replacement=REPLACEMENT, comparison=COMPARISON,
        start_window=START_WINDOW, baseline_window=BASELINE_WINDOW, windows=WINDOWS,
        intervention_starts=INTERVENTION_STARTS, n_perm=N_PERM,
        output_dir=OUTPUT_DIR, save_path=SAVE_PATH, plot_path=PLOT_PATH, pooled_title=POOLED_TITLE,
        style=STYLE,
        multilevel=MULTILEVEL, treatment_trajectory=TREATMENT_TRAJECTORY, ramp=RAMP,
        model_selection=MODEL_SELECTION, bc_smd=BC_SMD, bc_smd_trend=BC_SMD_TREND,
        plot_vaior=PLOT_VAIOR, vaior_pooled=VAIOR_POOLED, plot_panels=PLOT_PANELS,
        brinley_mode=BRINLEY_MODE, bounds=BOUNDS, impute_col=IMPUTE_COL,
        mcid=MCID)   # pas d'inférence bayésienne ici (design-based) -> templates bayésiens dédiés


if __name__ == "__main__":
    main()
