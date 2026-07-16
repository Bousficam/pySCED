"""
GUIDE D'ANALYSE — SCED alternant (UNIFIÉ : n-of-1 · groupe · multivarié)
=======================================================================

Un seul template pour TOUS les designs alternants. Le moteur est choisi
automatiquement (``run_sced_alternating``) selon les paramètres renseignés :

    UNIT_COL = None      + 1 outcome   → n-of-1 (un patient)
    UNIT_COL = "patient" + 1 outcome   → groupe (test stratifié, patient = bloc)
    OUTCOMES = [≥2 colonnes]           → multivarié (PERMANOVA)

Distinction clé (à ne pas confondre) :
  • OUTCOMES   = ce qui est MESURÉ (réponse). ≥2 → analyse multivariée conjointe.
  • COVARIATES = nuisance EXOGÈNE à neutraliser (NON affectée par la condition).

Données — format LONG, une ligne par séance :
    [patient] | session (entier) | condition | outcome(s) [+ covariables]

Inférence : test de RANDOMISATION (permutation de l'affectation des conditions selon le
schéma réellement tirable → exact pour le plan ; Edgington & Onghena 2007).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.sced.alternating.run import report_sced_alternating

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES ET COLONNES
# =========================================================================== #
CSV_PATH = None          # ▸ chemin du fichier .csv ou .xlsx. None = jeu de démo intégré.
# COL = NOM EXACT de chaque colonne dans ton fichier (pas la valeur).
COL = {
    "unit":      None,          # ▸ colonne PATIENT/unité. None = UN seul patient (n-of-1). Ex: "Participant".
    "session":   "session",     # ▸ colonne de l'ORDRE TEMPOREL (entier). Calculée si DATE_COL renseigné.
    "condition": "condition",   # ▸ colonne de CONDITION (= la dose/le bras alterné par séance). Ex: "condition".
    # --- Format des outcomes : TIDY (outcome_name+value) OU LARGE (1 colonne/outcome) ---
    "outcome_name": None,       # ▸ (TIDY) colonne qui NOMME l'outcome. Ex: "outcome". None = format large.
    "value":        None,       # ▸ (TIDY) colonne du SCORE numérique. Ex: "value". None = format large.
    #   Si renseignés : le tableau est dépivoté en large (1 colonne par outcome). Mettre alors
    #   OUTCOMES=None pour analyser TOUS les outcomes empilés (≥2 → multivarié), ou OUTCOMES=["x"]
    #   pour n'en garder qu'un. TRIAL_COL (ci-dessous) préserve les mesures répétées par visite.
}
CASE_INSENSITIVE = True  # ▸ harmonise casse/espaces des CONDITIONS et résout les noms de colonnes
#                            sans tenir compte de la casse ("a"/" A " -> "A" ; "Session" -> "session")
DATE_COL   = None        # ▸ si la colonne temporelle est une DATE : la convertir en session
#                            ordonnée 1..n (par patient si unit). La session DOIT être un entier
#                            d'ordre temporel ; ex. DATE_COL="date" -> remplit COL["session"].
DATE_DAYFIRST = False    # ▸ True si les dates sont au format JJ/MM/AAAA
GROUP_COL  = None        # ▸ COHORTE clinique (ex. "phase_clinique" aigu/chronique) : si renseigné,
#                            chaque groupe est analysé SÉPARÉMENT (figures + rapports dans un
#                            sous-dossier par groupe). ≠ unit (patient) et ≠ condition (dose).

# =========================================================================== #
#  ▸▸▸ 2. OUTCOME(S) vs COVARIABLE(S)  — rôles OPPOSÉS dans le modèle
# =========================================================================== #
OUTCOMES   = ["score"]   # ▸ valeur DÉJÀ AGRÉGÉE par visite. 1 → uni-/groupe ; ≥2 → MULTIVARIÉ.
#                            Ignoré si MEASURES est fourni (on agrège alors les mesures).
#                            FORMAT TIDY (COL["outcome_name"]/["value"]) : mettre None = TOUS les
#                            outcomes empilés ; ou une liste = sous-ensemble à analyser.
OUTCOME_MODE = "joint"   # ▸ si ≥2 outcomes : "joint" = analyse MULTIVARIÉE conjointe (PERMANOVA,
#                            profil commun ; suppose les outcomes mesurés ensemble) ; "separate" =
#                            chaque outcome analysé INDÉPENDAMMENT (1 rapport + figures par outcome,
#                            dans un sous-dossier). "separate" convient quand les outcomes ont des
#                            calendriers/patients différents (sinon le multivarié écarte les cas incomplets).
MEASURES   = None        # ▸ OU les mesures BRUTES par visite, SANS énumérer :
#                            • un PRÉFIXE  : "run"      (prend run_1, run_2, …)
#                            • un GLOB     : "trial_*"
#                            • une LISTE   : ["m1","m2"] (pour une sélection explicite)
#                            • None + format LONG : si plusieurs lignes partagent la même
#                              visite (session+condition[+patient]), elles sont détectées
#                              comme mesures répétées et agrégées AUTOMATIQUEMENT (rien à nommer).
AGG        = "mean"      # ▸ agrégat par visite : "mean" | "median"
LEVEL3     = False        # ▸ (mesures multiples + GROUPE) décomposition 3 NIVEAUX
#                            mesures▸visite▸patient (variance between-patient / between-visite /
#                            within-visite = bruit de mesure) + p au niveau visite. Ajoute aussi :
#                            le test de DISPERSION (la condition change-t-elle la RÉGULARITÉ
#                            intra-visite ?) et la PENTE intra-visite (fatigue/warm-up).
#                            Sans mesures multiples ou en n-of-1 : ignoré.
DISPERSION = None        # ▸ test de dispersion (régularité intra-visite) — secondaire/non-standard,
#                            OPTIONNEL : None = off (défaut) ; "sd" | "cv" pour l'activer
WITHIN_VISIT_SLOPE = False  # ▸ test de pente intra-visite (fatigue/warm-up) — secondaire, OPTIONNEL (off)
TRIAL_COL  = None        # ▸ (format LONG) colonne d'ordre de la mesure DANS la visite (1..k) pour
#                            la pente intra-visite ; None = ordre des lignes. (En colonnes : auto.)
COVARIATES = None        # ▸ nuisance EXOGÈNE à neutraliser (≠ outcome), ex. ["vigilance"].
#                            Ne JAMAIS mettre ici une variable affectée par la condition.

# =========================================================================== #
#  ▸▸▸ 3. PARAMÈTRES D'ANALYSE  
# =========================================================================== #
CONDITIONS   = ("A", "B", "C")  # ▸ référence en premier. ≥3 → test global + post-hoc + tendance
OUTCOME_TYPE = "continuous"     # ▸ "continuous" | "count" | "binary"
IMPROVEMENT  = "increase"       # ▸ "increase" si plus haut = mieux ; sinon "decrease"

# --- Provision du SCHÉMA DE RANDOMISATION (la permutation doit refléter le tirage réel) ---
MAX_CONSEC   = None             # ▸ max d'une même condition consécutive (None = pas de contrainte)
BLOCK_SIZE   = None             # ▸ randomisation PAR BLOCS (ex. len(CONDITIONS)) ; None = aucune

# --- Test de TENDANCE ORDONNÉE (dose-réponse) — si conditions = doses ordonnées ----------
TREND_TEST      = True          # ▸ ajoute la dose-réponse (si ≥3 conditions) EN PLUS du test global
CONDITION_ORDER = ("A", "B", "C")  # ▸ de la plus basse à la plus haute dose
DOSE            = (4, 5, 6)      # ▸ niveau numérique par condition (ex. s de MI) ; None = 0,1,2…
TEST_TIME_TREND = True          # ▸ TESTE la pente TEMPORELLE (session) comme covariable FIXE ajustée
#                                    pour la condition - l'axe "dérive au fil des séances" (≠ dose).
#                                    Schéma exact petit-n Huh-Jhun+W (recommend_scheme). Rapporté
#                                    EN PLUS ; l'effet de condition (randomisé) reste primaire.

TIME_COVARIATE = "auto"         # ▸ temps en COVARIABLE du modèle (ex-"detrend") : "auto"
#                                    (suit le diagnostic) | "none" | "linear" | "log".
#                                    Met la pente temporelle DANS le modèle (pas un pré-detrend).
PERM_METHOD  = "freedman-lane"  # ▸ "freedman-lane" (défaut) | "draper-stoneman" | "auto".
#                                    "auto" = choix guidé par la collinéarité condition-temps
#                                    (choose_permutation_method, Winkler 2014, seuil R2=0.15) :
#                                    Draper-Stoneman quand l'affectation est ~orthogonale au temps
#                                    (exact pour le plan), Freedman-Lane quand elle est confondue avec
#                                    la tendance. GROUP-AWARE : dès que COL["unit"] est renseigné
#                                    (analyse de groupe), le diagnostic passe automatiquement à
#                                    choose_permutation_method_group (moyenne des collinéarités
#                                    per-patient, pondérée par taille) ; sinon la forme single-unit.
STANDARDIZE  = True             # ▸ (groupe) z-score intra-patient
HIERARCHICAL = False            # ▸ (GROUPE uniquement) ajoute un MODÈLE MIXTE (intercept +
#                                    pente de condition ALÉATOIRES par patient) → effet de
#                                    population, SD de pente inter-patients, ICC, effets
#                                    rétrécis par patient. p par PERMUTATION (les ~5 patients
#                                    rendent le p asymptotique du mixte non fiable). Ignoré en
#                                    n-of-1 / multivarié. NB : pour des mesures répétées DANS la
#                                    visite (3 niveaux), voir template_sced_multilevel.py.
N_PERM       = 5000             # ▸ nombre de permutations (≥5000 conseillé)
OUTPUT_DIR   = None             # ▸ racine UNIQUE, arbo PAR OUTCOME : <outcome>/Analyse/alternating/ (xlsx) ;
#                                   <outcome>/Plot/ (figures). 'multivariate' si ≥2 outcomes. Supersede SAVE_PATH/PLOT_PATH.
SAVE_PATH    = None             # ▸ (avancé) dossier de sortie du rapport .xlsx seul (None = pas de fichier)

# =========================================================================== #
#  ▸▸▸ 4. VIZ
# =========================================================================== #

PLOT_PATH    = "."              # ▸ (avancé) dossier de la figure .png (None = pas de figure)
STYLE        = None             # ▸ style GLOBAL des figures (légende/couleurs/axes). None = défauts ; dict
#                                  de surcharges OU functions.common.plotstyle.PlotStyle.
PLOT_KIND    = "series"         # ▸ représentation : "series" (chaque valeur) | "box" | "violin"
#                                    | "mean" | "mean_sd"
PLOT_BY      = "session"        # ▸ AXE : "session" (vue temporelle standard) | "condition"
#                                    (regroupé par condition). "series" reste toujours temporel.
PLOT_CENTER  = "mean"           # ▸ tendance centrale (box/violin/mean) : "mean" | "median"
PLOT_POINTS  = True             # ▸ superposer chaque valeur (nuage) sur box/violin/mean
PLOT_COND_LINE = True           # ▸ tracer la ligne horizontale de moyenne/médiane PAR CONDITION
#                                    (True/False) — niveau de référence de chaque condition
# --- Aides visuelles de Manolov (complémentaires, non substitut au test) ---
PLOT_VAIOR   = False            # ▸ VAIOR (Manolov & Vannest 2019) : tendance Theil-Sen + bande ±MAD
#                                    de la condition de référence, projetée ; points comparés colorés
#                                    (vert=hors bande / jaune=hors tendance / rouge=ni l'un ni l'autre).
#                                    OFF par défaut en ALTERNANT : VAIOR suppose une trajectoire de
#                                    phase soutenue (AB/MBD) ; en alternance rapide les sessions sont
#                                    entrelacées, la tendance/bande mêle les conditions et n'est pas
#                                    adaptée. Mettre True seulement si pertinent. (Brinley reste actif.)
VAIOR_REF    = "A"              # ▸ condition de référence (baseline / dose la plus basse)
VAIOR_POOLED = False            # ▸ groupe : False = une figure VAIOR PAR PATIENT (mode canonique
#                                    single-case) ; True = vue poolée agrégée (bande ±MAD mêle
#                                    variabilité intra- et inter-patients — à interpréter prudemment)
PLOT_BRINLEY = True             # ▸ Brinley plot vs diagonale y=x (au-dessus = condition > référence)
BRINLEY_MODE = "classic"        # ▸ "classic" (1 point/patient = moyennes) | "paired" (apparié k-à-k, Manolov 2021)
BOUNDS       = None             # ▸ (lo, hi) : borne l'axe outcome à l'échelle RÉELLE (ex. (0,100)) ; None = AUTO
Y_PERCENT    = False            # ▸ True = affiche l'axe outcome en POURCENTAGE (données 0–1 -> 40%..100%)
ROW_HEIGHT   = 1.7              # ▸ hauteur (pouces) d'une rangée de série dans le tableau de bord groupe
PLOT_PANELS  = True             # ▸ FUSIONNE série + VAIOR + Brinley en UNE figure (tableau de bord)
#                                    + une grille VAIOR par patient (groupe). False = figures séparées.


def load_data():
    """Chargement : CSV_PATH si défini, sinon jeu de démo (groupe, MI 4/5/6 s)."""
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(0); rows = []
    for u in range(5):
        base = rng.normal(0, 3); s = 0
        for _ in range(8):
            for c in rng.permutation(["A", "B", "C"]):
                s += 1
                rows.append({"patient": f"P{u+1}", "session": s, "condition": c,
                             "score": base + {"A": 0, "B": 2, "C": 4}[c] + rng.normal(20, 2)})
    df = pd.DataFrame(rows)
    return df if COL["unit"] else df[df["patient"] == "P1"].drop(columns="patient")




def main():
    """Charge les données et délègue TOUTE l'orchestration + analyse + rapport à
    functions/SCED_alternating_run.py (report_sced_alternating)."""
    df = load_data()
    report_sced_alternating(
        df, unit_col=COL["unit"], session_col=COL["session"], condition_col=COL["condition"],
        outcome_name_col=COL.get("outcome_name"), value_col=COL.get("value"),
        outcomes=OUTCOMES, outcome_mode=OUTCOME_MODE, case_insensitive=CASE_INSENSITIVE,
        date_col=DATE_COL, date_dayfirst=DATE_DAYFIRST, group_col=GROUP_COL,
        measures=MEASURES, agg=AGG, level3=LEVEL3, dispersion=DISPERSION,
        within_visit_slope=WITHIN_VISIT_SLOPE, trial_col=TRIAL_COL, covariates=COVARIATES,
        conditions=CONDITIONS, outcome_type=OUTCOME_TYPE, improvement=IMPROVEMENT,
        max_consecutive=MAX_CONSEC, block_size=BLOCK_SIZE, trend_test=TREND_TEST,
        test_time_trend=TEST_TIME_TREND,
        condition_order=CONDITION_ORDER, dose=DOSE, time_covariate=TIME_COVARIATE,
        perm_method=PERM_METHOD, standardize=STANDARDIZE, hierarchical=HIERARCHICAL,
        n_perm=N_PERM, output_dir=OUTPUT_DIR, save_path=SAVE_PATH, plot_path=PLOT_PATH,
        plot_kind=PLOT_KIND, plot_by=PLOT_BY, plot_center=PLOT_CENTER, plot_points=PLOT_POINTS,
        plot_cond_line=PLOT_COND_LINE, plot_vaior=PLOT_VAIOR, vaior_ref=VAIOR_REF,
        vaior_pooled=VAIOR_POOLED, plot_brinley=PLOT_BRINLEY, brinley_mode=BRINLEY_MODE,
        bounds=BOUNDS, y_percent=Y_PERCENT, row_height=ROW_HEIGHT, plot_panels=PLOT_PANELS, style=STYLE)


if __name__ == "__main__":
    main()
