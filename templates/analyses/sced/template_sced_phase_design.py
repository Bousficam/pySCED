"""
GUIDE D'ANALYSE — SCED Plan de phases (AB / ABA / ABAB) à points de changement randomisés
==========================================================================================

Objet : un cas (ou plusieurs, répliqué) alterne des phases A (baseline) et B (traitement)
— AB, ABA, ABAB… — et les MOMENTS de changement de phase ont été randomisés dans des
fenêtres (longueur de phase minimale respectée). Plan de phases canonique (Onghena 1992 ;
Edgington & Onghena 2007).

Inférence : test de randomisation sur les moments de changement — on ré-découpe les sessions
à tous les découpages admissibles et on compare la statistique observée.
  - statistic="contrast" : phases B vs phases A (orienté) ;
  - statistic="omnibus"  : variance inter-phases (la phase a-t-elle UN effet quelconque).
+ tailles d'effet de non-recouvrement (Tau-U corrigé de la tendance, NAP) baseline vs traitement.

Confort aligné sur les autres templates : CASE_INSENSITIVE, DATE_COL (date→session), format TIDY
(chaque outcome à part), GROUP_COL (cohortes), rapport .xlsx (Données & design + Glossaire + Guide).

Toute la logique d'analyse + d'écriture du rapport consolidé stylé vit dans
functions/SCED_phase_design.py (report_sced_phase_design) ; ce template ne fait que charger
les données et appeler la fonction avec les paramètres ci-dessous.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.sced.phase_design import report_sced_phase_design

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES, COLONNES
# =========================================================================== #
# Chaque knob *_COL = le NOM EXACT de la colonne dans ton fichier (pas la valeur).
CSV_PATH    = None               # ▸ chemin du fichier .csv ou .xlsx. None = jeu de démo ABAB.
UNIT_COL    = None               # ▸ colonne SUJET. None = un seul cas ; sinon plan répliqué -> test combiné. Ex: "Participant".
SESSION_COL = "session"          # ▸ colonne de l'ORDRE TEMPOREL (entier). Calculée si tu renseignes DATE_COL.
PHASE_COL   = "phase"            # ▸ colonne de PHASE : la séquence réelle A/B/A/B… est LUE ici.
# --- Format des outcomes : TIDY (OUTCOME_NAME_COL+VALUE_COL) OU LARGE (1 colonne/outcome) ---
OUTCOME_NAME_COL = None          # ▸ (TIDY) colonne qui NOMME l'outcome. Ex: "Epreuve". None = format large.
VALUE_COL        = None          # ▸ (TIDY) colonne du SCORE numérique. Ex: "Score". None = format large.
OUTCOMES    = ["y"]              # ▸ LARGE: liste des colonnes-mesures. TIDY: None = tous, ou liste. Chacun à part.
CASE_INSENSITIVE = True          # ▸ True/False : tolère casse/espaces sur colonnes + phases.
DATE_COL    = None               # ▸ nom d'une colonne DATE -> session 1..n (par unité si UNIT_COL). None = pas de dates.
DATE_DAYFIRST = False            # ▸ True/False : True si dates JJ/MM/AAAA.
GROUP_COL   = None               # ▸ colonne de COHORTE (ex: "profil") analysée séparément (+ « (tous) »). None = une seule.

# =========================================================================== #
#  ▸▸▸ 2. PHASES, STATISTIQUE, INFÉRENCE
# =========================================================================== #
PHASES      = ("A", "B")         # ▸ étiquettes canoniques (baseline d'abord) — pour harmoniser la casse
BASELINE    = None               # ▸ étiquette(s) baseline du contraste ; None = PHASES[0]
MIN_LEN     = 3                  # ▸ longueur minimale par phase (fenêtre de randomisation)
STATISTIC   = "contrast"        # ▸ "contrast" (B vs A, MD) | "itei" (|3 derniers A − 3 premiers B|,
#                                  + puissant sous tendance, Michiels 2018) | "omnibus" (effet quelconque)
IMPROVEMENT = "increase"        # ▸ "increase" si plus haut = mieux ; sinon "decrease"
N_PERM      = 5000

OUTPUT_DIR  = None              # ▸ racine UNIQUE, arbo PAR OUTCOME : <outcome>/Analyse/phase_design/ ;
#                                 supersede SAVE_PATH
SAVE_PATH   = None              # ▸ (avancé) dossier rapport .xlsx (None = pas de fichier)


def load_data():
    if CSV_PATH:
        p = str(CSV_PATH)
        return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)
    rng = np.random.RandomState(1); rows = []
    labs = ["A", "B", "A", "B"]; bounds = [0, 6, 11, 16, 20]      # changements à 6,11,16
    for k, lb in enumerate(labs):
        for i in range(bounds[k], bounds[k + 1]):
            rows.append({"session": i + 1, "phase": lb,
                         "y": round(10 + (6 if lb == "B" else 0) + rng.normal(0, 2), 2)})
    return pd.DataFrame(rows)


def main():
    """Charge les données et délègue TOUTE l'analyse + l'écriture du rapport à
    functions/SCED_phase_design.py (report_sced_phase_design)."""
    df = load_data()
    report_sced_phase_design(
        df, unit_col=UNIT_COL, session_col=SESSION_COL, phase_col=PHASE_COL,
        outcome_name_col=OUTCOME_NAME_COL, value_col=VALUE_COL, outcomes=OUTCOMES,
        case_insensitive=CASE_INSENSITIVE, date_col=DATE_COL, date_dayfirst=DATE_DAYFIRST,
        group_col=GROUP_COL, phases=PHASES, baseline=BASELINE, min_len=MIN_LEN,
        statistic=STATISTIC, improvement=IMPROVEMENT, n_perm=N_PERM,
        output_dir=OUTPUT_DIR, save_path=SAVE_PATH)


if __name__ == "__main__":
    main()
