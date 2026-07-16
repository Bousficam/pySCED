"""
TEMPLATE LONGITUDINAL / APPARIÉ — guide pas-à-pas (user-friendly)
=================================================================

Objectif
--------
Partir de données pré/post (2 temps) ou à >2 temps, avec ou sans groupe, et
obtenir EN UN APPEL toutes les étapes d'une analyse appariée + un rapport Excel
prêt à l'emploi (descriptif par temps, tests appariés, modèle mixte/GEE).

Une pipeline par type d'outcome :
    - pipeline_paired_ols   -> outcome CONTINU (score, mesure...)
    - pipeline_paired_logit -> outcome BINAIRE (succès/échec, présent/absent)

Deux formats d'entrée acceptés indifféremment :
    - LARGE (wide) : une ligne par sujet, une colonne par temps
        -> on passe `time_map={"pre": "col_pre", "post": "col_post"}`
    - LONG : une ligne par sujet x temps (colonnes sujet, temps, outcome)
        -> on passe `time_col="nom_colonne_temps"`

Comment l'utiliser
------------------
1. Lance-le tel quel : `python templates/template_longitudinal.py`
   Il tourne sur un MINI JEU DE DÉMO synthétique (pré/post continu à 2 bras +
   pré/post binaire) et affiche les sorties, sans fichier réel.
2. Pour une étude réelle : mettre DEMO = False et brancher le `db` réel + les noms
   de colonnes dans les sections balisées « ▸ À ADAPTER ».
"""
# Permet de lancer le script directement depuis n'importe quel dossier
# (ajoute la racine du dépôt à sys.path).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.analysis import (
    pipeline_paired_ols,
    pipeline_paired_logit,
    pipeline_paired_manova,
)

DEMO = True
SAVE_PATH = None  # ex. "resultats/" pour écrire les .xlsx ; None = pas d'export


def _demo_data():
    """Jeu synthétique : pré/post continu (2 bras) et pré/post binaire."""
    rng = np.random.RandomState(0)
    n = 60
    base = rng.normal(50, 8, n)
    arm = rng.randint(0, 2, n)
    post = base + 3 + 4 * arm + rng.normal(0, 3, n)  # le bras 1 progresse plus
    continuous_wide = pd.DataFrame({
        "subject": np.arange(n), "arm": arm,
        "score_pre": base, "score_post": post,
    })

    pre = rng.binomial(1, 0.3, n)
    post_b = np.where(pre == 1, 1, rng.binomial(1, 0.55, n))
    binary_long = pd.concat([
        pd.DataFrame({"subject": np.arange(n), "visit": "pre", "success": pre}),
        pd.DataFrame({"subject": np.arange(n), "visit": "post", "success": post_b}),
    ], ignore_index=True)

    # Multivarié : 3 mesures (fréq. cardiaque, PAS, fréq. resp.) à 2 temps.
    bsl = rng.normal(0, 1, (n, 3))
    pre_m = bsl + rng.normal(0, 1, (n, 3))
    post_m = bsl + np.array([1.0, 0.8, 1.2]) + rng.normal(0, 1, (n, 3))
    multi_wide = pd.DataFrame({
        "subject": np.arange(n),
        "hr_pre": pre_m[:, 0], "hr_post": post_m[:, 0],
        "sbp_pre": pre_m[:, 1], "sbp_post": post_m[:, 1],
        "rr_pre": pre_m[:, 2], "rr_post": post_m[:, 2],
    })
    return continuous_wide, binary_long, multi_wide


def main():
    if DEMO:
        continuous_wide, binary_long, multi_wide = _demo_data()
    else:
        # ▸ À ADAPTER : charger le DataFrame réel (cf. template_preprocessing.py)
        raise SystemExit("Mettre DEMO=False et brancher les données réelles ici.")

    print("=" * 70)
    print("CAS 1 — Outcome CONTINU, format LARGE, 2 temps, comparaison de 2 bras")
    print("=" * 70)
    # ▸ À ADAPTER : id_col / outcome / time_map / group_col
    info_c, model_c = pipeline_paired_ols(
        continuous_wide,
        id_col="subject",
        outcome="score",
        time_map={"pre": "score_pre", "post": "score_post"},
        group_col="arm",            # mettre None si un seul groupe
        save_path=SAVE_PATH,
        verbose=True,
    )
    print("\nInfo modèle (extrait) :")
    for k in ["Outcome type", "Number of timepoints", "Number of subjects",
              "Balanced design", "Model", "Converged"]:
        print(f"  - {k}: {info_c.get(k)}")
    if model_c is not None:
        print("\nEffets fixes (β) du modèle mixte :")
        print(model_c.summary().tables[1])

    print("\n" + "=" * 70)
    print("CAS 2 — Outcome BINAIRE, format LONG, 2 temps (McNemar + GEE logit)")
    print("=" * 70)
    # ▸ À ADAPTER : id_col / outcome / time_col
    info_b, model_b = pipeline_paired_logit(
        binary_long,
        id_col="subject",
        outcome="success",
        time_col="visit",
        save_path=SAVE_PATH,
        verbose=True,
    )
    print("\nInfo modèle (extrait) :")
    for k in ["Outcome type", "Number of timepoints", "Number of subjects",
              "Model", "Working correlation"]:
        print(f"  - {k}: {info_b.get(k)}")

    print("\n" + "=" * 70)
    print("CAS 3 — MULTIVARIÉ, mesures répétées 2 temps (PERMANOVA appariée)")
    print("=" * 70)
    # ▸ À ADAPTER : id_col / outcome_maps {outcome: {temps: colonne}}
    info_m, perm = pipeline_paired_manova(
        multi_wide,
        id_col="subject",
        outcomes=["hr", "sbp", "rr"],
        outcome_maps={
            "hr": {"pre": "hr_pre", "post": "hr_post"},
            "sbp": {"pre": "sbp_pre", "post": "sbp_post"},
            "rr": {"pre": "rr_pre", "post": "rr_post"},
        },
        n_perm=999,
        save_path=SAVE_PATH,
        verbose=True,
    )
    print("\nInfo PERMANOVA (extrait) :")
    for k in ["Outcome type", "Outcomes", "Omnibus test",
              "PERMANOVA Pseudo-F (overall)", "PERMANOVA P (overall)"]:
        print(f"  - {k}: {info_m.get(k)}")

    print("\nRÉSUMÉ FINAL : les trois pipelines ont tourné de bout en bout.")
    if SAVE_PATH:
        print(f"Rapports Excel écrits sous : {SAVE_PATH}")


if __name__ == "__main__":
    main()
