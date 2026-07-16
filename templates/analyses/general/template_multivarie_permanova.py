"""
ANALYSE TYPE — Profil MULTIVARIÉ en mesures répétées (PERMANOVA appariée)
=========================================================================

Scénario
--------
Plusieurs mesures continues enregistrées à chaque temps (ex. panel de constantes
vitales : FC, PAS, FR), et on veut savoir si le PROFIL GLOBAL change dans le temps
plutôt que de tester chaque variable isolément (ce qui multiplierait les tests).

Pourquoi la PERMANOVA appariée
------------------------------
- gère une réponse VECTORIELLE (là où t-test / RM-ANOVA sont mono-variables),
- sans hypothèse de normalité ni de sphéricité (test de permutation par distances),
- permutation RESTREINTE intra-sujet : le pairing est respecté.
Un suivi par outcome (test apparié + correction de Holm) localise ensuite QUELLES
variables portent l'effet, sans gonfler le risque familial.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.analysis import pipeline_paired_manova

SAVE_PATH = None


def _demo_wide():
    rng = np.random.RandomState(3)
    n = 60
    bsl = rng.normal(0, 1, (n, 3))
    pre = bsl + rng.normal(0, 1, (n, 3))
    post = bsl + np.array([1.0, 0.7, 1.3]) + rng.normal(0, 1, (n, 3))  # décalage de profil
    return pd.DataFrame({
        "id": np.arange(n),
        "FC_pre": pre[:, 0], "FC_post": post[:, 0],
        "PAS_pre": pre[:, 1], "PAS_post": post[:, 1],
        "FR_pre": pre[:, 2], "FR_post": post[:, 2],
    })


def main():
    db = _demo_wide()  # ▸ À ADAPTER

    info, perm = pipeline_paired_manova(
        db,
        id_col="id",
        outcomes=["FC", "PAS", "FR"],         # ▸ noms logiques des mesures
        outcome_maps={                         # ▸ {mesure: {temps: colonne}}
            "FC": {"pre": "FC_pre", "post": "FC_post"},
            "PAS": {"pre": "PAS_pre", "post": "PAS_post"},
            "FR": {"pre": "FR_pre", "post": "FR_post"},
        },
        n_perm=4999,
        distance="euclidean",                  # "manhattan" pour atténuer les outliers
        standardize=True,                      # z-score : aucune mesure ne domine
        save_path=SAVE_PATH, verbose=True,
    )

    print("\n== Synthèse PERMANOVA multivariée ==")
    for k in ["Outcomes", "PERMANOVA Pseudo-F (overall)", "PERMANOVA P (overall)"]:
        print(f"  - {k}: {info.get(k)}")
    print(f"  - R² (part de variance temporelle) : {perm.get('R2')}")
    print("\nRÉSUMÉ FINAL : analyse multivariée terminée.")
    print("Note : feuille 'Per-Outcome Tests' -> colonne 'P (Holm)' = quelles")
    print("mesures portent l'effet, après correction du risque multiple.")


if __name__ == "__main__":
    main()
