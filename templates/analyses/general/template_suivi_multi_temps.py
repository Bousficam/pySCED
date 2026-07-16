"""
ANALYSE TYPE — Suivi de cohorte à PLUSIEURS temps (outcome continu)
===================================================================

Scénario
--------
Une cohorte suivie à 4 visites (M0, M3, M6, M12), un biomarqueur continu.
Question : « le biomarqueur évolue-t-il significativement au cours du suivi ? »

Particularités > 2 temps gérées automatiquement
-----------------------------------------------
- ANOVA à mesures répétées AVEC correction de sphéricité Greenhouse-Geisser,
- Friedman (non paramétrique) + PERMANOVA appariée,
- modèle mixte avec le temps en FACTEUR (`time_as="factor"`) pour capturer une
  évolution non linéaire ; passe `time_as="numeric"` pour tester une pente/tendance,
- diagnostic d'AUTOCORRÉLATION intra-sujet (Durbin-Watson) : informatif ici car
  plus de 2 temps.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.analysis import pipeline_paired_ols

SAVE_PATH = None


def _demo_long():
    rng = np.random.RandomState(1)
    n = 50
    visites = ["M0", "M3", "M6", "M12"]
    rows = []
    for i in range(n):
        base = rng.normal(10, 2)
        for k, v in enumerate(visites):
            # progression non linéaire (plateau) + corrélation intra-sujet
            val = base + 3 * np.log1p(k) + rng.normal(0, 1)
            rows.append({"id": i, "visite": v, "biomarqueur": val})
    df = pd.DataFrame(rows)
    df["visite"] = pd.Categorical(df["visite"], categories=visites, ordered=True)
    return df


def main():
    db = _demo_long()  # ▸ À ADAPTER (format LONG : une ligne par sujet x visite)

    info, model = pipeline_paired_ols(
        db,
        id_col="id",
        outcome="biomarqueur",
        time_col="visite",        # ▸ format long -> on donne la colonne de temps
        time_as="factor",         # "numeric" pour tester une tendance linéaire
        save_path=SAVE_PATH, verbose=True,
    )

    print("\n== Synthèse suivi multi-temps ==")
    for k in ["Number of timepoints", "Number of subjects",
              "Residual normality p (>0.05 = ok)",
              "Within-subject Durbin-Watson (~2 = no autocorr.)"]:
        print(f"  - {k}: {info.get(k)}")
    if model is not None:
        print("\nEffets fixes du modèle mixte :")
        print(model.summary().tables[1])
    print("\nRÉSUMÉ FINAL : analyse de suivi multi-temps terminée.")
    print("Astuce : ouvre la feuille 'Paired Tests' -> compare RM-ANOVA brute vs")
    print("Greenhouse-Geisser ; si ε << 1, se fier à la version corrigée / au mixte.")


if __name__ == "__main__":
    main()
