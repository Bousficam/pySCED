"""
ANALYSE TYPE — Essai randomisé contrôlé, mesure CONTINUE pré/post
=================================================================

Scénario
--------
Deux bras (traitement vs contrôle), un score continu mesuré AVANT et APRÈS.
Question clinique : « le traitement améliore-t-il le score PLUS que le contrôle ? »
-> c'est l'INTERACTION temps × bras qui répond, pas l'effet temps brut.

Ce que fait ce template
-----------------------
- reshape large -> long,
- descriptif par temps ET par bras (+ Δ intra-sujet),
- tests appariés par bras (t/Wilcoxon) + PERMANOVA appariée,
- modèle mixte `score ~ temps * bras` (intercept aléatoire / sujet) : le terme
  d'interaction est la réponse formelle,
- diagnostics (normalité résidus, autocorrélation).

À lancer tel quel (DEMO synthétique) ; pour une étude réelle, voir « ▸ À ADAPTER ».
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.analysis import pipeline_paired_ols

SAVE_PATH = None  # ex. "resultats/" pour exporter les .xlsx


def _demo():
    rng = np.random.RandomState(0)
    n = 80
    base = rng.normal(50, 8, n)
    arm = rng.randint(0, 2, n)                      # 0 = contrôle, 1 = traitement
    # effet temps commun (+3) ; le traitement ajoute +5 -> interaction
    post = base + 3 + 5 * arm + rng.normal(0, 4, n)
    return pd.DataFrame({"patient": np.arange(n), "bras": arm,
                         "score_J0": base, "score_J90": post})


def main():
    db = _demo()  # ▸ À ADAPTER : le DataFrame réel

    info, model = pipeline_paired_ols(
        db,
        id_col="patient",                          # ▸ identifiant sujet
        outcome="score",                           # ▸ nom logique de l'outcome
        time_map={"J0": "score_J0", "J90": "score_J90"},  # ▸ colonnes par temps
        group_col="bras",                          # ▸ facteur inter-sujet (None si 1 bras)
        save_path=SAVE_PATH, verbose=True,
    )

    print("\n== Synthèse essai randomisé pré/post ==")
    for k in ["Number of subjects", "Balanced design", "Model", "Converged"]:
        print(f"  - {k}: {info.get(k)}")
    if model is not None:
        print("\nEffets fixes (cherche le terme d'INTERACTION temps:bras) :")
        print(model.summary().tables[1])
        inter = [t for t in model.params.index if ":" in t]
        if inter:
            t = inter[0]
            print(f"\nInteraction {t}: β={model.params[t]:.3f}, p={model.pvalues[t]:.4f}")
            print("  -> p<0.05 = le traitement change l'évolution différemment du contrôle.")
    print("\nRÉSUMÉ FINAL : analyse essai randomisé pré/post terminée.")


if __name__ == "__main__":
    main()
