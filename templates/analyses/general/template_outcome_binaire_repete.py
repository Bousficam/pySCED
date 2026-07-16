"""
ANALYSE TYPE — Outcome BINAIRE répété (succès/échec, présence/absence)
======================================================================

Scénario A (pré/post) : un statut binaire mesuré avant et après une intervention
  (ex. « symptôme présent »). Question : la proportion change-t-elle ? -> McNemar.

Scénario B (>2 temps) : le même statut à plusieurs visites -> Cochran's Q.

Dans les deux cas, un GEE logistique (corrélation intra-sujet, clusterisé sur le
sujet) fournit l'odds ratio d'évolution, plus robuste qu'un simple test.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.analysis import pipeline_paired_logit

SAVE_PATH = None


def _demo_prepost():
    rng = np.random.RandomState(2)
    n = 90
    pre = rng.binomial(1, 0.6, n)                    # symptôme présent au départ
    # l'intervention fait disparaître le symptôme chez beaucoup de positifs
    post = np.where(pre == 1, rng.binomial(1, 0.35, n), rng.binomial(1, 0.1, n))
    return pd.concat([
        pd.DataFrame({"sujet": np.arange(n), "temps": "avant", "symptome": pre}),
        pd.DataFrame({"sujet": np.arange(n), "temps": "apres", "symptome": post}),
    ], ignore_index=True)


def main():
    db = _demo_prepost()  # ▸ À ADAPTER (format long : sujet, temps, statut 0/1)

    info, model = pipeline_paired_logit(
        db,
        id_col="sujet",
        outcome="symptome",
        time_col="temps",
        cov_struct="exchangeable",   # "ar1" si suivi temporel rapproché à >2 temps
        save_path=SAVE_PATH, verbose=True,
    )

    print("\n== Synthèse outcome binaire répété ==")
    for k in ["Number of timepoints", "Number of subjects",
              "Model", "Working correlation"]:
        print(f"  - {k}: {info.get(k)}")
    if model is not None:
        print("\nOdds ratios (effet du temps sur la cote du symptôme) :")
        print(model.summary())
    print("\nRÉSUMÉ FINAL : analyse outcome binaire répété terminée.")
    print("Note : feuille 'Paired Tests' -> McNemar (2 temps) ou Cochran-Q (>2).")


if __name__ == "__main__":
    main()
