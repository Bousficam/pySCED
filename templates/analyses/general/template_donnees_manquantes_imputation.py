"""
ANALYSE TYPE — Données MANQUANTES, complétude & imputation (MICE/LOCF)
=====================================================================

Scénario
--------
Suivi longitudinal avec des perdus de vue / visites manquées. Avant de conclure,
il faut (1) SAVOIR combien on perd en complete-case, (2) choisir une stratégie.

Ce que montre ce template
-------------------------
1) missingness_report : complétude par temps (observé / manquant / % écarté) ;
2) trois stratégies comparées sur la même cohorte :
   - complete-case (impute="none") : valide si MCAR, sinon biaisé ;
   - MICE (impute="mice") : référence sous MAR — le rapport PRÉCISE qu'une
     imputation a été réalisée et signale la limite (imputation unique, pas de
     règles de Rubin) ;
   - LOCF (impute="locf") : simple mais biaisé (signalé comme tel).
Rappel : le modèle mixte reste valide sous MAR SANS imputation — comparer.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.analysis import missingness_report, pipeline_paired_ols

SAVE_PATH = None


def _demo():
    rng = np.random.RandomState(0)
    n = 80
    visites = ["M0", "M3", "M6", "M12"]
    rows = []
    for i in range(n):
        b = rng.normal(10, 2)
        for k, v in enumerate(visites):
            rows.append({"id": i, "t": v, "y": b + 2 * np.log1p(k) + rng.normal(0, 1)})
    df = pd.DataFrame(rows)
    # 18% de manquants (perdus de vue plus fréquents en fin de suivi)
    p = np.where(df["t"].isin(["M6", "M12"]), 0.28, 0.06)
    df.loc[rng.rand(len(df)) < p, "y"] = np.nan
    return df


def main():
    db = _demo()  # ▸ À ADAPTER

    print("=" * 70)
    print("ÉTAPE 1 — Rapport de complétude")
    print("=" * 70)
    info, tbl = missingness_report(db, "id", "t", "y")
    print(tbl.to_string(index=False))
    print("\nRésumé :", {k: info[k] for k in info if k != "Missingness warning"})
    if "Missingness warning" in info:
        print("⚠", info["Missingness warning"])

    print("\n" + "=" * 70)
    print("ÉTAPE 2 — Trois stratégies comparées (effet du temps)")
    print("=" * 70)
    for strat in ["none", "mice", "locf"]:
        i, m = pipeline_paired_ols(db, id_col="id", outcome="y", time_col="t",
                                   impute=strat, compare_ar1=False,
                                   save_path=SAVE_PATH, verbose=False)
        note = i.get("Imputation note", "complete-case (aucune imputation)")
        n_sub = i.get("Number of subjects")
        print(f"\n  impute={strat:5s} | sujets analysés={n_sub} | {note[:70]}")
        if m is not None:
            tt = [t for t in m.pvalues.index if "t" in t.lower() and t != "Intercept"]
            ps = ", ".join(f"{t}:p={m.pvalues[t]:.3g}" for t in tt[:3])
            print(f"             effets temps -> {ps}")

    print("\nRÉSUMÉ FINAL : complétude rapportée + imputation exposée (MICE/LOCF).")
    print("Le rapport Excel précise toujours si une imputation a été réalisée.")


if __name__ == "__main__":
    main()
