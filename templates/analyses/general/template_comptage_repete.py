"""
ANALYSE TYPE — Outcome de COMPTAGE répété + TAILLES D'EFFET
===========================================================

Scénario
--------
Un nombre d'événements compté à chaque temps (crises, hospitalisations, lésions),
et/ou un score continu pour lequel on veut la taille d'effet, pas seulement la p.

Ce que montre ce template
-------------------------
1) pipeline_paired_count : GEE Poisson/binomial négatif (gère la surdispersion
   automatiquement), effets exportés en RATE RATIO (IRR) ;
2) paired_effect_sizes : Cohen's dz / Hedges' g / rank-biserial (2 temps) ou
   η²/η²p (>2 temps), à reporter à côté des p-values.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import numpy as np
import pandas as pd

from functions.general.longitudinal.effects import pipeline_paired_count, paired_effect_sizes

SAVE_PATH = None


def _demo():
    rng = np.random.RandomState(0)
    n = 80
    # comptage surdispersé (mélange Poisson-Gamma) qui baisse après traitement
    lam = rng.gamma(2.0, 2.0, n)
    pre = rng.poisson(lam)
    post = rng.poisson(lam * 0.5)
    counts = pd.concat([
        pd.DataFrame({"id": np.arange(n), "t": "pre", "crises": pre}),
        pd.DataFrame({"id": np.arange(n), "t": "post", "crises": post}),
    ], ignore_index=True)
    # un score continu apparié pour la démo des tailles d'effet
    s_pre = rng.normal(50, 8, n)
    s_post = s_pre + 5 + rng.normal(0, 4, n)
    scores = pd.concat([
        pd.DataFrame({"id": np.arange(n), "t": "pre", "score": s_pre}),
        pd.DataFrame({"id": np.arange(n), "t": "post", "score": s_post}),
    ], ignore_index=True)
    return counts, scores


def main():
    counts, scores = _demo()  # ▸ À ADAPTER

    print("=" * 70)
    print("CAS COMPTAGE — GEE Poisson/NB, rate ratios (IRR)")
    print("=" * 70)
    info, model = pipeline_paired_count(
        counts, id_col="id", outcome="crises", time_col="t",
        family="auto",                 # auto-bascule NB si surdispersion
        save_path=SAVE_PATH, verbose=True,
    )
    for k in ["Family", "Poisson Pearson dispersion (>1.5 => overdispersed)",
              "NB alpha (dispersion)", "Converged"]:
        if k in info:
            print(f"  - {k}: {info[k]}")
    if model is not None:
        print("\nRate ratios (exp β) :")
        print(model.summary())

    print("\n" + "=" * 70)
    print("CAS TAILLES D'EFFET — score continu apparié")
    print("=" * 70)
    es = paired_effect_sizes(scores, "id", "t", "score")
    print(es.to_string(index=False))

    print("\nRÉSUMÉ FINAL : comptage + tailles d'effet terminés.")
    print("Repère : IRR<1 = baisse du taux d'événements ; dz>0.8 = effet fort.")


if __name__ == "__main__":
    main()
