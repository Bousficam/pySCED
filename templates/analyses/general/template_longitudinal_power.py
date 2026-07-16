"""
GUIDE — PUISSANCE par simulation pour une COURBE DE CROISSANCE longitudinale
============================================================================

Estime la puissance de détecter une PENTE de temps (ou une interaction groupe×temps) dans un
modèle mixte linéaire, par simulation de Monte-Carlo : on génère ``N_SIM`` jeux conformes au
design (``N_SUBJ`` sujets × ``N_TIMES`` temps), on ajuste le modèle mixte (random intercept +
pente) et on compte la fraction où l'effet ciblé est significatif (IC de Wald excluant 0).

Sert à dimensionner une étude (combien de sujets / de temps ?) en faisant varier les paramètres.
Logique : functions/Longitudinal_growth.power_growth.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import pandas as pd
from functions.general.longitudinal.growth import power_growth

# =========================================================================== #
#  ▸▸▸ DESIGN & EFFET
# =========================================================================== #
N_SUBJ      = 30        # ▸ nombre de sujets
N_TIMES     = 5         # ▸ nombre de temps (0..N_TIMES-1)
BETA_TIME   = 0.4       # ▸ pente de temps à détecter (unité outcome / temps)
BY_EFFECT   = None      # ▸ effet d'interaction groupe×temps à détecter (None = pente de temps simple)
SD_RESID    = 1.0       # ▸ écart-type résiduel
SD_INTERCEPT = 1.0      # ▸ SD des intercepts aléatoires (différences de niveau entre sujets)
SD_SLOPE    = 0.3       # ▸ SD des pentes aléatoires (différences de changement entre sujets)
N_SIM       = 500       # ▸ nombre de simulations (≥500 conseillé)
ALPHA       = 0.05
SEED        = 0

# Balayage optionnel (None = un seul scénario ci-dessus)
SWEEP_N_SUBJ = None     # ▸ ex. [20, 30, 40, 60] -> table de puissance vs N_SUBJ


def main():
    if SWEEP_N_SUBJ:
        rows = []
        for n in SWEEP_N_SUBJ:
            r = power_growth(n_subj=n, n_times=N_TIMES, beta_time=BETA_TIME, by_effect=BY_EFFECT,
                             sd_resid=SD_RESID, sd_intercept=SD_INTERCEPT, sd_slope=SD_SLOPE,
                             n_sim=N_SIM, alpha=ALPHA, seed=SEED)
            rows.append({"N_SUBJ": n, "puissance": r["power"], "effet": r["target_effect"]})
        print(pd.DataFrame(rows).to_string(index=False))
    else:
        r = power_growth(n_subj=N_SUBJ, n_times=N_TIMES, beta_time=BETA_TIME, by_effect=BY_EFFECT,
                         sd_resid=SD_RESID, sd_intercept=SD_INTERCEPT, sd_slope=SD_SLOPE,
                         n_sim=N_SIM, alpha=ALPHA, seed=SEED)
        print(f"Puissance ({r['target_effect']}, beta_time={BETA_TIME}, N_SUBJ={N_SUBJ}, "
              f"N_TIMES={N_TIMES}) = {r['power']}  [{r['n_sim_ok']} simulations valides]")


if __name__ == "__main__":
    main()
