"""
GUIDE DE PLANIFICATION — Puissance A PRIORI (SCED alternant / N-of-1)
====================================================================

Objet : dimensionner une étude AVANT de collecter les données. Aucune donnée requise :
les hypothèses (effet visé, bruit, nombre de séances/unités) sont renseignées dans les
zones ▸. La puissance d'un test de randomisation n'a pas de formule fermée : elle est
estimée par SIMULATION en réutilisant le test réel.

Deux sorties :
  - puissance a priori pour un effet visé, selon le nombre d'unités ;
  - MDES (plus petit effet détectable à la puissance cible) selon le nombre d'unités.

Pour un plan de PHASES (AB)ᵏ (AB, ABAB…) répliqué sur m cas, une **puissance fermée**
(Hedges, Shadish & Natesan Batley 2022) est disponible en bas de fichier (section B) :
plus rapide, sur l'effet design-comparable δ (= d de Cohen), + vérification des standards
What Works Clearinghouse.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from functions.sced.power import (power_sced_alternating, mdes_sced_alternating,
                                  power_abk_design, mdes_abk_design, wwc_design_check)

# =========================================================================== #
#  ▸▸▸ HYPOTHÈSES DE PLANIFICATION  (critère de choix en commentaire)
# =========================================================================== #
CONDITIONS    = ("A", "B")      # ▸ référence en premier ; ("A","B","C") pour ≥3 conditions
EFFECT        = 4.0             # ▸ effet visé en unités brutes (écart cible entre conditions)
SD            = 6.0             # ▸ écart-type résiduel attendu (bruit séance à séance)
N_SESSIONS    = 18             # ▸ nombre de séances par unité (divisible par le nb de conditions)
N_UNITS_GRID  = (1, 3, 5, 8)   # ▸ tailles de groupe à comparer (1 = N-of-1)
LEARNING      = 0.0            # ▸ pente d'apprentissage attendue par séance (0 si aucune)
DETREND       = "none"         # ▸ "none" | "linear" | "log" (mettre "linear"/"log" si LEARNING≠0)
MAX_CONSEC    = 2             # ▸ contrainte du schéma (max d'une même condition d'affilée), ou None
TARGET_POWER  = 0.80          # ▸ puissance cible (pour le MDES)
ALPHA         = 0.05          # ▸ seuil de significativité
N_SIMS        = 300           # ▸ nb de simulations (↑ = estimation plus stable, plus lent)
N_PERM        = 300           # ▸ nb de permutations par test


def main():
    common = dict(sd=SD, n_sessions=N_SESSIONS, conditions=CONDITIONS, learning=LEARNING,
                  detrend=DETREND, max_consecutive=MAX_CONSEC, alpha=ALPHA,
                  n_sims=N_SIMS, n_perm=N_PERM, random_state=0)

    print(f"Effet visé = {EFFECT} (d = {EFFECT / SD:.2f}), sd = {SD}, "
          f"{N_SESSIONS} séances, conditions {CONDITIONS}\n")

    print("PUISSANCE A PRIORI selon le nombre d'unités :")
    for nu in N_UNITS_GRID:
        r = power_sced_alternating(effect=EFFECT, n_units=nu, **common)
        flag = "OK" if r["power"] >= TARGET_POWER else "sous-puissé"
        print(f"  n_units={nu:>2} : puissance = {r['power']:.3f}  (±{r['mc_se']:.3f})  [{flag}]")

    print(f"\nMDES (plus petit effet détectable à {int(TARGET_POWER*100)}%) :")
    for nu in N_UNITS_GRID:
        m = mdes_sced_alternating(n_units=nu, target_power=TARGET_POWER, **common)
        print(f"  n_units={nu:>2} : d = {m['mdes_cohens_d']}  (brut = {m['mdes_raw']})")

    print("\n→ Choisir le nombre d'unités/séances qui atteint la puissance cible pour "
          "l'effet jugé pertinent ; le MDES indique l'effet limite que le design peut détecter.")


# =========================================================================== #
#  ▸▸▸ SECTION B — PLAN DE PHASES (AB)ᵏ : puissance FERMÉE (Hedges 2022)
# =========================================================================== #
ABK_DELTA = 0.75      # ▸ effet visé en δ (= d de Cohen) ; 0.75+ courant en SCED
ABK_K     = 2         # ▸ nombre de paires AB (1 = AB, 2 = ABAB ; WWC veut k≥2)
ABK_N     = 5         # ▸ observations par phase (WWC : ≥3 avec réserves, ≥5 sans)
ABK_M_GRID = (2, 3, 4, 5, 6)   # ▸ nombre de cas à comparer (la puissance exige m≥2)
ABK_PHI   = 0.5       # ▸ autocorrélation supposée (défaut conservateur 0.5)
ABK_RHO   = 0.5       # ▸ ICC = part de variance inter-cas (défaut conservateur 0.5)


def main_abk():
    print(f"\n=== PLAN (AB)^{ABK_K} — puissance fermée (Hedges 2022) ===")
    print(f"δ = {ABK_DELTA} (d de Cohen), n = {ABK_N}/phase, φ = {ABK_PHI}, ρ = {ABK_RHO}\n")
    print("Puissance selon le nombre de cas m :")
    for m in ABK_M_GRID:
        r = power_abk_design(delta=ABK_DELTA, k=ABK_K, n=ABK_N, m=m, phi=ABK_PHI,
                             rho=ABK_RHO, alpha=ALPHA)
        flag = "OK" if r["power"] >= TARGET_POWER else "sous-puissé"
        print(f"  m={m} : puissance = {r['power']:.3f}  [{flag}]")
    md = mdes_abk_design(k=ABK_K, n=ABK_N, m=max(ABK_M_GRID), phi=ABK_PHI, rho=ABK_RHO,
                         alpha=ALPHA, target_power=TARGET_POWER)
    print(f"\nMDES à {int(TARGET_POWER*100)}% (m={max(ABK_M_GRID)}) : δ = {md['mdes_delta']}")

    print("\nStandards What Works Clearinghouse :")
    summ, sheet = wwc_design_check(k=ABK_K, n_per_phase=ABK_N, m=max(ABK_M_GRID),
                                   design="phase")
    for _, row in sheet.iterrows():
        print(f"  [{row['OK']}] {row['Critère']} = {row['Valeur']} — {row['Interprétation']}")
    verdict = ("sans réserves" if summ["meets_without_reservations"]
               else ("avec réserves" if summ["meets_with_reservations"] else "hors standard"))
    print(f"→ Statut WWC : {verdict}.")


if __name__ == "__main__":
    main()
    main_abk()
