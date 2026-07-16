# Organigrammes de construction — pipelines SCED

Comment chaque modèle/test se construit à partir des paramètres choisis. `[PARAM]` = zone `▸`.

---

## sced_alternating
```
DATA long (session, condition, outcome [, UNIT_COL, GROUP_COL, COVARIATES])
   │  prep : harmonise conditions · DATE_COL→sessions · detrend (none/linear/log/AUTO via diagnostic)
   │
   ├─ choix du MOTEUR (auto) ──────────────────────────────────────────────
   │      OUTCOMES = [>=2]        → PERMANOVA (réponse vectorielle, distances)
   │      UNIT_COL = None         → N-of-1   (permutation des conditions/sessions, 1 patient)
   │      UNIT_COL = "patient"    → groupe   (permutation STRATIFIÉE intra-patient)
   │      GROUP_COL set           → boucle par cohorte + (tous)
   │
   ├─ statistique = F partielle de CONDITION ajustée du temps   (>=3 conditions → omnibus)
   ├─ schéma de permutation : Freedman-Lane (défaut) | Draper-Stoneman | AUTO (colinéarité)
   │
   ▼  p = (1 + #stat>=obs) / (1 + B)        + Tau-U / NAP / Hedges g
      [+ modèle mixte si HIERARCHICAL]   [+ 3e niveau si LEVEL3]
```

## sced_phase_design
```
DATA (1 cas, ou répliqué via UNIT_COL) → prep [casse · DATE_COL→sessions · unstack] → (cohorte × outcome)
   │  fenêtres de changement de phase admissibles (longueur de phase minimale MIN_LEN respectée)
   │
   ├─ STATISTIC = "contrast"  → moyenne(B) − moyenne(A)   (orienté)
   ├─ STATISTIC = "omnibus"   → variance inter-phases     (effet quelconque)
   │
   ├─ énumère tous les découpages admissibles → distribution de permutation
   │     p = rang de la stat observée
   └─ tailles d'effet : Tau-U (corr. tendance) + NAP, baseline (toutes phases A) vs traitement
   ▼  rapport xlsx : Données & design · Récapitulatif (p + Tau-U/NAP) · Glossaire · Guide
```

## sced_multiple_baseline
```
DATA (paliers × sessions × phase, intervention DÉCALÉE)
   │
   ├─ fenêtre de début : BASELINE_WINDOW(n,n)→début n+1  OU  START_WINDOW(s,s)  OU  AUTO
   ├─ SCHEME : MB (remise, ∏kᵢ) | WW (cas, N!) | MB-R (sans remise) | KL | Rev
   ├─ STATISTIC : level Σ(moy_B−moy_A) | slope | combined
   │
   ▼  p = randomisation du MOMENT d'introduction (concordance avec le décalage)
   ├─ + Tau-U brut ET corrigé-tendance (Tarlow) par palier
   ├─ + multiniveau si MULTILEVEL : b2 (saut) / b3 (pente) / ICC  (p permutation)
   └─ + BC-SMD si BC_SMD :  b2 / sqrt(tau_intercas² + sigma²)   (~ d de RCT)
```

## sced_bayesian  (effet de condition + Bayes factor)
```
DATA long (session, condition, outcome [, GROUP_COLS, COVARIATES])
   │
   ├─ famille/lien selon OUTCOME_TYPE :
   │      continuous → Normal (d) | robust → StudentT(ν) | binary → logit (OR) | count → log (IRR)
   ├─ prédicteur : μ + β·condition (TOUS les contrastes vs réf) + covariables + TIME_COVARIATE
   ├─ hiérarchie : GROUP_COLS → intercept(s) aléatoire(s)  [+ pente si RANDOM_SLOPE]
   ├─ priors : JZS-Cauchy (défaut)  OU  informatif (PRIOR_MEAN/PRIOR_SD en unités brutes)
   │
   ▼  MCMC (PyMC) → effet a posteriori (par famille) + HDI + P(bénéfique)
      + BF10 (Savage-Dickey ; <1 = appui H0)   [+ décision ROPE]
```

## sced_bayesian_mbd  ★  (bayésien de PHASES, unifié — 3 axes POOLING · ONSET · AR)
```
DATA tidy → [resolve cols · DATE_COL→sessions · unstack] → boucle (GROUP_COL × outcome)
   │   moteur DÉRIVÉ des axes (plus de MODE/MODEL) :
   │
   ├─ ONSET = "known"  (bascule = étiquette de phase → modèle hiérarchique)
   │     1) η = b0_i [+ b1_i·time si 'trend'] [+ b2_i·phase si 'level'] [+ b3_i·tsince si 'slope']
   │     2) EFFETS ALÉATOIRES selon POOLING :
   │           none (par cas indép.) | random_intercept | partial (RI+RS) | correlated (MVN LKJ) | meta (2-stage)
   │     3) VRAISEMBLANCE selon FAMILY (+AR1 si AR) :
   │           gaussian/student → Normal/StudentT(η,σ)   → BC-SMD = μ_b2/sqrt(τ_b0²+σ²)
   │           beta/binomial    → logit(η)               → effet en POINTS (AR ignoré, pas de BC-SMD)
   │     ▼ MCMC → μ POPULATION + b_i (per-cas) + τ + fin-B + IP prédiction (meta) + forest (plots/)
   │
   └─ ONSET = "unknown"  (point de bascule BUCP estimé → NON poolable → POOLING forcé "none")
         η = α + β·phase  [+ β3·(temps depuis B) si TREND] (+ AR1 si AR) ; correction tendance si BASELINE_TREND
         ▼ MCMC → es = β/σ + HDI + pd + P(es>ROPE) [+ es_end si TREND] + posterior du CP (immédiateté)
```
> `pooling="none"` (+`ar`) = ex per_case simple/BITS (mais sortie b1/b2/b3 méta-analysable) ;
> `onset="unknown"` = ex per_case BUCP. Rétro-compat `mode`/`model` (dépréciés, warning).

## sced_model_compare  (LOO + WAIC)
```
pour chaque (cohorte × outcome) :
   │
   ├─ ajuster l'ensemble MODELS = {M0 trend, Mi +level, Mg +slope, Mf full}
   │      chacun = bayes_hier_sced(terms=…)        [AR et POOLING FIXÉS = sensibilité]
   │      → chaque fit émet son log_likelihood
   │
   ├─ az.compare : PSIS-LOO + WAIC ; poids stacking + pseudo-BMA+ ; Pareto-k
   ├─ complexité : n_params (nominal) · n_par/n_cas · p_eff/N_obs
   │
   ▼  DÉCISION : Δelpd > SE_MULT×dse ? → gagnant décisif
                 sinon → INDISTINGUABLE (garder le + simple / effet model-averaged)
      sauvegarde chaque modèle → models/*.nc
```

## sced_power_planning  (a priori — aucune donnée)
```
HYPOTHÈSES : effet visé · bruit · n_unités · n_sessions   (zones ▸, pas de données)
   │
   ├─ pour chaque (taille d'effet, n) :
   │      simuler N jeux sous H1  →  relancer le TEST DE RANDOMISATION RÉEL sur chacun
   │      puissance = P(p <= alpha)
   │
   ▼  courbe puissance vs n   +   MDES (plus petit effet à la puissance cible)
      [+ forme fermée pour plans (AB)ᵏ : t non-central, Hedges 2022]
```
