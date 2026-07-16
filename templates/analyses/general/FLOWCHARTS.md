# Organigrammes de construction — pipelines généraux (non-SCED)

Comment chaque modèle se construit à partir des paramètres choisis. `[PARAM]` = zone `▸`.

---

## essai_randomise_pre_post  (RCT continu pré/post)
```
DATA large (1 ligne/sujet : y_pre, y_post, bras) → LONG (sujet, temps, bras, y)
   │
   ├─ modèle mixte :  y ~ temps * bras + (1 | sujet)
   │      (intercept aléatoire = corrélation intra-sujet pré/post)
   │
   ▼  effet du traitement = coefficient d'INTERACTION  temps × bras
      (PAS l'effet temps brut)   + descriptif par temps×bras, Δ intra-sujet
```

## suivi_multi_temps  (cohorte à >2 temps, continu)
```
DATA (sujet × visites M0/M3/M6/…, y continu)
   │
   ├─ RM-ANOVA → test de sphéricité (Mauchly)
   │      sphéricité OK    → df bruts
   │      sphéricité violée → correction GREENHOUSE-GEISSER (df ajustés)
   ├─ + Friedman (non paramétrique)   + PERMANOVA appariée
   │
   ▼  effet du TEMPS  [+ modèle mixte / structure de corrélation échangeable vs AR1 (QIC) si demandé]
```

## comptage_repete  (comptages répétés)
```
DATA (sujet × temps, comptage d'événements)
   │
   ├─ GEE Poisson → test de SURDISPERSION
   │      pas de surdispersion → Poisson
   │      surdispersion        → binomial négatif (NB)
   ├─ corrélation de travail intra-sujet (clusterisé)
   │
   ▼  effet = exp(β) = RATE RATIO (IRR)   + tailles d'effet (dz / Hedges g / rank-biserial)
```

## outcome_binaire_repete  (binaire répété)
```
DATA (sujet × temps, statut 0/1)
   │
   ├─ nombre de temps :
   │      2 temps  → McNemar
   │      >2 temps → Cochran Q
   ├─ + GEE logistique clusterisé sujet
   │
   ▼  effet = exp(β) = ODDS RATIO d'évolution
```

## multivarie_permanova  (profil multivarié en mesures répétées)
```
DATA (sujet × temps × plusieurs outcomes = vecteur réponse)
   │
   ├─ matrice de DISTANCES euclidiennes entre profils
   ├─ pseudo-F (= F de trace MANOVA, valable même si p>n)
   ├─ permutation STRATIFIÉE intra-sujet → p   (sans normalité ni sphéricité)
   │
   ▼  test global du déplacement de profil
      + suivis UNIVARIÉS par outcome (Holm / BH) pour localiser QUI porte l'effet
```

## donnees_manquantes_imputation  (complétude & imputation)
```
DATA longitudinale AVEC manquants
   │
   ├─ rapport de COMPLÉTUDE par temps (observé / manquant / % écarté en complete-case)
   ├─ 3 stratégies sur LA MÊME cohorte :
   │      complete-case  (valide si MCAR, sinon biaisé)
   │      LOCF           (report de la dernière valeur)
   │      MICE           (imputation multiple, valide sous MAR)
   │
   ▼  comparer les estimés OLS appariés entre stratégies → robustesse de la conclusion
```

---

## Quel template longitudinal MULTINIVEAU ? (couplage vs trajectoire)
```
Plusieurs temps par sujet, modèle mixte
   │
   ├─ Question = « 2 variables variant dans le temps évoluent-elles ENSEMBLE ? »
   │     (ex. fatigue ↑ ⇒ performance ↓ intra-séance)           → longitudinal_coupling
   │
   ├─ Question = « quelle est la FORME de la trajectoire d'UN outcome dans le temps ?
   │     qui change plus vite ? un groupe diffère-t-il ? »        → longitudinal_growth
   │
   └─ Question = « l'outcome diffère-t-il ENTRE quelques vagues ? » (contraste, pas trajectoire)
         → suivi_multi_temps (continu) / comptage_repete / outcome_binaire_repete
```

## longitudinal_coupling  (couplage intra-cluster, covariable variant dans le temps)
```
DATA long (sujet/séance, temps/bloc, x = covariable focale, y = outcome [, patient])
   │
   ├─ CONTRÔLE colinéarité :  VIF(x ~ time)
   │      VIF < 5  → x se sépare du temps (x non-monotone, ex. pause)
   │      VIF ≥ 5  → x ~ confondu au temps : interpréter « à temps égal » avec prudence
   │
   ├─ [A] rmcorr (primaire)   : y ~ x + C(séance)  → pente COMMUNE intra-séance
   │            r_rm + IC (Fisher)        ⚠ ignore la nidification patient
   ├─ [B] pente-vs-pente      : pente(x~temps) vs pente(y~temps) par séance, puis corrélées
   │            garde : variance des pentes x ~0 → repli t-tests 1-échantillon
   ├─ [C] mixte fréquentiste  : y ~ x + C(bloc) [+ C(patient) FIXE si <6] + (1|séance)
   │            + df Satterthwaite/KR (R/lmerTest si dispo) ; ICC, R²
   └─ [D] mixte bayésien      : y ~ x + C(bloc) + (1|patient/séance), gaussian + beta
                prior SD = student_t(3,0, 0.1·SD(y))  → HDI + ROPE + pd  (pas de p)
   │
   ▼  estimés COMPARÉS de la pente de couplage (forest : rmcorr / LMM / bayés)
      Cible = ESTIMATION (pilote, petit n) — pas NHST de groupe
```

## longitudinal_growth  (courbe de croissance, forme de la trajectoire)
```
DATA long (sujet, temps continu, y [, by = groupe] [, predictors] [, covariable temporelle])
   │
   ├─ [within/between] covariable variant dans le temps → x_within (intra) + x_between (inter)
   │
   ├─ SÉLECTION DE FORME (ML) :  linear ⊂ poly2  vs  spline
   │      AIC / BIC + LRT emboîté  → forme retenue (SHAPE="auto" = meilleur AIC)
   │
   ├─ AJUSTEMENT (REML) :  y ~ f(temps) [+ C(by):f(temps)] [+ predictors] + (random | sujet)
   │      random = intercept | int_slope (PENTE ALÉATOIRE)   ; 3 niveaux → + (1|cluster)
   │      → effets fixes (Wald [+KR]) · ICC · R² Nakagawa (marginal / conditionnel)
   │
   ├─ TRAJECTOIRE MARGINALE (emmeans-like) : prédiction sur grille de temps × by + IC (delta)
   ├─ CATERPILLAR des BLUP : effets aléatoires par sujet (qui part haut / change vite)
   │
   ├─ [BAYES=True] PARITÉ BAYÉSIENNE (PyMC, même forme) : priors auto-échelle, random int/pente,
   │      pd/HDI/ROPE sur les termes de temps · LOO/WAIC · TRAJECTOIRE POSTÉRIEURE (médiane + HDI)
   │      → <oc>/Analyse/Bayes/ + diagnostics MCMC
   │      ├─ SHAPE="pspline" → GAMM (base spline + pénalité de lissage tau_smooth)
   │      └─ BAYES_FAMILY = gaussian | poisson | binomial | ordinal  (GLMM growth)
   │
   ▼  forme + différences inter-individuelles + effet différentiel (by×temps) + couplage (within)
      [+ forecast_individual : trajectoire prédite d'un sujet ± HDI ; power_growth : puissance par simu]
```
