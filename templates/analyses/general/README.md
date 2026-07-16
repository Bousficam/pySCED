# Templates généraux (NON-SCED : univarié / multivarié / longitudinal)

Designs de **groupe** classiques (cohorte / RCT, comparaison entre/intra-sujets), par opposition
au single-case (`../sced/`). Unité d'analyse = le **groupe** (sujets × temps). Inférence :
modèles de groupe (mixte / GEE / permutation).

Lancer : `python templates/analyses/general/<template>.py`. Rapport `.xlsx` : 1re feuille
**Données & design**, + Glossaire + Guide d'interprétation (ASCII).

> **Construction du modèle, pipeline par pipeline → [`FLOWCHARTS.md`](FLOWCHARTS.md)** (organigrammes).

**Cadres généraux en amont** (racine `templates/`, hors `analyses/`) :
- `template_preprocessing` — nettoyage + rôles de variables (`db`, `var_preprocessed` :
  Continuous / Dicho / …) ; porte d'entrée **univariée / multivariée** (moteurs
  `Univariate_Analysis`, `Multivariate_Analysis`, `Multivariate_OLS`).
- `template_longitudinal` — mécanique brute appariée/longitudinale (continu OLS-mixte, binaire
  GEE, comptage) ; les scénarios ci-dessous en sont des habillages prêts à l'emploi.

---

## Scénarios

### `longitudinal_coupling` — couplage multiniveau de deux variables variant dans le temps
**Question** : un outcome `y` (ex. accuracy) décline-t-il quand une covariable focale `x` variant
dans le temps (ex. fatigue EVA) monte, **à temps égal** (confond temporel `bloc` en covariable) ?
**4 analyses** (estimation, pas NHST) : **rmcorr** (corrélation à mesures répétées intra-cluster,
Bakdash 2017, primaire) · **pente-vs-pente** (pentes `x~temps`/`y~temps` par cluster puis corrélées)
· **mixte fréquentiste** `y ~ x + C(bloc) [+ C(patient) fixe si <6] + (1|session)` + df
Satterthwaite/KR via R · **mixte bayésien** `y ~ x + C(bloc) + (1|patient/session)` (gaussian + beta,
HDI/ROPE/pd). **Contrôle** : VIF `x~bloc` (si EVA monotone → colinéaire, `x` ne se sépare du temps
que si non-monotone). Sorties par outcome : `Analyse/{Inferentielle,Bayes}/` + `Plot/`.
**Quand** : couplage intra-séance, petit n de sujets (cible = estimation d'effet). Moteurs :
`functions/Longitudinal_growth.py` + `Longitudinal_report.py`.

### `longitudinal_growth` — courbe de croissance multiniveau (forme de la trajectoire)
**Question** : comment l'outcome évolue dans le temps (forme), les sujets diffèrent-ils dans leur
changement, un prédicteur module-t-il la trajectoire ? **Pipeline** : (1) **sélection de forme**
`linear` vs `poly2` vs `spline` (AIC/BIC + **LRT** emboîté) ; (2) LMM
`y ~ f(temps) [+ by×f(temps)] [+ predictors] + (random | sujet)`, `random` = intercept | **pente
aléatoire** ; 3 niveaux possible ; (3) **trajectoire marginale** (emmeans-like) ± IC + **caterpillar
des BLUP** ; option **within/between** d'une covariable variant dans le temps. ICC + **R² Nakagawa**
(marginal/conditionnel). df Satterthwaite via R (option). **`BAYES=True`** ajoute la parité
bayésienne (PyMC, même forme) : pd/HDI/ROPE, **LOO/WAIC**, **trajectoire postérieure** (médiane+HDI),
diagnostics MCMC -> `Analyse/Bayes/`. `SHAPE="pspline"` = **GAMM** (base spline + pénalité de lissage) ;
`BAYES_FAMILY` = gaussian | **poisson | binomial | ordinal** (GLMM growth). Prévision individuelle :
`forecast_individual` ; puissance : voir `longitudinal_power`. Sorties : `Analyse/{Inferentielle,Bayes}/` + `Plot/`.
**Quand** : ≥3 temps, focus sur la TRAJECTOIRE (vs contrastes par vague). Moteurs :
`functions/Longitudinal_growth.py` (`fit_growth_curve`, `compare_growth_models`, `marginal_trajectory`,
`fit_growth_bayes_curve`, `posterior_trajectory`).

### `essai_randomise_pre_post` — RCT continu pré/post
**Formule** : `y ~ temps * bras + (1|sujet)` -> effet du traitement = **interaction temps×bras**
(pas l'effet temps brut). Δ intra-sujet par bras en descriptif.

### `suivi_multi_temps` — cohorte à >2 temps (continu)
**Formule** : RM-ANOVA (**Greenhouse-Geisser**) + Friedman (non param.) + PERMANOVA appariée.
**Hypothèse** : sphéricité (corrigée par GG).

### `comptage_repete` — comptages répétés (+ tailles d'effet)
**Formule** : GEE **Poisson / binomial négatif** (surdispersion auto) -> **IRR** `= exp(beta)` ;
+ dz / Hedges g / rank-biserial.

### `outcome_binaire_repete` — binaire répété
```
2 temps -> McNemar   |   >2 temps -> Cochran Q
```
**Formule** : GEE logistique clusterisé sujet -> **odds ratio** `= exp(beta)` d'évolution.

### `multivarie_permanova` — profil multivarié en mesures répétées
**Formule** : **pseudo-F** sur matrice de distances euclidiennes, p par permutation (stratifiée
intra-sujet). Valable même si p>n, sans normalité/sphéricité ; suivis univariés (Holm/BH).

### `longitudinal_power` — puissance par simulation (courbe de croissance)
**Question** : combien de sujets/temps pour détecter une pente de temps (ou interaction groupe×temps) ?
**Méthode** : Monte-Carlo — simule `N_SIM` jeux conformes au design (random intercept+pente, résidu),
ajuste le modèle mixte, compte la fraction où l'effet ciblé est significatif (IC de Wald excluant 0).
**Sortie** : puissance (scalaire) ou balayage `SWEEP_N_SUBJ` (table puissance vs N). Moteur :
`functions/Longitudinal_growth.power_growth`.

### `donnees_manquantes_imputation` — complétude & imputation
**Formule** : rapport de complétude + 3 stratégies sur la même cohorte : complete-case (valide si
MCAR) · LOCF · **MICE** (valide sous MAR), sur le modèle OLS apparié.

---

## Choix transversaux (group-based) — hypothèse posée & implication

| Choix | Hypothèse posée | Implication / quand |
|---|---|---|
| **modèle mixte** `(1|sujet)` | corrélation intra-sujet par intercept aléatoire | mesures répétées appariées ; effet = terme fixe |
| **GEE** (Poisson/NB/logit) | structure de corrélation de travail (échangeable/AR1) | effet marginal (population), robuste (sandwich) |
| **surdispersion** (NB vs Poisson) | variance > moyenne | comptages sur-dispersés -> NB |
| **Greenhouse-Geisser** | sphéricité violée | RM-ANOVA >2 temps -> correction des df |
| **PERMANOVA** | pas de normalité/sphéricité ; distances | réponse **multivariée** (profil), même si p>n |
| **imputation** (MICE/LOCF) | MAR (MICE) / report (LOCF) | perdus de vue ; complete-case valide seulement si MCAR |
| **interaction temps×bras** | l'effet = différence d'évolution | RCT pré/post : tester l'interaction, pas le temps brut |
| **rmcorr** (cluster intra) | pente de couplage COMMUNE intra-cluster | couplage de 2 variables variant dans le temps, sans pseudoréplication |
| **confond temporel en covariable** `C(bloc)` | l'effet de `x` est « à temps égal » | sépare fatigue (ressenti) du temps si `x` non-monotone (vérifier VIF<5) |
| **sujet de niveau-3 en FIXE** (`C(patient)`, <6) | variance inter-sujets non fiable à petit n | pilote : cible = estimation d'effet, pas généralisation populationnelle |
| **mixte bayésien** `(1|patient/session)` | pooling partiel, prior SD régularisant | meilleur à petit n ; rapport HDI/ROPE/pd (pas de p) |

Moteurs : `Univariate_Analysis`, `Multivariate_Analysis`, `Multivariate_OLS` (univarié/multivarié
généraux) ; `Longitudinal_Analysis` (mixte / GEE logit / PERMANOVA), `Longitudinal_Effects`
(comptage IRR + tailles d'effet) ; `preprocessing`.
