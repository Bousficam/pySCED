# Templates SCED (single-case experimental designs)

Unité d'analyse = **LE patient** (séries A/B, sessions). Inférence : **test de randomisation**
(design-based, primaire) **+** **bayésien** (model-based, secondaire). Pour les designs de
groupe classiques (univarié/multivarié/longitudinal), voir `../general/`.

> **Méthodes statistiques (fiche détaillée, EN) : [`METHODS.md`](METHODS.md)**. Pour chaque
> approche d'inférence - randomisation par permutation des labels, permutation des résidus
> (Freedman-Lane) avec contrôle de nuisance, GLM piecewise MBD (b1/b2/b3), bayésien par cas /
> hiérarchique / méta-analyse + comparaison de modèles - la fiche donne le problème, le principe,
> la fonction et le template associés, un pont R, la règle de lecture de la sortie, et les
> références (Crossref).

Lancer : `python templates/analyses/sced/<template>.py` (ou piloter via un script qui surcharge
les globals et appelle `main()`). Rapport `.xlsx` : 1re feuille **Données & design**, +
**Glossaire** + **Guide d'interprétation** (ASCII).

> **Architecture (depuis 2026)** : les **templates ne sont que des paramétrables** — config +
> `load_data()` + UN appel à la pipeline versionnée. **Toute** l'analyse + l'écriture des rapports
> stylés vit dans `functions/` :
> `SCED_mbd_report.report_sced_multiple_baseline`, `SCED_bayes_report.report_sced_model_compare` /
> `report_sced_bayesian_mbd`, `SCED_phase_design.report_sced_phase_design`,
> `SCED_alternating_run.report_sced_alternating`, `SCED_bayesian.report_sced_bayesian_condition`.
> Pour scripter une étude : `import` la fonction `report_*` directement (pas besoin du template).

> **Rangement standard (`output_dir=`)** — arborescence INTÉGRÉE à toute pipeline (pas seulement
> aux regen), **ORGANISÉE PAR OUTCOME** : la racine de chaque outcome contient `Plot/` + `Analyse/` →
> ```
> <output_dir>/<outcome>/
>   Plot/                       descriptifs (étagé, panneaux, VAIOR, Brinley) — indép. du modèle
>     forest/<bN…>/             forests (un sous-dossier par modèle)
>     poolé/                    poolés inférentiel + bayésien (hier & meta, systématiques)
>   Analyse/
>     permutation_test/         test de randomisation MBD + tailles d'effet (.xlsx)
>     phase_design/ alternating/ (selon la pipeline)
>     bayes/
>       hier/                   model_compare_summary.xlsx + <bN…>/ (rapports par modèle)
>                               + models/ (= cache) + diagnostics/
>       meta/                   méta two-stage : <bN…>/ + models/ + diagnostics/
>       condition/ per_case/    (selon la pipeline)
> ```
> `model_compare` ne garde que la **comparaison** ; les rapports par modèle + le cache `.nc` +
> les diagnostics vivent dans le dossier `bayes/<hier|meta>/` correspondant (selon le pooling
> des modèles comparés). Le **plot poolé bayésien** est produit systématiquement (hier ET meta)
> dans `<outcome>/Plot/poolé/`. (`SAVE_PATH`/`PLOT_PATH`/`CACHE_DIR`/`FOREST_PATH` : mode avancé,
> dossier unique consolidé multi-outcomes.)

> **Cache fit-or-load (`CACHE_DIR=`)** : cache PARTAGÉ des modèles bayésiens (`SCED_model_cache`).
> Avant chaque MCMC, on vérifie si un modèle **identique** (clé = spec famille/pooling/termes/AR/
> draws… + **hash des données**, via manifeste) existe → si oui **on recharge** (pas de MCMC),
> sinon **on ajuste + sauvegarde**. Un modèle périmé (spec/données différentes) n'est JAMAIS
> réutilisé ; `FORCE_REFIT=True` force le recalcul. Couvre hier/méta/model_compare (`bayes_hier_sced`)
> et per-cas (`bayes_phase_model`). Nommage : `outcome__termes__famille__pooling__<hash>.nc`.
> Défaut (sous `output_dir`) = `Analyse/Bayes/models/` ; pointer `CACHE_DIR` ailleurs pour partager
> un cache **inter-études**.

> OUTCOMES = ce qui est mesuré (≥2 → multivarié) · COVARIATES = nuisance exogène à neutraliser ·
> GROUP_COL = cohorte (sous-ensemble analysé à part) · POOLING = comment les patients se relient.

> **Construction du modèle, pipeline par pipeline → [`FLOWCHARTS.md`](FLOWCHARTS.md)** (organigrammes).

> **Plots** (tous sous `<outcome>/Plot/`) : les pipelines MBD produisent les **plots SCED des données**
> (étagé, panneaux, VAIOR) dans `Plot/` + le **poolé inférentiel** dans `Plot/poolé/`. Les pipelines
> bayésiennes produisent le **poolé bayésien** (`_pooled_bayesien_{hier,meta}`, systématique) dans
> `Plot/poolé/` et le(s) **forest(s)** dans `Plot/forest/<bN…>/` — `FOREST_ESTIMAND="auto"` (défaut)
> émet **un forest par estimand du modèle** : level+slope → **b2 + b3 + fin-B** (3 forests) · b1+b3 →
> **b3** · b1+b2 → **b2**. Override par un estimand unique (`effect_end`|`slope`|`level`). + en option
> (`DIAGNOSTICS=True`) les **diagnostics MCMC** dans `Analyse/bayes/<hier|meta>/diagnostics/`.

> **Style des figures `article-proof` (`STYLE=`)** — un objet central `PlotStyle`
> (`functions/SCED_plotstyle.py`) rend **paramétrables** légende, couleurs (par rôle : baseline,
> traitement, population, répondeur…), libellés (`phase_a`/`phase_b`/`pooled_mean`…), axes
> (`xlabel`/`ylabel`), titre, tailles de police, `figsize`, `dpi`, grille. Chaque template expose
> `STYLE` (un `dict` de surcharges ou une instance `PlotStyle` ; `None` = défauts) qui descend
> jusqu'à toutes les figures de la pipeline (étagé, poolés, forests, VAIOR, Brinley, panels,
> alternant). Champs `soumission` : `fmt` (**`"pdf"`/`"svg"`/`"eps"`** vectoriel | `"png"`), `dpi`
> (300/600), `font_family` (`"serif"`, `"Arial"`…), `rcparams` (passthrough matplotlib : épaisseurs,
> mathtext…), `linewidth`/`markersize`, `legend_loc`/`legend_ncol`/`legend_frameon`,
> `condition_colors={cond: couleur}` (alternant/Brinley), `transparent`, `band_hatch` (hachure des
> bandes HDI pour le **N&B**), `integer_ticks`, `decimal_comma` (virgule FR sur les axes). Couvre aussi
> les couleurs **VAIOR** (`vaior_pass/warn/fail`), **`meta_forest`** (`meta_raw/shrunk/diamond/pi`) et
> les **diagnostics MCMC** (fmt/dpi/police via `STYLE`). Ex. :
> `STYLE = {"fmt": "pdf", "dpi": 300, "font_family": "serif", "colors": {"treatment": "#0a7d4f"},
> "labels": {"phase_b": "tDCS"}, "ylabel": "FM-UE", "fontsize": 12}`. **Presets de thème** :
> `STYLE = "journal"` (serif, PDF 300 dpi, traits fins, sans cadre + spines retirées) | `"colorblind"`
> (palette Wong) | `"grayscale"` (N&B, PDF) — combinables : `PlotStyle.preset("journal", ylabel="FM-UE")`.
> Les **clés inconnues** de `colors`/`labels` (ou un champ mal orthographié) déclenchent un *warning*
> (plus de faute de frappe silencieuse). **Échappatoire** :
> `post=callable(fig, ax)` (champ `PlotStyle.post`) est appelé juste avant la sauvegarde de
> chaque figure (`ax` = axe principal si unique, sinon `None` → utiliser `fig.axes`) pour des
> retouches matplotlib arbitraires (annotations, lignes, spines…).

> **Où styliser : template d'ANALYSE vs `visualize`.** Les figures sont générées **pendant
> l'analyse** (sauf celles désactivées par les flags `PLOT_VAIOR`/`PLOT_PANELS`/`BAYESIAN`…) et se
> stylisent **directement via le `STYLE` du template d'analyse** — `visualize` n'est PAS requis pour
> ça. `template_sced_visualize.py` est un chemin INDÉPENDANT qui RE-génère des figures (depuis le CSV
> ou des `.nc` sauvegardés) sans relancer l'analyse. Les deux partagent les mêmes fonctions `plot_*`
> et le même `STYLE`.
>
> | Besoin | Outil |
> |---|---|
> | Plots stylés produits pendant l'analyse | `STYLE` dans le **template d'analyse** (défaut) |
> | Re-styliser pour l'article **sans relancer** le MCMC (lent) | `template_sced_visualize` (relit les `.nc`/CSV) |
> | Tracer les **diagnostics MCMC** depuis le cache `models/` | `template_sced_visualize` |
> | Produire des figures sans avoir relancé l'analyse | `template_sced_visualize` |
>
> Les **diagnostics MCMC** (trace/rank/PPC/panel) respectent aussi `STYLE` (fmt/dpi/police), via le
> template d'analyse (`DIAGNOSTICS=True`) comme via `visualize`.

## Quel template ? (arbre de décision)
```
Conditions ALTERNÉES par séance (switch possible chaque séance) ? ─ oui ─→ sced_alternating
Phases A/B (ou ABA/ABAB) d'un cas, bascule(s) randomisée(s) ?      ─ oui ─→ sced_phase_design
Intervention introduite à des moments DÉCALÉS entre paliers ?      ─ oui ─→ sced_multiple_baseline

Estimation BAYÉSIENNE (en complément du test de randomisation) :
   design de CONDITION (alternant) + Bayes factor ?               ─→ sced_bayesian
   design en PHASES (A→B) ?                                        ─→ sced_bayesian_mbd
        POOLING : partage entre cas (partial/RI/correlated/none/meta)
        ONSET="known"   : bascule = étiquette de phase (modèle hiérarchique)
        ONSET="unknown" : point de bascule INCONNU (BUCP) → force POOLING="none" (single-case)

Comparer des structures de modèle (saut vs pente…) par LOO/WAIC ? ─→ sced_model_compare
Dimensionner AVANT de collecter (puissance / MDES) ?              ─→ sced_power_planning
```
> Le design se choisit à la PLANIFICATION (selon ce qui a été randomisé), pas d'après la forme des
> données. Les templates fréquentistes ET bayésiens se séparent **par design** (même logique).

---

## Statistiques de test (tests de randomisation)

Un test de randomisation peut envelopper **n'importe quelle** statistique (Heyvaert 2014). Le choix
change la **puissance**, pas la validité :

| Statistique | Définition | Quand / note |
|---|---|---|
| **MD** (mean difference) = `level`/`contrast` | moyenne(B) − moyenne(A) | défaut ; bon en alternance |
| **NAP** | P(point B > point A) | ≈ MD en puissance (Michiels 2017) ; bornée 0-1 |
| **ITEI** | \|moyenne 3 derniers A − 3 premiers B\| | **AB-phase AVEC tendance** : plus puissant que MD (Michiels 2018) |
| **Tau-U** (corr. tendance) | non-recouvrement de rang, détrendé (Tarlow) | effet ET inférence en une stat |
| `slope` | changement de pente A→B | effet progressif |
| `combined` | niveau + pente (ITS) | effet mixte |
| ~~PND~~ | % points non-recouvrants | **à éviter** : ~16 % moins puissant (Michiels 2017) |

> Sous **tendance** : préférer **ITEI** (AB-phase) ou la stat trend-robuste ; MD reste valide (Type I
> garanti) mais le `level` est confondu avec la tendance (cf. eCALAP). Paramètre `STATISTIC=` dans
> `phase_design` / `multiple_baseline` (level / slope / combined / **itei** / **tau_u**).
>
> **Primaire vs sensibilité (important)** : pré-spécifier UNE statistique primaire — **MD (`level`)** par
> défaut, **ITEI** si tendance attendue. `slope` / `combined` / **`tau_u`** = analyses de **sensibilité**.
> Tau-U comme statistique de *randomisation* est **non-standard** (Tau-U est d'ordinaire une *taille
> d'effet* avec sa propre inférence) → à présenter comme exploratoire. **Ne PAS** lancer plusieurs stats
> et garder la significative (multiplicité / p-hacking) : si la conclusion change selon la stat (eCALAP
> deno : MD/Tau-U sig., ITEI ns), le **dire**.

## Famille (vraisemblance) selon l'outcome

Le modèle bayésien / multiniveau choisit sa **vraisemblance** d'après le **type d'outcome** :

| Outcome | Famille | Pourquoi / effet |
|---|---|---|
| Continu, ~milieu d'échelle, variance ~constante | **gaussian** | défaut ; effet en **points**, **AR1** modélisable, BC-SMD (scdhlm) |
| Continu **avec outliers** | **student** | gaussien **robuste** (queues lourdes), garde l'AR1 |
| Score **borné** (plancher/plafond, /100, %) | **beta** | lien **logit**, effet en **points**, **gère le plafond** ; pas d'AR1 |
| **Comptage** k/N (items réussis) | **binomial** | lien logit, effet en points ; requiert `N_TRIALS` |

> Astuce : un score sur 100 qui sature (patients proches du plafond) → **beta** ; sinon **gaussian**.
> `FAMILY` accepte un **dict {outcome: famille}** pour mélanger selon les épreuves.

## Tailles d'effet — estimateurs, avantages / inconvénients

La **détection** d'effet revient au test de randomisation (design-based, robuste à la maturation si le
début est randomisé) ; la **taille** se lit avec une métrique. Toutes ne **corrigent pas la tendance**
baseline, ce qui change tout sous récupération spontanée (cf. eCALAP) :

| Métrique | Trend-corr. ? | Avantage | Inconvénient |
|---|---|---|---|
| **Tau-U baseline-corrigé** (Tarlow) | **oui** | non-param., robuste outliers, corrige la tendance, effet + inférence en une stat | borné 0-1 (pas en unité clinique) ; dépend du codage de correction |
| **NAP** | **non** | simple, interprétable `P(B>A)`, robuste outliers | **gonflé sous tendance** (comme `level` brut) ; plafonne (effet plafond) |
| **Hedges' g** (intra-cas) | **non** | familier, standardisé | suppose la normalité ; ni tendance ni AR ; dépend du SD intra-cas |
| **BC-SMD** `scdhlm` (brut, `y~phase`) | **non** | **design-comparable** (= d de RCT), méta-analysable ; REML + corAR1 + df Satterthwaite (**canonique**) | « B vs A » non trend-corrigé ; ≥ 2 cas ; nécessite R |
| **BC-SMD** `scdhlm(trend=True)` | **oui** | idem, **net** d'une tendance baseline linéaire | un peu moins puissant ; suppose pentes parallèles |
| **Bayésien** (b1+b2+b3) | **oui** | effet en **points** + `pd`/HDI, **hétérogénéité τ**, **plafond** (Beta), shrinkage per-cas | plus lourd, sensible aux priors ; comparaison de modèles peu puissante sur séries courtes |

> Règle pratique : comparer **BC-SMD brut vs `trend=True`** — un g qui s'effondre une fois la tendance
> retirée = effet largement porté par la maturation. `multiple_baseline` sort **les deux** colonnes
> (`d (BC-SMD)` / `d (BC-SMD trend-corr.)`). Au-dessus de Tau-U corrigé + BC-SMD trend-aware, le bayésien
> sert surtout à l'**hétérogénéité par patient** (τ, random slopes) et au **plafond** (Beta).

## Estimation des pentes — OLS vs Theil-Sen

| | OLS (moindres carrés) | **Theil-Sen** (défaut SCED) |
|---|---|---|
| Pente | minimise Σ résidus² | **médiane des pentes de toutes les paires** de points |
| Hypothèses | erreurs ~normales, homoscédastiques | **aucune** (non-paramétrique) |
| Outliers | **sensible** (un point tire la droite) | **robuste** (point de rupture ~29 %) |
| Usage SCED | risqué sur baseline courte/bruitée | **recommandé** : projection de la tendance baseline, pente du Tau-U (Tarlow) |

> Dans les panneaux MBD : **tendance A** Theil-Sen (navy pointillé) **projetée** dans B comme
> contrefactuel « si rien n'avait changé », **+ tendance B** Theil-Sen sur les points B (**vert
> pointillé**) ; l'écart entre les deux = la cassure A→B. Le **modèle**, lui, sépare aussi les pentes :
> **pente A = b1**, **pente B = b1 + b3**, donc **b3 = pente B − pente A** (effet progressif).

### Correction de tendance du Tau-U — `adj` (Theil-Sen) vs `trend_a` (Brossard)

La correction de tendance du Tau-U (`tau_u`, `method="auto"`) est **conditionnelle ET choisie par la
longueur de baseline `n_A`** :

| `n_A` | Variante | Mécanisme | Compromis |
|---|---|---|---|
| **≥ 7** | **`adj`** (Tarlow 2016, Theil-Sen) | estime la pente baseline et la **retire de toute la série**, puis Tau sur résidus | **meilleur** contrôle de tendance, mais exige une **pente fiable** |
| **< 7** | **`trend_a`** (Brossard 2018, borné) | **soustrait** la tendance dans la formule `(S_AB−S_A)/(n_AB+n_A)`, **sans estimer de pente** | **robuste** sur baseline courte, mais correction **plus faible** |
| (pas de tendance) | **`none`** | aucune correction | Tau-U brut (détrender une baseline plate ajoute du bruit) |

> « Tendance présente » = `p_A < 0.05` **OU** `|Tau-A| ≥ 0.40` (le critère de magnitude rattrape le
> manque de puissance du `p` sur baseline courte). Seuil `min_baseline_for_adj = 7` (Tarlow : le
> contrôle Theil-Sen ne tient qu'à partir de ~7 points baseline). Le rapport MBD note la variante
> appliquée par palier (colonne `Variante corr.`). Sur baselines courtes, **toute** correction reste
> imparfaite → l'inférence primaire reste le **test de randomisation** (design-based).

## Modèle multiniveau (ML) & BC-SMD design-comparable

Deux **tailles d'effet model-based** complètent le test de randomisation (qui, lui, fournit le **p**).

**ML — modèle multiniveau piecewise (interrupted time-series)**. Idée (Van den Noortgate & Onghena
2003) : un SCED est une **régression multiniveau** où les **mesures (niveau 1)** sont nichées dans les
**cas/patients (niveau 2)** — et les cas dans les **études (niveau 3)** pour une méta. On ajuste **tous
les cas ensemble** (*partial pooling*) :
`y_it = b0 + b1·temps + b2·phase + b3·(trajectoire B) + (effets aléatoires par cas) + e`.
- **b1** = tendance baseline · **b2** = saut de niveau · **b3** = changement de pente (effet progressif) ·
  **ICC** = part de variance inter-cas. Le modèle donne l'**effet MOYEN** sur les cas **+** sa
  **variabilité** inter-cas, en empruntant de la force entre patients — estimés chiffrés **même sans
  randomisation**. Réfs : **Van den Noortgate & Onghena (2003)** ; extension méta 3 niveaux
  **Moeyaert, Ferron, Beretvas & Van den Noortgate (2014)**, *J. School Psychology*.
- C'est la version **fréquentiste** (REML/ML) de notre **bayésien hiérarchique** (même structure, un
  modèle conjoint sur tous les cas) — cf. `sced_bayesian_mbd`.
- NB **Inférence** : on **n'utilise PAS** le p asymptotique du ML (non fiable avec peu de cas) → le **p
  vient de la randomisation**. Limites `statsmodels` : pas d'AR(1), pas de ddl Kenward-Roger → pour une
  inférence modèle aux normes, passer par `nlme` / SAS.

**BC-SMD — between-case SMD `g_AB`** (Hedges, Pustejovsky & Shadish) : le saut **standardisé par
l'écart-type INTER-cas** `√(τ_intercept² + σ²)` → **même métrique qu'un Cohen's d de RCT**, donc
**méta-analysable**. Calculé par le package **R `scdhlm`** (REML, corAR1, ddl de Satterthwaite) —
estimateur **canonique** (`bc_smd_scdhlm`). Repères Cohen 0.2 / 0.5 / 0.8.
- **Brut** (`y~phase`) vs **trend-corrigé** (`trend=True`, `y~time+phase`) : un `g` qui **s'effondre**
  une fois la tendance retirée = effet largement **porté par la maturation** (cf. eCALAP deno
  0.59 → 0.17). `multiple_baseline` sort **les deux** colonnes.

**ML vs BC-SMD** : le ML donne les coefficients **en unités d'origine** (points) ; le BC-SMD les
**standardise** par la dispersion inter-cas pour la **comparabilité entre études**. Les deux sont
**model-based** ; la **randomisation** reste l'inférence primaire.

## Synthèse multi-cas : hiérarchique (1 étape) vs méta-analyse (2 étapes)

Deux façons d'agréger plusieurs cas, **toutes deux** des méta-analyses bayésiennes :

| | **Hiérarchique 1 étape** (`pooling="partial"`) | **Méta-analyse 2 étapes** |
|---|---|---|
| Estimation | tout conjointement ; per-cas **rétréci** vers la population | stage 1 = fit **indépendant** par cas (`pooling="none"`) → effet_i + SE_i ; stage 2 = random-effects sur (effet_i, SE_i) |
| Per-cas | stable mais **lissé** (emprunt de force) | **non contaminé** par le groupe ; rétréci seulement au stage 2 (visible : forest brut-vs-rétréci) |
| μ / τ | estimés ensemble | μ / τ / **I²** + intervalle de prédiction au stage 2 |
| Risque | sur-rétrécit ; **masque** l'hétérogénéité de *type* d'effet | instable si les SE par cas sont peu fiables (séries très courtes) |

**Règle de décision :**
- **peu** de points/cas (< ~8-10/phase), effet **homogène**, énoncé de **population** primaire → **hiérarchique** (emprunt de force, τ stabilisé par les priors).
- **assez** de points/cas (≥ ~10-12), **hétérogénéité de type** (un cas « niveau », un autre « pente »), décision **par cas** (répondeurs) primaire → **méta-analyse 2 étapes**.
- dans tous les cas : la **détection** s'ancre sur le test de **randomisation** (trend-robuste) ; le bayésien sert la **magnitude + crédibilité**. Si 1-étape et 2-étapes **divergent** fortement sur μ → signal d'hétérogénéité à **rapporter**, pas un bug.

**Estimand recommandé (les deux approches) : l'effet TOTAL en fin de phase `b2 + b3·T_B`**, en points
(colonne *Effet total fin-B*). Le compromis niveau↔pente s'**annule dans la somme**, donc il est bien
mieux identifié que b2 ou b3 isolément (c'est aussi le titre du script brms `effet_end`). Le moteur le
sort par cas **et** population, tous poolings/familles ; `meta_from_idata(..., estimand="effect_end")`
le prend par défaut.

> **Note `pooling="none"`** : il n'y a **aucun** modèle de population → le rapport bayésien **ne publie
> pas** de ligne *Population* (ce serait une simple moyenne des cas, pas une inférence). μ/τ/I²
> s'obtiennent par `functions/SCED_mbd_meta.bayes_meta_analysis` (stage 2).

### Identifier des répondeurs (binariser les patients) : BRUT, pas rétréci

Le rapport `POOLING="meta"` sort **deux pages par cas** : *Par cas (brut)* (stage 1, individuel,
non contaminé) et *Par cas (retreci)* (stage 2, tiré vers μ). Pour **trancher répondeur /
non-répondeur**, on **binarise sur le BRUT** (HDI/pd du stage 1), idéalement :

- **seuil pré-spécifié vs un MCID**, pas seulement vs 0 ;
- **adossé au test de randomisation par cas** (exact, design-based, trend-robuste) comme épine
  dorsale de la détection — le brut bayésien donne alors la magnitude + crédibilité ;
- en tenant compte de la **multiplicité** (k patients) et de l'**incertitude large** (peu de points).

Le **rétréci**, lui, sert à : l'**énoncé de groupe** (μ), un **classement stabilisé** (meilleure MSE
sous échangeabilité), et une **estimation régularisée** — **pas** à trancher répondeur / non-répondeur.
Raison : le rétréci aspire tous les θ_i vers μ → il homogénéise (faux positifs près de μ, faux négatifs
sur les répondeurs idiosyncratiques) et son pd reflète en partie l'évidence de **groupe**, pas celle du
patient. *Brut = qui a répondu ; rétréci/μ = ce que dit la population.*

### Hétérogénéité : τ, I², intervalle de prédiction

Les trois quantifient **à quel point les patients diffèrent**, mais répondent à des questions distinctes :

- **τ (tau)** — écart-type **inter-cas des effets vrais** (`θ_i ~ Normal(μ, τ)`), **net du bruit
  intra-cas**, en **unité d'effet** (points). τ≈0 = effets homogènes ; τ grand = les effets diffèrent
  réellement. Mal estimé quand *k* est petit (→ priors).
- **I²** — **proportion** de la variabilité totale due à l'hétérogénéité vraie : `I² = τ²/(τ²+s²_typ)`
  (Higgins-Thompson). Sans unité ; repères 0.25 / 0.50 / 0.75 = faible / modérée / forte. Dépend aussi
  de la **précision** des cas → lire **avec τ**.
- **Intervalle de prédiction** — fourchette de l'effet d'un **NOUVEAU patient** (`μ ± dispersion τ`).
  **Plus large** que le HDI de μ (qui ne porte que sur la moyenne). S'il **croise 0**, un futur patient
  peut ne pas répondre **même si μ est crédible** → c'est le **pronostic individuel**.

> `μ` = effet moyen · `HDI(μ)` = précision sur la moyenne · `τ` = ampleur de la variation (points) ·
> `I²` = part de variance « vraie » (proportion) · intervalle de prédiction = à quoi s'attendre pour
> le prochain patient. **Caveat** : à petit *k* (< ~5 cas), τ/I²/prédiction sont instables.

**Outils.** `functions/SCED_mbd_meta.py` : `bayes_meta_analysis(effects, ses, ...)` (random-effects
**non centré** `θ_i=μ+τ·z_i` → pas d'entonnoir/divergences ; μ, τ, **I²** Higgins-Thompson, intervalle
de prédiction, θ_i rétrécis) ; `meta_from_idata(idata, estimand=...)` part d'un modèle stage-1
`pooling="none"` ; `plot_meta_forest(result)` = forest **brut vs rétréci** + losange μ + bande de
prédiction.

### Bonnes pratiques validées (audit vs littérature, 2024)

Audit du pipeline bayésien contre la littérature méta/multiniveau SCED. **Ce qui est confirmé :**

- **Bayésien > REML pour les composantes de variance** : à petit *k*, REML/DL sous-estiment τ²
  (estimés au bord) → CI de μ trop étroits → **type-I gonflé** ; le bayésien à prior faiblement
  informatif corrige (Williams, Rast & Bürkner 2018 ; Moeyaert et al. 2017 ; Chung et al. 2013).
- **μ (effet moyen) fiable quel que soit N** : l'effet fixe est non biaisé et précis sous toutes les
  méthodes ; ce sont les **composantes de variance** qui sont fragiles (Baek et al. 2014 ; Ferron et
  al. 2009 ; Moeyaert et al. 2017). → se reposer sur μ + per-cas, prudence sur τ/I².
- **Prior τ = Half-Normal ou Half-Cauchy** : les deux recommandés à **J ≥ 5** ; Half-Normal(0,·) =
  plus petit biais à J=3 ; Half-Cauchy = queue plus lourde, défaut méta-analytique (Moeyaert et al.
  2017 ; Williams et al. 2018 ; Gelman 2006). *(Notre défaut : Half-Normal ; Half-Cauchy en option.)*
- **Codage ITS** : interaction centrée au 1er point d'intervention → b2 = effet **immédiat**
  (Baek et al. 2014, d'après Huitema & McKean 2000). Conforme (`tsince=0` au 1er B).

**Seuils / caveats à respecter :**

- **τ, I², intervalle de prédiction non fiables si k < 5** (couverture CI adéquate seulement à
  **J ≥ 5** en bayésien, J ≥ 7 en ML — Moeyaert et al. 2017 ; Baek et al. 2014). À petit *k*, ne pas
  conclure sur l'hétérogénéité.
- **Empirical-Bayes (rétréci) ≠ brut** : le rétréci a une **meilleure MSE** pour l'**estimation**
  individuelle à séries courtes (Baek et al. 2014 ; Van den Noortgate 2024) ; le **brut** reste la
  preuve per-patient défendable pour la **décision** répondeur (adossée à la randomisation). Deux
  usages distincts (cf. section répondeurs).
- **Échelle** : la méta en **points** n'est valable qu'**intra-échelle** (même outcome). Pour méta
  **entre études/échelles**, les ES SCED standardisés par la variance intra-cas sont **inflés** vs
  Cohen's d (Ugille et al. 2012 ; Van den Noortgate & Onghena 2008) → utiliser **BC-SMD** (`scdhlm`,
  Pustejovsky et al. 2014 ; Valentine et al. 2016).
- **Autocorrélation** : AR(1) recommandé par défaut en SCED (Baek et al. 2014 ; Ferron et al. 2009).
  Le chemin **gaussien/student** modélise l'AR1 (`ar=True`). Le chemin **Beta** ne le fait PAS — non
  par oubli mais parce qu'un AR1 ADDITIF n'est pas défini sur une réponse bornée : le bruit beta est
  porté par la précision **φ** (var = μ(1−μ)/(1+φ)), et la dépendance sérielle se modéliserait par un
  latent-AR sur le logit / un **β-ARMA** (Da-Silva, Migon & Correia 2011 ; Rocha & Cribari-Neto 2009),
  non identifiable aux longueurs SCED. Le rapport sort **`phi (Beta)`** (le « bruit » du modèle) et un
  **`autocorr PPC p`** (posterior predictive check de l'autocorr lag-1, BARG) : p≈0.5 = indépendance
  conditionnelle adéquate ; extrême (<0.05 ou >0.95) = misfit. Le **signe** de ρ compte : ρ>0 → pd un
  peu optimiste ; ρ<0 → conservateur.

**Références (méta-analyse & SCED).** Van den Noortgate & Onghena (2003, *Behav. Res. Methods* ;
2008, *EBCAI*) et Van den Noortgate (2024, *review*) — modèles multiniveaux pour intégrer des effets
single-case ; Baek et al. (2014, *Neuropsychol. Rehabil.*) — tutoriel 2/3 niveaux ; Moeyaert et al.
(2014, *J. School Psychol.*) — 3 niveaux ; Moeyaert, Rindskopf, Onghena & Van den Noortgate (2017,
*Behav. Res. Methods*) — ML vs bayésien, priors τ ; Ugille et al. (2012, *Behav. Res. Methods*) —
inflation des ES standardisés à séries courtes ; Williams, Rast & Bürkner (2018, *Psychol. Methods*)
— priors faiblement informatifs (half-Cauchy) pour τ ; Pustejovsky, Hedges & Shadish (2014, *JEBS*)
+ Hedges, Pustejovsky & Shadish (2012/2013, *RSM*) + Valentine et al. (2016, *Campbell*) — effets
design-comparables (`scdhlm`) ; Shadish, Rindskopf & Hedges (2008, *EBCAI*) — état de l'art méta SCED
; Rindskopf (2014) — estimation bayésienne ; Burke, Ensor & Riley (2017, *Stat. Med.*) — one-stage vs
two-stage ; Higgins & Thompson (2002, *Stat. Med.*) — I².

## Pipelines

### `sced_alternating` — alternance randomisée (ATD / N-of-1 / groupe / multivarié)
```
UNIT_COL=None, 1 outcome  -> N-of-1 (1 patient)
UNIT_COL="patient"        -> groupe (permutation stratifiée intra-patient, patient = bloc)
OUTCOMES=[>=2]            -> multivarié (PERMANOVA)
GROUP_COL set             -> par cohorte + (tous)   ;   HIERARCHICAL=True -> mixte (ICC)
```
**Formule** : stat = écart de moyennes entre conditions vs distribution des **ré-attributions de
conditions** tirables -> `p=(1+#>=obs)/(1+B)`. + Tau-U, NAP, g. **Hypothèse** : conditions
échangeables sous H0 (tirage réel) -> exact, sans loi.

### `sced_phase_design` — plans de phases (AB / ABA / ABAB)
```
STATISTIC="contrast" -> phases B vs A (orienté)   |   "omnibus" -> variance inter-phases
+ Tau-U (corr. tendance) / NAP baseline vs traitement ; prep TIDY/DATE_COL/GROUP_COL ; rapport xlsx
```
**Formule** : re-découpe à tous les changements admissibles -> permutation de la stat.
**Hypothèse** : la fenêtre de randomisation (MIN_LEN) doit refléter le tirage réel.

### `sced_multiple_baseline` — lignes de base décalées (MBD)
```
START_WINDOW=(s_min,s_max)  OU  BASELINE_WINDOW=(n_min,n_max)  [|A|=n -> début n+1]
STATISTIC: level | slope | combined    SCHEME: MB | WW | MB-R | KL | Rev
MULTILEVEL=True -> b2/b3/ICC      BC_SMD=True -> d design-comparable
```
**Formule** : stat = Σ_paliers (moy_B − moy_A) **concordante avec les débuts décalés** ; p par
randomisation du moment. BC-SMD = `b2 / sqrt(tau_intercas^2 + sigma^2)` (~ d de RCT).
**Hypothèse** : le staggering contrôle la tendance commune ; `level` reste confondu avec la
tendance -> croiser avec `slope` / Tau-U corrigé. (⚠ piège fréquent : « baseline 4-9 points » =
`BASELINE_WINDOW=(4,9)` = `START_WINDOW=(5,10)`, pas (4,9).)

### `sced_bayesian` — effet de condition + Bayes factor (alternant)
```
GROUP_COLS=None -> cas unique | ["patient"]/["site","patient"] -> multiniveau (intercept alea.)
OUTCOME_TYPE: continuous (d) | robust (Student-t) | binary (OR) | count (IRR)
```
**Formule** : `y ~ Famille(lien(mu + beta.condition + (1|sujet)))` -> postérieur, HDI,
P(bénéfique), **BF10** (H1 vs H0 ; <1 = appui H0), décision ROPE optionnelle.

### `sced_bayesian_mbd` — bayésien de PHASES, unifié (★ phare) : 3 axes `POOLING` · `ONSET` · `AR`
Un seul template pour l'estimation bayésienne des designs en phases (A→B). Le moteur est **dérivé**
de trois axes orthogonaux (plus de `MODE`/`MODEL`) :
```
POOLING : partage entre cas
   partial(RI+RS) | random_intercept(RI) | correlated(LKJ) | none (PAR CAS indépendant) | meta(two-stage)
ONSET   : known   → bascule = étiquette de phase (modèle hiérarchique ; HYPOTHESIS/TERMS = b1/b2/b3)
          unknown → point de bascule ESTIMÉ (BUCP) ; NON poolable → force POOLING="none"
AR      : bruit AR1 intra-cas (= l'ancien "BITS") ; gaussian/student ; IGNORÉ en beta/binomial
FAMILY  : gaussian | student | beta | binomial         (dict {outcome: famille} possible)
```
- `pooling="none"` (+`ar`) **remplace** l'ancien per_case simple/BITS (mêmes fits indépendants, mais
  sortie b1/b2/b3 en points, méta-analysable).
- `onset="unknown"` **remplace** l'ancien per_case BUCP (la seule capacité vraiment distincte).
- Rétro-compat : `mode`/`model` (ancienne API) restent acceptés avec un **warning de dépréciation**.

**Formule (onset=known)** : `y_it = b0_i + b1_i·t_c + b2_i·phase + b3_i·tsince + AR1`,
`(b)_i ~ Normal(μ, Σ)` → μ population + per-cas + τ + BC-SMD `μ_b2/√(τ_b0²+σ²)`.
**Formule (onset=unknown)** : `y_t = α + β·phase (+ ρ·AR1) + point de bascule inconnu` ; `es = β/σ`.
Choix : population/partage → `pooling` ; immédiateté/bascule inconnue → `onset="unknown"`. Voir §Choix.

### `sced_model_compare` — comparaison de modèles (LOO + WAIC)
```
Compare la MOYENNE (M0 trend / Mi +level / Mg +slope / Mf full) ; AR & POOLING FIXÉS.
Comparateurs : PSIS-LOO + WAIC ; poids stacking + pseudo-BMA+.
```
**Décision** : décisif si `Δelpd > SE_MULT × dse` ; sinon indistinguable -> plus simple /
model-averaged. Complexité : `n_params`, `n_par/n_cas`, `p_eff/N_obs`. Notation **b1/b2/b3**
(`Termes (b)`) dans LOO/WAIC + Décisions.
> **Wrapper** (`PER_MODEL_REPORTS=True`, défaut) : en plus de la comparaison, sort le **rapport
> bayes hier de CHAQUE modèle** dans un sous-dossier (`b1/`, `b1b2/`, `b1b3/`, `b1b2b3/`) — réutilise
> le **cache** (instantané). Les `.nc` sont fittés une fois, partagés avec hier/méta.

### `sced_power_planning` — puissance a priori / MDES (aucune donnée)
**Formule** : **simulation** du test de randomisation réel sous un effet visé -> puissance vs nb
d'unités, et MDES. (+ forme fermée plans (AB)ᵏ, Hedges 2022.)

---

## Choix transversaux (bayésien) — hypothèse posée & implication

| Choix | Si activé… | Hypothèse posée | Implication / quand |
|---|---|---|---|
| **hiérarchique** (pooling≠none) | un modèle, effets tirés d'une loi de groupe | cas **échangeables** | rétrécit le bruit, donne μ + τ ; défaut multi-cas |
| **random intercept** (RI) | niveau varie/cas, **effet commun** | l'intervention agit **pareil** chez tous | n petit, on veut juste μ |
| **random slopes** (RS = `partial`) | l'**effet** varie/cas (τ par effet) | même **direction**, magnitudes différentes | n ≥ ~6-8, on veut l'hétérogénéité |
| **correlated** (LKJ) | RI+RS **corrélés** | niveau et effet **covarient** | capte le **plafond** (niveau haut→effet faible) ; très gourmand |
| **none** | k cas indépendants | aucun partage | cas non échangeables (= ex per_case simple/BITS) |
| **meta** | stage-1 `none` → méta-analyse stage-2 | échangeables, SE par cas fiables | μ/τ/I² + intervalle de prédiction ; brut vs rétréci par cas |
| **ONSET=unknown** (BUCP) | le moment de l'effet est **estimé** | bascule inconnue (immédiateté testée) | **non poolable** → force `pooling="none"` ; immédiateté / effet différé |
| **AR1** | corrélation lag-1 des résidus | résidus auto-corrélés | gaussian/student ; **ignoré en beta** (logit) ; **danger** séries courtes |
| **gaussian** | Normal | continu, milieu d'échelle | défaut ; donne BC-SMD |
| **student** | queues lourdes | outliers présents | robuste ; garde BC-SMD |
| **beta** | lien logit, borné | score **borné** (plancher/plafond) | effet en POINTS ; pas de BC-SMD |
| **binomial** | k/N | comptage d'items | requiert N_TRIALS ; effet en POINTS |
| **HYPOTHESIS / terms** | quels termes b1/b2/b3 | la forme (saut vs pente) | **pré-spécifier** ; sinon comparer (`sced_model_compare`) |
| **ROPE** | seuil d'équivalence pratique | effet < ROPE = négligeable | nombre OU **`"auto"`** (=0.1·SD, conv. Kruschke/bayestestR) ; rapporte **% in ROPE** + **décision HDI vs ROPE** (effet / équivalence / indécis ; Kruschke 2018) + P(>ROPE) ; régler au **MCID** |

**Règles d'or** : (1) randomisation = inférence **primaire** (assumption-light) ; bayésien =
**secondaire** riche. (2) Statistique **trend-robuste** (slope / Tau-U corrigé) si la baseline
monte (récupération spontanée). (3) Sur petit N, « indistinguable » = **manque de puissance**,
pas preuve d'absence -> lire le **HDI**, pas seulement le « gagnant ».

Moteurs (calcul) : `SCED_core`, `SCED_alternating(_group/_run)`, `SCED_multiple_baseline`,
`SCED_mbd_procedures`, `SCED_phase_design`, `SCED_multivariate`, `SCED_mbd_multilevel`,
`SCED_hierarchical`, `SCED_multilevel`, `SCED_bayesian`, `SCED_mbd_bayesian`, `SCED_mbd_meta`,
`SCED_power`.
Rapports (orchestration + écriture xlsx stylée, appelés par les templates) :
`SCED_mbd_report` (MBD), `SCED_bayes_report` (model_compare + bayesian_mbd),
`SCED_phase_design` (report_*), `SCED_alternating_run` (report_*), `SCED_bayesian` (report_*).
