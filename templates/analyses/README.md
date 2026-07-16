# Templates d'analyse — index

Chaque template = un bloc **PARAMÈTRES** (zones `▸`) puis `main()`. On l'édite, ou on le pilote
via un script qui surcharge les globals et appelle `main()`. Tout rapport `.xlsx` commence par la
feuille **Données & design** (outcomes, variables, N unités, phases, sessions/unité), + Glossaire
+ Guide d'interprétation (ASCII).

Deux familles, en **sous-dossiers** :

| Dossier | Famille | Unité | Inférence | Quand |
|---|---|---|---|---|
| [`sced/`](sced/README.md) | **SCED** (single-case) | LE patient (séries A/B) | randomisation (design-based) **+** bayésien (model-based) | n petit, mesures répétées intensives, intervention décalée/randomisée |
| [`general/`](general/README.md) | **Général / group-based** (univarié · multivarié · longitudinal) | LE groupe (sujets × temps) | modèles de groupe (mixte / GEE / permutation) | cohorte / RCT, comparaison entre/intra-sujets |

> « group-based » s'oppose à « single-case » : analyses **classiques de groupe** (univariées et
> multivariées, longitudinales), **non** appliquées aux SCED.

- **SCED** → [`sced/README.md`](sced/README.md) : 7 pipelines (alternant, phases, MBD, bayésien
  de condition, bayésien de phases unifié — 3 axes `POOLING`/`ONSET`/`AR`, comparaison de modèles,
  puissance). Templates = **paramétrables** (config + appel) ; analyse/rapports dans `functions/`
  (`report_*`) ; rangement standard `output_dir` (plots/ résultats · diagnostics/ · models/).
  + table « hypothèse & paramètres » (hiérarchique / RI / RS / correlated / AR /
  familles / terms / ROPE).
- **Général** → [`general/README.md`](general/README.md) : 6 scénarios (RCT pré/post, suivi
  multi-temps, comptage, binaire, PERMANOVA, données manquantes) + cadres `template_preprocessing`
  et `template_longitudinal` (racine `templates/`).

**Organigrammes de construction** (comment chaque modèle s'assemble selon les paramètres) :
[`sced/FLOWCHARTS.md`](sced/FLOWCHARTS.md) · [`general/FLOWCHARTS.md`](general/FLOWCHARTS.md).
