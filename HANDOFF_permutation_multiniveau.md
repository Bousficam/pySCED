# Permutation multi-niveaux : etat, limite, et ce qu'il y a a faire

Note de passation pour un agent qui travaille sur pySCED. Ecrite le 2026-08-05, depuis l'analyse
ARM1 (BCINET) qui consomme ces primitives.

---

## 1. Ou est le code

| Quoi | Ou |
|---|---|
| Primitive de test par permutation | `BCINET/projects/bcinet/scripts/BCI_SESSION/stats/sced_templates/hier_model.py::block_test` |
| Assemblage des deux niveaux | meme fichier, `_fit_level` et `fit` |
| Statistique elementaire, TFCE, composantes | `pySCED/functions/sced/cluster/core.py` (`_glm_statmap`, `_tfce_map`, `adjacency_components`) |

`block_test` est la brique. `fit` decide de la structure. La distinction compte pour la suite.

---

## 2. Ce qui est implemente, et correctement

D'apres [[winkler-2014-permutation-inference-general-linear-model]] (page wiki
`methods/neuroimaging/Randomise.md`) :

- **Freedman-Lane** pour les termes continus, **Draper-Stoneman** pour les facteurs randomises.
  Ce sont deux des 8 strategies du catalogue, et FL est celle que Winkler recommande.
- **Blocs d'echangeabilite** : `block_test(blocks=...)` fixe entre quelles lignes la permutation
  peut echanger, independamment de `units` qui fixe les indicatrices d'effet fixe. Les deux
  parametres ont longtemps ete confondus parce qu'ils avaient la meme valeur ; ils sont maintenant
  distincts et documentes.
- **Groupes de variance et statistique G** : `block_test(stat="G", vg=...)` existe et est teste.
  La regle de Winkler 2015 - deux lignes echangeables partagent leur groupe de variance, donc
  `vg == blocks` - est ecrite dans le docstring.

---

## 3. LA limite : le regime whole-block n'existe pas

Winkler distingue deux regimes de permutation des blocs : **within-block** (echanger les lignes a
l'interieur d'un bloc) et **whole-block** (echanger des blocs entiers en conservant leur ordre
interne).

`_perm_within` ne sait faire que le premier :

```python
def _perm_within(idx_by_unit, rng, n):
    out = np.arange(n)
    for rows in idx_by_unit.values():
        out[rows] = rng.permutation(rows)     # melange DANS le bloc
    return out
```

Consequence directe : sur une table au niveau run, on ne peut pas tester un terme de niveau
seance, puisque cela demanderait de deplacer les 9 runs d'une seance ENSEMBLE.

---

## 4. Ce que le code fait a la place, et pourquoi ca tient

Faute de whole-block, `fit` fait **deux ajustements separes sur deux tables** :

```python
out += _fit_level(df_sess, ..., level="seance", blocks=subject)
out += _fit_level(df_run,  ..., level="run",    blocks=subject|session)
```

Sur `df_sess`, une seance EST une ligne : "permuter des seances entieres" redevient "permuter des
lignes", et `_perm_within` suffit. **L'agregation prealable est un contournement de l'absence de
whole-block, pas un choix statistique premier.** Le commentaire du code ne le dit pas.

C'est l'approche **summary statistics** en deux niveaux, dont la reference canonique est
[[beckmann-2003-multilevel-linear-modeling-group-fmri]] (page wiki
`methods/neuroimaging/MultilevelGLM.md`), et non le multi-level block permutation de Winkler.

### Ce qui la rend valide ici, et ce qui la rendrait invalide ailleurs

Beckmann donne la condition d'equivalence exacte entre le fit en deux niveaux et le fit poole :

```
V_G2 = V_G + (X' V^-1 X)^-1
```

soit la covariance inter-unites PLUS la covariance des estimations de premier niveau. **Le code
n'ajoute pas le second terme** : il moyenne les runs et ignore leur precision. C'est la version
homoscedastique, celle que Beckmann decrit comme le standard anterieur.

Elle donne malgre tout des estimations ponctuelles exactes **sous equilibre du design**. Sur ARM1
c'est le cas : 9 runs dans chacune des 73 seances, verifie. Les estimations de niveau seance sont
bit-a-bit identiques qu'on inclue ou non le niveau run - ecart mesure 0.00e+00 sur 37 termes.

> **A verifier avant toute reutilisation** : si le design est desequilibre (runs manquants,
> nombre de runs variable), l'equivalence tombe et les estimations de niveau haut deviennent
> biaisees par la ponderation implicite. Rien dans le code ne controle l'equilibre ni ne previent.

---

## 5. Ce qu'il y a a faire, par ordre de rapport benefice/cout

### 5.1 Cabler la statistique G (facile, effet immediat)

`block_test` accepte deja `stat="G"` et `vg`, mais **aucun appelant ne les passe** : `hier_model.py`
ne les renseigne nulle part, donc la regle historique s'applique (t pour un bloc a une colonne,
F au-dela, variance supposee homogene).

Sur ARM1 l'heterogeneite inter-patients est forte et documentee. Winkler montre que sous
heteroscedasticite le F classique n'est pas pivotal alors que le G l'est. Passer `stat="G",
vg=blocks` dans `_fit_level` est un changement de deux lignes.

**Ne pas le faire en aveugle** : mesurer d'abord l'effet sur les p d'un ou deux outcomes. Si les p
ne bougent pas, l'heterogeneite n'a pas de consequence et le changement n'apporte rien.

### 5.2 Ajouter le regime whole-block (moyen)

Environ quinze lignes dans `_perm_within` : un mode qui permute l'ORDRE des blocs en conservant
l'ordre interne, au lieu de melanger a l'interieur.

Utile en soi, meme sans le point 5.3 : c'est le regime correct pour tout terme de niveau haut teste
sur une table de niveau bas.

### 5.3 Modele unique multi-niveaux (lourd, benefice incertain ici)

Trois changements, dont un seul dans `block_test` :

1. `_perm_within` gagne le mode whole-block (point 5.2) ;
2. `_fit_level` construit UN design portant les six termes sur la table de run ;
3. `fit` route l'echangeabilite PAR TERME : whole-block pour `session`, `phase`, `fat_base` ;
   within-block pour `run`, `run:phase`, `run:fat_base`.

**Ce que ca changerait** : les estimations ponctuelles, non (design equilibre). Les erreurs
standard, oui - le modele unique utiliserait la variance intra-seance que l'agregation jette.

**Le risque a connaitre** : sur un modele unique au niveau run, si l'echangeabilite des termes de
seance est mal cablee, on retombe en pseudoreplication et les p de niveau seance deviennent
anticonservateurs. L'approche a deux tables est immunisee contre cette erreur par construction.
C'est son principal merite, et la raison de ne pas la remplacer sans necessite.

---

## 6. Ce qu'il faut ecrire dans un article, et ne pas ecrire

**Ne pas ecrire** "modele hierarchique multi-niveaux (Winkler)". C'est faux sur deux plans : ce
n'est pas un ajustement unique, et le regime whole-block qui le rendrait possible n'est pas code.

**Ecrire**, en trois temps :

1. schemas de permutation (Freedman-Lane, Draper-Stoneman) et blocs d'echangeabilite ->
   **Winkler et al. 2014** ;
2. structure en deux niveaux -> **Beckmann et al. 2003**, version homoscedastique, valide ici
   parce que le design est equilibre ;
3. dire explicitement que le multi-level block permutation de Winkler n'est PAS utilise.

Un reviewer qui connait Winkler verrait immediatement que le whole-block manque.

---

## 7. Autres points ouverts sur ces primitives

- **TFCE** : `TFCE_E_2D = 2/3`, `TFCE_H = 2.0`, 50 pas. Balayage mene sur une seance ARM1 (grille
  21 x 13, 162 essais apparies, 2000 permutations) : E de 0.5 a 1.0 change le masque de 194 a 203
  cellules sur 273, et **le pic ne bouge pas**. Raison structurelle : le seuil vient du maximum de
  la carte TFCE PERMUTEE, enhancee avec les memes exposants, donc le rapport est insensible au
  reglage. Ne pas esperer resserrer un cluster en touchant E ou H.
- **Graine non rejouable** : `bootstrap_pooled.py:139` utilise `abs(hash(...))`, randomise par
  processus depuis Python 3.3. Les npz pooles ne sont pas reproductibles. Non corrige.
