"""
TEMPLATE DE PREPROCESSING — guide pas-à-pas (user-friendly)
===========================================================

Objectif
--------
Partir d'un fichier de données brut (Excel) et produire les deux objets attendus
par le pipeline d'analyse :

    db                -> le DataFrame nettoyé/recodé
    var_preprocessed  -> un dict {role: [variables]} (Continuous, Dicho, ...)

Comment l'utiliser
------------------
1. Lance-le tel quel : `python templates/template_preprocessing.py`
   Il tourne sur un MINI JEU DE DÉMO synthétique et affiche `data_overview`,
   pour illustrer le workflow de bout en bout sans fichier réel.
2. Pour une étude réelle : mettre DEMO = False, renseigner XLSX_PATH et
   VARIABLES_A_EXTRAIRE, puis adapte chaque section balisée « ▸ À ADAPTER ».

Principes user-friendly mis en avant
------------------------------------
- `data_overview(db)` : UNE ligne pour comprendre le jeu de données (types, manquants,
  constantes, suggestions de recodage) AVANT de coder quoi que ce soit.
- `find_categorical_variables(db, as_result=True)` : retour NOMMÉ
  (`.dichotomous`, `.multiclass`, `.continuous`) au lieu de tuples à mémoriser.
- Les fonctions de recodage NE MUTENT PAS le DataFrame : réassigner toujours
  `db = recode_*(db, ...)`. Réexécuter une cellule ne crée pas de doublons.
- En cas de faute de frappe sur un nom de colonne, l'erreur te suggère le bon nom.
"""

import pandas as pd

# Rend le package `functions` importable quel que soit le dossier d'où on lance
# le script (ajoute la racine du dépôt à sys.path).
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import du module de preprocessing. `import *` est le style des scripts d'étude
# existants ; pour du code neuf, préférer importer explicitement le nécessaire.
from functions.general.preprocessing import (
    extraire_variables_excel,
    check_variable_names,
    data_overview,
    find_categorical_variables,
    find_time_variables_splitting,
    encode_datetime,
    recode_time_difference,
    recode_categorial_var,
    recode_formula_var,
    replace_target_value,
    modify_list,
    rename_columns,
)

# ===========================================================================
# 0) CONFIGURATION  ▸ À ADAPTER
# ===========================================================================
DEMO = True  # True = jeu synthétique intégré ; False = lecture du fichier Excel réel

XLSX_PATH = "/chemin/vers/ta_base.xlsx"  # utilisé seulement si DEMO = False
VARIABLES_A_EXTRAIRE = [
    # "age", "sexe", "creat", "poids", ...
]
SEUIL_UNIQUE = 10  # >= ce nb de modalités distinctes -> variable considérée continue


def _charger_demo():
    """Mini cohorte synthétique pour faire tourner le template sans fichier réel."""
    import numpy as np
    rng = np.random.RandomState(0)
    n = 60
    return pd.DataFrame({
        "age": rng.randint(40, 90, n),
        "sexe": rng.randint(0, 2, n),                         # 0/1
        "poids": rng.uniform(50, 100, n),
        "creat": pd.Series(rng.uniform(50, 120, n)).where(rng.rand(n) > 0.15),  # ~15% manquants
        "pnn": rng.uniform(1, 9, n),
        "lympho": rng.uniform(0.5, 4, n),
        "traitement": rng.choice(["None", "AAG", "AVK", "AOD"], n),  # texte
        "stade": rng.randint(0, 4, n),                        # multiclasse
        "date_diag": pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.randint(0, 300, n), "D"),
        "date_ttt":  pd.to_datetime("2021-01-01") + pd.to_timedelta(rng.randint(5, 320, n), "D"),
        "deces": rng.randint(0, 2, n),                        # outcome
    })


# ===========================================================================
# 1) CHARGEMENT
# ===========================================================================
if DEMO:
    db = _charger_demo()
else:
    db = extraire_variables_excel(XLSX_PATH, VARIABLES_A_EXTRAIRE)

# Vérifie que les noms de colonnes sont des identifiants Python valides
# (les noms invalides cassent les formules de recode_formula_var).
valid, invalid = check_variable_names(db)
if invalid:
    print(f"⚠ Noms de colonnes à renommer (caractères spéciaux/espaces) : {invalid}")
    # Exemple : db = rename_columns(db, [("ancien nom", "nouveau_nom")])


# ===========================================================================
# 2) COMPRENDRE LE JEU DE DONNÉES  (le réflexe à avoir EN PREMIER)
# ===========================================================================
# Un seul appel = un tableau récap trié par % de manquants, avec pour chaque
# colonne : type détecté, n unique, manquants, exemples, drapeaux qualité et
# une suggestion de recodage. Idéal pour repérer constantes, types mixtes, dates.
apercu = data_overview(db, unique_threshold=SEUIL_UNIQUE)
print("\n===== APERÇU DU JEU DE DONNÉES =====")
print(apercu.to_string(index=False))


# ===========================================================================
# 3) DÉTECTER AUTOMATIQUEMENT LES TYPES DE VARIABLES  (retour nommé)
# ===========================================================================
# `as_result=True` renvoie un objet avec attributs explicites (autocomplétion)
# au lieu d'un tuple dont il faut retenir l'ordre.
types = find_categorical_variables(db, unique_threshold=SEUIL_UNIQUE, as_result=True)
print("\n===== TYPES DÉTECTÉS =====")
print(f"Dichotomiques : {types.dichotomous}")
print(f"Multiclasses  : {types.multiclass}")
print(f"Continues     : {types.continuous}")

# On part de la détection automatique, puis on AJUSTE manuellement les cas
# particuliers connus du métier (modify_list ne mute pas la liste d'origine).
#  ▸ À ADAPTER : déplace ici les variables mal classées par l'auto-détection.
dicho_var = list(types.dichotomous)
multiclass_var = list(types.multiclass)
cont_var = list(types.continuous)

# Les colonnes datetime tombent dans « continues » (beaucoup de valeurs uniques)
# mais ne sont pas des prédicteurs continus : on les retire ici, elles seront
# transformées en délais dans la section chronologie ci-dessous.
date_like = [c for c in cont_var if pd.api.types.is_datetime64_any_dtype(db[c])]
cont_var = modify_list(cont_var, date_like)

# Exemple : 'stade' est détecté multiclasse, on le garde en multiclasse ; 'sexe'
# reste dichotomique. Pour forcer un déplacement :
# cont_var = modify_list(cont_var, ["variable_a_retirer"])
# multiclass_var = modify_list(multiclass_var, ["variable_a_ajouter"], operation="add")


# ===========================================================================
# 4) CHRONOLOGIE : combiner dates+heures puis calculer des délais
# ===========================================================================
# Repère automatiquement les colonnes de type date / heure (optionnel, indicatif).
date_cols, time_cols = find_time_variables_splitting(db)
# print("Colonnes date détectées :", date_cols, "| heures :", time_cols)

#  ▸ À ADAPTER : si dates ET heures sont SÉPARÉES, les combiner en datetime.
# date_hour_pairs = [("datetime_admission", "date_adm", "heure_adm"), ...]
# db = encode_datetime(db, date_hour_pairs)

# Calcule un délai entre deux colonnes datetime (ne mute pas db).
if "date_diag" in db.columns and "date_ttt" in db.columns:
    db = recode_time_difference(
        db, new_variable_name="delai_diag_ttt_j",
        datetime1="date_ttt", datetime2="date_diag",
        unit="days", drop_original=False,
    )
    cont_var = modify_list(cont_var, ["delai_diag_ttt_j"], operation="add")


# ===========================================================================
# 5) RECODAGES MÉTIER
# ===========================================================================
# 5a) Recodage catégoriel via une fonction (ou un dict) -> nouvelle colonne.
#  ▸ À ADAPTER : transforme un texte libre en dichotomique 0/1, regroupe des
#     modalités, crée un score ordonné (ordered=True), etc.
if "traitement" in db.columns:
    db = recode_categorial_var(
        db, variable="traitement",
        recode_func=lambda x: 1 if "AAG" in str(x) else 0,
        new_variable_name="antiaggregant",
    )
    dicho_var = modify_list(dicho_var, ["antiaggregant"], operation="add")

# 5b) Recodage par FORMULE : ratios, deltas, seuils, combinaisons logiques.
#     Format : "nouvelle_var ~ expression".  output_type='Float64' pour du continu,
#     'Int64' pour des entiers. Les formules logiques (> < == | &) donnent du 0/1.
formules_continues = [
    "ratio_pnn_lympho ~ pnn / lympho",     # ratio
    # "delta_creat ~ creat_j2 - creat_j0", # delta
]
for f in formules_continues:
    db = recode_formula_var(db, f, output_type="Float64")
    cont_var = modify_list(cont_var, [f.split("~")[0].strip()], operation="add")

formules_dicho = [
    # "insuffisance_renale ~ creat > 100",
    # "evenement ~ deces == 1 | recidive == 1",
]
for f in formules_dicho:
    db = recode_formula_var(db, f, output_type="Float64")
    dicho_var = modify_list(dicho_var, [f.split("~")[0].strip()], operation="add")

# 5c) Remplacer une valeur cible (utile pour les codes 0 = "absent", etc.).
# db = replace_target_value(db, "traitement", target_value=0, new_value="None")


# ===========================================================================
# 6) RÔLES D'ANALYSE  ▸ À ADAPTER
# ===========================================================================
# Renseigner ici, selon la question de recherche, ce qui est exposition / outcome /
# variable d'ajustement. Ces listes alimentent directement `var_preprocessed`.
baseline_var = [v for v in ["age", "sexe", "poids", "creat"] if v in db.columns]
outcome_var = [v for v in ["deces"] if v in db.columns]
only_describe = []   # variables seulement décrites (pas dans les modèles)


# ===========================================================================
# 7) RENOMMAGE LISIBLE (optionnel, pour des tableaux/figures propres)
# ===========================================================================
#  ▸ À ADAPTER : noms « publication ». On répercute le renommage sur les listes.
rename_map = {
    "age": "Age",
    "sexe": "Sexe",
    "creat": "Creatinine",
    # ...
}
db = rename_columns(db, list(rename_map.items()))
for liste in (cont_var, dicho_var, multiclass_var, baseline_var, outcome_var, only_describe):
    liste[:] = [rename_map.get(v, v) for v in liste]


# ===========================================================================
# 8) SORTIES POUR LE PIPELINE D'ANALYSE
# ===========================================================================
var_preprocessed = {
    "Continuous": cont_var,
    "Dicho": dicho_var,
    "Multiclass": multiclass_var,
    "Baseline": baseline_var,
    "Outcomes": outcome_var,
    "Describe": only_describe,
}

if __name__ == "__main__":
    print("\n===== RÉSUMÉ FINAL =====")
    print(f"db : {db.shape[0]} lignes × {db.shape[1]} colonnes")
    for role, variables in var_preprocessed.items():
        print(f"  {role:<11}: {variables}")
    # `db` et `var_preprocessed` sont prêts à être passés au pipeline
    # (ex. pipeline_multiv_logit de functions.general.multivariate.selection).
