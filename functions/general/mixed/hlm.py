import bambi as bmb
import pandas as pd

def build_hlm_model(
    data: pd.DataFrame,
    response: str,
    fixed_effects: list[str],
    random_effects: dict[str, list[str]],
    family: str = "gaussian",
    priors: dict[str, dict] | None = None
) -> bmb.Model:
    """
    Construits un modèle hiérarchique bayésien avec Bambi.

    Args:
        data: DataFrame contenant les données.
        response: nom de la variable dépendante.
        fixed_effects: liste de termes d'effets fixes, e.g. ["x1", "x2 * x3"].
        random_effects: dict mapping nom de groupe -> liste de variables pour pentes aléatoires.
            e.g. {"Sujet": ["1", "Session_c", "ERD_mean_c"]}.
        family: famille de distribution ("gaussian", "beta", "binomial", etc.).
        priors: structure de priors, cf. bambi.Prior.

    Returns:
        Un objet bambi.Model prêt à être ajusté.

    References: Gelman & Hill 2007 (hierarchical / multilevel models); Capretto et al. 2022 (Bambi).
    R equivalent: brms::brm / rstanarm::stan_glmer.
    """
    # Construire la partie fixe
    fixed = " + ".join(fixed_effects)

    # Construire la partie aléatoire
    random_terms = []
    for group, slopes in random_effects.items():
        # slopes list may include '1' for intercept
        term = f"({' + '.join(slopes)} | {group})"
        random_terms.append(term)
    random = " + ".join(random_terms)

    formula = f"{response} ~ {fixed} + {random}"

    # Préparer priors si fournis
    bambi_priors = None
    if priors:
        from bambi import Prior
        bambi_priors = {}
        for term, prior_spec in priors.items():
            bambi_priors[term] = Prior(**prior_spec)

    # Construire le modèle
    model = bmb.Model(
        formula,
        data=data,
        family=family,
        priors=bambi_priors
    )
    return model


def fit_hlm_model(
    model: bmb.Model,
    draws: int = 2000,
    tune: int = 1000,
    target_accept: float = 0.9
) -> bmb.backends.base.Result:
    """
    Ajuste un modèle HLM bayésien.

    Args:
        model: objet bambi.Model.
        draws: nombre d'échantillons après tuning.
        tune: nombre d'itérations de tuning.
        target_accept: taux d'acceptation pour NUTS.

    Returns:
        Résultat de l'ajustement (trace, summary, etc.).

    References: Hoffman & Gelman 2014 (No-U-Turn Sampler, the NUTS/HMC backend).
    R equivalent: brms::brm (Stan HMC/NUTS backend).
    """
    results = model.fit(
        draws=draws,
        tune=tune,
        target_accept=target_accept
    )
    return results

# Exemples d'utilisation
# df: DataFrame long avec colonnes ['Sujet','Session_c','ERD_mean_c','ERD_sd_c','Fatigue_c','Accuracy']
#
# Helper d'utilisation de la librairie
# ------------------------------
# HLM_help() donne un guide rapide pour construire un modèle hiérarchique bayésien :
#
# fixed_effects : liste de termes R-style, p. ex. ['ERD_mean_c', 'ERD_sd_c', 'Fatigue_c', 'Session_c']
# random_effects : dict mapping groupe -> pentes aléatoires, ex. {'Sujet':['1','Session_c']} signifie (1 + Session_c | Sujet)
# family        : 'gaussian','beta','binomial', etc.
# priors        : dictionnaire term->{dist, paramètres}, ex.
#     {
#       'Intercept': {'dist':'Normal','mu':0,'sigma':5},
#       'ERD_mean_c': {'dist':'Normal','mu':0,'sigma':1},
#       'sigma': {'dist':'HalfNormal','sigma':1}
#     }
# draws, tune   : options NUTS
#
# Utilisation typique :
# mod = build_hlm_model(
#     data=df,
#     response='Accuracy',
#     fixed_effects=['ERD_mean_c','ERD_sd_c','Fatigue_c','Session_c'],
#     random_effects={'Sujet':['1','ERD_mean_c','ERD_sd_c','Session_c']},
#     family='beta',
#     priors={
#         'Intercept':{'dist':'Normal','mu':0,'sigma':5},
#         'ERD_mean_c':{'dist':'Normal','mu':0,'sigma':1},
#         'sigma':{'dist':'HalfNormal','sigma':1}
#     }
# )
# res = fit_hlm_model(mod, draws=1000, tune=500, target_accept=0.8)
# print(res.summary())


def HLM_help():
    """
    Affiche un guide rapide pour construire et ajuster un HLM bayésien.
    """
    help_text = '''
    ===== HLM Bayésien Helper =====

1) build_hlm_model(data, response, fixed_effects, random_effects,
                   family='gaussian', priors=None)
   - data           : pandas.DataFrame
   - response       : nom de la variable dépendante (str)
   - fixed_effects  : list de str, termes d'effets fixes (ex. ['x1','x2*x3'])
   - random_effects : dict, clé = groupe, valeur = list de str pour pentes aléatoires
                      (inclure '1' pour l'intercept aléatoire)
   - family         : 'gaussian','beta','binomial', etc.
   - priors         : dict où chaque clé est un terme de la formule et la valeur est un dict :
       {
         '<terme>': {
             'dist': '<DistributionName>',   # e.g. 'Normal','HalfNormal','Gamma','Uniform','Beta'
             # paramètres :
             # Normal:    mu=<float>, sigma=<float>
             # HalfNormal: sigma=<float>
             # Gamma:     alpha=<float>, beta=<float>
             # Uniform:   lower=<float>, upper=<float>
             # Beta:      alpha=<float>, beta=<float>
         },
         'sd(<Groupe>|<terme>)': {'dist':'HalfNormal','sigma':1},  # sd aléatoire
       }
   Ex. :
       priors = {
           'Intercept':              {'dist':'Normal','mu':0,'sigma':5},
           'ERD_mean_c':             {'dist':'Normal','mu':0,'sigma':1},
           'sigma':                  {'dist':'HalfNormal','sigma':1},
           'sd(Subject|Intercept)': {'dist':'HalfNormal','sigma':2},
           'sd(Subject|ERD_mean_c)': {'dist':'HalfNormal','sigma':1}
       }

2) fit_hlm_model(model, draws=2000, tune=1000, target_accept=0.9)
   - model          : objet retourné par build_hlm_model
   - draws, tune    : NUTS sampling parameters
   - target_accept  : taux d'acceptation pour NUTS

3) Choix des priors
   - **Priori par défaut de Bambi** (si priors=None) :  
     • **Intercept** : Normal(0, 2.5)  
     • **Slopes (fixed effects)** : Normal(0, 1)  
     • **sigma** (gaussian) : HalfNormal(1)  
     • **kappa** (beta) : Gamma(1, 0.1)  
     • **sd(random)** : HalfNormal(1)  

   - **Les priors ne sont pas tous obligatoires** : surcharge uniquement ce qui t’intéresse.

   - **Étape 1 : Standardisation**-centrer et réduire les covariables pour rendre ces defaults pertinents.

   - **Intercept** :  
     • Remplacer si la moyenne de y est connue (logit pour beta)  

   - **Pentes** :  
     • `Normal(0,1)` par défaut  
     • `Normal(0,0.5)` pour plus de régularisation  

   - **Variance résiduelle** :  
     • `HalfNormal(1)`  

   - **Concentration beta (kappa)** :  
     • `Gamma(1,0.1)`  

   - **Effets aléatoires** :  
     • `sd(…)=HalfNormal(1)` par défaut  

   - **Vérifier** avec **prior predictive checks** pour t’assurer que les defaults sont appropriés.

   - **Exemples** :  
     1. *Defaults* (aucun prior spécifié)  
        → Intercept~N(0,2.5), slopes~N(0,1), sigma~HN(1), sd(random)~HN(1).  
     2. *Surcharge intercept* pour beta  
        ```python
        priors={'Intercept':{'dist':'Normal','mu':np.log(0.6/0.4),'sigma':2}}
        ```
     3. *Surcharge random slope*  
        ```python
        priors['sd(Subject|Session_c)']={'dist':'HalfNormal','sigma':0.5}
        ```

Exemple minimal :
    mod = build_hlm_model(
        data=df, response='Accuracy',
        fixed_effects=['ERD_mean_c','Session_c'],
        random_effects={'Subject':['1','Session_c']},
        family='beta'
    )
    # utilise defaults de Bambi
    res = fit_hlm_model(mod)
    print(res.summary())
'''
    print(help_text)

def HLM_for_SCED_help():
    """
    Guide général pour construire et ajuster un HLM bayésien dans un SCED (Single-Case Experimental Design).

    Structure SCEDs courants :
      * ATD (Alternating Treatments Design) : phases A, B, (± C) randomisées ou séquentielles
      * MBD (Multiple Baseline Design) : plusieurs lignes de base avant intervention
      * Reversal Design (ABAB) : bascule entre conditions

    Niveaux hiérarchiques :
      1) **Niveau 1**-Observations répétées (trials, runs) au sein de chaque session/phase
      2) **Niveau 2**-Sessions ou phases traitées comme groupes
      (pas de niveau sujet si N=1, mais on peut imbriquer runs → sessions)

    Variables clés à définir :
      - **response** : variable dépendante (p.ex. Accuracy, ERD)
      - **time_var** : temps ou visite (V1, V2, V3…), centré → créer `time_c = time_var - mean(time_var)`
      - **phase_var** : indicatrice ou facteur de condition (A, B, C codés 0/1 ou 3 niveaux)
      - **covariables** : fatigue, dose, etc., idéalement centrées et réduites (`x_c`)
      - **group IDs** : `session_id` ou `phase_id`, et `run_id` si plusieurs mesures intra-session

    1) **Effets fixes** (fixed_effects) :
       - `time_c`                     # tendance linéaire du temps
       - `phase_var`                  # effet de la condition (A vs B vs C)
       - `phase_var:time_c`           # interaction pour changement de pente par phase
       - autres covariables centrées (# fatigue_c, dose_c, etc.)

    2) **Effets aléatoires** (random_effects) :
       - **Session/Phase**  : intercept (`'1'`) ± pente de `time_c` ou de `phase_var`
         ex: `{'Phase': ['1', 'time_c']}` → `(1 + time_c | Phase)`
       - **Run** (optionnel) : intercept par run pour capturer variabilité intra-session
         ex: `{'Run': ['1']}` → `(1 | Run)`
       - Combiner comme :
         ```python
         random_effects = {
             'Phase': ['1','time_c'],
             'Run':   ['1']
         }
         ```

    3) **Famille (family)** :
       - `gaussian` : LMM pour réponse continue (ERD, niveaux de signal)
       - `beta`     : pour proportions dans (0,1) (accuracy continue)
       - `binomial`: pour comptages succès/essais (accuracy nombre de succès)
       - autres selon type de réponse (poisson, gamma…)

    4) **Codage des variables** :
       - **time_var → time_c** : centré (évite corrélation intercept-pente). **Manuellement** créer :
         ```python
         df['time_c'] = df['time_var'] - df['time_var'].mean()
         ```
       - **phase_var** : facteur ordonné ou indicatrices dummy (manuellement) :
         ```python
         df['phase'] = pd.Categorical(df['phase'], categories=['A','B','C'])
         df = pd.get_dummies(df, columns=['phase'], drop_first=True)
         ```
       - **covariables continues (x)** : centrer et éventuellement standardiser :
         ```python
         df['x_c'] = (df['x'] - df['x'].mean()) / df['x'].std()
         ```
       - **Remarque** : Bambi ne centre **pas** automatiquement les variables; il faut **toujours** créer explicitement les versions `_c` dans le DataFrame.

5) **Priors recommandés** (weakly informative par défaut Bambi si non spécifiés) : **Priors recommandés** (weakly informative par défaut Bambi si non spécifiés) :
       - Intercept      ~ Normal(0,2.5)
       - Slopes fixes   ~ Normal(0,1)
       - sigma (gaussian)      ~ HalfNormal(1)
       - kappa (beta)          ~ Gamma(1,0.1)
       - sd(random effects)    ~ HalfNormal(1)

    6) **Workflow** :
       ```python
       # préparation des données
       df['time_c'] = df['visit'] - df['visit'].mean()
       df['phase']  = pd.Categorical(df['phase'], categories=['A','B','C'])
       # définir fixed et random
       fixed_effects = ['time_c', 'phase', 'phase:time_c', 'fatigue_c']
       random_effects= {'Phase': ['1','time_c'], 'Run': ['1']}
       # construire et ajuster
       mod = build_hlm_model(df, 'response', fixed_effects,
                             random_effects, family='beta')
       res = fit_hlm_model(mod)
       print(res.summary())
       ```
    """
    import textwrap
    print(textwrap.dedent(HLM_for_SCED_help.__doc__))

