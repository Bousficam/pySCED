"""
TEMPLATE DE VISUALISATION SCED — dédié aux FIGURES (séparé des pipelines d'analyse)
==================================================================================

Les pipelines d'analyse (MBD, bayésien, comparaison de modèles) ne produisent plus de plots :
elles écrivent des RAPPORTS .xlsx + des MODÈLES .nc. Ce template centralise TOUTES les figures
dans un seul dossier ``PLOT_PATH`` :

  1. **Plots SCED des données** (panneaux étagés A/B, tendance baseline projetée, Brinley, VAIOR)
     — lus depuis les données brutes ; type selon ``DESIGN`` (mbd | alternating).
  2. **Diagnostics MCMC** des modèles bayésiens (trace, rank, posterior+HDI, PPC) — lus depuis
     les fichiers ``*.nc`` sauvegardés par les pipelines bayésiennes (``MODELS_DIRS``).

Usage : éditer les zones ▸ puis `python template_sced_visualize.py`, ou piloter via un script.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import os
import glob

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =========================================================================== #
#  ▸▸▸ 1. DONNÉES (pour les plots SCED)
# =========================================================================== #
# Chaque knob *_COL = le NOM EXACT de la colonne dans ton fichier (pas la valeur).
CSV_PATH     = None              # ▸ chemin du fichier .csv ou .xlsx. None = pas de plot de données.
TIER_COL     = "tier"            # ▸ colonne du PALIER/cas (1 série par palier). Ex: "Participant".
SESSION_COL  = "session"         # ▸ colonne de l'ORDRE TEMPOREL (entier). Calculée si DATE_COL renseigné.
PHASE_COL    = "phase"           # ▸ colonne de PHASE (valeurs de PHASES, défaut "A"/"B").
# --- Format des outcomes : TIDY (OUTCOME_NAME_COL+VALUE_COL) OU LARGE (1 colonne/outcome) ---
OUTCOME_NAME_COL = None          # ▸ (TIDY) colonne qui NOMME l'outcome. Ex: "Epreuve". None = format large.
VALUE_COL        = None          # ▸ (TIDY) colonne du SCORE numérique. Ex: "Score". None = format large.
OUTCOMES     = ["accuracy"]      # ▸ LARGE: liste des colonnes-mesures. TIDY: None = toutes, ou liste.
CASE_INSENSITIVE = True          # ▸ True/False : tolère casse/espaces sur colonnes + phases.
DATE_COL     = None              # ▸ nom d'une colonne DATE -> session 1..n par palier. None = pas de dates.
DATE_DAYFIRST = False            # ▸ True/False : True si dates JJ/MM/AAAA.
GROUP_COL    = None              # ▸ colonne de COHORTE (+ « (tous) »). None = une seule cohorte.
PHASES          = ("A", "B")
BASELINE_PHASE  = None
TREATMENT_PHASE = None
IMPROVEMENT  = "increase"
DESIGN       = "mbd"             # ▸ "mbd" (lignes de base décalées) | "alternating"
BOUNDS       = None              # ▸ (lo, hi) : borne l'axe outcome à l'échelle RÉELLE (ex. (0,100)) —
#                                  tuple OU dict {outcome: (lo,hi)} ; None = auto-échelle

# =========================================================================== #
#  ▸▸▸ 2. MODÈLES BAYÉSIENS (.nc) pour les diagnostics MCMC
# =========================================================================== #
MODELS_DIRS  = None              # ▸ liste de dossiers contenant des *.nc (ex. [".../models"]) ; None = aucun
VAR_NAMES    = None              # ▸ paramètres à tracer (None = auto : pop_level, es, …)

# =========================================================================== #
#  ▸▸▸ 3. SORTIE
# =========================================================================== #
PLOT_PATH    = "plots"           # ▸ dossier UNIQUE de toutes les figures
STYLE        = None              # ▸ style GLOBAL des figures SCED (légende/couleurs/axes/tailles). None =
#                                  défauts ; dict de surcharges OU functions.common.plotstyle.PlotStyle.
#                                  (les diagnostics MCMC gardent un style fixe.)
SCED_PLOTS   = True              # ▸ produire les plots SCED des données
BRINLEY_MODE = "classic"         # ▸ "classic" (1 point/patient, moyennes A/B) | "paired" (apparié k-à-k)
DIAGNOSTICS  = True              # ▸ produire les diagnostics MCMC depuis les .nc
DIAG_IN_MODEL_DIR = True         # ▸ diagnostics dans le dossier de CHAQUE modèle (à côté de models/) ;
#                                  False = tout dans PLOT_PATH/diagnostics/
PANEL        = True              # ▸ aussi un tableau de bord des diagnostics les plus importants (1 figure)
VAIOR        = True              # ▸ aussi la grille VAIOR des données (baseline vs intervention)
GROUPED_PPC  = ("phase", "case") # ▸ PPC groupé (par "phase" et/ou "case") dans diagnostics/grouped/ ; () = aucun


def _load_data():
    p = str(CSV_PATH)
    return pd.read_csv(p) if p.endswith(".csv") else pd.read_excel(p)


def _sced_plots():
    from functions.sced.prep import resolve_columns, harmonize_conditions
    from functions.sced.plots.panels import plot_mbd_panels, plot_sced_panels
    df = _load_data()
    tc, sc, pc = TIER_COL, SESSION_COL, PHASE_COL
    date_col, group_col, on, vc = DATE_COL, GROUP_COL, OUTCOME_NAME_COL, VALUE_COL
    baseline = BASELINE_PHASE if BASELINE_PHASE is not None else PHASES[0]
    treatment = TREATMENT_PHASE if TREATMENT_PHASE is not None else PHASES[-1]
    if CASE_INSENSITIVE:
        r = resolve_columns(df, {"tier": tc, "session": sc, "phase": pc, "date": date_col,
                                 "group": group_col, "outcome_name": on, "value": vc})
        tc, sc, pc, date_col, group_col, on, vc = (r["tier"], r["session"], r["phase"], r["date"],
                                                   r["group"], r["outcome_name"], r["value"])
        if pc:
            df = harmonize_conditions(df, pc, conditions=PHASES)
    if date_col:
        from functions.sced.prep import dates_to_sessions
        sc = sc or "session"
        df = dates_to_sessions(df, date_col, unit_col=tc, new_col=sc, dayfirst=DATE_DAYFIRST)
    outcomes = list(OUTCOMES) if OUTCOMES else []
    if on and vc:
        from functions.sced.prep import unstack_outcomes
        keep = [group_col] if group_col else []
        df, names = unstack_outcomes(df, outcome_name_col=on, value_col=vc, id_cols=[tc, sc, pc], keep_cols=keep)
        outcomes = outcomes or names
    cohorts = ([(g, gd) for g, gd in df.groupby(group_col)] if group_col else [(None, df)])
    if group_col and len(cohorts) > 1:
        cohorts = cohorts + [("(tous)", df)]
    n = 0
    for gname, gd in cohorts:
        gtag = "(tous)" if (gname is None or gname == "(tous)") else str(gname)
        for oc in outcomes:
            d = gd.dropna(subset=[oc])
            if d.empty:
                continue
            try:
                bnd = BOUNDS.get(oc) if isinstance(BOUNDS, dict) else BOUNDS
                if DESIGN == "alternating":
                    fig = plot_sced_panels(d, session_col=sc, condition_col=pc, outcome_col=oc,
                                           conditions=PHASES, unit_col=tc, brinley_mode=BRINLEY_MODE,
                                           bounds=bnd, style=STYLE)
                    fig.savefig(os.path.join(PLOT_PATH, f"sced_{oc}_{gtag}.png"), dpi=130, bbox_inches="tight")
                    plt.close(fig); n += 1
                else:
                    starts = d[d[pc].astype(str) == str(treatment)].groupby(tc)[sc].min()
                    if starts.empty:
                        continue
                    # plot_mbd_panels sauvegarde lui-même (wrap >5, lots >10) -> name = préfixe fichier
                    plot_mbd_panels(d, tier_col=tc, session_col=sc, outcome_col=oc,
                                    starts={k: int(v) for k, v in starts.items()}, phase_col=pc,
                                    baseline=baseline, treatment=treatment, improvement=IMPROVEMENT,
                                    brinley_mode=BRINLEY_MODE, bounds=bnd,
                                    name=f"sced_{oc}_{gtag}", save_path=PLOT_PATH, style=STYLE)
                    plt.close("all"); n += 1
                    if VAIOR:                                  # VAIOR aussi pour les MBD (baseline vs B), y borné
                        try:
                            from functions.sced.plots.vaior import plot_vaior_grid
                            plot_vaior_grid(d, unit_col=tc, session_col=sc, condition_col=pc,
                                            outcome_col=oc, reference=baseline, compared=treatment,
                                            improvement=IMPROVEMENT, bounds=bnd,
                                            name=f"vaior_{oc}_{gtag}", save_path=PLOT_PATH, style=STYLE)
                            plt.close("all")
                        except Exception as e:
                            print(f"  [VAIOR ignoré] {gtag}/{oc}: {type(e).__name__}: {e}")
            except Exception as e:
                print(f"  [SCED ignoré] {gtag}/{oc}: {type(e).__name__}: {e}")
    print(f"Plots SCED des données : {n} figures → {PLOT_PATH}/")


def _diagnostics():
    import arviz as az
    from functions.sced.plots.panels import plot_bayesian_diag, plot_bayesian_panel, plot_ppc_grouped
    from functions.sced.glossary import bayesian_diag_legend
    n = 0
    for mdir in (MODELS_DIRS or []):
        ncs = sorted(glob.glob(os.path.join(mdir, "*.nc")))
        if not ncs:
            continue
        # diagnostics dans le dossier du MODÈLE (parent de models/) ; sinon centralisé
        diagdir = (os.path.join(os.path.dirname(mdir.rstrip("/")), "diagnostics")
                   if DIAG_IN_MODEL_DIR else os.path.join(PLOT_PATH, "diagnostics"))
        os.makedirs(diagdir, exist_ok=True)
        try:                                              # légende d'interprétation à côté des figures
            bayesian_diag_legend().to_csv(os.path.join(diagdir, "_legende_diagnostics.csv"), index=False)
        except Exception:
            pass
        for nc in ncs:
            try:
                idata = az.from_netcdf(nc)
                name = os.path.splitext(os.path.basename(nc))[0]
                plot_bayesian_diag(idata, save_path=diagdir, name=name, var_names=VAR_NAMES, style=STYLE)
                if PANEL:
                    plot_bayesian_panel(idata, save_path=diagdir, name=name, var_names=VAR_NAMES, style=STYLE)
                for by in (GROUPED_PPC or ()):            # PPC groupé -> sous-dossier dédié
                    plot_ppc_grouped(idata, by=by, save_path=os.path.join(diagdir, "grouped"), name=name,
                                     style=STYLE)
                n += 1
            except Exception as e:
                print(f"  [diag ignoré] {os.path.basename(nc)}: {type(e).__name__}: {e}")
    print(f"Diagnostics MCMC : {n} modèles (legende + panel" + (" + PPC groupé" if GROUPED_PPC else "") + ")")


def main():
    os.makedirs(PLOT_PATH, exist_ok=True)
    if SCED_PLOTS and CSV_PATH:
        _sced_plots()
    if DIAGNOSTICS and MODELS_DIRS:
        _diagnostics()
    print(f"\n=== Figures dans {PLOT_PATH}/ ===")


if __name__ == "__main__":
    main()
