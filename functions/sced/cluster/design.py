"""Declarative SCED ANCOVA over an element map : NAME each variable ``randomized`` or ``fixed`` and
let the router pick the inference engine PER TERM - randomized -> Draper-Stoneman (permute that
term's assignment schedule) ; fixed -> Freedman-Lane (residual permutation of the reduced model).
Every term is tested adjusting for all the OTHERS. A factor term yields one signed contrast axis per
non-reference level ; a continuous term yields one slope axis.

This puts the design knowledge in ONE place (which variable was randomized, with which schedule),
instead of hard-coding FL/DS per axis in each template. The scalar toolbox and the cluster / network
templates can route inference identically from the same declaration.
"""
import numpy as np

from .spatial import spatial_freedman_lane, spatial_huh_jhun, spatial_contrast

__all__ = ["term", "run_ancova"]


def term(name, values, *, role, kind, cond_order=None, dose=None, block_size=None,
         max_consecutive=None, auto=False):
    """Declare one model variable.

    role  : 'randomized' (its assignment schedule is permuted -> Draper-Stoneman) or 'fixed'
            (a non-randomized covariate -> Freedman-Lane).
    kind  : 'continuous' (one slope axis) ; 'factor' (one signed contrast axis per non-reference
            level ; requires cond_order, reference = cond_order[0]) ; 'ordered' (an ORDERED factor
            tested as ONE linear trend across the dose scores `dose` - a single multiplicity-free
            test, exact under the complete null, preferred over splitting an ordered dose into two
            change-point axis on `values` = time, always Freedman-Lane).
    dose  : ordered scores per cond_order level (e.g. [4, 5, 6]) for a 'ordered' factor.
    block_size / max_consecutive : the randomized term's schedule constraints (force / shape DS).
    auto  : for a randomized term, resolve DS vs FL from its collinearity with the fixed covariates
            (choose_permutation_method, Winkler 2014) instead of always forcing DS."""
    return {"name": name, "values": np.asarray(values), "role": role, "kind": kind,
            "cond_order": cond_order, "dose": dose, "block_size": block_size,
            "max_consecutive": max_consecutive, "auto": auto}


def _resolve_perm(t, terms):
    """Permutation method for a term, delegated to the UNIFIED dispatcher (core.recommend_scheme) so
    DS/FL/HJ resolution lives in one place. Fixed -> Freedman-Lane here (a fixed covariate's
    HJ grid is handled by _fixed_axis_grid, which also queries recommend_scheme for its primary).
    Randomized -> Draper-Stoneman, or Freedman-Lane when ``auto`` finds temporal collinearity."""
    if t["role"] != "randomized":
        return "freedman-lane"
    from ..core import recommend_scheme                         # scalar toolbox (statsmodels env)
    fixed_time = next((x for x in terms if x["role"] == "fixed" and x["kind"] == "continuous"), None)
    sessions = fixed_time["values"] if fixed_time is not None else np.arange(len(t["values"]))
    detrend = "linear" if t.get("auto") else "none"            # only run the collinearity check under auto
    return recommend_scheme(role="randomized", kind=t["kind"], values=t["values"],
                            sessions=sessions, detrend=detrend)["primary"]


def run_ancova(Y, adjacency, terms, *, primary_p, n_perm, cluster_stat, tail, seed=0,
               stat=None, vg=None):
    """Run every ANCOVA axis declared by ``terms`` on the element map Y (n_obs, n_elements), each
    with the null adapted to its role and adjusting for all other terms. Returns
    (res_by_axis, clabel_by_axis, perm_by_axis). Axis names : a continuous term keeps its name ; a
    factor term gives '<name>_<level>v<ref>' per contrast.

    stat : None (defaut) laisse chaque axe garder sa statistique historique - 'auto' pour un terme
           ordonne ou continu randomise, 't' pour un contraste de facteur, le t de la charniere pour
           la grille {t, W} pour une covariable fixe. Une valeur explicite ('W',
           'G') remplace cette statistique sur tous les axes ; pour la covariable fixe elle remplace
           la cellule ROBUSTE de la grille, la cellule classique restant 't'.
    vg   : etiquettes de groupe de variance par observation, transmises a la statistique de chaque
           axe (utiles a 'G' seulement). Le blanchiment Huh-Jhun dissout ces groupes, aussi un `vg`
           fourni bascule l'axe covariable fixe sur Freedman-Lane (cf. _fixed_axis_grid)."""
    sv = {"vg": vg} if vg is not None else {}          # ne rien passer quand rien n'est demande
    st = {} if stat is None else {"stat": stat}        # None = defaut propre a chaque axe
    res, clab, pm_by = {}, {}, {}
    for i, t in enumerate(terms):
        # Nuisance = the other terms.
        others = [x for j, x in enumerate(terms) if j != i]
        nuis = [x["values"] for x in others]
        nuis_kind = ["discrete" if x["kind"] in ("factor", "ordered") else "continuous" for x in others]
        pm = _resolve_perm(t, terms)
        tag = "DS" if pm != "freedman-lane" else "FL"
        if t["kind"] == "ordered":                        # ordered factor -> ONE dose-linear trend
            ax = f"{t['name']}_dose"
            lut = dict(zip([str(c) for c in t["cond_order"]], t["dose"]))
            dose_vec = np.array([lut[str(v)] for v in t["values"]], dtype=float)
            res[ax] = spatial_freedman_lane(Y, adjacency, effect=dose_vec, nuisance=nuis,
                effect_kind="continuous", nuisance_kind=nuis_kind, perm_method=pm,
                block_size=t["block_size"], max_consecutive=t["max_consecutive"],
                primary_p=primary_p, n_perm=n_perm, cluster_stat=cluster_stat, tail=tail, seed=seed,
                **st, **sv)
            _tag_axis(res[ax], "ordered", t["name"])
            clab[ax] = f"{t['name']} dose slope /step [{tag}] (red = increase)"; pm_by[ax] = pm
        elif t["kind"] == "factor":
            ref = str(t["cond_order"][0])
            for c in [str(x) for x in t["cond_order"][1:]]:
                ax = f"{t['name']}_{c}v{ref}"
                res[ax] = spatial_contrast(Y, adjacency, factor=t["values"], contrast=c,
                    cond_order=t["cond_order"], nuisance=nuis, nuisance_kind=nuis_kind, perm_method=pm,
                    block_size=t["block_size"], max_consecutive=t["max_consecutive"],
                    primary_p=primary_p, n_perm=n_perm, cluster_stat=cluster_stat, tail=tail, seed=seed,
                    **st, **sv)
                _tag_axis(res[ax], "factor", t["name"])    # all contrasts of a factor share its family
                clab[ax] = f"{c}-{ref} contrast [{tag}] (red = higher at {c})"; pm_by[ax] = pm
        elif t["role"] == "fixed":                          # fixed covariate (trend) -> robust grid
            ax = t["name"]
            res[ax] = _fixed_axis_grid(Y, adjacency, t["values"], nuis, nuis_kind,
                primary_p=primary_p, n_perm=n_perm, cluster_stat=cluster_stat, tail=tail, seed=seed,
                stat=stat, vg=vg)
            _tag_axis(res[ax], "continuous", t["name"])
            sch = res[ax].get("scheme", "huh-jhun")          # primary scheme from recommend_scheme (HJ/FL)
            lab = "HJ" if sch == "huh-jhun" else "FL"
            clab[ax] = f"{t['name']} slope /session [{lab}+W robust] (red = increase)"
            pm_by[ax] = sch
        else:                                               # randomized continuous covariate -> DS
            ax = t["name"]
            res[ax] = spatial_freedman_lane(Y, adjacency, effect=t["values"], nuisance=nuis,
                effect_kind="continuous", nuisance_kind=nuis_kind, perm_method=pm,
                block_size=t["block_size"], max_consecutive=t["max_consecutive"],
                primary_p=primary_p, n_perm=n_perm, cluster_stat=cluster_stat, tail=tail, seed=seed,
                **st, **sv)
            _tag_axis(res[ax], "continuous", t["name"])
            clab[ax] = f"{t['name']} slope /session [{tag}] (red = increase)"; pm_by[ax] = pm
    return res, clab, pm_by


def _fixed_axis_grid(Y, adjacency, effect, nuis, nuis_kind, *, primary_p, n_perm, cluster_stat,
                     tail, seed, stat=None, vg=None):
    """A FIXED nuisance-adjusted covariate (e.g. the time trend) : run the
    {Freedman-Lane, Huh-Jhun} x {F/t, W} grid and keep HUH-JHUN + W as the PRIMARY result - best
    small-n type I control (exact-exchangeability whitening) plus heteroscedasticity robustness
    (White HC), the combination Helwig (2019) recommends. The other three cells become a
    SENSITIVITY list and a Breusch-Pagan homoscedasticity diagnostic is attached. FL and HJ estimate
    the SAME adjusted slope (Frisch-Waugh-Lovell), and F vs W differ only in robustness, so the four
    cells are directly comparable : agreement -> homoscedastic, safe to read F ; divergence -> trust
    the robust W. This is a REPORTED sensitivity, never a data-dependent switch (a pre-test on the
    same data would itself distort the type I error). Randomized terms do NOT come here : their
    Draper-Stoneman randomization, not exchangeability, is the reference set.

    The PRIMARY scheme (HJ at small n, else FL) is decided by the unified dispatcher
    core.recommend_scheme - the SAME rule the randomized terms route through - so scheme choice is
    centralised, not hard-coded here ; the grid always runs both schemes, recommend_scheme only
    picks which cell is the headline. Paired with the robust W statistic.

    ``stat`` remplace la cellule ROBUSTE de la grille (W par defaut) ; la cellule classique reste
    't'/'F', pour que la comparaison garde son sens. ``vg`` ne concerne que la cellule
    Freedman-Lane : le blanchiment Huh-Jhun remplace les observations par des combinaisons
    lineaires de toutes les lignes, ou un etiquetage defini sur les observations n'a plus de
    correspondant. Un ``vg`` fourni fixe donc le schema primaire a Freedman-Lane, faute de quoi le
    resultat de tete ignorerait en silence les groupes de variance demandes."""
    from .core import breusch_pagan_map, _fl_design
    from ..core import recommend_scheme
    prim_scheme = recommend_scheme(role="fixed", kind="continuous",
                                   n=int(np.asarray(effect).shape[0]))["primary"]   # HJ (small n) vs FL
    robust = "W" if stat is None else stat             # cellule robuste de la grille
    cells = tuple(dict.fromkeys(("t", robust)))        # dedoublonne si l'appelant demande 't'
    if vg is not None:
        prim_scheme = "freedman-lane"                  # HJ ne sait pas porter les groupes de variance
    prim, sens = None, []
    for scheme, fn in (("freedman-lane", spatial_freedman_lane), ("huh-jhun", spatial_huh_jhun)):
        for st in cells:
            kw = {"vg": vg} if (vg is not None and scheme == "freedman-lane") else {}
            r = fn(Y, adjacency, effect=effect, nuisance=nuis, effect_kind="continuous",
                   nuisance_kind=nuis_kind, stat=st, primary_p=primary_p, n_perm=n_perm,
                   cluster_stat=cluster_stat, tail=tail, seed=seed, **kw)
            mp = min(r["comp_pvals"].values()) if r["comp_pvals"] else 1.0
            # 'vg' dit si la cellule a bien porte les groupes de variance demandes : une cellule
            # Huh-Jhun ne le peut pas, et une statistique 'G' sans groupe est exactement F, ce que
            # le seul nom de la statistique laisserait croire acquis.
            sens.append({"scheme": scheme, "stat": ("F" if st == "t" else st),
                         "min_p": float(mp), "n_sig": int(r["sig_elements"].sum()),
                         "n_clusters": len(r["comp_pvals"]), "vg": bool(kw)})
            if scheme == prim_scheme and st == robust:            # primary = recommended scheme + robust cell
                prim = r
    design, _tc, _nc, _st = _fl_design(Y, effect, nuis, "continuous", nuis_kind, "t")
    prim["sensitivity"] = sens
    prim["homoscedasticity"] = breusch_pagan_map(Y, design, sig_mask=prim["sig_elements"])
    prim["scheme"] = prim_scheme
    prim["stat_kind"] = robust
    return prim


def _tag_axis(res, kind, family):
    """Stamp the term kind + report family onto an axis result, so downstream code (family folders,
    onset detection) reads them from the RESULT instead of sniffing axis-name conventions."""
    res["kind"] = kind
    res["family"] = family


