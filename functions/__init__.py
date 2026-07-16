"""Reusable statistics pipeline.

Function library for the statistical analysis of cohorts and single cases:
descriptive analysis, univariate tests and regressions, collinearity (VIF,
correlograms), multivariate selection (stepwise, Lasso/Elastic-Net, stability
selection), paired/longitudinal analysis (pre/post or >2 time points: paired
tests, mixed/GEE models, multivariate PERMANOVA, count outcomes and effect
sizes), single-case experimental designs (SCED: randomization test, Tau-U/NAP,
segmented regression, Bayesian models), hierarchical models and visualization.

The code is grouped into three sub-packages:

- ``functions.common`` - cross-family plotting helpers (plotstyle, viz).
- ``functions.sced`` - single-case designs (core, prep, glossary, power,
  intravisit, plots/, alternating/, mbd/, bayes/).
- ``functions.general`` - group designs (preprocessing, univariate,
  collinearity, multivariate/, longitudinal/, mixed/).

Sub-modules are imported explicitly by the analysis scripts. This ``__init__``
deliberately performs no eager imports, so that matplotlib, bambi or PyMC are
loaded only when actually needed.
"""
