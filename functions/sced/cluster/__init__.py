"""
Cluster-based permutation for longitudinal / SCED designs (edge and element space).

Array-in counterpart of the scalar SCED toolbox : a per-element GLM, thresholded, grouped
into connected components on an adjacency, with cluster-level FWER by a Freedman-Lane
max-component permutation null.

  - core    : the shared engine (statistic map, cluster_run, term design).
  - network : graph edges, shared-node adjacency (NBS ; Zalesky 2010).
  - spatial : element grid, generic adjacency (electrode x freq ; Maris-Oostenveld 2007).

Reporting a cluster result - three rules this layer now supports explicitly, all from Meyer,
Lamers, Kayhan, Hunnius & Oostenveld (2021, Dev. Cogn. Neurosci. 52:101036), i.e. guidance from a
co-author of the original Maris-Oostenveld method:

  1. EFFECT SIZE as a BOUND PAIR, never one number. ``cluster_effect_size_bounds(res, n_obs=...,
     grid_shape=...)`` returns, per significant cluster, the maximum-within-cluster d (positively
     biased UPPER bound - it is the element the cluster was selected for) and the
     circumscribing-rectangle d (LOWER bound), with the exact extent each covers. On the paper's own
     data these differ by roughly 2x (1.064 vs 0.557), so a single figure would overstate precision.
  2. TWO-TAILED CONVENTION. A ``tail="both"`` run makes two one-sided passes, each against its own
     same-sign null at alpha, so the per-direction p is NOT already two-tailed. The result now
     carries ``tail_convention`` saying so: halve alpha, or double the p, for a two-tailed claim.
     The p itself is deliberately left unchanged (doubling it here would silently move every
     previously reported number).
  3. NEIGHBOURHOOD CHECK. ``check_adjacency(adjacency, names=...)`` verifies symmetry, flags
     isolated elements (which can never join a cluster) and degree inhomogeneity, before any test
     runs. The adjacency is what clustering operates on, so a wrong neighbour definition changes
     which clusters can form at all, silently.

What this layer still does NOT do, by design: it does not estimate an effect's ONSET or extent as
an inferential quantity. Cluster extent is fixed at the non-inferential thresholding stage, so it
carries no error-rate guarantee (Sassenhagen & Draschkow 2019), and Rousselet (2024, bioRxiv
preprint) measures every candidate onset estimator - change point, cluster-depth, cluster-sum, FDR
BH95/BY01, MAX - as biased late by 21 to 45 ms, with cluster-depth among the worst despite its
strong FWER control. Read a cluster as "a difference exists somewhere in this window", nothing more.

That last clause deserves a correction, because it judges a method on a criterion it was not built
for. Frossard & Renaud (2022, NeuroImage 247:118824) designed the cluster depth tests for PER-POINT
error control, not for onset ESTIMATION, and prove asymptotic STRONG FWER control (their Theorem 1,
p. 6) where cluster mass and TFCE give only weak control. What such a result licenses is a bounded
claim, not a point estimate : significant points from 50 to 70 mean "there is at least one region
of true effect from 50 to 70, beginning no later than 50 and ending no earlier than 70" (p. 6).
Rousselet's verdict and theirs are answers to two different questions, and the same method can
reasonably be poor at the first and sound at the second. This layer implements neither : it has no
per-point procedure at all, which is precisely the gap those tests fill.
"""

from .core import (triu_edges, edge_components, adjacency_components, freq_product_adjacency,
                   check_adjacency, block_index, block_permuter, scheme_by_block,
)
from .network import (nbs_glm, nbs_trend, nbs_freedman_lane, nbs_step,
                      edge_adjacency, nbs_freq_trend, nbs_freq_freedman_lane)
from .spatial import (spatial_glm, spatial_trend, spatial_freedman_lane,
                      spatial_contrast)
from .design import term, run_ancova
from .report import (report_sced_cluster, describe_clusters, summary_rows,
                     grouped_axis_reports, cluster_effect_size_bounds,
                     apriori_effect_size)

__all__ = ["triu_edges", "edge_components", "adjacency_components", "freq_product_adjacency",
           "check_adjacency", "block_index", "block_permuter", "scheme_by_block",
           "nbs_glm", "nbs_trend", "nbs_freedman_lane", "nbs_step",
           "edge_adjacency", "nbs_freq_trend", "nbs_freq_freedman_lane",
           "spatial_glm", "spatial_trend", "spatial_freedman_lane",
           "spatial_contrast", "term", "run_ancova",
           "report_sced_cluster", "describe_clusters", "summary_rows", "grouped_axis_reports",
           "cluster_effect_size_bounds", "apriori_effect_size"]
