"""
Cluster-based permutation for longitudinal / SCED designs (edge and element space).

Array-in counterpart of the scalar SCED toolbox : a per-element GLM, thresholded, grouped
into connected components on an adjacency, with cluster-level FWER by a Freedman-Lane
max-component permutation null.

  - core    : the shared engine (statistic map, cluster_run, term design).
  - network : graph edges, shared-node adjacency (NBS ; Zalesky 2010).
  - spatial : element grid, generic adjacency (electrode x freq ; Maris-Oostenveld 2007).
"""

from .core import (triu_edges, edge_components, adjacency_components, freq_product_adjacency,
                   relu_run, scalar_relu)
from .network import (nbs_glm, nbs_trend, nbs_freedman_lane, nbs_step, nbs_relu,
                      edge_adjacency, nbs_freq_trend, nbs_freq_freedman_lane, nbs_freq_relu)
from .spatial import (spatial_glm, spatial_trend, spatial_freedman_lane, spatial_relu,
                      spatial_contrast)
from .design import term, run_ancova
from .report import report_sced_cluster, describe_clusters, summary_rows, grouped_axis_reports

__all__ = ["triu_edges", "edge_components", "adjacency_components", "freq_product_adjacency",
           "relu_run", "scalar_relu",
           "nbs_glm", "nbs_trend", "nbs_freedman_lane", "nbs_step", "nbs_relu",
           "edge_adjacency", "nbs_freq_trend", "nbs_freq_freedman_lane", "nbs_freq_relu",
           "spatial_glm", "spatial_trend", "spatial_freedman_lane", "spatial_relu",
           "spatial_contrast", "term", "run_ancova",
           "report_sced_cluster", "describe_clusters", "summary_rows", "grouped_axis_reports"]
