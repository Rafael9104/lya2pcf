"""
lya2pcf computes the correlation functions of the Lyman-alpha forest.

This top-level import stays free of GPU-only dependencies: gpu_support,
correlation_procedures_pycuda and distortion_procedures_pycuda import
pycuda at module level and are meant to be imported explicitly by callers
that need them, so `import lya2pcf` still works on CPU-only nodes.

Public API for building other correlation code on top of lya2pcf's forests
(see IMPROVEMENTS.md #1):

- `lya2pcf.quasar` — the per-forest data object, and `lya2pcf.cosmology`,
  `lya2pcf.parameters` (the config loader) for the data pieces every
  extraction/correlation needs.
- `lya2pcf.gpu_support.compile_kernels` and the device checks in the same
  module, and `lya2pcf.correlation_procedures_pycuda.upload_forests` for
  the reusable GPU-side pieces — import those submodules explicitly.
  `upload_forests` takes a `lya2pcf.pixel_partition.ForestPlan` (see
  `plan_rank_data`), not an in-memory dict: the forests are streamed to the
  GPU one data file at a time.
"""
from . import parameters
from . import cosmology
from .forest_class import quasar

__all__ = ["parameters", "cosmology", "quasar"]
