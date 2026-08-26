"""ElasticVis — admission-time per-request visual-token budget allocation.

Submodules:
  predictors  — LatencyPred(Σk,load) + AccuracyTerm(k), fit from v2 probe data.
  allocator   — Greedy / Lagrangian / Fixed allocators (goodput@SLO policy).
  sim         — offline discrete-event simulator (zero-GPU go/no-go).

See notes/elasticvis_design.md for the spec.
"""
