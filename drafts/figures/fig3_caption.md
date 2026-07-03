**Figure 3.** The load-adaptive controller tracks engine load. Per-decision
realized prune rate ``r`` (blue) and the controller's load signal — concurrency
fraction ``n_running / max_num_seqs`` (aqua) — over decision index, under the
low→high→low step profile on GQA (141 segments; ``max_num_seqs=12``). Both
quantities share the [0,1] unit interval, so they are drawn on a single axis.
The controller sits at ``r_min`` (0.25) through the low-load phase, rises to
``r_max`` (0.50) for the segment following the high-batch step (the one-segment
reactive lag of §4.3), and returns to ``r_min`` for the low-load tail. The
concurrency fraction spans the full [0,1] range (versus ~0.00-0.04 for the
abandoned KV-occupancy signal), so the controller exercises its full
[r_min, r_max] swing. *Source: Table F2; `runs/p2_d/p3s1_gqa_adaptive_step_mt32.json`.*
