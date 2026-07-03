**Figure 2.** Concurrency amplifies the prune-speedup: served req/s on GQA
across prune rate {r0, r50, r75} for three concurrency levels
(``max_num_seqs`` ∈ {1, 4, 12}), batch-submit mode, n=100 per cell. The prune
speedup *grows* with concurrency: at r50 it rises 1.17× (c1) → 1.42× (c12);
at r75 it rises 1.26× (c1) → **1.76× (c12)** — the headline served-throughput
result. A compressor that looks only mildly useful in serial latency (1.26× at
r75/c1) becomes substantially useful (1.76×) under the continuous batching a
real deployer uses. Offline FLOPs measurement, which is independent of
concurrency, cannot see this effect. *Source: Table A; `notes/p2_d_measurements.md` M2.*
