# Development notes

## Current optimizer bookkeeping

In some `GreedyClimbOptimizer` modes using the `best` improvement strategy,
`best_score` and `best_labels` remain at their initial values even after
accepted improvements. The current `score` and `labels` fields are updated;
use those fields to retrieve a result. The example and retained paper
experiments use the current fields. Fixing the separate `best_*` bookkeeping
remains a follow-up.

## Parallel graph checks

Cached objectives divide graph edges into batches whose sizes differ by at
most one and start at most one worker per edge. Empty graphs and empty
projection coefficient lists are rejected explicitly. Worker exceptions and
exits are reported to the caller, with processes cleaned up before retrying.

The suite contains 154 tests, including checks for nine requested workers on
21- and 24-dimensional graphs, fewer edges than workers, spawn-based execution,
and worker failure handling. All 154 passed on Python 3.12 during repository
preparation on 14 October 2026.

## Paper experiment cleanup — 14 October 2026

The retained experiments are the HEPMASS and kin8nm comparison scripts, their
data-preparation scripts, and the large HEPMASS appendix diagnostic. HEPMASS's
final comparison scripts were moved out of `silhouette-experiments/`; filenames
now describe each method. See [the experiment guide](../experiments/README.md).

Identifiable final results, original splits, the large appendix diagnostic,
and synthetic paper partitions and figures were preserved in `paper_results/`.
Exploratory archives, MNIST, MiniBooNE, performance probes, the old three-cubes
scripts, STEM scripts, unused configuration helpers, and superseded HEPMASS
and kin8nm experiments were removed. The standalone synthetic example remains
in `examples/`.

Retained scripts resolve their split and output paths relative to their own
location and record optimizer losses through `score`. kin8nm's rounding was
set to 0.25 to match the paper and saved final result archive; other fitting
parameters were preserved.
The paper source and its dimension counts were left unchanged.
