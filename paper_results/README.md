# Preserved paper results

This folder contains the identifiable final result archives and supporting
files from the original checkout. The NumPy archives and figure PDFs were
copied unchanged; [manifest.json](manifest.json) records their original paths
and SHA-256 checksums.

## Readable comparison results

- [summary.csv](summary.csv): mean and population standard deviation of test
  log-likelihood across the 20 splits for each available dataset/method pair.
- [per_split_scores.csv](per_split_scores.csv): individual test scores, with
  validation-selected component counts for SME and GMM. Splits are numbered
  from 1 to 20.

To regenerate these tables after installing the library:

```sh
python paper_results/summarize.py
```

For SME, the export selects between `k = 2, 3, 4` by validation likelihood, as
in the paper. The original kin8nm archive also contains `k = 5, 6` evaluations;
they remain in the unchanged archive but are excluded from the CSV export.
The CSV files report the original numerical values without rounding them to
the paper's display precision.

## Archive contents

| Folder | Preserved material |
| --- | --- |
| `hepmass/` | Main SME, GMM, GMM-AA, KDE, and NB-KDE result archives; original splits; large appendix diagnostic. |
| `kin8nm/` | Main SME, GMM, GMM-AA, KDE, NB-KDE, and MAF result archives; original splits. |
| `synthetic/` | Original 900-point synthetic dataset and the true/1/2/4/6-projection partitions, plus their five figure PDFs. |

Each dataset's `splits.npz` is the original saved split archive. The other
method archives retain their original fields, including fitted states where
those were saved. The final HEPMASS MAF result archive was not identified in
this checkout, so no replacement result has been substituted.

`hepmass/objective_likelihood.npz` is the 20,000-training-sample, 80-component
single-run appendix diagnostic. Its script is
[`objective_likelihood.py`](../experiments/hepmass/objective_likelihood.py).
The separate 20-run diagnostic with 500 training samples and five components
was not located among the available scripts and result archives.

See [the experiment guide](../experiments/README.md) for the retained fitting
scripts and instructions for using the original splits. The fresh synthetic
demonstration in `examples/` is separate from the preserved paper figures here.
