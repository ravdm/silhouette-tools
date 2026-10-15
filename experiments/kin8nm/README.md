# kin8nm

These are the final full-experiment scripts. Earlier STEM scripts and the
obsolete unstandardized experiment have been removed.

## Input and split preparation

To use the original saved splits, run this from the project root:

```sh
cp paper_results/kin8nm/splits.npz experiments/kin8nm/kin8nm_splits.npz
```

Then proceed to the comparisons below. To generate fresh splits instead,
use the data-preparation script as follows.

`prepare_data.py` expects the numeric CSV file, with a header, at:

```text
~/Documents/datasets/kin8nm.csv
```

Change `path` near the top of the script if the file is elsewhere. As in the
paper, all columns are standardized using the mean and standard deviation of
the full dataset before splitting.

From the project root:

```sh
python experiments/kin8nm/prepare_data.py
```

This writes `experiments/kin8nm/kin8nm_splits.npz`: 20 splits, each with 1,000
training observations, 300 validation observations, and the remaining
observations for testing. Generate it once so all comparison methods use the
same splits.

## Comparisons

```sh
python experiments/kin8nm/fit_sme.py
python experiments/kin8nm/fit_kde.py
python experiments/kin8nm/fit_nb_kde.py
python experiments/kin8nm/fit_maf.py
```

For the Gaussian-mixture comparisons, set `axis_aligned` near the top of
`fit_gmm.py` and run:

```sh
python experiments/kin8nm/fit_gmm.py
```

Use `axis_aligned = True` for GMM-AA and `False` for full-covariance GMM.
Both select the number of components from 1 through 32 by validation likelihood.

SME selects between 2, 3, and 4 components by validation likelihood, with a
4-regular graph, four normalized projections per edge, coordinate rounding of
0.25 during partition fitting, and nine requested workers. Density estimators
use the unrounded observations. The SME and NB-KDE bandwidth settings are
15-fold cross-validation over 80 logarithmically spaced values from 0.001 to 1.

The MAF comparison requires the additional dependencies described in the
[experiment setup](../README.md). Results are saved in
`experiments/kin8nm/artifacts/`.
