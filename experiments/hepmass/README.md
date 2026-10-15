# HEPMASS

These are the final comparison scripts previously stored in
`hepmass/silhouette-experiments/`, plus the large appendix diagnostic.

## Input and split preparation

To use the original saved splits, run this from the project root:

```sh
cp paper_results/hepmass/splits.npz experiments/hepmass/hepmass_splits.npz
```

Then proceed to the main comparisons below. To generate fresh splits instead,
use the data-preparation script as follows.

`prepare_data.py` expects the original preprocessed input archive at:

```text
~/Documents/datasets/hepmass/hepmass_1000_discrete_removed.npz
```

The archive must contain `X_train` and `X_test` arrays. This is the preprocessed
input used by the research scripts; raw HEPMASS CSV files require preprocessing
before they can be used here. Change `base` in the preparation and diagnostic
scripts if your input is stored elsewhere.

From the project root:

```sh
python experiments/hepmass/prepare_data.py
```

This writes `experiments/hepmass/hepmass_splits.npz`: 20 training/validation
splits of 250/250 observations and a fixed test set of 20,000 observations.
Generate it once so all comparison methods use the same splits.

## Main comparisons

```sh
python experiments/hepmass/fit_sme.py
python experiments/hepmass/fit_kde.py
python experiments/hepmass/fit_nb_kde.py
python experiments/hepmass/fit_maf.py
```

For the two Gaussian-mixture comparisons, set `axis_aligned` near the top of
`fit_gmm.py` and run:

```sh
python experiments/hepmass/fit_gmm.py
```

Use `axis_aligned = True` for GMM-AA and `False` for the full-covariance GMM.
Both select the number of components from 1 through 32 by validation likelihood.

SME selects between 2, 3, and 4 components by validation likelihood, with a
4-regular graph, four normalized projections per edge, coordinate rounding of
0.125 during partition fitting, and nine requested workers. Density estimators
use the unrounded observations. The SME and NB-KDE bandwidth settings are
15-fold cross-validation over 80 logarithmically spaced values from 0.001 to 1.

The MAF comparison requires the additional dependencies described in the
[experiment setup](../README.md). Results are saved in
`experiments/hepmass/artifacts/`.

## Appendix diagnostic

```sh
python experiments/hepmass/objective_likelihood.py
```

This is the original large run: 20,000 training observations, 10,000 test
observations, and 80 components, recording density-estimation scores every 100
accepted label changes. It uses a 40-value bandwidth grid and can take a long
time. Its saved histories support the single-run objective-versus-likelihood
diagnostic; it is separate from the main comparison table.
