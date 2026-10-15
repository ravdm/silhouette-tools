# Paper experiments

Density-estimation scripts for the HEPMASS and kin8nm comparisons in
[The Silhouette Operator](https://arxiv.org/abs/2610.09687), plus the large
HEPMASS objective-versus-likelihood diagnostic from the appendix. The
[two-dimensional example](../examples/README.md) covers the synthetic example.

## Layout

```text
experiments/
    hepmass/
        prepare_data.py
        fit_sme.py
        fit_gmm.py
        fit_kde.py
        fit_nb_kde.py
        fit_maf.py
        objective_likelihood.py
    kin8nm/
        prepare_data.py
        fit_sme.py
        fit_gmm.py
        fit_kde.py
        fit_nb_kde.py
        fit_maf.py
```

Each dataset has a [HEPMASS guide](hepmass/README.md) or
[kin8nm guide](kin8nm/README.md) explaining its inputs and commands. The scripts
can be launched from the project root: split files and output directories are
resolved relative to the script's location.

## Setup

With your project virtual environment activated:

```sh
python -m pip install -e .
```

The MAF comparisons additionally require PyTorch and Zuko:

```sh
python -m pip install torch zuko
```

The parallel SME scripts retain the original macOS/Linux setup, which uses the
`fork` multiprocessing context. The small synthetic example runs with one worker.

## Running and interpreting the experiments

Use the original splits preserved in [paper_results](../paper_results/README.md),
or generate new splits with the preparation script. Run all comparison methods
against the same splits. Each script prints its results and writes a timestamped
NumPy archive into that dataset's `artifacts/` directory. GMM and GMM-AA use
the same script: its `axis_aligned` setting selects the covariance model.

The fitting scripts include random choices, so a new fit need not reproduce
the original saved scores exactly, even with the original splits. The selected
original results remain in `paper_results/`; new experiment split files and
generated results are ignored by Git.

kin8nm's SME rounding is set to 0.25, matching the paper and saved result
archive. Other fitting parameters are preserved. Recorded optimizer losses use
the current `score` field; the library's separate `best_score` bookkeeping issue
is described in
[development notes](../docs/development-notes.md).
