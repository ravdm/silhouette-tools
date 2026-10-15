import datetime
import time
from pathlib import Path

import numpy as np

from silhouette_tools.metrics.wasserstein import wasserstein_1d
from silhouette_tools.models import naive_bayes_kde
from silhouette_tools.optimizers.hillclimb import GreedyClimbOptimizer
from silhouette_tools.silhouettes.objectives.pairwise_nb_fast import (
    CachedPairwiseSilhouetteNBMixObjective,
)
from silhouette_tools.silhouettes.projections import graphs, pairwise


def quantize(X: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0:
        raise ValueError("delta must be positive")
    return np.round(X / delta) * delta


print("Loading splits...")
data = np.load(Path(__file__).resolve().parent / "hepmass_splits.npz")
print("Data loaded!")

train_splits = data["train_splits"]
val_splits = data["val_splits"]
X_test = data["X_test"]

num_experiments = int(data["num_experiments"])
n_train = int(data["n_train"])
n_val = int(data["n_val"])
n_test = int(data["n_test"])

# ----------------------------
# method settings
# ----------------------------
delta = 0.125

grid = np.logspace(-3, 0, 80)
bandwidth_n_folds = 15

r = 4
k_min = 2
k_max = 4
k_grid = np.arange(k_min, k_max + 1)

k_coefs = 2
n_workers = 9

max_iters = 5 * n_train
max_run_time = 1000.0 * 3600

# ----------------------------
# storage
# ----------------------------
val_scores = np.empty((num_experiments, len(k_grid)))
test_scores = np.empty((num_experiments, len(k_grid)))
objective_scores = np.empty((num_experiments, len(k_grid)))
run_times = np.empty((num_experiments, len(k_grid)))

best_k = np.empty(num_experiments, dtype=int)
best_val_scores = np.empty(num_experiments)
best_test_scores = np.empty(num_experiments)

labels_all = np.empty((num_experiments, len(k_grid)), dtype=object)
bandwidths_all = np.empty((num_experiments, len(k_grid)), dtype=object)
graphs_all = np.empty((num_experiments, len(k_grid)), dtype=object)

# ----------------------------
# main loop
# ----------------------------
for exp_idx in range(num_experiments):
    print(f"\n=== Experiment {exp_idx + 1}/{num_experiments} ===")

    X_train = train_splits[exp_idx]
    X_val = val_splits[exp_idx]

    if delta is not None:
        X_train_labeling = quantize(X_train, delta).astype(np.float32)
    else:
        X_train_labeling = X_train

    d = X_train.shape[1]

    for k_idx, k_components in enumerate(k_grid):
        print(f"\n--- k_components={k_components} ---")

        print(f"Constructing {r}-regular graph...")
        graph = graphs.random_regular_graph(d, r)
        print("Graph constructed!")

        coefs = pairwise.silhouette_coefficients(a=-2, b1=1, b2=1, k=k_coefs)
        labels0 = np.random.randint(0, k_components, n_train)

        print("Constructing Objective...")
        obj = CachedPairwiseSilhouetteNBMixObjective(
            X_train_labeling,
            k_components,
            metric=wasserstein_1d,
            graph=graph,
            coefs=coefs,
            n_workers=n_workers,
        )

        def objective(labels: np.ndarray) -> float:
            scores = obj.evaluate_labels(labels=labels)
            return float(np.mean(scores))

        print("Objective constructed!")

        opt = GreedyClimbOptimizer(
            labels=labels0,
            objective=objective,
            improvement_strategy="best",
            verbose=True,
            on_step=None,
        )

        print("Beginning optimization...")
        t0 = time.perf_counter()

        try:
            opt.run(max_iters=max_iters, max_run_time=max_run_time)
        finally:
            obj.close()

        run_times[exp_idx, k_idx] = time.perf_counter() - t0
        objective_scores[exp_idx, k_idx] = float(opt.score)

        print("Fitting final density estimator...")
        mix_nb_kde = naive_bayes_kde.NaiveBayesKDEMixture(
            X_train,
            opt.labels,
        )
        mix_nb_kde.fit_bandwidths(
            method="likelihood",
            grid=grid,
            n_folds=bandwidth_n_folds,
        )

        val_ll = float(np.mean(mix_nb_kde.log_density(X_val)))
        test_ll = float(np.mean(mix_nb_kde.log_density(X_test)))

        val_scores[exp_idx, k_idx] = val_ll
        test_scores[exp_idx, k_idx] = test_ll

        labels_all[exp_idx, k_idx] = opt.labels.copy()
        bandwidths_all[exp_idx, k_idx] = mix_nb_kde.bandwidths.copy()
        graphs_all[exp_idx, k_idx] = graph

        print(
            f"k={k_components} | "
            f"objective={opt.score:.6f} | "
            f"val_ll={val_ll:.6f} | "
            f"test_ll={test_ll:.6f}"
        )

    best_idx = int(np.argmax(val_scores[exp_idx]))
    best_k[exp_idx] = int(k_grid[best_idx])
    best_val_scores[exp_idx] = val_scores[exp_idx, best_idx]
    best_test_scores[exp_idx] = test_scores[exp_idx, best_idx]

    print(
        f"\nSelected k={best_k[exp_idx]} | "
        f"best val ll={best_val_scores[exp_idx]:.6f} | "
        f"test ll={best_test_scores[exp_idx]:.6f}"
    )

# ----------------------------
# save results
# ----------------------------
out_dir = Path(__file__).resolve().parent / "artifacts"
out_dir.mkdir(exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"silhouette_k_selected_results_{ts}.npz"

np.savez_compressed(
    out_path,
    # dataset info
    num_experiments=num_experiments,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,

    # method parameters
    delta=delta,
    grid=grid,
    bandwidth_n_folds=bandwidth_n_folds,
    r=r,
    k_grid=k_grid,
    k_coefs=k_coefs,
    n_workers=n_workers,
    max_iters=max_iters,
    max_run_time=max_run_time,

    # scores
    val_scores=val_scores,
    test_scores=test_scores,
    objective_scores=objective_scores,
    run_times=run_times,

    # selected-by-validation outputs
    best_k=best_k,
    best_val_scores=best_val_scores,
    best_test_scores=best_test_scores,

    # fitted state
    labels_all=labels_all,
    bandwidths_all=bandwidths_all,
    graphs_all=graphs_all,
)

print("\n===================================")
print(f"Saved results to {out_path}")
print(f"Mean validation-selected test LL: {best_test_scores.mean():.6f}")
print(f"Std validation-selected test LL:  {best_test_scores.std():.6f}")
