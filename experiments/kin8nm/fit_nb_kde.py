import datetime
from pathlib import Path

import numpy as np
from silhouette_tools.models import naive_bayes_kde

print("Loading splits...")
data = np.load(Path(__file__).resolve().parent / "kin8nm_splits.npz")
print("Data loaded!")

train_splits = data["train_splits"]
val_splits = data["val_splits"]
test_splits = data["test_splits"]

num_experiments = int(data["num_experiments"])
n_train = int(data["n_train"])
n_val = int(data["n_val"])
n_test = int(data["n_test"])

grid = np.logspace(-3, 0, 80)
bandwidth_n_folds = 15

nb_val_scores = np.empty(num_experiments)
nb_test_scores = np.empty(num_experiments)
nb_bandwidths = np.empty(num_experiments, dtype=object)
nb_lower_hits = np.empty(num_experiments, dtype=int)
nb_upper_hits = np.empty(num_experiments, dtype=int)

for exp_idx in range(num_experiments):
    print(f"\n=== Experiment {exp_idx + 1}/{num_experiments} ===")

    X_train = train_splits[exp_idx]
    X_val = val_splits[exp_idx]
    X_test = test_splits[exp_idx]
    labels = np.zeros(X_train.shape[0], dtype=int)

    nb_kde = naive_bayes_kde.NaiveBayesKDEMixture(
        X_train,
        labels,
    )

    print("Fitting naive Bayes KDE bandwidths...")
    nb_kde.fit_bandwidths(
        method="likelihood",
        grid=grid,
        n_folds=bandwidth_n_folds,
    )

    bw = nb_kde.bandwidths.copy()
    nb_bandwidths[exp_idx] = bw

    at_lower = np.isclose(bw, grid[0])
    at_upper = np.isclose(bw, grid[-1])

    nb_lower_hits[exp_idx] = int(at_lower.sum())
    nb_upper_hits[exp_idx] = int(at_upper.sum())

    if np.any(at_lower) or np.any(at_upper):
        print(
            f"🚨 Naive Bayes KDE: bandwidth grid boundary hit: "
            f"{at_lower.sum()} lower, {at_upper.sum()} upper"
        )
    else:
        print("✅ Naive Bayes KDE: all bandwidths interior.")

    val_score = float(np.mean(nb_kde.log_density(X_val)))
    test_score = float(np.mean(nb_kde.log_density(X_test)))

    nb_val_scores[exp_idx] = val_score
    nb_test_scores[exp_idx] = test_score

    print(f"val_ll={val_score:.6f}, test_ll={test_score:.6f}")

out_dir = Path(__file__).resolve().parent / "artifacts"
out_dir.mkdir(exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"nb_kde_results_{ts}.npz"

np.savez_compressed(
    out_path,
    nb_val_scores=nb_val_scores,
    nb_test_scores=nb_test_scores,
    nb_bandwidths=nb_bandwidths,
    nb_lower_hits=nb_lower_hits,
    nb_upper_hits=nb_upper_hits,
    grid=grid,
    bandwidth_n_folds=bandwidth_n_folds,
    num_experiments=num_experiments,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)

print("\n===================================")
print(f"Saved results to {out_path}")
print(f"Mean test LL: {nb_test_scores.mean():.6f}")
print(f"Std test LL:  {nb_test_scores.std():.6f}")
