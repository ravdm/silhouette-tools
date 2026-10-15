import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

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

print(f"train_splits shape: {train_splits.shape}")
print(f"val_splits shape:   {val_splits.shape}")

# ----------------------------
# KDE settings
# ----------------------------
grid = np.logspace(-3, 0, 80)
num_bw = len(grid)

print(f"Bandwidth grid: [{grid[0]:.4f}, ..., {grid[-1]:.4f}]")

# ----------------------------
# storage
# ----------------------------
kde_test_scores = np.empty(num_experiments)
kde_best_bw = np.empty(num_experiments)
kde_best_val_scores = np.empty(num_experiments)
kde_val_scores = np.empty((num_experiments, num_bw))

# ----------------------------
# main loop
# ----------------------------
for exp_idx in range(num_experiments):
    print(f"\n=== Experiment {exp_idx + 1}/{num_experiments} ===")

    X_train = train_splits[exp_idx]
    X_val = val_splits[exp_idx]
    X_test = test_splits[exp_idx]
    best_val = -np.inf
    best_kde = None
    best_bw = None

    for j, bw in enumerate(grid):
        kde = KernelDensity(kernel="gaussian", bandwidth=bw)
        kde.fit(X_train)

        val_score = float(np.mean(kde.score_samples(X_val)))
        kde_val_scores[exp_idx, j] = val_score

        print(f"[exp {exp_idx:02d} | bw={bw:.4f}] val_ll={val_score:.6f}")

        if val_score > best_val:
            best_val = val_score
            best_kde = kde
            best_bw = bw

    assert best_kde is not None

    kde_best_bw[exp_idx] = best_bw
    kde_best_val_scores[exp_idx] = best_val

    kde_test_scores[exp_idx] = float(
        np.mean(best_kde.score_samples(X_test))
    )

    print(
        f"Best bw = {best_bw:.6f}, "
        f"val_ll = {best_val:.6f}, "
        f"test_ll = {kde_test_scores[exp_idx]:.6f}"
    )

# ----------------------------
# save results
# ----------------------------
out_dir = Path(__file__).resolve().parent / "artifacts"
out_dir.mkdir(exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"kde_results_{ts}.npz"

np.savez_compressed(
    out_path,
    kde_test_scores=kde_test_scores,
    kde_best_bw=kde_best_bw,
    kde_best_val_scores=kde_best_val_scores,
    kde_val_scores=kde_val_scores,
    bandwidth_grid=grid,
    num_experiments=num_experiments,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)

print("\n===================================")
print(f"Saved results to {out_path}")
print(f"Mean test LL: {kde_test_scores.mean():.6f}")
print(f"Std test LL:  {kde_test_scores.std():.6f}")
