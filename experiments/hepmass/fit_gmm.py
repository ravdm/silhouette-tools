import datetime
from pathlib import Path

import numpy as np
from sklearn.mixture import GaussianMixture

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

print(f"train_splits shape: {train_splits.shape}")
print(f"val_splits shape:   {val_splits.shape}")
print(f"X_test shape:       {X_test.shape}")

# ----------------------------
# GMM settings
# ----------------------------
axis_aligned = True
covariance_type = "diag" if axis_aligned else "full"
component_grid = list(range(1, 33))
num_k = len(component_grid)

print(f"Using covariance_type={covariance_type!r}")

# ----------------------------
# storage
# ----------------------------
gmm_test_scores = np.empty(num_experiments)
gmm_best_k = np.empty(num_experiments, dtype=int)
gmm_best_val_scores = np.empty(num_experiments)
gmm_val_scores = np.empty((num_experiments, num_k))

# ----------------------------
# main loop
# ----------------------------
for exp_idx in range(num_experiments):
    print(f"\n=== Experiment {exp_idx + 1}/{num_experiments} ===")

    X_train = train_splits[exp_idx]
    X_val = val_splits[exp_idx]

    best_val = -np.inf
    best_gmm = None
    best_k = None

    for j, k in enumerate(component_grid):
        gmm = GaussianMixture(
            n_components=k,
            covariance_type=covariance_type,
            reg_covar=1e-6,
            max_iter=500,
            n_init=3,
        )

        gmm.fit(X_train)
        val_score = float(np.mean(gmm.score_samples(X_val)))
        gmm_val_scores[exp_idx, j] = val_score

        print(f"[exp {exp_idx:02d} | k={k:2d}] val_ll={val_score:.6f}")

        if val_score > best_val:
            best_val = val_score
            best_gmm = gmm
            best_k = k

    assert best_gmm is not None
    assert best_k is not None

    gmm_best_k[exp_idx] = best_k
    gmm_best_val_scores[exp_idx] = best_val
    gmm_test_scores[exp_idx] = float(np.mean(best_gmm.score_samples(X_test)))

    print(
        f"Best k = {best_k}, "
        f"best val ll = {best_val:.6f}, "
        f"test ll = {gmm_test_scores[exp_idx]:.6f}"
    )

# ----------------------------
# save results
# ----------------------------
out_dir = Path(__file__).resolve().parent / "artifacts"
out_dir.mkdir(exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"gmm_results_{ts}.npz"

np.savez_compressed(
    out_path,
    gmm_test_scores=gmm_test_scores,         # (num_experiments,)
    gmm_best_k=gmm_best_k,                   # (num_experiments,)
    gmm_best_val_scores=gmm_best_val_scores, # (num_experiments,)
    gmm_val_scores=gmm_val_scores,           # (num_experiments, num_k)
    component_grid=np.array(component_grid),
    covariance_type=covariance_type,
    axis_aligned=axis_aligned,
    num_experiments=num_experiments,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)

print("\n===================================")
print(f"Saved results to {out_path}")
print(f"Mean test LL: {gmm_test_scores.mean():.6f}")
print(f"Std test LL:  {gmm_test_scores.std():.6f}")
