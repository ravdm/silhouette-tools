import os
from pathlib import Path

import numpy as np

base = os.path.expanduser("~/Documents/datasets/hepmass")

print("Loading data...")
data = np.load(os.path.join(base, "hepmass_1000_discrete_removed.npz"))
print("Data loaded!")

X_train_raw = data["X_train"]
X_test_raw = data["X_test"]

# ----------------------------
# experiment settings
# ----------------------------
num_experiments = 20
n_train = 250
n_val = 250
n_test = 20_000

rng = np.random.default_rng()

# permute once up front
perm_train = rng.permutation(X_train_raw.shape[0])
perm_test = rng.permutation(X_test_raw.shape[0])

X_train_raw = X_train_raw[perm_train]
X_test_raw = X_test_raw[perm_test]

# fixed test set shared across experiments
X_test = X_test_raw[:n_test, :]

# ----------------------------
# build train/val splits
# ----------------------------
train_splits = np.empty((num_experiments, n_train, X_train_raw.shape[1]), dtype=X_train_raw.dtype)
val_splits = np.empty((num_experiments, n_val, X_train_raw.shape[1]), dtype=X_train_raw.dtype)

for exp_idx in range(num_experiments):
    idx = rng.choice(X_train_raw.shape[0], size=n_train + n_val, replace=False)
    X_sub = X_train_raw[idx]
    train_splits[exp_idx] = X_sub[:n_train]
    val_splits[exp_idx] = X_sub[n_train:]

# ----------------------------
# save in current directory
# ----------------------------
out_path = Path(__file__).resolve().parent / "hepmass_splits.npz"

np.savez_compressed(
    out_path,
    train_splits=train_splits,
    val_splits=val_splits,
    X_test=X_test,
    num_experiments=num_experiments,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)

print(f"Saved splits to {out_path}")
print(f"train_splits shape: {train_splits.shape}")
print(f"val_splits shape:   {val_splits.shape}")
print(f"X_test shape:       {X_test.shape}")
