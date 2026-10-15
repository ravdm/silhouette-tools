import os
from pathlib import Path

import numpy as np


path = os.path.expanduser("~/Documents/datasets/kin8nm.csv")

# assumes standard CSV with header
X_raw = np.loadtxt(path, delimiter=",", skiprows=1)


d = X_raw.shape[1]


rng = np.random.default_rng()
perm = rng.permutation(X_raw.shape[0])

X = X_raw[perm,:]
X = (X - X.mean(axis=0)) / X.std(axis=0)

n_total = X.shape[0]
n_train = 1000
n_val = 300

num_experiments = 20
n_test = n_total - (n_train + n_val)

assert n_test > 0

# ----------------------------
# build train/val/test splits
# ----------------------------
train_splits = np.empty((num_experiments, n_train, d), dtype=X.dtype)
val_splits   = np.empty((num_experiments, n_val, d), dtype=X.dtype)
test_splits  = np.empty((num_experiments, n_test, d), dtype=X.dtype)

for exp_idx in range(num_experiments):
    idx = rng.permutation(n_total)

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train + n_val]
    test_idx  = idx[n_train + n_val:]

    train_splits[exp_idx] = X[train_idx]
    val_splits[exp_idx]   = X[val_idx]
    test_splits[exp_idx]  = X[test_idx]
# ----------------------------
# save in current directory
# ----------------------------
out_path = Path(__file__).resolve().parent / "kin8nm_splits.npz"

np.savez_compressed(
    out_path,
    train_splits=train_splits,
    val_splits=val_splits,
    test_splits=test_splits,
    num_experiments=num_experiments,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)

print(f"Saved splits to {out_path}")
print(f"train_splits shape: {train_splits.shape}")
print(f"val_splits shape:   {val_splits.shape}")
print(f"test_splits shape:  {test_splits.shape}")
