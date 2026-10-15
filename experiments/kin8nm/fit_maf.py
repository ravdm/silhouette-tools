import datetime
from pathlib import Path

import numpy as np
import torch
import zuko

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
# MAF settings
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transforms = 5
hidden_features = (32, 32)
batch_size = 50
max_epochs = 1000
patience = 30
lr = 1e-3
weight_decay = 1e-6

print(f"Using device={device}")

# ----------------------------
# storage
# ----------------------------
maf_test_scores = np.empty(num_experiments)
maf_best_val_scores = np.empty(num_experiments)
maf_epochs_run = np.empty(num_experiments, dtype=int)

maf_val_nll_history = np.empty(num_experiments, dtype=object)
maf_train_nll_history = np.empty(num_experiments, dtype=object)

# ----------------------------
# main loop
# ----------------------------
for exp_idx in range(num_experiments):
    print(f"\n=== Experiment {exp_idx + 1}/{num_experiments} ===")

    X_train = train_splits[exp_idx]
    X_val = val_splits[exp_idx]
    X_test = test_splits[exp_idx]

    x_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    x_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    x_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    flow = zuko.flows.MAF(
        features=X_train.shape[1],
        context=0,
        transforms=transforms,
        hidden_features=hidden_features,
        randperm=True,
    ).to(device)

    optimizer = torch.optim.Adam(
        flow.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_val_nll = float("inf")
    best_state = None
    bad_epochs = 0

    train_nll_history = []
    val_nll_history = []

    for epoch in range(max_epochs):
        flow.train()

        perm = torch.randperm(x_train_t.shape[0], device=device)
        train_loss_sum = 0.0

        for i in range(0, x_train_t.shape[0], batch_size):
            idx = perm[i:i + batch_size]
            xb = x_train_t[idx]

            optimizer.zero_grad()
            loss = -flow().log_prob(xb).mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.detach()) * xb.shape[0]

        train_nll = train_loss_sum / x_train_t.shape[0]

        flow.eval()
        with torch.no_grad():
            val_nll = -flow().log_prob(x_val_t).mean().item()

        train_nll_history.append(train_nll)
        val_nll_history.append(val_nll)

        print(
            f"[exp {exp_idx:02d} | epoch {epoch + 1:03d}] "
            f"train_nll={train_nll:.6f} "
            f"val_nll={val_nll:.6f}"
        )

        if val_nll < best_val_nll:
            best_val_nll = val_nll
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in flow.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("MAF early stopping.")
                break

    if best_state is not None:
        flow.load_state_dict(best_state)

    flow.eval()
    with torch.no_grad():
        test_ll = flow().log_prob(x_test_t).mean().item()

    maf_test_scores[exp_idx] = test_ll
    maf_best_val_scores[exp_idx] = -best_val_nll
    maf_epochs_run[exp_idx] = len(val_nll_history)

    maf_train_nll_history[exp_idx] = np.array(train_nll_history)
    maf_val_nll_history[exp_idx] = np.array(val_nll_history)

    print(
        f"Best val ll = {-best_val_nll:.6f}, "
        f"test ll = {test_ll:.6f}"
    )

# ----------------------------
# save results
# ----------------------------
out_dir = Path(__file__).resolve().parent / "artifacts"
out_dir.mkdir(exist_ok=True)

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"maf_results_{ts}.npz"

np.savez_compressed(
    out_path,
    maf_test_scores=maf_test_scores,
    maf_best_val_scores=maf_best_val_scores,
    maf_epochs_run=maf_epochs_run,
    maf_train_nll_history=maf_train_nll_history,
    maf_val_nll_history=maf_val_nll_history,
    transforms=transforms,
    hidden_features=np.array(hidden_features),
    batch_size=batch_size,
    max_epochs=max_epochs,
    patience=patience,
    lr=lr,
    weight_decay=weight_decay,
    num_experiments=num_experiments,
    n_train=n_train,
    n_val=n_val,
    n_test=n_test,
)

print("\n===================================")
print(f"Saved results to {out_path}")
print(f"Mean test LL: {maf_test_scores.mean():.6f}")
print(f"Std test LL:  {maf_test_scores.std():.6f}")
