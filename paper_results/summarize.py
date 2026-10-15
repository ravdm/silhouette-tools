"""Export readable table scores from the preserved final result archives."""

import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
METHODS = {
    "SME": ("sme.npz", None),
    "GMM": ("gmm.npz", "gmm_test_scores"),
    "GMM-AA": ("gmm_aa.npz", "gmm_test_scores"),
    "KDE": ("kde.npz", "kde_test_scores"),
    "NB-KDE": ("nb_kde.npz", "nb_test_scores"),
    "MAF": ("maf.npz", "maf_test_scores"),
}


def main():
    rows = []
    summary = []
    for dataset in ("hepmass", "kin8nm"):
        for method, (filename, score_key) in METHODS.items():
            path = ROOT / dataset / filename
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                if method == "SME":
                    # The paper selects k from 2, 3, 4 by validation likelihood.
                    # The original kin8nm archive also contains exploratory k=5,6.
                    columns = np.flatnonzero(np.isin(data["k_grid"], [2, 3, 4]))
                    best = data["val_scores"][:, columns].argmax(axis=1)
                    scores = data["test_scores"][:, columns][np.arange(len(best)), best]
                    selected_k = data["k_grid"][columns][best]
                else:
                    scores = data[score_key]
                    selected_k = data["gmm_best_k"] if "gmm_best_k" in data else None

                for split, score in enumerate(scores, start=1):
                    k = int(selected_k[split - 1]) if selected_k is not None else ""
                    rows.append([dataset, split, method, float(score), k])
                summary.append([
                    dataset, method, len(scores), float(scores.mean()), float(scores.std())
                ])

    for filename, header, content in (
        ("per_split_scores.csv",
         ["dataset", "split", "method", "test_log_likelihood", "selected_k"], rows),
        ("summary.csv",
         ["dataset", "method", "runs", "mean_test_log_likelihood",
          "std_test_log_likelihood"], summary),
    ):
        with (ROOT / filename).open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(content)
        print(f"Wrote {ROOT / filename}")


if __name__ == "__main__":
    main()
