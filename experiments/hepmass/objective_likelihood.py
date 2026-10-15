import datetime
import os
import time
from pathlib import Path

import numpy as np
from silhouette_tools.metrics.wasserstein import wasserstein_1d
from silhouette_tools.models import naive_bayes_kde
from silhouette_tools.optimizers.hillclimb import GreedyClimbOptimizer
from silhouette_tools.silhouettes.objectives.pairwise_nb_fast import \
    CachedPairwiseSilhouetteNBMixObjective
from silhouette_tools.silhouettes.projections import graphs, pairwise
from scipy.stats import special_ortho_group
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

base = os.path.expanduser("~/Documents/datasets/hepmass")

def quantize(X: np.ndarray, delta: float) -> np.ndarray:
    if delta <= 0:
        raise ValueError("delta must be positive")
    return np.round(X / delta) * delta

print("Loading data...")
data = np.load(os.path.join(base, "hepmass_1000_discrete_removed.npz"))
print("Data loaded!")

X_train_raw = data["X_train"]
X_test_raw = data["X_test"]

d = X_train_raw.shape[1]

rng = np.random.default_rng(123)
perm_train = rng.permutation(X_train_raw.shape[0])
perm_test = rng.permutation(X_test_raw.shape[0])

X_train_raw = X_train_raw[perm_train]
X_test_raw = X_test_raw[perm_test]


n_train = 20_000
n_test = 10_000

# print("Creating random rotation...")
# R = special_ortho_group.rvs(d)

print("Not rotating data")
R = np.eye(d)

X_train = X_train_raw[:n_train, :]
X_test = X_test_raw[:n_test, :]

# print("aligning to principal components")
# Sigma = np.cov(X_train,rowvar= False)
# eigvals, eigvecs = np.linalg.eigh(Sigma)
# R = eigvecs

X_train = X_train@R
X_test = X_test@R
note = "back to identity for rotation"

delta = 0.125  # set to None to disable

########## Optimization statistics

step_times: list[float] = []
silhouette_scores  : list[float] = []
estimator_scores: list[float] = []
steps_per_test = 100

grid = np.logspace(-3, 0, 40)  # 0.001 → ~0.63
t0 = time.perf_counter()

def on_step(
    n_steps: int,
    score: float,
    labels: np.ndarray,
) -> None:
    # record elapsed wall-clock time
    step_times.append(time.perf_counter() - t0)
    silhouette_scores.append(float(score))
    if n_steps % steps_per_test == 0:
        temp_mix_nb_kde = naive_bayes_kde.NaiveBayesKDEMixture(
            X_train,
            labels,
        )

        temp_mix_nb_kde.fit_bandwidths(
            method="likelihood",
            grid=grid,
        )

        test_score = np.mean(temp_mix_nb_kde.log_density(X_test))
        test_score = float(np.mean(temp_mix_nb_kde.log_density(X_test)))
        estimator_scores.append(test_score)

        print(
            f"[step {n_steps}] " f"objective={score:.6f} | " f"test_ll={test_score:.6f}"
        )

##############
if delta is not None:
    X_train_labeling = quantize(X_train, delta).astype(np.float32)
else:
    X_train_labeling = X_train

d = X_train.shape[1]
r = 4
k_components = 80
k_coefs = 2
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
    n_workers=9,
)

def objective(labels: np.ndarray) -> float:
    scores = obj.evaluate_labels(
        labels=labels,
    )
    return float(np.mean(scores))

print("Objective constructed!")

opt = GreedyClimbOptimizer(
    labels=labels0,
    objective=objective,
    improvement_strategy="best",
    verbose=False,
    on_step=on_step,
)
print("Calculating initial score...")
initial_score = opt.score
print("Initial score calculated!")
print("Beginning optimization...")
opt.run(max_iters=5 * n_train, max_run_time=1000.0 * 3600)
# opt.run(max_iters=3, max_run_time=10) #Just for testing
labels = opt.labels

obj.close()

mix_nb_kde = naive_bayes_kde.NaiveBayesKDEMixture(
    X_train,
    labels,
)
mix_nb_kde.fit_bandwidths("likelihood", grid=grid)

bw = mix_nb_kde.bandwidths

at_lower = np.isclose(bw, grid[0])
at_upper = np.isclose(bw, grid[-1])

if np.any(at_lower) or np.any(at_upper):
    print(
        f"🚨 Naive Bayes Mixture: Bandwidth grid boundary hit: "
        f"{at_lower.sum()} at lower bound ({grid[0]}), "
        f"{at_upper.sum()} at upper bound ({grid[-1]}). "
        "Consider expanding the grid."
    )
else:
    print("✅ Naive Bayes Mixture: all bandwidths are interior to the grid.")
mix_nb_score = np.mean(mix_nb_kde.log_density(X_test))

print(f"📊 Final mean log likelihood mixture of naive Bayes is {mix_nb_score:.6f}")


# ----------- Naive Bayes KDE --------------
nb_kde = naive_bayes_kde.NaiveBayesKDEMixture(
    X_train,
    np.zeros(labels.shape),
)
print("Fitting naive Bayes KDE bandwitdths...")
nb_kde.fit_bandwidths("likelihood", grid=grid)
print("Finished!")
bw = nb_kde.bandwidths

at_lower = np.isclose(bw, grid[0])
at_upper = np.isclose(bw, grid[-1])

if np.any(at_lower) or np.any(at_upper):
    print(
        f"🚨 Naive Bayes: Bandwidth grid boundary hit: "
        f"{at_lower.sum()} at lower bound ({grid[0]}), "
        f"{at_upper.sum()} at upper bound ({grid[-1]}). "
        "Consider expanding the grid."
    )
else:
    print("✅ Naive Bayes: all bandwidths are interior to the grid.")

nb_score = np.mean(nb_kde.log_density(X_test))

print(f"📊 Final mean log likelihood of naive Bayes is {nb_score:.6f}")

# -------------Standard KDE ---------------

print("Fitting standard KDE bandwidth (CV over scalar bw)...")
kde = KernelDensity(kernel="gaussian")

kde_search = GridSearchCV(
    kde,
    {"bandwidth": grid},   # reuse your grid
    cv=5,                  # or make this a parameter
    n_jobs=1,              # avoid nested parallelism
)
kde_search.fit(X_train)

kde_bw = float(kde_search.best_params_["bandwidth"])
print(f"✅ Standard KDE best bandwidth = {kde_bw:g}")

if np.isclose(kde_bw, grid[0]) or np.isclose(kde_bw, grid[-1]):
    print(
        f"🚨 Standard KDE: bandwidth hit grid boundary: "
        f"{kde_bw:g} (grid [{grid[0]:g}, {grid[-1]:g}]). Consider expanding the grid."
    )

kde_score = float(np.mean(kde_search.best_estimator_.score_samples(X_test)))
print(f"📊 Final mean log likelihood of standard KDE is {kde_score:.6f}")

# ---- Full-covariance Gaussian ------------
gmm = GaussianMixture(
    n_components=1,
    covariance_type="full",
    random_state=0,
)
gmm.fit(X_train)

gauss_score = np.mean(gmm.score_samples(X_test))

print(f"📊 Final mean log likelihood of full Gaussian is {gauss_score:.6f}")

# mu = X_train.mean(axis=0)
# Sigma = np.cov(X_train, rowvar=False, bias=True)  # bias=True gives MLE
final_score = opt.score

out_dir = Path(__file__).resolve().parent / "artifacts"
out_dir.mkdir(exist_ok=True)
#
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_path = out_dir / f"run_{ts}.npz"
#
np.savez_compressed(
    out_path,
    X_train=X_train,
    X_test=X_test,
    k_components=k_components,
    k_coefs=k_coefs,
    coefs=coefs,
    r=r,
    graph=graph,
    labels=opt.labels,
    delta=delta,
    grid=grid,
    initial_score=initial_score,
    final_score=final_score,
    step_times=step_times,
    estimator_scores=estimator_scores,
    mix_nb_score=mix_nb_score,
    nb_score=nb_score,
    kde_score = kde_score,
    gauss_score = gauss_score,
    steps_per_test=steps_per_test,
    silhouette_scores = silhouette_scores,
    R = R,
    note = note,
)
