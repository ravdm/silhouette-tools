"""Fit three mixture-of-products components to a two-dimensional sample.

From the project root, after installing ``.[examples]``, run:
    python examples/three_components.py

The data-generating distributions are from the paper's synthetic experiment.
This smaller demonstration generates its own data and saves a comparison plot.
"""

import argparse
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linear_sum_assignment

from silhouette_tools.metrics.wasserstein import wasserstein_1d
from silhouette_tools.optimizers.hillclimb import GreedyClimbOptimizer
from silhouette_tools.silhouettes.objectives.pairwise_nb_fast import (
    CachedPairwiseSilhouetteNBMixObjective,
)
from silhouette_tools.silhouettes.projections.pairwise import silhouette_coefficients


N_COMPONENTS = 3
GRID_SPACING = 0.05


def make_data(samples_per_component, rng):
    """Draw a central square and two groups of four corner squares."""
    shape = (samples_per_component, 2)
    center = rng.uniform(-5 / 6, 5 / 6, size=shape)
    inner = rng.uniform(1 / 2, 3 / 2, size=shape)
    inner *= rng.choice([-1.0, 1.0], size=shape)
    outer = rng.uniform(7 / 6, 13 / 6, size=shape)
    outer *= rng.choice([-1.0, 1.0], size=shape)

    # Within each component, x and y are sampled independently.
    X = np.vstack([center, inner, outer])
    true_labels = np.repeat(np.arange(N_COMPONENTS), samples_per_component)
    order = rng.permutation(len(X))
    return X[order], true_labels[order]


def fit_partition(X, rng, projections, max_steps):
    """Minimize the mean projected Wasserstein distance without true labels."""
    if projections == 1:
        coefs = [(1.0, 1.0)]
    else:
        # Match the synthetic experiment's unnormalized projection directions.
        coefs = silhouette_coefficients(
            a=-2, b1=1, b2=1, k=projections // 2, normalize=False
        )

    # Rounding reduces the number of distinct support points and speeds up fitting.
    # The plot still uses the original, unrounded observations.
    X_fit = np.round(X / GRID_SPACING) * GRID_SPACING
    objective = CachedPairwiseSilhouetteNBMixObjective(
        X_fit,
        N_COMPONENTS,
        metric=wasserstein_1d,
        graph=[(0, 1)],  # The only coordinate pair in two dimensions.
        coefs=coefs,
        n_workers=1,
    )

    def loss(labels):
        return float(np.mean(objective.evaluate_labels(labels)))

    history = []

    def record_step(step, score, labels):
        history.append(score)

    try:
        # Start with a random, balanced partition, independent of true membership.
        initial_labels = rng.permutation(np.arange(len(X)) % N_COMPONENTS)
        optimizer = GreedyClimbOptimizer(
            initial_labels,
            loss,
            rng=rng,
            improvement_strategy="best",
            index_policy="queue",
            on_step=record_step,
        )
        history.append(optimizer.score)
        optimizer.run(max_iters=max_steps)
        return optimizer.labels.copy(), np.asarray(history)
    finally:
        objective.close()


def align_colors(true_labels, fitted_labels):
    """Permute fitted labels for matching plot colors; do not change the fit."""
    counts = np.zeros((N_COMPONENTS, N_COMPONENTS), dtype=int)
    np.add.at(counts, (true_labels, fitted_labels), 1)
    true_ids, fitted_ids = linear_sum_assignment(-counts)
    color_ids = np.empty(N_COMPONENTS, dtype=int)
    color_ids[fitted_ids] = true_ids
    return color_ids[fitted_labels]


def make_figure(X, true_labels, fitted_labels, history, projections):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    colors = ["#0072B2", "#D55E00", "#009E73"]
    markers = ["o", "^", "s"]
    fitted_colors = align_colors(true_labels, fitted_labels)

    for ax, labels, title in zip(
        axes[:2], [true_labels, fitted_colors], ["True components", "Fitted partition"]
    ):
        for component, (color, marker) in enumerate(zip(colors, markers)):
            points = X[labels == component]
            ax.scatter(
                points[:, 0], points[:, 1], c=color, marker=marker,
                s=13, alpha=0.65, linewidths=0, label=f"Component {component + 1}",
            )
        ax.set(title=title, xlabel="x", ylabel="y", xlim=(-2.4, 2.4), ylim=(-2.4, 2.4))
        ax.set_aspect("equal", adjustable="box")
        ax.spines[["top", "right"]].set_visible(False)

    axes[2].plot(np.arange(len(history)), history, color="#333333", linewidth=1.8)
    axes[2].set(
        title="Optimization", xlabel="Accepted label changes",
        ylabel="Mean projected Wasserstein loss",
    )
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"Silhouette Tools · {len(X)} samples · {projections} projections", fontsize=14
    )
    return fig


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--samples-per-component", type=positive_int, default=200,
        help="observations in each of the three components (default: 200)",
    )
    parser.add_argument(
        "--projections", type=int, choices=[1, 2, 4, 6], default=6,
        help="number of non-axis projections (default: 6)",
    )
    parser.add_argument(
        "--max-steps", type=positive_int, default=3000,
        help="maximum accepted label changes before stopping (default: 3000)",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(__file__).resolve().parent / "output" / "three_components.png",
        help="path for the saved figure",
    )
    parser.add_argument("--show", action="store_true", help="also open the plot window")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    X, true_labels = make_data(args.samples_per_component, rng)
    print(f"Fitting {len(X)} samples with {args.projections} projections…", flush=True)
    started = perf_counter()
    fitted_labels, history = fit_partition(X, rng, args.projections, args.max_steps)
    elapsed = perf_counter() - started
    print(f"Loss: {history[0]:.6f} → {history[-1]:.6f} (lower is better)")
    print(f"Accepted label changes: {len(history) - 1}; fitting time: {elapsed:.1f}s")
    if len(history) - 1 == args.max_steps:
        print("Step limit reached; increase --max-steps to continue a longer fit.")

    fig = make_figure(X, true_labels, fitted_labels, history, args.projections)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"Saved figure: {args.output.resolve()}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
