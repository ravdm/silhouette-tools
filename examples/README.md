# Examples

## Three components in two dimensions

[`three_components.py`](three_components.py) is a self-contained version of the
synthetic experiment in the paper. It generates observations from three equally
weighted mixture-of-products components: a central square, four inner corner
squares, and four outer corner squares. Each group of four squares is **one**
component. The component supports overlap.

![True components, fitted partition, and optimization loss](three_components.png)

From the project root, with your virtual environment activated, install the
library and its optional plotting dependency, then run:

```sh
python -m pip install -e ".[examples]"
python examples/three_components.py
```

The script prints the initial and final loss and saves the figure to
`examples/output/three_components.png`. Add `--show` to also display a plot
window. Generated output is excluded from Git.

### What the script does

1. Generates 200 observations per component, using a fixed random seed.
2. Initializes a random, balanced partition into three components.
3. Minimizes the mean projected 1-Wasserstein loss using six non-axis
   projections and the cached objective, with one worker.
4. Plots the true components, fitted partition, and loss after each accepted
   label change.

The true labels are used only in the figure. After fitting, the fitted component
numbers are permuted to align their colors with the true components; this does
not alter the partition or its loss.

The distributions and unnormalized projection directions follow the paper's
synthetic experiment. For a quicker demonstration, this script uses 600 samples
rather than 900 and rounds coordinates to a grid of spacing 0.05 during fitting.
The figure uses the original, unrounded observations. This is a fresh example,
not an exact reproduction of the paper's saved run.

### Things to try

```sh
# Use a single projection instead of six.
python examples/three_components.py --projections 1

# Generate 900 observations and try another random seed.
python examples/three_components.py --samples-per-component 300 --seed 1

# Choose a separate output file and open the plot window.
python examples/three_components.py --output examples/output/my_fit.png --show

# List all options.
python examples/three_components.py --help
```

The optimizer stops when no single-sample reassignment improves the loss or
when it reaches `--max-steps` (default: 3000). It finds a local optimum, so
different seeds can give different partitions. Lower loss is better, but losses
from different projection counts use different directions and scales and should
not be compared directly. Overlapping components also mean that exact recovery
of every sample's generating label is not expected.

