import cuml
import time
from typing import Optional
import numpy as np
import pickle
import os
from skopt import gp_minimize
from skopt.space import Integer
import pandas as pd


from typing import Optional
import numpy as np
import pandas as pd
import hdbscan
from skopt import gp_minimize
from skopt.space import Integer
from tqdm import tqdm

# key: upper bound on number of points e.g. if num_points=3000, we'd use the values for 5000.
DEFAULT_CLUSTERING_PARAMS = {
    1000: {
        "min_cluster_size": 10,
        "min_samples": 3,
    },
    5000: {
        "min_cluster_size": 15,
        "min_samples": 5,
    },
    15000: {
        "min_cluster_size": 20,
        "min_samples": 7,
    },
    30000: {
        "min_cluster_size": 25,
        "min_samples": 10,
    },
    50000: {
        "min_cluster_size": 40,
        "min_samples": 10,
    },
    75000: {
        "min_cluster_size": 50,
        "min_samples": 15,
    },
    100000: {
        "min_cluster_size": 75,
        "min_samples": 20,
    },
    150000: {
        "min_cluster_size": 100,
        "min_samples": 25,
    },
    200000: {
        "min_cluster_size": 125,
        "min_samples": 30,
    },
    300000: {
        "min_cluster_size": 150,
        "min_samples": 35,
    },
    500000: {
        "min_cluster_size": 200,
        "min_samples": 40,
    },
}


def _evaluate_clustering(
    clusterer, embeddings, ref_num_clusters: float, noise_ratio_threshold: float = 0.45
):
    """
    Evaluate the clustering results and compute a cost value.

    Parameters
    ----------
    clusterer : cuml.cluster.hdbscan.HDBSCAN object (fitted)
        The HDBSCAN clusterer that was just run.
    embeddings : np.ndarray
        The data embeddings that were clustered.
    ref_num_clusters : float
        Reference number of clusters based on dataset size (for penalty calculation).
    noise_ratio_threshold : float, optional
        Threshold above which noise is penalized. Default is 0.45.

    Returns
    -------
    cost : float
        The computed cost for the given clusterer parameters. Lower is better.
    metrics : dict
        Dictionary containing intermediate metrics for logging.
    """
    labels = clusterer.labels_
    n_tweets = len(embeddings)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    # If too few clusters form, return a large penalty.
    if n_clusters < 3:
        return 1000.0, {
            "n_clusters": n_clusters,
            "noise_ratio": None,
            "persistence": None,
            "mean_cluster_size": None,
            "max_cluster_size": None,
            "n_cluster_penalty": None,
            "noise_penalty": None,
            "large_cluster_penalty": None,
        }

    # Compute cluster metrics for cost calculation
    noise_ratio = np.sum(labels == -1) / n_tweets
    persistence = np.mean(clusterer.cluster_persistence_) * 2
    cluster_sizes = [np.sum(labels == i) for i in set(labels) if i != -1]
    mean_cluster_size = np.mean(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)

    # Penalties:
    # 1. Noise penalty: Penalize if noise ratio exceeds noise_ratio_threshold.
    noise_penalty = max(0, noise_ratio - noise_ratio_threshold) * 2

    # 2. Number of clusters penalty:
    #    We want the cluster count to be close to ref_num_clusters.
    #    If n_clusters is outside of bound, heavily penalize.
    if 0.3 * ref_num_clusters < n_clusters < 1.7 * ref_num_clusters:
        n_cluster_penalty = (abs(n_clusters - ref_num_clusters) / ref_num_clusters) ** 2
    else:
        n_cluster_penalty = (
            50 + ((abs(n_clusters - ref_num_clusters) / ref_num_clusters) ** 2) * 10
        )

    # 3. Large cluster penalty:
    #    Penalize heavily if any cluster is larger than half the dataset.
    excess_ratio = max(0, (max_cluster_size / (n_tweets / 2)) - 1)
    large_cluster_penalty = 50 * excess_ratio**2  # Smooth, smaller penalty

    # Combine into a single cost.
    # Higher persistence is good, so we subtract penalties from persistence.
    cost = -(persistence - noise_penalty - n_cluster_penalty - large_cluster_penalty)

    metrics = {
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "persistence": float(
            persistence / 2
        ),  # original persistence before scaling by 2
        "mean_cluster_size": float(mean_cluster_size),
        "max_cluster_size": float(max_cluster_size),
        "n_cluster_penalty": float(n_cluster_penalty),
        "noise_penalty": float(noise_penalty),
        "large_cluster_penalty": float(large_cluster_penalty),
    }
    return cost, metrics


def generate_initial_points(
    initial_min_cluster_size: int,
    min_min_cluster_size: int,
    max_cluster_size: int,
    min_min_samples: int,
    max_min_samples: int,
    n_random: int = 20,
) -> list[list[int]]:
    """Generate initial points for Bayesian optimization.

    Returns a list of [min_cluster_size, min_samples] points combining:
    1. Points with fixed initial_min_cluster_size and varying min_samples
    2. Random points within the parameter bounds
    """
    # Generate initial points with fixed min_cluster_size
    initial_points = [
        [initial_min_cluster_size, min_samples]
        for min_samples in range(min_min_samples, max_min_samples)
    ]

    # Generate random points
    random_points = set()
    while len(random_points) < n_random:
        point = (
            np.random.randint(min_min_cluster_size, max_cluster_size),
            np.random.randint(min_min_samples, max_min_samples),
        )
        random_points.add(point)

    return initial_points + [list(p) for p in random_points]


def get_default_params(n_tweets: int) -> tuple[int, int]:
    """Get default parameters based on dataset size from DEFAULT_CLUSTERING_PARAMS"""
    sorted_keys = sorted(DEFAULT_CLUSTERING_PARAMS.keys())
    selected_key = next((k for k in sorted_keys if k >= n_tweets), sorted_keys[-1])
    params = DEFAULT_CLUSTERING_PARAMS[selected_key]
    return params["min_cluster_size"], params["min_samples"]


def evaluate_params(
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_method: str,
) -> tuple[Optional[dict], Optional[float]]:
    """Run clustering with given params and return results if valid"""
    print(
        f"Trying parameters: min_cluster_size={min_cluster_size}, min_samples={min_samples}, method={cluster_selection_method}"
    )

    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
        prediction_data=True,
        cluster_selection_method=cluster_selection_method,
    )
    clusterer.fit(embeddings)

    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    ref_num_clusters = int(len(embeddings) ** (1 / 2.5))

    print(
        f"Got {n_clusters} clusters (target range: {int(0.3 * ref_num_clusters)} to {int(1.7 * ref_num_clusters)})"
    )

    if 0.3 * ref_num_clusters <= n_clusters <= 1.7 * ref_num_clusters:
        noise_ratio = np.sum(labels == -1) / len(labels)
        persistence = float(np.mean(clusterer.cluster_persistence_))
        cluster_sizes = [np.sum(labels == i) for i in set(labels) if i != -1]
        max_csize = max(cluster_sizes) if cluster_sizes else 0

        print(
            f"Parameters valid! Noise ratio: {noise_ratio:.2f}, Persistence: {persistence:.2f}"
        )
        return {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "n_clusters": n_clusters,
            "noise_ratio": float(noise_ratio),
            "persistence": persistence,
            "max_cluster_size": float(max_csize),
            "score": persistence - (noise_ratio * 0.5),  # Simple quality score
            "cluster_selection_method": cluster_selection_method,
            "source": "default_params",
        }, None
    return None, n_clusters


def run_bayesian_optimization(
    embeddings: np.ndarray,
    min_min_cluster_size: int,
    max_cluster_size: int,
    min_min_samples: int,
    max_min_samples: int,
    initial_min_cluster_size: int,
    n_calls: int,
    random_state: int,
    cluster_selection_method: str,
) -> dict:
    """Run Bayesian optimization to find optimal clustering parameters.

    Returns
    -------
    dict
        Dictionary containing the best parameters found and metrics
    """
    # Define parameter search space
    space = [
        Integer(min_min_cluster_size, max_cluster_size, name="min_cluster_size"),
        Integer(min_min_samples, max_min_samples, name="min_samples"),
    ]

    results = []
    ref_num_clusters = int(len(embeddings) ** (1 / 2.5))

    def objective(params):
        min_cluster_size, min_samples = params
        clusterer = cuml.cluster.hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            gen_min_span_tree=True,
            prediction_data=True,
            cluster_selection_method=cluster_selection_method,
        )
        clusterer.fit(embeddings)

        cost, metrics = _evaluate_clustering(clusterer, embeddings, ref_num_clusters)

        results.append(
            {
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "n_clusters": metrics["n_clusters"],
                "noise_ratio": metrics["noise_ratio"],
                "persistence": metrics["persistence"],
                "mean_cluster_size": metrics["mean_cluster_size"],
                "max_cluster_size": metrics["max_cluster_size"],
                "score": float(-cost),
                "n_cluster_penalty": metrics["n_cluster_penalty"],
                "noise_penalty": metrics["noise_penalty"],
                "large_cluster_penalty": metrics["large_cluster_penalty"],
            }
        )
        return cost

    # Generate initial points for optimization
    x0 = generate_initial_points(
        initial_min_cluster_size,
        min_min_cluster_size,
        max_cluster_size,
        min_min_samples,
        max_min_samples,
    )

    # Run optimization
    with tqdm(total=n_calls + len(x0), desc="Optimizing clustering params") as pbar:

        def callback(res):
            pbar.update(1)

        result = gp_minimize(
            objective,
            space,
            x0=x0,
            n_calls=n_calls + len(x0),
            random_state=random_state,
            callback=callback,
        )

    # Evaluate best parameters
    best_min_cluster_size, best_min_samples = result.x
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=best_min_cluster_size,
        min_samples=best_min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
        prediction_data=True,
        cluster_selection_method=cluster_selection_method,
    )
    clusterer.fit(embeddings)

    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    noise_ratio = sum(labels == -1) / len(labels)
    persistence = float(np.mean(clusterer.cluster_persistence_))
    cluster_sizes = [sum(labels == i) for i in set(labels) if i != -1]
    max_csize = max(cluster_sizes) if cluster_sizes and n_clusters > 0 else 0

    return {
        "min_cluster_size": int(best_min_cluster_size),
        "min_samples": int(best_min_samples),
        "n_clusters": n_clusters,
        "noise_ratio": float(noise_ratio),
        "persistence": persistence,
        "max_cluster_size": float(max_csize),
        "score": float(-result.fun),
        "results_df": pd.DataFrame(results).to_dict(orient="records"),
        "cluster_selection_method": cluster_selection_method,
        "source": "bayesian_optimization",
    }


def find_optimal_clustering_params(
    embeddings: np.ndarray,
    min_min_cluster_size: Optional[int] = 10,
    lower_max_min_cluster_size: Optional[int] = 20,
    upper_max_min_cluster_size: Optional[int] = 100,
    min_min_samples: Optional[int] = 6,
    max_min_samples: Optional[int] = 20,
    n_calls: int = 25,
    random_state: int = 42,
    cluster_selection_method: Optional[str] = "eom",
):
    """Find optimal HDBSCAN parameters, trying default parameters before optimization."""
    n_tweets = len(embeddings)
    ref_cluster_size = int(n_tweets ** (1 / 3))
    ref_num_clusters = int(n_tweets ** (1 / 2.5))

    print(f"\nStarting parameter search for {n_tweets} points")
    print(
        f"Target number of clusters: {ref_num_clusters} (range: {int(0.3 * ref_num_clusters)} to {int(1.7 * ref_num_clusters)})"
    )

    def try_default_params(selection_method: str) -> Optional[dict]:
        """Try default parameters with given selection method"""
        print(f"\nPhase: Default parameters with {selection_method} selection method")

        # Try default parameters
        default_mcs, default_ms = get_default_params(n_tweets)
        result, n_clusters = evaluate_params(
            embeddings, default_mcs, default_ms, selection_method
        )
        if result:
            return result

        # Try adjusted parameters if needed
        sorted_keys = sorted(DEFAULT_CLUSTERING_PARAMS.keys())
        current_key = next(k for k in sorted_keys if k >= n_tweets)
        key_index = sorted_keys.index(current_key)

        if n_clusters < 0.3 * ref_num_clusters and key_index > 0:
            print("\nTrying smaller parameters from default set")
            # Try smaller parameters
            adj_params = DEFAULT_CLUSTERING_PARAMS[sorted_keys[key_index - 1]]
            result, _ = evaluate_params(
                embeddings,
                adj_params["min_cluster_size"],
                adj_params["min_samples"],
                selection_method,
            )
            if result:
                return result
        elif n_clusters > 1.7 * ref_num_clusters and key_index < len(sorted_keys) - 1:
            print("\nTrying larger parameters from default set")
            # Try larger parameters
            adj_params = DEFAULT_CLUSTERING_PARAMS[sorted_keys[key_index + 1]]
            result, _ = evaluate_params(
                embeddings,
                adj_params["min_cluster_size"],
                adj_params["min_samples"],
                selection_method,
            )
            if result:
                return result

        print(f"No valid parameters found with {selection_method} selection method")
        return None

    # Try default parameters with initial selection method
    result = try_default_params(cluster_selection_method)
    if result:
        return result

    # If initial method failed and was 'eom', try with 'leaf'
    if cluster_selection_method == "eom":
        print("\nPhase: Switching to leaf selection method with default parameters")
        result = try_default_params("leaf")
        if result:
            return result

    # Fall back to Bayesian optimization
    print("\nPhase: Falling back to Bayesian optimization")
    min_min_cluster_size = max(min_min_cluster_size, int(n_tweets ** (1 / 3.5)))
    max_cluster_size = min(
        int(
            max(
                lower_max_min_cluster_size,
                int(ref_cluster_size * 4),
                min_min_cluster_size + 1,
            )
        ),
        upper_max_min_cluster_size,
    )
    initial_min_cluster_size = min(
        max(min_min_cluster_size, ref_cluster_size),
        max_cluster_size,
    )
    max_min_samples = max(max_min_samples, min_min_samples + 1)

    print(f"Parameter bounds for optimization:")
    print(f"min_cluster_size: {min_min_cluster_size} to {max_cluster_size}")
    print(f"min_samples: {min_min_samples} to {max_min_samples}")

    result = run_bayesian_optimization(
        embeddings=embeddings,
        min_min_cluster_size=min_min_cluster_size,
        max_cluster_size=max_cluster_size,
        min_min_samples=min_min_samples,
        max_min_samples=max_min_samples,
        initial_min_cluster_size=initial_min_cluster_size,
        n_calls=n_calls,
        random_state=random_state,
        cluster_selection_method=cluster_selection_method,
    )

    # If score is too low with 'eom', retry with 'leaf' selection method
    if result["score"] < -2 and cluster_selection_method == "eom":
        print("\nPhase: Score too low, retrying everything with leaf selection method")
        return find_optimal_clustering_params(
            embeddings=embeddings,
            min_min_cluster_size=min_min_cluster_size,
            lower_max_min_cluster_size=lower_max_min_cluster_size,
            upper_max_min_cluster_size=upper_max_min_cluster_size,
            min_min_samples=min_min_samples,
            max_min_samples=max_min_samples,
            n_calls=n_calls,
            random_state=random_state,
            cluster_selection_method="leaf",
        )

    return result
