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
    #    If n_clusters is outside [0.3 * ref_num_clusters, 2 * ref_num_clusters], heavily penalize.
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
    """
    Find optimal HDBSCAN parameters focusing on cluster quality rather than count using Bayesian optimization.

    Parameters
    ----------
    embeddings : np.ndarray
        A 2D array of embeddings (e.g., from text or images).
    min_min_cluster_size : int, optional
        The smallest minimum cluster size to consider.
    lower_max_min_cluster_size : int, optional
        A lower baseline for the maximum min_cluster_size.
    min_min_samples : int, optional
        The smallest minimum samples to consider.
    max_min_samples : int, optional
        A baseline maximum min_samples to consider.
    n_calls : int, optional
        Number of calls to gp_minimize.
    random_state : int, optional
        Random state for reproducibility.

    Returns
    -------
    dict
        Dictionary containing the best parameters found and various metrics.
    """
    n_tweets = len(embeddings)

    # Reference values for penalty calculations
    ref_cluster_size = int(n_tweets ** (1 / 3))
    ref_num_clusters = int(n_tweets ** (1 / 2.5))

    min_min_cluster_size = max(min_min_cluster_size, int(n_tweets ** (1 / 3.5)))
    # Initial values based on user request

    # Ensure max values are larger than min values and are integers
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

    # Define parameter search space
    space = [
        Integer(min_min_cluster_size, max_cluster_size, name="min_cluster_size"),
        Integer(min_min_samples, max_min_samples, name="min_samples"),
    ]

    results = []

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

    # Run Bayesian optimization with initial point

    x0 = generate_initial_points(
        initial_min_cluster_size,
        min_min_cluster_size,
        max_cluster_size,
        min_min_samples,
        max_min_samples,
    )
    print(f"x0: {x0}")
    print(f"min_min_cluster_size: {min_min_cluster_size}")
    print(f"max_cluster_size: {max_cluster_size}")
    print(f"min_min_samples: {min_min_samples}")
    print(f"max_min_samples: {max_min_samples}")
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

    # If score is too low, retry with 'leaf' selection method
    if -result.fun < -2 and cluster_selection_method == "eom":
        print("Score too low, retrying with leaf selection method")
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

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Extract best parameters
    best_min_cluster_size, best_min_samples = result.x

    # Evaluate once more with the best parameters
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
    if n_clusters > 0:
        cluster_sizes = [sum(labels == i) for i in set(labels) if i != -1]
        max_csize = max(cluster_sizes) if cluster_sizes else 0
    else:
        max_csize = 0

    return {
        "min_cluster_size": int(best_min_cluster_size),
        "min_samples": int(best_min_samples),
        "n_clusters": n_clusters,
        "noise_ratio": float(noise_ratio),
        "persistence": persistence,
        "max_cluster_size": float(max_csize),
        "score": float(-result.fun),
        "results_df": results_df.to_dict(orient="records"),
        "cluster_selection_method": cluster_selection_method,
    }
