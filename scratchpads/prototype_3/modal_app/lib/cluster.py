import time
import umap
import hdbscan
import numpy as np
import pickle
import os


def save_or_load_df(file_path, data_func, *args, **kwargs):
    try:
        with open(file_path, "rb") as f:
            data = pd.read_csv(f)
        print(f"Data loaded from {file_path}.")
    except Exception:
        print(f"Data not found at {file_path}. Running data function...")
        data = data_func(*args, **kwargs)
        data.to_csv(file_path, index=False)
    return data


def create_2d_mapper(embeddings):
    """Create a 2D UMAP mapper for visualization."""
    reducer_2d = umap.UMAP(n_components=2, random_state=42)
    mapper = reducer_2d.fit(embeddings)
    return mapper


def save_or_load_pickle(file_path, data_func, *args, rerun=False, **kwargs):
    try:
        if rerun:
            print("Enforced running of clustering again")
            raise Exception("rerun")
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        print(f"Data loaded from {file_path}.")
    except Exception:
        print(f"Data not found at {file_path}. Running data function...")
        data = data_func(*args, **kwargs)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
    return data


def get_or_create_and_fit_clusterer(embeddings, filepath, rerun=False):
    if rerun:
        print("Enforced running of clustering again")
        return run_clustering(embeddings)

    return save_or_load_pickle(filepath, run_clustering, embeddings, rerun)


def reduce_embeddings(embeddings, n_components=5):
    """Perform dimensionality reduction with UMAP."""
    start_time = time.time()
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    reduced_embeddings = reducer.fit_transform(embeddings)
    end_time = time.time()

    print(f"UMAP dimensionality reduction took {end_time - start_time:.2f} seconds")
    return reduced_embeddings, reducer


def run_clustering(embeddings):
    reduced_embeddings, reducer = reduce_embeddings(embeddings)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=15, prediction_data=True
    )
    clusterer.fit(reduced_embeddings)
    return clusterer


def assign_tweet_clusters(cluster_labels, cluster_probs, tweets_df, filepath):
    if tweets_df is None:
        raise ValueError("Must provide tweets_df when creating new clustered DataFrame")

    tweets_df["cluster"] = cluster_labels
    tweets_df["cluster_prob"] = cluster_probs
    # tweets_df.to_csv(filepath, index=False)
    print("Saved clustered tweets to Google Drive.")
    print(f"Number of unique clusters: {len(tweets_df.cluster.unique())}")
    return tweets_df


def load_or_create_clustered_tweets_df(
    cluster_labels, cluster_probs, filepath, tweets_df=None, rerun=True
):
    if rerun:
        print("Enforced running of clustering again")
        return assign_tweet_clusters(cluster_labels, cluster_probs, tweets_df, filepath)

    return save_or_load_df(
        filepath, assign_tweet_clusters, cluster_labels, cluster_probs, tweets_df
    )


import numpy as np
import pandas as pd
import hdbscan


def get_cluster_centroids(clusterer):
    # Get unique cluster labels (excluding noise labeled as -1)
    cluster_ids = np.unique(clusterer.labels_)
    cluster_ids = cluster_ids[cluster_ids >= 0]
    centroids = []
    for cluster_id in cluster_ids:
        centroid = clusterer.weighted_cluster_centroid(cluster_id)
        centroids.append(centroid)
    centroids = np.array(centroids)
    return centroids


def cluster_centroids(clusterer):
    centroids = cluster_centroids(clusterer)
    higher_clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        prediction_data=True,
        alpha=1.3,
        cluster_selection_method="leaf",
    )
    higher_clusterer.fit(centroids)
    return higher_clusterer


def build_hierarchy_df(cluster_df):
    # Create groups df with level 1
    groups_df = cluster_df[cluster_df["group"] >= 0].copy()
    groups_df["id"] = groups_df.apply(lambda x: f"1-{x.name}", axis=1)
    groups_df["level"] = 1
    groups_df["parent"] = None

    # Update clusters df with level 0
    clusters_df = cluster_df.copy()
    clusters_df["id"] = clusters_df.index
    clusters_df["level"] = 0
    clusters_df = clusters_df.rename(columns={"group": "parent"})

    # Combine and select columns
    result = pd.concat([clusters_df.sort_values("id"), groups_df])[
        ["id", "name", "summary", "num_tweets", "parent", "level"]
    ].reset_index(drop=True)

    return result
