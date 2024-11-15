# ---
# deploy: true
# cmd: ["modal", "serve", "10_integrations/streamlit/serve_streamlit.py"]
# ---
#
# # Run and share Streamlit apps
#
# This example shows you how to run a Streamlit app with `modal serve`, and then deploy it as a serverless web app.
#
# ![example streamlit app](./streamlit.png)
#
# This example is structured as two files:
#
# 1. This module, which defines the Modal objects (name the script `serve_streamlit.py` locally).
# 2. `app.py`, which is any Streamlit script to be mounted into the Modal
# function ([download script](https://github.com/modal-labs/modal-examples/blob/main/10_integrations/streamlit/app.py)).

import shlex
import subprocess
from pathlib import Path
import modal
import sys
import json
from .lib.label_clusters import (
    get_cluster_tweet_texts,
    label_all_clusters,
    label_cluster,
    label_cluster_groups,
    query_anthropic_model,
    extract_special_tokens,
    parse_extracted_data,
)
from .lib.get_tweets import (
    check_and_process_tweets,
    patch_tweets_with_note_tweets,
    create_tweets_df,
)
import json
import socket
import subprocess
from pathlib import Path
import pandas as pd
import time
from tqdm import tqdm
import numpy as np
import asyncio
import httpx


# Configure GPU and model constants
GPU_CONFIG = modal.gpu.T4()
MODEL_ID = "BAAI/bge-base-en-v1.5"
BATCH_SIZE = 32
DOCKER_IMAGE = "ghcr.io/huggingface/text-embeddings-inference:turing-1.5"  # Turing for T4s  # Create the app before using it in decorators
app = modal.App(name="twitter-archive-analysis")


# Create image with dependencies
image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "streamlit~=1.35.0",
    "numpy~=1.26.4",
    "pandas~=2.2.2",
    "supabase",
    "tqdm",
    "seaborn",
    "hdbscan",
    "umap-learn",
    "anthropic",
)

# Import supabase only inside the container where it's installed
with image.imports():
    from supabase import create_client
    import pandas as pd
    import logging
    import pickle
    import os
    import sys
    import re
    from collections import defaultdict, deque
    from tqdm import tqdm
    import re
    import seaborn as sns
    import time
    import numpy as np

# Create persistent volume for data
volume = modal.Volume.from_name("twitter-archive-data", create_if_missing=True)


# Add TEI server setup functions
def spawn_server() -> subprocess.Popen:
    process = subprocess.Popen(
        [
            "text-embeddings-router",
            "--model-id",
            MODEL_ID,
            "--port",
            "8000",
            "--max-client-batch-size",
            "128",
            "--dtype",
            "float16",
            "--auto-truncate",
        ]
    )

    while True:
        try:
            socket.create_connection(("127.0.0.1", 8000), timeout=1).close()
            print("Webserver ready!")
            return process
        except (socket.timeout, ConnectionRefusedError):
            retcode = process.poll()
            if retcode is not None:
                raise RuntimeError(f"launcher exited unexpectedly with code {retcode}")


def download_model():
    spawn_server().terminate()


# Create TEI image with all necessary dependencies
tei_image = (
    modal.Image.from_registry(DOCKER_IMAGE, add_python="3.10")
    .dockerfile_commands("ENTRYPOINT []")
    .pip_install(
        "httpx",
        "tqdm",
        "pandas",
        "streamlit~=1.35.0",
        "numpy~=1.26.4",
        "pandas~=2.2.2",
        "supabase",
        "tqdm",
        "seaborn",
        "hdbscan",
        "umap-learn",
        "anthropic",
    )
    .run_function(download_model, gpu=GPU_CONFIG)
)

with tei_image.imports():
    import httpx

rapids_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/rapidsai/base:24.10-cuda12.0-py3.12", add_python="3.12"
    )
    .pip_install(
        "numpy~=1.26.4",
        "pandas~=2.2.2",
        "supabase",
        "tqdm",
        "seaborn",
        "hdbscan",
        "umap-learn",
        "anthropic",
    )
    .entrypoint([])
)  # removes default entrypoint

with rapids_image.imports():
    import cuml
    import pandas as pd
    import logging
    import pickle
    import os
    import sys
    import re
    from collections import defaultdict, deque
    from tqdm import tqdm
    import re
    import seaborn as sns
    import time
    from pprint import pprint
    import numpy as np


# Now we can use app.cls since app is defined
@app.cls(
    gpu=GPU_CONFIG,
    image=tei_image,
    concurrency_limit=20,
    allow_concurrent_inputs=10,
)
class TextEmbeddingsInference:
    @modal.enter()
    def setup_server(self):
        self.process = spawn_server()
        self.client = httpx.AsyncClient(base_url="http://127.0.0.1:8000")

    @modal.exit()
    def teardown_server(self):
        self.process.terminate()

    @modal.method()
    async def embed(self, inputs: list[str]):
        retries = 3  # Number of retries
        for attempt in range(retries):
            try:
                resp = await self.client.post("/embed", json={"inputs": inputs})
                resp.raise_for_status()
                return resp.json()
            except httpx.ReadTimeout as e:
                if attempt < retries - 1:  # If not the last attempt
                    print(f"Timeout occurred, retrying... (Attempt {attempt + 1})")
                    await asyncio.sleep(1)  # Wait before retrying
                else:
                    raise e  # Raise the last exception if all retries fail


@app.function(image=image, volumes={"/twitter_data": volume}, timeout=600)
def process_archive(username: str, force_recompute: bool = False):
    volume.reload()
    # Set up paths
    data_dir = Path("/twitter_data")
    user_dir = data_dir / username
    user_dir.mkdir(exist_ok=True)
    processed_path = user_dir / "convo_tweets_tweets_df.parquet"

    # Check if processed file exists and we're not forcing recompute
    if processed_path.exists() and not force_recompute:
        print(f"Found cached processed tweets for {username}")
        df = pd.read_parquet(processed_path)
        return {
            "username": username,
            "tweet_count": df.shape[0],
            "trees_count": None,
            "incomplete_trees_count": None,
            "source": "cache",
        }

    SUPABASE_URL = "https://fabxmporizzqflnftavs.supabase.co"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # Use volume path for data storage
    data_dir = Path("/twitter_data")
    user_dir = data_dir / username
    user_dir.mkdir(exist_ok=True)

    archive_path = user_dir / "archive.json"

    print(f"Attempting to download archive for {username}")

    # Download from Supabase with better error handling
    try:
        # First check if file exists
        bucket = supabase.storage.from_("archives")
        files = bucket.list(f"{username}/")
        print(f"Files in bucket for {username}:", files)

        # Try download
        print(f"Downloading from path: {username}/archive.json")
        res = bucket.download(f"{username}/archive.json")

        print(f"Download successful, writing {len(res)} bytes")
        with open(archive_path, "wb") as f:
            f.write(res)

    except Exception as e:
        print(f"Error downloading archive: {str(e)}")
        print(f"Error type: {type(e)}")
        raise e

    # Load and verify archive
    try:
        with open(archive_path) as f:
            archive = json.load(f)
        tweet_count = len(archive["tweets"])

    except Exception as e:
        raise e

    patched_tweets = patch_tweets_with_note_tweets(
        archive.get("note-tweet", []), archive["tweets"]
    )
    print(f"Patched {len(patched_tweets)} tweets")

    username = archive["account"][0]["account"]["username"]
    accountId = archive["account"][0]["account"]["accountId"]

    print(f"Creating tweets dataframe for user {username}")

    tweets_df = create_tweets_df(patched_tweets, username, accountId)
    # filter df to tweets after 01-2019
    if username == "exgenesis":
        tweets_df = tweets_df[
            tweets_df["created_at"] > pd.Timestamp("2019-01-01", tz="UTC")
        ]

    procesed_tweets_df, trees, incomplete_trees = check_and_process_tweets(
        supabase, tweets_df, user_dir / "convo_tweets"
    )

    # Save processed tweets to correct path
    processed_path = user_dir / "convo_tweets_tweets_df.parquet"
    procesed_tweets_df.to_parquet(processed_path)

    volume.commit()

    # Return stats before embedding
    return {
        "username": username,
        "tweet_count": procesed_tweets_df.shape[0],
        "trees_count": len(trees),
        "incomplete_trees_count": len(incomplete_trees),
        "source": "computed",
    }


@app.function(
    # gpu=GPU_CONFIG,
    image=image,
    concurrency_limit=2,
    allow_concurrent_inputs=10,
    volumes={"/twitter_data": volume},
    timeout=600,
)
def embed_tweets(username: str, force_recompute: bool = False):
    data_dir = Path("/twitter_data")
    user_dir = data_dir / username
    embeddings_path = user_dir / "convo_tweets_embeddings.npy"

    volume.reload()

    # Check if embeddings exist and we're not forcing recompute
    if embeddings_path.exists() and not force_recompute:
        embeddings = np.load(embeddings_path)
        return {"embedded_tweets": len(embeddings), "source": "cache"}

    df_path = user_dir / "convo_tweets_tweets_df.parquet"

    # Check if embeddings already exist
    if embeddings_path.exists():
        embeddings = np.load(embeddings_path)
        return {"embedded_tweets": len(embeddings), "source": "cache"}

    # Load and process tweets
    df = pd.read_parquet(df_path)
    texts = df["emb_text"].tolist()

    # Initialize embedder and process in batches
    embedder = TextEmbeddingsInference()
    embeddings = []

    # Create batches for tqdm
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    print(f"Starting embeddings for {len(texts)} texts")
    batch_embeddings = embedder.embed.map(batches, order_outputs=True)
    for batch_embedding in batch_embeddings:
        embeddings.extend(batch_embedding)

    # Convert to numpy array and save
    embeddings = np.array(embeddings)
    temp_path = embeddings_path.with_suffix(".tmp.npy")
    try:
        print(f"Saving {len(embeddings)} embeddings to temporary file")
        np.save(temp_path, embeddings)

        # Atomic rename
        temp_path.rename(embeddings_path)
        print(f"Successfully saved embeddings to {embeddings_path}")

    except Exception as e:
        print(f"Error saving embeddings: {e}")
        if temp_path.exists():
            temp_path.unlink()
        raise

    volume.commit()

    return {"embedded_tweets": len(embeddings), "source": "computed"}


# @app.function(image=rapids_image, volumes={"/twitter_data": volume}, timeout=600)
@app.function(
    gpu="T4", image=rapids_image, volumes={"/twitter_data": volume}, timeout=600
)
def cluster_tweets(username: str, force_recompute: bool = False):
    print(f"Starting clustering for {username}")
    data_dir = Path("/twitter_data")
    user_dir = data_dir / username
    embeddings_path = user_dir / "convo_tweets_embeddings.npy"
    reduced_embeddings_path = user_dir / "reduced_embeddings.npy"
    clusterer_path = user_dir / "clusterer.pkl"
    clusters_path = user_dir / "clusters.json"
    df_path = user_dir / "convo_tweets_tweets_df.parquet"

    volume.reload()
    # Load embeddings and tweets df
    print("Loading embeddings and tweets...")
    embeddings = np.load(embeddings_path)
    df = pd.read_parquet(df_path)
    print(f"Loaded embeddings {embeddings.shape}")

    # Try to load reduced embeddings first if not forcing recompute
    if reduced_embeddings_path.exists() and not force_recompute:
        print("Loading cached reduced embeddings...")
        reduced_embeddings = np.load(reduced_embeddings_path)
        umap_time = 0
        print("Loaded reduced embeddings from cache")
    else:
        # Reduce dimensions with UMAP if not cached
        print("Running UMAP dimension reduction...")
        start_time = time.time()
        reducer = cuml.manifold.UMAP(n_components=5)
        reduced_embeddings = reducer.fit_transform(embeddings)
        umap_time = time.time() - start_time
        print(f"UMAP reduction completed in {umap_time:.2f} seconds")

        # Save reduced embeddings
        print("Saving reduced embeddings...")
        np.save(reduced_embeddings_path, reduced_embeddings)

    # Run HDBSCAN clustering
    print("Running HDBSCAN clustering...")
    start_time = time.time()
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=15, min_samples=15, metric="euclidean", prediction_data=True
    )
    clusterer.fit(reduced_embeddings)
    clustering_time = time.time() - start_time
    print(
        f"HDBSCAN clustering completed in {clustering_time:.2f} seconds, {clusterer.labels_.shape} labels"
    )
    cluster_labels = clusterer.labels_
    print(f"Cluster labels int: {cluster_labels.shape}")
    cluster_labels = clusterer.labels_.astype(str)
    print(f"Cluster labels str: {cluster_labels.shape}")
    cluster_probs = clusterer.probabilities_

    # Get cluster assignments and probabilities
    print("Computing soft clusters...")
    start_time = time.time()
    soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(clusterer)
    soft_clustering_time = time.time() - start_time
    print(f"Soft clustering completed in {soft_clustering_time:.2f} seconds")

    # Get cluster centroids
    print("Computing cluster centroids...")
    start_time = time.time()

    # Compute centroids
    unique_labels = np.unique(cluster_labels[cluster_labels != "-1"])
    centroids = np.array(
        [
            reduced_embeddings[cluster_labels == label].mean(axis=0)
            for label in unique_labels
        ]
    )
    # Cluster the centroids
    centroid_clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=1,
        prediction_data=True,
        alpha=1.3,
        cluster_selection_method="leaf",
    )
    centroid_clusterer.fit(centroids)

    # Convert group_labels to string
    group_labels = centroid_clusterer.labels_.astype(str)
    print(f"Group labels: {group_labels.shape}, {group_labels}")

    # Create hierarchy dataframe with string parent
    hierarchy_df = pd.DataFrame(
        {
            "cluster_id": unique_labels,
            "parent": [(f"1-{l}" if l != "-1" else "-1") for l in group_labels],
            "level": 0,  # All base clusters are level 0
        }
    )

    # Add parent clusters (level 1) with parent as string
    parent_clusters = pd.DataFrame(
        {
            "cluster_id": [
                f"{1}-{i}" for i in np.unique([l for l in group_labels if l != "-1"])
            ],
            "parent": "-1",  # Changed to string
            "level": 1,
        }
    )

    hierarchy_df = pd.concat([hierarchy_df, parent_clusters], ignore_index=True)

    print("After clustering, hierarchy")
    print(hierarchy_df)

    # Get cluster assignments and stats for base clusters (level 0)
    base_n_clusters = len(np.unique(cluster_labels[cluster_labels != "-1"]))
    base_n_noise = sum(cluster_labels == "-1")

    # Get stats for parent clusters (level 1)
    parent_n_clusters = len(np.unique(group_labels[group_labels != "-1"]))
    parent_n_noise = sum(group_labels == "-1")

    print(f"Found {base_n_clusters} base clusters with {base_n_noise} noise points")
    print(
        f"Grouped into {parent_n_clusters} parent clusters with {parent_n_noise} ungrouped clusters"
    )

    # Ensure 'parent' column is string before saving
    hierarchy_df["parent"] = hierarchy_df["parent"].astype(str)

    # Save hierarchy with explicit schema
    import pyarrow as pa

    schema = pa.schema(
        [
            ("cluster_id", pa.string()),
            ("parent", pa.string()),
            ("level", pa.int64()),
        ]
    )

    hierarchy_df.to_parquet(user_dir / "cluster_hierarchy.parquet", schema=schema)

    hierarchy_time = time.time() - start_time
    print(f"Hierarchy computation completed in {hierarchy_time:.2f} seconds")

    # Save clusterer
    print("Saving results...")
    with open(clusterer_path, "wb") as f:
        pickle.dump(clusterer, f)

    # Save cluster assignments
    clusters_data = {
        "labels": cluster_labels.tolist(),
        "probabilities": cluster_probs.tolist(),
        "soft_clusters": soft_clusters.tolist(),
    }
    with open(clusters_path, "w") as f:
        json.dump(clusters_data, f)

    # Add clusters to dataframe and save
    df["cluster"] = cluster_labels.astype(str)
    df["cluster_prob"] = cluster_probs
    print(f"Saving df with clusters to {df_path}, cols {df.columns}")
    print(df)
    df.to_parquet(df_path)

    volume.commit()

    return {
        "base_clusters": {"n_clusters": base_n_clusters, "n_noise": base_n_noise},
        "parent_clusters": {"n_clusters": parent_n_clusters, "n_noise": parent_n_noise},
        "timing": {
            "umap": umap_time,
            "clustering": clustering_time,
            "soft_clustering": soft_clustering_time,
            "hierarchy": hierarchy_time,
        },
        # "hierarchy_df": hierarchy_df.to_dict("records"),
    }


@app.function(
    image=image,
    volumes={"/twitter_data": volume},
    secrets=[modal.Secret.from_name("anthropic-secret")],
)
def label_one_cluster(cluster_id: str, tweet_texts: str):
    """Labels a single cluster"""
    return label_cluster(cluster_id, tweet_texts)


@app.function(
    image=image,
    volumes={"/twitter_data": volume},
    secrets=[modal.Secret.from_name("anthropic-secret")],
)
def label_clusters(username: str, force_recompute: bool = False):
    """Labels both level 0 clusters and their parent groups (level 1)"""
    print(f"Starting cluster labeling for {username}")
    data_dir = Path("/twitter_data")
    user_dir = data_dir / username
    df_path = user_dir / "convo_tweets_tweets_df.parquet"
    hierarchy_path = user_dir / "cluster_hierarchy.parquet"

    # Load data
    df = pd.read_parquet(df_path)
    hierarchy_df = pd.read_parquet(hierarchy_path)
    print(f"Hierarchy df shape: {hierarchy_df.shape}, columns: {hierarchy_df.columns}")
    print(hierarchy_df)

    # Check if level 0 clusters already have labels and we're not forcing recompute
    level0_clusters = hierarchy_df[hierarchy_df["level"] == 0]
    has_level0_labels = (
        not force_recompute
        and "name" in hierarchy_df.columns
        and not level0_clusters["name"].isna().all()
    )

    if has_level0_labels:
        print("Level 0 clusters already labeled, skipping...")
        level0_labels = level0_clusters[
            ["cluster_id", "name", "summary", "level", "bad_group_flag"]
        ]
    else:
        # Label level 0 clusters
        print("Labeling level 0 clusters...")
        cluster_id_text_tuples = [
            (cluster_id, get_cluster_tweet_texts(df, cluster_id))
            for cluster_id in level0_clusters["cluster_id"]
        ]
        level0_results = label_one_cluster.starmap(
            cluster_id_text_tuples, order_outputs=True
        )
        level0_labels = pd.DataFrame(
            [
                {
                    "cluster_id": str(label_res["cluster_id"]),
                    "name": label_res["name"],
                    "summary": label_res["summary"],
                    "level": 0,
                    "bad_group_flag": label_res["bad_group_flag"],
                }
                for label_res in level0_results
            ]
        )

    # Label parent clusters using hierarchy_df directly since it has the level column
    print("Labeling level 1 clusters...")
    merged_df = hierarchy_df.merge(
        level0_labels[["cluster_id", "name", "summary", "bad_group_flag"]],
        on="cluster_id",
        how="left",
        suffixes=("_old", ""),  # Keep the new labels without suffix
    )

    # Drop old label columns if they exist
    cols_to_drop = [col for col in merged_df.columns if col.endswith("_old")]
    merged_df = merged_df.drop(columns=cols_to_drop)

    print(f"Merged df shape: {merged_df.shape}, columns: {merged_df.columns}")
    print(merged_df[["cluster_id", "parent", "level", "name", "summary"]])
    print(merged_df[["cluster_id", "parent", "level", "name", "summary"]].head(20))
    level1_labels = label_cluster_groups(
        merged_df,
    )
    level1_labels["level"] = 1

    # Add bad_group_flag to level0_labels if not present
    if "bad_group_flag" not in level0_labels.columns:
        level0_labels["bad_group_flag"] = False

    # Combine level 0 and level 1 labels
    all_labels = pd.concat([level0_labels, level1_labels], ignore_index=True)

    # Convert bad_group_flag to boolean
    all_labels["bad_group_flag"] = all_labels["bad_group_flag"].map(
        {"0": False, "1": True, 0: False, 1: True}
    )
    all_labels["bad_group_flag"] = all_labels["bad_group_flag"].fillna(False)

    # Create final hierarchy DataFrame with consistent suffix handling
    hierarchy_df = hierarchy_df.merge(
        all_labels[["cluster_id", "name", "summary", "bad_group_flag"]],
        on="cluster_id",
        how="left",
        suffixes=("_old", ""),  # Keep new labels without suffix
    )

    # Drop any old columns from previous merges
    cols_to_drop = [col for col in hierarchy_df.columns if col.endswith("_old")]
    hierarchy_df = hierarchy_df.drop(columns=cols_to_drop)

    print(
        f"Merged hierarchy_df shape: {hierarchy_df.shape}, columns: {hierarchy_df.columns}"
    )

    # Only fill bad_group_flag with False as that's a valid default
    hierarchy_df["bad_group_flag"] = hierarchy_df["bad_group_flag"].fillna(False)

    # Check for missing labels
    missing_labels = hierarchy_df[
        hierarchy_df["name"].isna() | hierarchy_df["summary"].isna()
    ]
    if not missing_labels.empty:
        print(f"WARNING: Found {len(missing_labels)} clusters with missing labels:")
        print(missing_labels[["cluster_id", "level", "name", "summary"]])

    # Save with explicit schema
    import pyarrow as pa

    schema = pa.schema(
        [
            ("cluster_id", pa.string()),
            ("parent", pa.string()),
            ("level", pa.int64()),
            ("name", pa.string()),
            ("summary", pa.string()),
            ("bad_group_flag", pa.bool_()),
        ]
    )

    hierarchy_df.to_parquet(
        user_dir / "labeled_cluster_hierarchy.parquet", schema=schema
    )

    volume.commit()

    return {
        "hierarchy_df": hierarchy_df,
        "stats": {
            "level0_clusters": len(level0_labels),
            "level1_clusters": len(level1_labels),
            "total_clusters": len(hierarchy_df),
            "error_clusters": len(hierarchy_df[hierarchy_df["bad_group_flag"]]),
        },
    }


@app.function(image=image, volumes={"/twitter_data": volume}, timeout=600)
def get_or_create_analysis(username_: str, force_recompute: bool = False):
    """Check for cached analysis results or run full pipeline if needed"""
    username = username_.lower()
    data_dir = Path("/twitter_data")
    user_dir = data_dir / username

    # Define paths for all required files
    required_files = {
        "hierarchy": user_dir / "labeled_cluster_hierarchy.parquet",
        "tweets": user_dir / "convo_tweets_tweets_df.parquet",
        "incomplete_trees": user_dir / "convo_tweets_incomplete_trees.pkl",
        "trees": user_dir / "convo_tweets_trees.pkl",
        "embeddings": user_dir / "reduced_embeddings.npy",
    }

    # Check if all files exist and we're not forcing recompute
    all_files_exist = all(path.exists() for path in required_files.values())

    if all_files_exist and not force_recompute:
        print(f"Found cached analysis for {username}")
        return {
            "status": "cached",
            "paths": {name: str(path) for name, path in required_files.items()},
        }

    # Run each step of the pipeline with force_recompute
    process_result = process_archive.remote(username, force_recompute)
    print("Processing complete:", process_result)

    embed_result = embed_tweets.remote(username, force_recompute)
    print("Embedding complete:", embed_result)

    cluster_result = cluster_tweets.remote(username, force_recompute)
    print("Clustering complete:", cluster_result)

    label_result = label_clusters.remote(username, force_recompute)
    print("Labeling complete:", label_result)

    volume.reload()

    # Verify all files were created
    missing_files = [name for name, path in required_files.items() if not path.exists()]
    if missing_files:
        raise RuntimeError(f"Pipeline completed but files missing: {missing_files}")

    return {
        "status": "computed",
        "paths": {name: str(path) for name, path in required_files.items()},
        "pipeline_results": {
            "process": process_result,
            "embed": embed_result,
            "cluster": cluster_result,
            "label": label_result,
        },
    }


@app.local_entrypoint()
def main():
    username = "nosilverv"
    # result = get_or_create_analysis.remote(username)

    # Run each step of the pipeline
    # process_result = process_archive.remote(username)

    # print("Processing complete:", process_result)
    # embed_result = embed_tweets.remote(username)
    # print("Embedding complete:", embed_result)

    cluster_result = cluster_tweets.remote(username, force_recompute=True)
    print("Clustering complete:", cluster_result)

    # label_result = label_clusters.remote(username, force_recompute=True)
    # print("Labeling complete:", label_result)
    # print("Analysis complete")
