# %%
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
# Add at the very top of your file
import sys
from pathlib import Path
import logging
from datetime import datetime

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

# Only modify path when running as notebook/interactive
if any("ipykernel" in arg for arg in sys.argv):
    sys.path.append(str(Path(__file__).parent))
    # Use local imports
    from lib.get_tweets import process_tweets
    from lib.const import (
        get_user_paths,
        SUPABASE_URL,
        SUPABASE_KEY,
        TWEETS_DF_SCHEMA,
        CLUSTERED_TWEETS_DF_SCHEMA,
        LABELED_HIERARCHY_DF_SCHEMA,
        HIERARCHY_DF_SCHEMA,
    )
else:
    # Use relative imports for Modal
    from .lib.get_tweets import process_tweets
    from .lib.const import (
        get_user_paths,
        SUPABASE_URL,
        SUPABASE_KEY,
        TWEETS_DF_SCHEMA,
        CLUSTERED_TWEETS_DF_SCHEMA,
        LABELED_HIERARCHY_DF_SCHEMA,
        HIERARCHY_DF_SCHEMA,
    )

import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple
import modal
import sys
import json


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

if any("ipykernel" in arg for arg in sys.argv):
    from lib.get_tweets import process_tweets
    from lib.const import (
        get_user_paths,
        SUPABASE_URL,
        SUPABASE_KEY,
        TWEETS_DF_SCHEMA,
        CLUSTERED_TWEETS_DF_SCHEMA,
        LABELED_HIERARCHY_DF_SCHEMA,
        HIERARCHY_DF_SCHEMA,
    )
    from lib.prompts import ONTOLOGY_GROUP_PROMPT, group_ontology
else:
    from .lib.get_tweets import process_tweets
    from .lib.const import (
        get_user_paths,
        SUPABASE_URL,
        SUPABASE_KEY,
        TWEETS_DF_SCHEMA,
        CLUSTERED_TWEETS_DF_SCHEMA,
        LABELED_HIERARCHY_DF_SCHEMA,
        HIERARCHY_DF_SCHEMA,
    )
    from .lib.prompts import ONTOLOGY_GROUP_PROMPT, group_ontology

# Configure GPU and model constants
GPU_CONFIG = modal.gpu.T4()
# MODEL_ID = "jxm/cde-small-v2"
MODEL_ID = "BAAI/bge-base-en-v1.5"  # 60 texts/sec on infinity on T4 , w bettertransformer, batch size 32
MODEL_ID = "dunzhang/stella_en_400M_v5"  # 56 texts/sec on infinity on T4 , w bettertransformer, batch size 32

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
    "openai",
    "toolz",
    "scikit-learn",
)

# Import supabase only inside the container where it's installed
with image.imports():
    from supabase import create_client
    import openai
    import pandas as pd
    import pickle
    import sys
    import time
    import numpy as np

    if any("ipykernel" in arg for arg in sys.argv):
        from lib.ontological_label_lib import (
            make_cluster_str,
            tfidf_label_clusters,
            label_with_ontology,
            validate_ontology_results,
            parallel_io_with_retry,
            label_one_cluster as label_one_cluster_lib,
        )
        from lib.prompts import (
            ONTOLOGY_LABEL_CLUSTER_PROMPT,
            ONTOLOGY_GROUP_PROMPT,
            ONTOLOGY_GROUP_EXAMPLES,
            group_ontology,
            ontology,
        )
        from lib.utils import pick, make_error_result
    else:
        from .lib.ontological_label_lib import (
            make_cluster_str,
            tfidf_label_clusters,
            label_with_ontology,
            validate_ontology_results,
            parallel_io_with_retry,
            label_one_cluster as label_one_cluster_lib,
        )
        from .lib.prompts import (
            ONTOLOGY_LABEL_CLUSTER_PROMPT,
            ONTOLOGY_GROUP_PROMPT,
            ONTOLOGY_GROUP_EXAMPLES,
            group_ontology,
            ontology,
        )
        from .lib.utils import pick, make_error_result
    import pyarrow as pa
    import toolz as tz


# Replace TEI image and server setup with Infinity setup
INFINITY_VERSION = "0.0.63"
infinity_image = (
    modal.Image.from_registry(f"michaelf34/infinity:{INFINITY_VERSION}")
    .pip_install(
        "infinity-emb[all]",
        "httpx",
        "numpy~=1.26.4",
        "pandas~=2.2.2",
        "supabase",
        "tqdm",
        "seaborn",
        "openai",
        "toolz",
        "pyarrow",
        "xformers",
    )
    .entrypoint([])
)

with infinity_image.imports():
    from infinity_emb import AsyncEngineArray, EngineArgs
    import httpx


rapids_image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/rapidsai/base:24.10-cuda12.0-py3.12", add_python="3.12"
    )
    .apt_install("build-essential")
    .pip_install(
        "numpy~=1.26.4",
        "pandas~=2.2.2",
        "supabase",
        "tqdm",
        "seaborn",
        "openai",
        "scikit-learn",
        "scikit-optimize",
        "hdbscan",
    )
    .entrypoint([])
)  # removes default entrypoint

with rapids_image.imports():
    import cuml
    from skopt import gp_minimize
    from skopt.space import Integer
    import pandas as pd
    import logging
    import pickle
    import os
    import sys
    import re
    from collections import defaultdict, deque
    import pyarrow as pa
    from tqdm import tqdm
    import re
    import seaborn as sns
    import time
    from pprint import pprint
    import numpy as np
    from modal_app.lib.cluster import find_optimal_clustering_params


# Add these constants near the top
MB_PER_TWEET = 0.5  # Conservative estimate of memory needed per tweet
BASE_MEMORY = 128  # Base memory in MB (1GB)
MAX_MEMORY = 32768  # Maximum memory in MB (32GB)


def calculate_memory_requirement(n_tweets: int) -> int:
    """Calculate memory requirement in MB based on number of tweets"""
    memory = BASE_MEMORY + int(n_tweets * MB_PER_TWEET)
    if memory > MAX_MEMORY:
        print(
            f"Warning: Required memory ({memory}MB) exceeds maximum ({MAX_MEMORY}MB). Using maximum available."
        )
    return min(memory, MAX_MEMORY)


@app.cls(
    gpu=GPU_CONFIG,
    image=infinity_image,
    concurrency_limit=10,
    allow_concurrent_inputs=10,
    memory=lambda args: calculate_memory_requirement(len(args[0])),
)
class InfinityEmbedder:

    def __init__(self):
        from infinity_emb import AsyncEngineArray, EngineArgs

        ENGINE_ARGS = EngineArgs(
            model_name_or_path=MODEL_ID,
            model_warmup=False,
            batch_size=32,
            bettertransformer=False,
            dtype="float16",
        )
        self.engine_args = ENGINE_ARGS

    def _get_array(self):
        from infinity_emb import AsyncEngineArray, EngineArgs

        return AsyncEngineArray.from_args([self.engine_args])

    @modal.build()
    async def download_model(self):
        print(f"downloading model {self.model_id} ...")
        self._get_array()

    @modal.method()
    async def embed(self, inputs: list[str]):
        logging.info(f"Embedding batch of {len(inputs)} texts")
        start = time.time()
        engine = self.engine_array[0]
        embeddings, _ = await engine.embed(sentences=inputs)
        logging.debug(f"Embedded batch in {time.time()-start:.3f}s")
        return embeddings

    @modal.enter()
    async def enter(self):
        logging.info("Starting embedding engine array")
        start_time = time.time()
        self.engine_array = self._get_array()
        await self.engine_array.astart()
        logging.info(f"Engine array started in {time.time()-start_time:.2f}s")

    @modal.exit()
    async def exit(self):
        print("Shutting down engine array...")
        if hasattr(self, "engine_array"):
            await self.engine_array.astop()
        print("Engine array shutdown complete")


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


@app.function(
    image=image,
    volumes={"/twitter-archive-data": volume},
    timeout=6000,
    memory=lambda args: calculate_memory_requirement(
        len(args[0].get("tweets", [])) if isinstance(args[0], dict) else BASE_MEMORY
    ),
)
def process_archive_data(archive: dict, username: str):
    """Process archive data and return processed dataframe and stats"""
    logging.info(f"Processing archive for {username}")
    tweet_count = len(archive.get("tweets", []))
    logging.info(f"Archive contains {tweet_count} tweets")

    # Extract username and account_id from archive if not provided
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    processed_tweets_df, trees, incomplete_trees, qts = process_tweets(
        supabase, archive, username, include_date=False
    )

    logging.info(f"Processed {processed_tweets_df.shape[0]} tweets")
    return {
        "tweets_df": processed_tweets_df,
        "trees": trees,
        "incomplete_trees": incomplete_trees,
        "qts": qts,
        "stats": {
            "username": username,
            "tweet_count": processed_tweets_df.shape[0],
            "trees_count": len(trees),
            "incomplete_trees_count": len(incomplete_trees),
        },
    }


def download_archive(username: str):
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    bucket = supabase.storage.from_("archives")
    bytes_data = bucket.download(f"{username}/archive.json")
    # Decode bytes to string and parse JSON
    json_str = bytes_data.decode("utf-8")
    return json.loads(json_str)


@app.function(gpu="T4", image=rapids_image, timeout=600)
def reduce_dimensions(
    embeddings: np.ndarray, n_components: int = 5
) -> tuple[np.ndarray, float]:
    """Reduces dimensionality of embeddings using UMAP

    Args:
        embeddings: Input embeddings array
        n_components: Number of dimensions to reduce to

    Returns:
        tuple of (reduced embeddings array, time taken in seconds)
    """
    print("Running UMAP dimension reduction...")
    start_time = time.time()

    reducer = cuml.manifold.UMAP(n_components=n_components)
    reduced_embeddings = reducer.fit_transform(embeddings)

    reduction_time = time.time() - start_time
    print(f"UMAP reduction completed in {reduction_time:.2f} seconds")

    return reduced_embeddings, reduction_time


# Modify the cluster_tweet_embeddings function
@app.function(
    gpu="T4",
    image=rapids_image,
    timeout=600,
    memory=lambda args: calculate_memory_requirement(
        len(args[0]) if isinstance(args[0], np.ndarray) else BASE_MEMORY
    ),
)
def cluster_tweet_embeddings(
    reduced_embeddings: np.ndarray,
    username: str = None,
    do_soft_clustering=False,
):
    """Clusters tweet embeddings using HDBSCAN"""

    print("Running HDBSCAN clustering...")
    start_time = time.time()

    # Check for account-specific params
    params = find_optimal_clustering_params(reduced_embeddings)

    print(
        f"clustering params: {pick(['min_cluster_size', 'min_samples', 'n_clusters', 'noise_ratio', 'persistence', 'score', 'cluster_selection_method'], params)}"
    )

    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=params["min_cluster_size"],
        min_samples=params["min_samples"],
        metric="euclidean",
        prediction_data=True,
        cluster_selection_method=params["cluster_selection_method"],
    )
    clusterer.fit(reduced_embeddings)

    clustering_time = time.time() - start_time
    print(f"HDBSCAN clustering completed in {clustering_time:.2f} seconds")

    # Get cluster assignments and probabilities
    print("Computing soft clusters...")
    soft_start = time.time()
    if do_soft_clustering:
        soft_clusters = np.array(clusterer.soft_clusters_)
    else:
        soft_clusters = np.array([])
    soft_clustering_time = time.time() - soft_start

    return {
        "labels": [str(l) for l in clusterer.labels_],
        "probabilities": clusterer.probabilities_.tolist(),
        "soft_clusters": soft_clusters,
        "n_clusters": len(np.unique(clusterer.labels_[clusterer.labels_ != "-1"])),
        "n_noise": sum(clusterer.labels_ == "-1"),
        "timing": clustering_time,
        "soft_clustering_time": soft_clustering_time,
        "params": params,
    }


@app.function(gpu="T4", image=rapids_image, timeout=600)
def cluster_centroids(reduced_embeddings: np.ndarray, cluster_labels: list[str]):
    """Clusters the centroids of the base clusters"""
    print("Computing cluster centroids...")
    start_time = time.time()
    print(f"len(cluster_labels): {len(cluster_labels)}")
    print(f"reduced: {reduced_embeddings.shape}")
    # Get unique non-noise labels and compute centroids
    unique_labels = np.unique([l for l in cluster_labels if l != "-1"])
    centroids = np.array(
        [
            reduced_embeddings[np.array(cluster_labels) == label].mean(axis=0)
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
    group_labels = centroid_clusterer.labels_.astype(str)

    # Create hierarchy dataframe
    hierarchy_df = pd.DataFrame(
        {
            "cluster_id": unique_labels,
            "parent": [(f"1-{l}" if l != "-1" else "-1") for l in group_labels],
            "level": 0,
        }
    )

    # Add parent clusters
    parent_clusters = pd.DataFrame(
        {
            "cluster_id": [
                f"1-{i}" for i in np.unique([l for l in group_labels if l != "-1"])
            ],
            "parent": "-1",
            "level": 1,
        }
    )

    hierarchy_df = pd.concat([hierarchy_df, parent_clusters], ignore_index=True)

    parent_n_clusters = len(np.unique(group_labels[group_labels != "-1"]))
    parent_n_noise = sum(group_labels == "-1")

    hierarchy_time = time.time() - start_time

    return {
        "hierarchy_df": hierarchy_df,
        "n_clusters": parent_n_clusters,
        "n_noise": parent_n_noise,
        "timing": hierarchy_time,
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("openrouter-api-key")],
    timeout=3600,
    concurrency_limit=100,
)
def label_one_cluster(
    cluster_id: str,
    cluster_str: str,
    model="meta-llama/llama-3.3-70b-instruct",
    # model="anthropic/claude-3.5-haiku-20241022:beta",
    max_tokens: int = 8000,
    temperature: float = 0.0,
    max_retries: int = 3,
):
    """Labels a single cluster with error handling and retries"""

    return label_one_cluster_lib(
        cluster_id, cluster_str, model, max_tokens, temperature, max_retries
    )


openai_TOKEN_LIMIT_PER_MINUTE = 100000
CHAR_PER_TOKEN = 4


# Make batches based on token limits
def make_batches(cluster_label_data):
    max_batch_chars = openai_TOKEN_LIMIT_PER_MINUTE * CHAR_PER_TOKEN
    batches = [[]]
    for cluster_id, cluster_str in cluster_label_data:
        cur_len = sum(len(cluster_str) for _, cluster_str in batches[-1])
        if cur_len + len(cluster_str) > max_batch_chars:
            batches.append([])
        batches[-1].append((cluster_id, cluster_str))
    return batches


def group_clusters(
    clusters_df: pd.DataFrame,
    prompt: str = ONTOLOGY_GROUP_PROMPT,
    ontology: dict = group_ontology,
    return_error: bool = True,
    max_validation_retries: int = 3,
    model="anthropic/claude-3.5-sonnet:beta",
) -> Dict[str, Any]:
    """Groups clusters based on their summaries using the provided ontology.

    Args:
        clusters_df: DataFrame containing cluster summaries with columns:
                    cluster_id, name, summary
        prompt: Template string for the grouping prompt
        ontology: Dictionary defining group structure schema
        return_error: If True, returns error info instead of raising
        max_validation_retries: Number of retries on validation failure

    Returns:
        Dict containing the grouped results matching ontology structure
    """
    # Format clusters into readable string
    clusters_str = ""
    for _, row in clusters_df.iterrows():
        cluster_str = (
            f"Cluster {row['cluster_id']}: {row['name']}\n{row['summary']}\n\n"
        )
        clusters_str += cluster_str

    return label_with_ontology(
        prompt=prompt,
        ontology=ontology,
        return_error=return_error,
        max_validation_retries=max_validation_retries,
        clusters_str=clusters_str,  # Pass formatted clusters string to prompt
        examples=ONTOLOGY_GROUP_EXAMPLES,
        model=model,
    )


# the phase from which to start recomputing
FORCE_RECOMPUTE_OPTIONS = ["all", "none", "process", "embed", "cluster", "label"]


def decide_force_recompute(username: str):
    paths = get_user_paths(username)
    trees_exist = paths["trees"].exists()
    incomplete_trees_exist = paths["incomplete_trees"].exists()
    clustered_tweets_df_exist = paths["clustered_tweets_df"].exists()
    cluster_ontology_items_exist = paths["cluster_ontology_items"].exists()
    labeled_hierarchy_exist = paths["labeled_hierarchy"].exists()
    hierarchy_df_exist = paths["hierarchy"].exists()
    tweets_df_exist = paths["tweets_df"].exists()
    embeddings_exist = paths["embeddings"].exists()

    if (
        trees_exist
        and incomplete_trees_exist
        and clustered_tweets_df_exist
        and cluster_ontology_items_exist
        and labeled_hierarchy_exist
    ):
        return {
            "status": f"Exists! Load from volume twitter-archive-data.",
            "username": username,
            "paths": {
                "trees": paths["trees"],
                "incomplete_trees": paths["incomplete_trees"],
                "clustered_tweets_df": paths["clustered_tweets_df"],
                "cluster_ontology_items": paths["cluster_ontology_items"],
                "labeled_hierarchy": paths["labeled_hierarchy"],
            },
        }
    elif (
        clustered_tweets_df_exist
        and hierarchy_df_exist
        and trees_exist
        and incomplete_trees_exist
    ):
        force_recompute = "label"
    elif embeddings_exist:
        force_recompute = "cluster"
    elif tweets_df_exist and trees_exist and incomplete_trees_exist:
        force_recompute = "embed"
    else:
        force_recompute = "process"
    return force_recompute


def save_post_hydration_data(
    username: str,
    tweets_df: pd.DataFrame,
    trees: dict,
    incomplete_trees: dict,
    qts: dict,
):
    paths = get_user_paths(username)
    tweets_df["created_at"] = pd.to_datetime(tweets_df["created_at"])
    tweets_df["created_at"] = tweets_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    tweets_df["retweet_count"] = tweets_df["retweet_count"].astype("int64")
    tweets_df["favorite_count"] = tweets_df["favorite_count"].astype("int64")
    tweets_df.to_parquet(paths["tweets_df"], schema=TWEETS_DF_SCHEMA)
    pickle.dump(trees, open(paths["trees"], "wb"))
    pickle.dump(incomplete_trees, open(paths["incomplete_trees"], "wb"))
    pickle.dump(qts, open(paths["qts"], "wb"))
    print(f"Saved post-hydration data for {username}")


def save_post_clustering_data(
    username: str,
    clustered_tweets_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    cluster_results: dict,
    clustering_params: dict,
    reduced_embeddings: np.ndarray,
):
    paths = get_user_paths(username)
    # Ensure IDs are strings
    clustered_tweets_df["tweet_id"] = clustered_tweets_df["tweet_id"].astype(str)
    clustered_tweets_df["account_id"] = clustered_tweets_df["account_id"].astype(str)
    clustered_tweets_df["reply_to_tweet_id"] = clustered_tweets_df[
        "reply_to_tweet_id"
    ].astype(str)
    clustered_tweets_df["reply_to_user_id"] = clustered_tweets_df[
        "reply_to_user_id"
    ].astype(str)
    if pd.api.types.is_datetime64_any_dtype(clustered_tweets_df["created_at"]):
        clustered_tweets_df["created_at"] = clustered_tweets_df[
            "created_at"
        ].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    clustered_tweets_df["retweet_count"] = clustered_tweets_df["retweet_count"].astype(
        "int64"
    )
    clustered_tweets_df["favorite_count"] = clustered_tweets_df[
        "favorite_count"
    ].astype("int64")

    # Save clustering results
    np.save(paths["cluster_labels_arr"], cluster_results["labels"])
    np.save(paths["cluster_probs"], cluster_results["probabilities"])
    np.save(paths["soft_clusters"], cluster_results["soft_clusters"])
    np.save(paths["reduced_embeddings"], reduced_embeddings)
    clustered_tweets_df.to_parquet(
        paths["clustered_tweets_df"], schema=CLUSTERED_TWEETS_DF_SCHEMA
    )

    schema = pa.schema(
        [
            ("cluster_id", pa.string()),
            ("parent", pa.string()),
            ("level", pa.int64()),
        ]
    )

    hierarchy_df.to_parquet(paths["hierarchy"], schema=schema)
    with open(paths["clustering_params"], "w") as f:
        json.dump(clustering_params, f, indent=2)
    print(f"Saved post-clustering data for {username}")


def save_post_labeling_data(
    username: str,
    labeled_hierarchy_df: pd.DataFrame,
    cluster_ontology_items: dict,
    local_tweet_id_maps: dict,
    group_results: dict,
):
    paths = get_user_paths(username)
    schema = LABELED_HIERARCHY_DF_SCHEMA
    labeled_hierarchy_df.loc[:, "low_quality_cluster"] = labeled_hierarchy_df[
        "low_quality_cluster"
    ].fillna("0")
    labeled_hierarchy_df.to_parquet(paths["labeled_hierarchy"], schema=schema)
    with open(paths["cluster_ontology_items"], "w") as f:
        json.dump(cluster_ontology_items, f, indent=2)

    with open(paths["local_tweet_id_maps"], "w") as f:
        json.dump(local_tweet_id_maps, f, indent=2)
    with open(paths["group_results"], "w") as f:
        json.dump(group_results, f, indent=2)
    print(f"Saved post-labeling data for {username}")


# Add near the top with other imports
from modal.exception import ExecutionError


# Add before orchestrator function
def get_running_jobs(username: str) -> list:
    """Get list of currently running jobs for a user"""
    try:
        f = modal.Function.lookup(
            app_name="twitter-archive-analysis", tag="orchestrator"
        )
        stats = f.get_stats()
        running_jobs = [
            call
            for call in stats.function_calls
            if call.args
            and call.args[0] == username  # First arg is username
            and call.status == "running"
        ]
        return running_jobs
    except Exception as e:
        print(f"Error checking running jobs: {e}")
        return []


# Add near the top with other constants
from functools import lru_cache


# Add after download_archive function
def get_archive_tweet_count(archive: dict) -> int:
    """Get number of tweets from archive dict"""
    return len(archive.get("tweets", []))


@app.function(
    image=image,
    volumes={"/twitter-archive-data": volume},
    secrets=[modal.Secret.from_name("openrouter-api-key")],
    timeout=36000,
    memory=lambda args: calculate_memory_requirement(
        len(download_archive(args[0]).get("tweets", []))
        if isinstance(args[0], str)
        else BASE_MEMORY
    ),
)
def orchestrator(username: str, force_recompute: str = "none", stop_after: str = None):
    logging.info(f"Starting pipeline for {username}")
    logging.info(f"Force recompute mode: {force_recompute}")
    username = username.lower()
    """Orchestrator function that runs the entire pipeline"""
    # Check for already running jobs
    running_jobs = get_running_jobs(username)
    if running_jobs:
        raise ExecutionError(
            f"Analysis already running for {username}. "
            f"Job started at {running_jobs[0].start_time}"
        )

    volume.reload()
    paths = get_user_paths(username)
    paths["user_dir"].mkdir(exist_ok=True)

    if force_recompute not in FORCE_RECOMPUTE_OPTIONS:
        raise ValueError(f"Invalid force_recompute option: {force_recompute}")

    if force_recompute == "none":
        force_recompute = decide_force_recompute(username)

    tweets_df = None
    trees = None
    incomplete_trees = None
    embeddings = None
    clustered_tweets_df = None
    hierarchy_df = None
    labeled_hierarchy_df = None
    cluster_ontology_items = None

    # STEP 1:  Hydrate archive: get replies and make thread trees
    # Check if processed file exists and we're not forcing recompute
    if force_recompute not in ["all", "process"]:
        print(f"Found cached processed tweets for {username}")
        tweets_df = pd.read_parquet(paths["tweets_df"])
        trees = pickle.load(open(paths["trees"], "rb"))
        incomplete_trees = pickle.load(open(paths["incomplete_trees"], "rb"))
        qts = pickle.load(open(paths["qts"], "rb"))
    else:
        logging.info(f"Processing raw archive data")
        archive = download_archive(username)
        logging.info(f"Downloaded archive with {len(archive.get('tweets', []))} tweets")

        # Compute
        result = process_archive_data.remote(archive, username)
        tweets_df = result["tweets_df"]
        trees = result["trees"]
        incomplete_trees = result["incomplete_trees"]
        qts = result["qts"]

        # Save
        save_post_hydration_data(username, tweets_df, trees, incomplete_trees, qts)

    if stop_after == "process":
        return

    # STEP 2: Embed tweets
    if force_recompute not in ["all", "embed", "process"]:
        print(f"Found cached embeddings for {username}")
        embeddings = np.load(paths["embeddings"])
    else:
        texts = tweets_df["emb_text"].tolist()
        embedder = InfinityEmbedder()
        embeddings = []

        # Create batches
        batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

        print(f"Starting embeddings for {len(texts)} texts")
        start_time = time.time()

        batch_embeddings = embedder.embed.map(batches, order_outputs=True)
        for batch_embedding in batch_embeddings:
            embeddings.extend(batch_embedding)

        total_time = time.time() - start_time
        texts_per_sec = len(texts) / total_time

        print(f"Embedding completed in {total_time:.2f} seconds")
        print(f"Average speed: {texts_per_sec:.1f} texts/second")

        # Convert to numpy array
        embeddings = np.array(embeddings)
        np.save(paths["embeddings"], embeddings)

    if stop_after == "embed":
        return

    # STEP 3: Cluster tweets
    if force_recompute not in ["all", "cluster", "embed", "process"]:
        print(f"Found cached clusters for {username}")
        clustered_tweets_df = pd.read_parquet(paths["clustered_tweets_df"])
        hierarchy_df = pd.read_parquet(paths["hierarchy"])
    else:
        reduced_embeddings, umap_time = reduce_dimensions.remote(
            embeddings, n_components=5
        )
        # Pass username to clustering function
        cluster_results = cluster_tweet_embeddings.remote(
            reduced_embeddings, username=username, do_soft_clustering=False
        )
        print(f"n_clusters: {cluster_results['n_clusters']}")
        print(f"n_noise: {cluster_results['n_noise']}")
        print(f"timing: {cluster_results['timing']}")
        print(f"soft_clustering_time: {cluster_results['soft_clustering_time']}")
        print(f"params: {cluster_results['params']}")
        # Update and save dataframe
        clustered_tweets_df = tweets_df.copy()
        clustered_tweets_df["cluster"] = cluster_results["labels"]
        clustered_tweets_df["cluster_prob"] = cluster_results["probabilities"]

        hierarchy_results = cluster_centroids.remote(
            reduced_embeddings, cluster_results["labels"]
        )
        hierarchy_df = hierarchy_results["hierarchy_df"]

        save_post_clustering_data(
            username,
            clustered_tweets_df,
            hierarchy_df,
            cluster_results,
            cluster_results["params"],
            reduced_embeddings,
        )

    if stop_after == "cluster":
        return

    # STEP 4: Label clusters
    if force_recompute not in ["all", "label", "cluster", "embed", "process"] or (
        stop_after and stop_after != "label"
    ):
        print(f"Found cached labels for {username}")
        labeled_hierarchy_df = pd.read_parquet(paths["labeled_hierarchy"])
        cluster_ontology_items = json.load(open(paths["cluster_ontology_items"]))
        # Clear any existing selected_ids when loading cached data
        labeled_hierarchy_df["selected_ids"] = None
    else:
        limit_clusters = len(set(clustered_tweets_df["cluster"]))
        # limit_clusters = 5
        tfidf_labels = tfidf_label_clusters(
            clustered_tweets_df,
            n_top_terms=5,
            exclude_words=["current", "root", "context"],
        )

        # Label level 0 clusters
        cluster_str_data = [
            (
                cluster_id,
                make_cluster_str(
                    clustered_tweets_df,
                    trees,
                    incomplete_trees,
                    tfidf_labels,
                    cluster_id,
                    qts,
                ),
            )
            for cluster_id in tqdm(
                [
                    c
                    for c in list(set(clustered_tweets_df["cluster"]))[:limit_clusters]
                    if str(c) != "-1"
                ],
                desc="Making cluster strings",
            )
        ]

        cluster_label_data = [
            (cluster_id, cluster_str)
            for cluster_id, (cluster_str, _) in cluster_str_data
        ]
        local_tweet_id_maps = {
            cluster_id: local_id_map
            for cluster_id, (_, local_id_map) in cluster_str_data
        }
        print(f"number of clusters: {len(cluster_label_data)}")

        # print total number of chars in all cluster_str
        print(
            f"total number of chars in all cluster_str: {sum([len(cluster_str) for _, cluster_str in cluster_label_data])}"
        )
        start_time = time.time()
        results = list(
            label_one_cluster.starmap(cluster_label_data, order_outputs=True)
        )
        print(f"Labeling clusters took {time.time() - start_time:.2f}s")
        print(f"got {len(results)} results from {len(cluster_label_data)} clusters")

        # Process results into cluster ontology items
        cluster_ontology_items = {
            result["cluster_id"]: result
            for result in results
            if result and (not result["is_error"])
        }

        error_clusters = [
            result["cluster_id"]
            for result in results
            if (not result) or result["is_error"]
        ]

        if error_clusters:
            print(f"Errors labeling {len(error_clusters)} clusters.")

        # Create labeled hierarchy dataframe by merging with cluster info
        cluster_info = [
            {
                "cluster_id": str(cid),
                "name": labels["cluster_summary"]["name"],
                "summary": labels["cluster_summary"]["summary"],
                "level": 0,
                "low_quality_cluster": labels["low_quality_cluster"],
            }
            for cid, labels in cluster_ontology_items.items()
        ]

        labeled_hierarchy_df = hierarchy_df.merge(
            pd.DataFrame(cluster_info)[
                ["cluster_id", "name", "summary", "low_quality_cluster"]
            ],
            on="cluster_id",
            how="left",
            suffixes=("_old", ""),
        )
        # Drop columns ending with _old
        old_cols = [col for col in labeled_hierarchy_df.columns if col.endswith("_old")]
        labeled_hierarchy_df = labeled_hierarchy_df.drop(columns=old_cols)

        # Label level 1 clusters and get final hierarchy

        group_results = group_clusters(
            clusters_df=labeled_hierarchy_df[
                labeled_hierarchy_df["name"].notna()
                & (labeled_hierarchy_df["level"] == 0)
                & (labeled_hierarchy_df["low_quality_cluster"] == "0")
            ],
            prompt=ONTOLOGY_GROUP_PROMPT,
            ontology=group_ontology,
            # model="meta-llama/llama-3.3-70b-instruct",
            model="anthropic/claude-3.5-sonnet:beta",
        )

        if group_results.get("is_error"):
            raise ValueError(
                f"Error grouping clusters: {group_results.get('error')}, {group_results}"
            )

        # Process results into DataFrame
        # Create group rows
        group_rows = []
        for i, group in enumerate(group_results["groups"]):
            group_id = f"1-{chr(65+i)}"  # Assign "1-A", "1-B", etc
            group_rows.append(
                {
                    "cluster_id": group_id,
                    "parent": "-1",
                    "level": 1,
                    "name": group["name"],
                    "summary": group["summary"],
                    "low_quality_cluster": "0",
                }
            )

        # Add group rows to DataFrame
        labeled_hierarchy_df = pd.concat(
            [labeled_hierarchy_df, pd.DataFrame(group_rows)], ignore_index=True
        )

        # Update member info
        for i, group in enumerate(group_results["groups"]):
            group_id = f"1-{chr(65+i)}"
            for member in group["members"]:
                mask = labeled_hierarchy_df["cluster_id"].astype(str) == str(
                    member["id"]
                )
                labeled_hierarchy_df.loc[mask, "parent"] = group_id

        print("labeled_hierarchy_df")
        print(labeled_hierarchy_df)
        # Initialize selected_ids as None for all clusters
        labeled_hierarchy_df["selected_ids"] = None
        save_post_labeling_data(
            username,
            labeled_hierarchy_df,
            cluster_ontology_items,
            local_tweet_id_maps,
            group_results,
        )

    try:
        logging.info(f"Pipeline completed for {username}")
        return {
            "status": "success",
            "username": username,
            "paths": {
                "trees": paths["trees"],
                "incomplete_trees": paths["incomplete_trees"],
                "clustered_tweets_df": paths["clustered_tweets_df"],
                "cluster_ontology_items": paths["cluster_ontology_items"],
                "labeled_hierarchy": paths["labeled_hierarchy"],
            },
        }
    except Exception as e:
        logging.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise


@app.local_entrypoint()
def main():
    usernames = ["sunriseoath"]
    usernames = ["DRMacIver"]
    usernames = ["silverarm0r"]
    for username in usernames:
        print(f"\n\n\nProcessing {username}")
        orchestrator.remote(username.lower(), force_recompute="label")


# %%
