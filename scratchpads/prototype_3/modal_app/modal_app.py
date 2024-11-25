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
from typing import List, Tuple
import modal
import sys
import json
from .lib.label_clusters import (
    LABEL_CLUSTER_PROMPT,
    get_cluster_tweet_texts,
    label_all_clusters,
    label_cluster,
    label_cluster_groups,
    query_anthropic_model,
    extract_special_tokens,
    parse_extracted_data,
    curry,
)
from .lib.get_tweets import (
    check_and_process_tweets,
    patch_tweets_with_note_tweets,
    create_tweets_df,
    process_tweets,
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
from .lib.const import (
    get_user_paths,
    TWEETS_DF_SCHEMA,
    SUPABASE_URL,
    SUPABASE_KEY,
    CLUSTERED_TWEETS_DF_SCHEMA,
)
import toolz as tz

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
    import anthropic
    from anthropic.types.beta.message_create_params import (
        MessageCreateParamsNonStreaming,
    )
    from anthropic.types.beta.messages.batch_create_params import Request
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
    from .lib.ontological_label_lib import (
        make_cluster_str,
        tfidf_label_clusters,
        group_clusters,
        label_with_ontology,
        validate_ontology_results,
    )
    from .lib.prompts import (
        ONTOLOGY_LABEL_CLUSTER_PROMPT,
        ONTOLOGY_GROUP_PROMPT,
        group_ontology,
        ontology,
    )
    import pyarrow as pa
    import toolz as tz


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
    import pyarrow as pa
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


@app.function(image=image, volumes={"/twitter-archive-data": volume}, timeout=600)
def process_archive_data(archive: dict, username: str) -> dict:
    """Process archive data and return processed dataframe and stats"""
    # Extract username and accountId from archive if not provided
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    processed_tweets_df, trees, incomplete_trees = process_tweets(
        supabase, archive, username, include_date=False
    )

    return {
        "tweets_df": processed_tweets_df,
        "trees": trees,
        "incomplete_trees": incomplete_trees,
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
    res = bucket.download(f"{username}/archive.json")
    return res


@app.function(image=image, timeout=600)
def compute_embeddings(texts: list[str]) -> np.ndarray:
    """Compute embeddings for a list of texts"""
    # Initialize embedder and process in batches
    embedder = TextEmbeddingsInference()
    embeddings = []

    # Create batches
    batches = [texts[i : i + BATCH_SIZE] for i in range(0, len(texts), BATCH_SIZE)]

    print(f"Starting embeddings for {len(texts)} texts")
    batch_embeddings = embedder.embed.map(batches, order_outputs=True)
    for batch_embedding in batch_embeddings:
        embeddings.extend(batch_embedding)

    # Convert to numpy array
    return np.array(embeddings)


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


@app.function(image=image, volumes={"/twitter-archive-data": volume}, timeout=600)
def cluster_tweets(username: str, force_recompute: bool = False):
    """Handles caching and I/O for tweet clustering"""
    print(f"Starting clustering for {username}")
    data_dir = Path("/twitter-archive-data")
    user_dir = data_dir / username

    # Define paths
    paths = get_user_paths(username)

    volume.reload()

    clustered_tweets_df = pd.read_parquet(paths["tweets_df"])
    # Get reduced embeddings
    if paths["reduced_embeddings"].exists() and not force_recompute:
        print("Loading cached reduced embeddings...")
        reduced_embeddings = np.load(paths["reduced_embeddings"])
        umap_time = 0
    else:
        # Load embeddings and tweets
        print("Loading embeddings and tweets...")
        embeddings = np.load(paths["embeddings"])
        print(f"Loaded embeddings {embeddings.shape}")
        # Reduce dimensions with UMAP
        reduced_embeddings, umap_time = reduce_dimensions.remote(
            embeddings, n_components=5
        )
        np.save(paths["reduced_embeddings"], reduced_embeddings)

    # Check for cached clustering results
    if (
        all(
            paths[p].exists()
            for p in [
                "cluster_ontology_items",
                "cluster_probs",
                "soft_clusters",
                "clustered_tweets_df",
            ]
        )
        and not force_recompute
    ):
        print("Loading cached clustering results...")
        cluster_results = {
            "labels": np.load(paths["cluster_labels_arr"]),
            "probabilities": np.load(paths["cluster_probs"]),
            "soft_clusters": np.load(paths["soft_clusters"]),
            "timing": 0,
            "soft_clustering_time": 0,
        }
        # Calculate stats from cached data
        cluster_labels = cluster_results["labels"].astype(str)
        cluster_results["n_clusters"] = len(
            np.unique(cluster_labels[cluster_labels != "-1"])
        )
        cluster_results["n_noise"] = sum(cluster_labels == "-1")
    else:
        reduced_embeddings, umap_time = reduce_dimensions.remote(
            embeddings, n_components=5
        )
        # Run clustering pipeline
        min_cluster_size = max(
            5, min(100, int(reduced_embeddings.shape[0] * 0.001) + 1)
        )
        min_samples = min_cluster_size
        cluster_results = cluster_tweet_embeddings.remote(
            reduced_embeddings, min_cluster_size, min_samples
        )

        # Save clustering results
        np.save(paths["cluster_labels_arr"], cluster_results["labels"])
        np.save(paths["cluster_probs"], cluster_results["probabilities"])
        np.save(paths["soft_clusters"], cluster_results["soft_clusters"])

        # Update and save dataframe
        clustered_tweets_df["cluster"] = cluster_results["labels"]
        clustered_tweets_df["cluster_prob"] = cluster_results["probabilities"]

        # Ensure IDs are strings
        clustered_tweets_df["tweet_id"] = clustered_tweets_df["tweet_id"].astype(str)
        clustered_tweets_df["accountId"] = clustered_tweets_df["accountId"].astype(str)
        clustered_tweets_df["reply_to_tweet_id"] = clustered_tweets_df[
            "reply_to_tweet_id"
        ].astype(str)
        clustered_tweets_df["reply_to_user_id"] = clustered_tweets_df[
            "reply_to_user_id"
        ].astype(str)

        print(clustered_tweets_df.columns)
        print(clustered_tweets_df)
        if pd.api.types.is_datetime64_any_dtype(clustered_tweets_df["created_at"]):
            clustered_tweets_df["created_at"] = clustered_tweets_df[
                "created_at"
            ].dt.strftime("%Y-%m-%d %H:%M:%S%z")
        clustered_tweets_df["retweet_count"] = clustered_tweets_df[
            "retweet_count"
        ].astype("int64")
        clustered_tweets_df["favorite_count"] = clustered_tweets_df[
            "favorite_count"
        ].astype("int64")
        # Define schema for tweets_df
        # Convert datetime to string format
        clustered_tweets_df.to_parquet(
            paths["clustered_tweets_df"], schema=CLUSTERED_TWEETS_DF_SCHEMA
        )

    # Check for cached hierarchy results
    if paths["hierarchy"].exists() and not force_recompute:
        print("Loading cached hierarchy results...")
        hierarchy_df = pd.read_parquet(paths["hierarchy"])
        hierarchy_results = {
            "hierarchy_df": hierarchy_df,
            "n_clusters": len(hierarchy_df[hierarchy_df["level"] == 1]),
            "n_noise": len(hierarchy_df[hierarchy_df["parent"] == "-1"]),
            "timing": 0,
        }
    else:
        # Run centroid clustering
        hierarchy_results = cluster_centroids.remote(
            reduced_embeddings, cluster_results["labels"]
        )

        # Save results

        # Save hierarchy with schema
        schema = pa.schema(
            [
                ("cluster_id", pa.string()),
                ("parent", pa.string()),
                ("level", pa.int64()),
            ]
        )

        hierarchy_results["hierarchy_df"].to_parquet(paths["hierarchy"], schema=schema)
    print(hierarchy_results["hierarchy_df"])

    volume.commit()

    return {
        "base_clusters": {
            "n_clusters": cluster_results["n_clusters"],
            "n_noise": cluster_results["n_noise"],
        },
        "parent_clusters": {
            "n_clusters": hierarchy_results["n_clusters"],
            "n_noise": hierarchy_results["n_noise"],
        },
        "timing": {
            "umap": umap_time,
            "clustering": cluster_results["timing"],
            "soft_clustering": cluster_results["soft_clustering_time"],
            "hierarchy": hierarchy_results["timing"],
        },
    }


@app.function(gpu="T4", image=rapids_image, timeout=600)
def cluster_tweet_embeddings(
    reduced_embeddings: np.ndarray, min_cluster_size: int = 15, min_samples: int = 15
):
    """Clusters tweet embeddings using HDBSCAN"""
    print("Running HDBSCAN clustering...")
    start_time = time.time()
    clusterer = cuml.cluster.hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        prediction_data=True,
    )
    clusterer.fit(reduced_embeddings)

    clustering_time = time.time() - start_time
    print(f"HDBSCAN clustering completed in {clustering_time:.2f} seconds")

    # Get cluster assignments and probabilities
    print("Computing soft clusters...")
    soft_start = time.time()
    soft_clusters = cuml.cluster.hdbscan.all_points_membership_vectors(clusterer)
    soft_clustering_time = time.time() - soft_start

    cluster_labels = clusterer.labels_.astype(str)
    base_n_clusters = len(np.unique(cluster_labels[cluster_labels != "-1"]))
    base_n_noise = sum(cluster_labels == "-1")

    return {
        "labels": cluster_labels,
        "probabilities": clusterer.probabilities_,
        "soft_clusters": soft_clusters,
        "n_clusters": base_n_clusters,
        "n_noise": base_n_noise,
        "timing": clustering_time,
        "soft_clustering_time": soft_clustering_time,
    }


@app.function(gpu="T4", image=rapids_image, timeout=600)
def cluster_centroids(reduced_embeddings: np.ndarray, cluster_labels: np.ndarray):
    """Clusters the centroids of the base clusters"""
    print("Computing cluster centroids...")
    start_time = time.time()

    # Get unique non-noise labels and compute centroids
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


from toolz import keyfilter


def pick(allowlist, d):
    return keyfilter(lambda k: k in allowlist, d)


# Create error result object
def make_error_result(
    cluster_id,
    text_results,
    error_msg,
):
    return {
        "is_error": True,
        "error": str(error_msg),
        "message": text_results,
        "cluster_id": cluster_id,
        "cluster_summary": {
            "name": f"Error: {error_msg[:30]}...",
            "summary": str(error_msg),
        },
        "ontology_items": {},
        "low_quality_cluster": "1",
    }


@app.function(
    image=image,
    secrets=[modal.Secret.from_name("anthropic-secret")],
    concurrency_limit=10,
)
def label_one_cluster(
    cluster_id: str,
    cluster_str: str,
    model="claude-3-5-haiku-20241022",
    max_tokens: int = 1000,
    temperature: float = 0.0,
    max_retries: int = 3,
):
    """Labels a single cluster with error handling and retries"""

    message = ONTOLOGY_LABEL_CLUSTER_PROMPT.format(
        ontology=ontology,
        tweet_texts=cluster_str,
        ontology=ontology,
        previous_ontology="",
    )
    for attempt in range(max_retries):
        try:
            client = anthropic.Anthropic()
            send_msg = {"role": "user", "content": message}
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[send_msg],
            )
            text_results = response.content[0].text
        except Exception as e:
            if attempt == max_retries - 1:
                result = make_error_result(cluster_id, "", f"Query failed: {e}")
            continue
        try:
            answer = extract_special_tokens(text_results, tokens=["ANSWER"])["ANSWER"]
        except Exception as e:
            result = make_error_result(
                cluster_id, text_results, f"Token extraction failed: {e}"
            )
            continue
        try:
            parsed_answer = parse_extracted_data(answer)["ANSWER"]
        except Exception as e:
            result = make_error_result(cluster_id, text_results, f"Parse failed: {e}")
            continue

        validation_result = validate_ontology_results(parsed_answer, ontology)
        if not validation_result["valid"]:
            result = make_error_result(
                cluster_id,
                text_results,
                f"Validation failed: {validation_result['info']}",
            )
            continue
        else:
            return {
                "cluster_id": cluster_id,
                "is_error": False,
                "message": text_results,
                "ontology_items": pick(
                    [
                        k
                        for k in ontology.keys()
                        if k
                        not in ["schema_info", "low_quality_cluster", "cluster_summary"]
                    ],
                    parsed_answer,
                ),
                "cluster_summary": pick(["name", "summary"], parsed_answer),
                "low_quality_cluster": parsed_answer["low_quality_cluster"]["value"],
            }
    return result


# TODO finish this when relevant
# def label_cluster_batch(
#     cluster_label_data: List[Tuple[str, str]],
#     model="claude-3-5-haiku-20241022",
#     max_tokens: int = 1024,
#     temperature: float = 0.0,
# ):
#     """Labels a batch of clusters using the Messages Batch API with proper retrieval handling"""

#     client = anthropic.Anthropic()

#     def make_request(cluster_id: str, prompt: str):
#         return Request(
#             custom_id=cluster_id,  # Use cluster_id as custom_id for tracking
#             params=MessageCreateParamsNonStreaming(
#                 model=model,
#                 max_tokens=max_tokens,
#                 messages=[{"role": "user", "content": prompt}],
#             ),
#         )

#     messages = [
#         ONTOLOGY_LABEL_CLUSTER_PROMPT.format(
#             ontology=ontology,
#             tweet_texts=cluster_str,
#             ontology=ontology,
#             previous_ontology="",
#         )
#         for cluster_id, cluster_str in cluster_label_data
#     ]

#     # Create the batch
#     batch = client.beta.messages.batches.create(
#         requests=[
#             make_request(cluster_id, message)
#             for (cluster_id, _), message in zip(cluster_label_data, messages)
#         ]
#     )

#     # Poll until batch is complete
#     while True:
#         status = client.beta.messages.batches.retrieve(batch.id)
#         if status.processing_status == "ended":
#             break
#         time.sleep(60)  # Wait 5 seconds before polling again

#     # Get results
#     results = []
#     for result in client.beta.messages.batches.results(
#         batch.id,
#     ):
#         results.append(result)

#     return results


ANTHROPIC_TOKEN_LIMIT_PER_MINUTE = 100000
CHAR_PER_TOKEN = 4


# Make batches based on token limits
def make_batches(cluster_label_data):
    max_batch_chars = ANTHROPIC_TOKEN_LIMIT_PER_MINUTE * CHAR_PER_TOKEN
    batches = [[]]
    for cluster_id, cluster_str in cluster_label_data:
        cur_len = sum(len(cluster_str) for _, cluster_str in batches[-1])
        if cur_len + len(cluster_str) > max_batch_chars:
            batches.append([])
        batches[-1].append((cluster_id, cluster_str))
    return batches


def label_level0_clusters(
    df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    trees: dict,
    incomplete_trees: dict,
    tfidf_labels: dict,
    existing_labels: dict = None,
    force_recompute: bool = False,
    limit_clusters: int = None,
) -> Tuple[pd.DataFrame, dict]:
    """Label individual clusters (level 0) with their topics and summaries.

    Args:
        df: Main tweets DataFrame
        hierarchy_df: Hierarchy information DataFrame
        trees: Conversation trees dict
        incomplete_trees: Incomplete conversation trees dict
        tfidf_labels: TF-IDF labels for clusters
        existing_labels: Optional dict of existing labels
        force_recompute: Whether to force recomputation of existing labels

    Returns:
        Tuple of (level0_labels DataFrame, full cluster_labels dict)
    """
    limit_clusters = limit_clusters or len(hierarchy_df[hierarchy_df["level"] == 0])
    # Check if level 0 clusters already have labels

    unique_clusters = list(set(df["cluster"]))
    # Label level 0 clusters
    print("Labeling level 0 clusters...")
    cluster_label_data = [
        (
            cluster_id,
            make_cluster_str(df, trees, incomplete_trees, tfidf_labels, cluster_id),
        )
        for cluster_id in unique_clusters[:limit_clusters]
        if cluster_id != -1
    ]

    batches = make_batches(cluster_label_data)

    # Get full ontology results
    label_cluster_results = []

    for i, batch in enumerate(batches):
        batch_results = label_one_cluster.starmap(batch, order_outputs=True)
        label_cluster_results.extend(batch_results)
        if i < len(batches) - 1:
            time.sleep(30)

    error_clusters = {}
    cluster_ontology_items = {}
    for result in label_cluster_results:
        if result["is_error"]:
            error_clusters[str(result["cluster_id"])] = result
        else:
            cluster_ontology_items[str(result["cluster_id"])] = result

    if error_clusters:
        print(f"Errors labeling {len(error_clusters)} clusters.")

    # Extract required fields for hierarchy
    level0_labels = pd.DataFrame(
        [
            {
                "cluster_id": str(cid),
                "name": labels["cluster_summary"]["name"],
                "summary": labels["cluster_summary"]["summary"],
                "level": 0,
                "low_quality_cluster": labels["low_quality_cluster"]["value"] == "1",
            }
            for cid, labels in cluster_ontology_items.items()
            if "cluster_summary" in labels
            and "name" in labels["cluster_summary"]
            and "summary" in labels["cluster_summary"]
            and "low_quality_cluster" in labels
        ]
    )
    print(f"Level 0 labels: {level0_labels}")

    return level0_labels, cluster_ontology_items


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
    username: str, tweets_df: pd.DataFrame, trees: dict, incomplete_trees: dict
):
    paths = get_user_paths(username)
    tweets_df["created_at"] = tweets_df["created_at"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    tweets_df["retweet_count"] = tweets_df["retweet_count"].astype("int64")
    tweets_df["favorite_count"] = tweets_df["favorite_count"].astype("int64")
    tweets_df.to_parquet(
        paths["clustered_tweets_df"], schema=CLUSTERED_TWEETS_DF_SCHEMA
    )
    pickle.dump(trees, open(paths["trees"], "wb"))
    pickle.dump(incomplete_trees, open(paths["incomplete_trees"], "wb"))
    print(f"Saved post-hydration data for {username}")


def save_post_clustering_data(
    username: str,
    clustered_tweets_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    cluster_results: dict,
):
    paths = get_user_paths(username)
    # Ensure IDs are strings
    clustered_tweets_df["tweet_id"] = clustered_tweets_df["tweet_id"].astype(str)
    clustered_tweets_df["accountId"] = clustered_tweets_df["accountId"].astype(str)
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
    print(f"Saved post-clustering data for {username}")


def save_post_labeling_data(
    username: str, labeled_hierarchy_df: pd.DataFrame, cluster_ontology_items: dict
):
    paths = get_user_paths(username)
    schema = pa.schema(
        [
            ("cluster_id", pa.string()),
            ("parent", pa.string()),
            ("level", pa.int64()),
            ("name", pa.string()),
            ("summary", pa.string()),
            ("low_quality_cluster", pa.bool_()),
        ]
    )
    labeled_hierarchy_df.to_parquet(paths["labeled_hierarchy"], schema=schema)
    with open(paths["cluster_ontology_items"], "w") as f:
        json.dump(cluster_ontology_items, f, indent=2)
    print(f"Saved post-labeling data for {username}")


# I'll make this a modal function but we can copy it to be a local function later by looking up the app and volume
@app.function(
    image=image,
    volumes={"/twitter-archive-data": volume},
    secrets=[modal.Secret.from_name("anthropic-secret")],
    timeout=1200,
)
def orchestrator(username: str, force_recompute: str = "none"):
    """Orchestrator function that runs the entire pipeline"""
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
    else:
        archive = download_archive(username)

        # Compute
        result = process_archive_data.remote(archive, username)
        tweets_df = result["tweets_df"]
        trees = result["trees"]
        incomplete_trees = result["incomplete_trees"]

        # Save
        save_post_hydration_data(username, tweets_df, trees, incomplete_trees)

    # STEP 2: Embed tweets
    if force_recompute not in ["all", "embed", "process"]:
        print(f"Found cached embeddings for {username}")
        embeddings = np.load(paths["embeddings"])
    else:
        texts = tweets_df["emb_text"]
        embeddings = compute_embeddings.remote(texts)
        np.save(paths["embeddings"], embeddings)

    # STEP 3: Cluster tweets
    if force_recompute not in ["all", "cluster", "embed", "process"]:
        print(f"Found cached clusters for {username}")
        clustered_tweets_df = pd.read_parquet(paths["clustered_tweets_df"])
        hierarchy_df = pd.read_parquet(paths["hierarchy"])
    else:
        reduced_embeddings, umap_time = reduce_dimensions.remote(
            embeddings, n_components=5
        )
        # Run clustering pipeline
        min_cluster_size = max(
            5, min(100, int(reduced_embeddings.shape[0] * 0.001) + 1)
        )
        min_samples = min_cluster_size
        cluster_results = cluster_tweet_embeddings.remote(
            reduced_embeddings, min_cluster_size, min_samples
        )

        # Update and save dataframe
        clustered_tweets_df["cluster"] = cluster_results["labels"]
        clustered_tweets_df["cluster_prob"] = cluster_results["probabilities"]

        hierarchy_results = cluster_centroids.remote(
            reduced_embeddings, cluster_results["labels"]
        )
        hierarchy_df = hierarchy_results["hierarchy_df"]

        save_post_clustering_data(
            username, clustered_tweets_df, hierarchy_df, cluster_results
        )

    # TODO STEP 4: Label clusters
    if force_recompute not in ["all", "label", "cluster", "embed", "process"]:
        print(f"Found cached labels for {username}")
        labeled_hierarchy_df = pd.read_parquet(paths["labeled_hierarchy"])
        cluster_ontology_items = json.load(open(paths["cluster_ontology_items"]))
    else:
        limit_clusters = 3
        tfidf_labels = tfidf_label_clusters(
            clustered_tweets_df,
            n_top_terms=5,
            exclude_words=["current", "root", "context"],
        )
        # Label level 0 clusters

        # Label level 0 clusters
        cluster_label_data = [
            (
                cluster_id,
                make_cluster_str(
                    clustered_tweets_df,
                    trees,
                    incomplete_trees,
                    tfidf_labels,
                    cluster_id,
                ),
            )
            for cluster_id in list(set(clustered_tweets_df["cluster"]))[:limit_clusters]
            if cluster_id != -1
        ]

        batches = make_batches(cluster_label_data)

        # Get full ontology results
        cluster_ontology_items = {}
        for i, batch in enumerate(batches):
            batch_results = label_one_cluster.starmap(batch, order_outputs=True)
            for result in batch_results:
                cluster_ontology_items[str(result["cluster_id"])] = result
            if i < len(batches) - 1:
                time.sleep(60)

        # Check for errors
        error_clusters = [
            cid for cid, labels in cluster_ontology_items.items() if labels["is_error"]
        ]
        if error_clusters:
            print(f"Errors labeling {len(error_clusters)} clusters.")

        # Create labeled hierarchy dataframe
        labeled_hierarchy_df = hierarchy_df.merge(
            pd.DataFrame(
                [
                    {
                        "cluster_id": str(cid),
                        "name": labels["cluster_summary"]["name"],
                        "summary": labels["cluster_summary"]["summary"],
                        "level": 0,
                        "low_quality_cluster": labels["low_quality_cluster"]["value"]
                        == "1",
                    }
                    for cid, labels in cluster_ontology_items.items()
                    if all(
                        key in labels.get("cluster_summary", {})
                        for key in ["name", "summary"]
                    )
                    and "low_quality_cluster" in labels
                ]
            )[["cluster_id", "name", "summary", "low_quality_cluster"]],
            on="cluster_id",
            how="left",
            suffixes=("_old", ""),
        ).drop(columns=lambda x: x.endswith("_old"))

        # Label level 1 clusters and get final hierarchy

        group_results = group_clusters(
            clusters_df=labeled_hierarchy_df[
                labeled_hierarchy_df["name"].notna()
                & (labeled_hierarchy_df["level"] == 0)
            ],
            prompt=ONTOLOGY_GROUP_PROMPT,
            ontology=group_ontology,
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
                    "low_quality_cluster": False,
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


@app.local_entrypoint()
def main():
    username = "exgenesis"
    orchestrator.remote(username, force_recompute="none")
