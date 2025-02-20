# %%
import modal
import numpy as np
import io
import pickle
import pandas as pd
import os

# from lib.cluster import *
from lib.ontological_label_lib import *
from lib.prompts import *
from dotenv import load_dotenv
import os
from supabase import create_client, Client
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from lib.get_tweets import *

# Add project root to Python path
project_root = str(
    Path(__file__).parent.parent.parent
)  # Go up 3 levels to project root
if project_root not in sys.path:
    sys.path.append(project_root)

# Now import using absolute path
from src.utils.load_data import *  # Or import specific functions you need
import hdbscan

# %%

from typing import Optional
import numpy as np
import pandas as pd
import hdbscan
from skopt import gp_minimize
from skopt.space import Integer


def _evaluate_clustering(
    clusterer, embeddings, ref_num_clusters: float, noise_ratio_threshold: float = 0.45
):
    """
    Evaluate the clustering results and compute a cost value.

    Parameters
    ----------
    clusterer : hdbscan.HDBSCAN object (fitted)
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
    persistence = np.mean(clusterer.cluster_persistence_) * 5
    cluster_sizes = [np.sum(labels == i) for i in set(labels) if i != -1]
    mean_cluster_size = np.mean(cluster_sizes)
    max_cluster_size = np.max(cluster_sizes)

    # Penalties:
    # 1. Noise penalty: Penalize if noise ratio exceeds noise_ratio_threshold.
    noise_penalty = max(0, noise_ratio - noise_ratio_threshold)

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
    n_random: int = 10,
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
    min_min_samples: Optional[int] = 6,
    max_min_samples: Optional[int] = 20,
    n_calls: int = 25,
    random_state: int = 42,
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

    min_min_cluster_size = max(min_min_cluster_size, int(n_tweets ** (1 / 3)))
    # Initial values based on user request
    initial_min_cluster_size = max(min_min_cluster_size, ref_cluster_size)

    # Ensure max values are larger than min values and are integers
    max_cluster_size = int(
        max(
            lower_max_min_cluster_size,
            max(int(ref_cluster_size * 3), int(n_tweets ** (1 / 2.5))),
            min_min_cluster_size + 1,
        )
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

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric="euclidean",
            gen_min_span_tree=True,
            prediction_data=True,
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
    result = gp_minimize(
        objective, space, x0=x0, n_calls=n_calls + len(x0), random_state=random_state
    )

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Extract best parameters
    best_min_cluster_size, best_min_samples = result.x

    # Evaluate once more with the best parameters
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=best_min_cluster_size,
        min_samples=best_min_samples,
        metric="euclidean",
        gen_min_span_tree=True,
        prediction_data=True,
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
    }


# %%
# Supabase config
url: str = "https://fabxmporizzqflnftavs.supabase.co"
key: str = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
)
supabase: Client = create_client(url, key)
# test supabase


# ANTHROPIC_API_KEY =


load_dotenv(
    "/Users/frsc/Documents/Projects/community-archive-personal/scratchpads/clustering_experiments/.env"
)


# Create error result object
import toolz as tz


def pick(allowlist, d):
    return tz.keyfilter(lambda k: k in allowlist, d)


# %%
volume = modal.Volume.lookup("twitter-archive-data")
# %%
username = "euxenus"
username = "nosilverv"
username = "octopusyarn"
username = "erikbjare"
username = "romeostevens76"
username = "johnsonmxe"
username = "mechanical_monk"
data = {}

# %%
usernames = [
    "mechanical_monk",
    "tyleralterman",
]
username = "exgenesis"
username = "bierlingm"

data = load_from_volume(
    username,
    volume,
    required_files=[
        "tweets_df.parquet",
        "reduced_embeddings.npy",
        # "clustered_tweets_df.parquet",
        # "qts.pkl",
        # "trees.pkl",
        # "incomplete_trees.pkl",
        # "cluster_hierarchy.parquet",
        # "labeled_cluster_hierarchy.parquet",
        # "cluster_ontology_items.json",
        # "group_results.json",
    ],
)

tweet_ids = data["tweets_df.parquet"]["tweet_id"].tolist()
tweets_df = data["tweets_df.parquet"]
print(tweets_df.shape)
reduced_embeddings = data["reduced_embeddings.npy"]
print(reduced_embeddings.shape)
# hierarchy_df = data["cluster_hierarchy.parquet"]
# labeled_hierarchy_df = data["labeled_cluster_hierarchy.parquet"]
# ontology_items = data["cluster_ontology_items.json"]
# group_results = data["group_results.json"]
# qts = data["qts.pkl"]
# trees = data["trees.pkl"]
# incomplete_trees = data["incomplete_trees.pkl"]
# now cluster with top parameters
# Find optimal parameters
import umap


umap_embeddings = umap.UMAP(n_components=2).fit_transform(
    data["reduced_embeddings.npy"]
)

# %%
params = find_optimal_clustering_params(
    data["reduced_embeddings.npy"],
    n_calls=25,
)
# %%
print(
    f"Params: {pick(['min_cluster_size', 'min_samples', 'n_clusters', 'noise_ratio', 'persistence', 'score'], params)}"
)
# display(pd.DataFrame(params["results_df"]))
# Create clusterer with optimal params
clusterer = hdbscan.HDBSCAN(
    min_cluster_size=44 or params["min_cluster_size"],
    min_samples=6 or params["min_samples"],
    metric="euclidean",
    gen_min_span_tree=True,
    cluster_selection_method="leaf",
)
clusterer.fit(data["reduced_embeddings.npy"])
print(f"N clusters: {clusterer.labels_.max()}")
print(f"N noise: {sum(clusterer.labels_ == -1)}")
print(f"params min_cluster_size: {params['min_cluster_size']}")
print(f"params min_samples: {params['min_samples']}")
print(f"params score: {params['score']}")
print(f"params persistence: {params['persistence']}")
import plotly.express as px

fig = px.scatter(
    x=umap_embeddings[:, 0],
    y=umap_embeddings[:, 1],
    color=clusterer.labels_,
    title="UMAP Projection of Embeddings",
    labels={"x": "UMAP1", "y": "UMAP2"},
    width=1200,
    height=800,
    render_mode="webgl",
    hover_data={"text": tweets_df["full_text"]},  # Add tweet text to tooltips
)
fig.update_traces(marker=dict(size=5))
fig.show()

# %%
tweets_df["cluster"] = clusterer.labels_
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 8), dpi=300)
# pd.Series(clusterer.labels_[clusterer.labels_ != -1]).value_counts(
#     normalize=False
# ).plot(kind="bar")
pd.Series(tweets_df[tweets_df["cluster"] != -1].value_counts("cluster")).plot(
    kind="bar"
)

plt.title("Cluster Label Distribution")
plt.xlabel("Cluster Label")
plt.ylabel("Count")
plt.tight_layout()
plt.show()
# plot distribution
# tweets_df = data["tweets_df.parquet"]
# tweets_df = data["clustered_tweets_df.parquet"]

from pprint import pprint


def tfidf_label_clusters(
    tweets_df, n_top_terms=3, exclude_words=["current", "quoted", "context", "root"]
):
    if exclude_words is None:
        exclude_words = []

    # Compute TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(tweets_df["emb_text"])

    # Get feature names
    feature_names = vectorizer.get_feature_names_out()

    # Create DataFrame with TF-IDF scores
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

    # Add cluster labels to TF-IDF DataFrame
    tfidf_df["cluster"] = tweets_df["cluster"]

    # Group by cluster and sum TF-IDF scores
    cluster_tfidf = tfidf_df.groupby("cluster").sum()

    # Get top terms for each cluster, excluding specified words
    cluster_labels = {}
    for cluster, row in cluster_tfidf.iterrows():
        filtered_row = row.drop(labels=exclude_words, errors="ignore")
        top_terms = filtered_row.nlargest(n_top_terms).index.tolist()
        cluster_labels[cluster] = top_terms

    return cluster_labels


tfidf_labels = tfidf_label_clusters(tweets_df, n_top_terms=5)
pprint(tfidf_labels)
# %%
# After running random search, add:
results_df = pd.DataFrame(params["results_df"])
plot_parameter_coverage(results_df, value_col="n_clusters")
plot_parameter_coverage(results_df, value_col="persistence")
plot_parameter_coverage(results_df, value_col="score")


# %%
# cluster_ontology_items = data["cluster_ontology_items.json"]
# %%
condensed_tree = clusterer.condensed_tree_
plt.figure(figsize=(15, 10), dpi=300)
condensed_tree.plot()
plt.tight_layout()
plt.show()

# %%
# import cProfile
# import pstats

# Example usage:
cluster_id = "5"  # Your cluster cluster_id

# profiler = cProfile.Profile()
# profiler.enable()

cluster_str, local_id_map = make_cluster_str(
    tweets_df,
    trees,
    incomplete_trees,
    tfidf_labels,
    cluster_id,
    qts,
    max_chars=50000,
)
# profiler.disable()

# # Print sorted stats
# stats = pstats.Stats(profiler).sort_stats("cumulative")
# stats.print_stats(20)  # Show top 20 time-consuming functions
# %%
print(cluster_str)
print(len(cluster_str))
# %%


def make_error_result(cluster_id: str, message: str, error: str) -> Dict[str, Any]:
    """Create an error result dict"""
    return {
        "is_error": True,
        "error": error,
        "message": message,
        "cluster_id": cluster_id,
        "cluster_summary": {"name": f"Error: {error[:50]}...", "summary": error},
        "ontology_items": {},
        "low_quality_cluster": "1",
    }


# %%
import os
from openai import OpenAI


def parse_extracted_data(extracted_data: Dict[str, str]) -> Dict[str, dict]:
    """
    Parses the extracted JSON data into Python dictionaries.

    Args:
        extracted_data (Dict[str, str]): The extracted JSON data as strings.

    Returns:
        Dict[str, dict]: A dictionary where the keys are tokens and the values are parsed JSON objects.
    """
    parsed_data = {}
    for token, json_string in extracted_data.items():
        if json_string:
            try:
                # Remove ```json and ``` if present
                cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
                parsed_data[token] = json.loads(cleaned)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON for token {token}: {e}")
                parsed_data[token] = None
        else:
            parsed_data[token] = None
    return parsed_data


def get_openrouter_client():
    """Get OpenRouter client with proper configuration"""
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        default_headers={
            "HTTP-Referer": "https://community-archive.org",
            "X-Title": "community-archive",
        },
    )


def query_llm(
    message: str,
    model: str = "anthropic/claude-3.5-haiku-20241022:beta",
    max_tokens: int = 8000,
    temperature: float = 0.0,
) -> str:
    """Query LLM through OpenRouter

    Args:
        message: Prompt to send
        model: Model to use
        max_tokens: Max tokens in response
        temperature: Sampling temperature

    Returns:
        Model response text
    """
    client = get_openrouter_client()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": message}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


def label_one_cluster(
    cluster_id: str,
    cluster_str: str,
    model="anthropic/claude-3.5-haiku-20241022:beta",
    max_tokens: int = 8000,
    temperature: float = 0.0,
    max_retries: int = 3,
):
    """Labels a single cluster with error handling and retries"""

    print(f"labeling {cluster_id}")
    message = ONTOLOGY_LABEL_CLUSTER_PROMPT.format(
        ontology=ontology,
        tweet_texts=cluster_str,
        previous_ontology="",
        example_answer=ONTOLOGY_LABEL_CLUSTER_EXAMPLE,
    )

    result = None  # Initialize result outside the loop

    for attempt in range(max_retries):
        try:
            start_time = time.time()
            text_results = query_llm(
                message=message,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            print(f"LLM call for {cluster_id} took {time.time() - start_time:.2f}s")

            answer = extract_special_tokens(text_results, tokens=["ANSWER"])
            if not answer or "ANSWER" not in answer:
                result = make_error_result(
                    cluster_id, text_results, "No ANSWER token found"
                )
                print(f"no answer token for {cluster_id}")
                continue

            parsed_answer = parse_extracted_data(answer)["ANSWER"]
            if parsed_answer is None:
                result = make_error_result(
                    cluster_id, text_results, "Failed to parse JSON"
                )
                print(f"failed to parse json for {cluster_id}")
                continue

            validation_result = validate_ontology_results(parsed_answer, ontology)
            if not validation_result["valid"]:
                result = make_error_result(
                    cluster_id,
                    text_results,
                    f"Validation failed: {validation_result['info']}",
                )
                print(f"validation failed for {cluster_id}")
                continue

            print(f"validated {cluster_id}")
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
                "cluster_summary": pick(
                    ["name", "summary"], parsed_answer["cluster_summary"]
                ),
                "low_quality_cluster": parsed_answer["low_quality_cluster"]["value"],
            }

        except Exception as e:
            result = make_error_result(
                cluster_id,
                text_results if "text_results" in locals() else "",
                f"Error: {str(e)}",
            )
            continue

    # If we get here, all attempts failed
    return result or make_error_result(cluster_id, "", "Max retries exceeded")


# %%
import nest_asyncio  # Add this import

nest_asyncio.apply()  # Allow nested event loops
import asyncio

clusters = [c for c in tweets_df["cluster"].unique() if str(c) != "-1"]

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed


def process_cluster(cluster_id):
    cluster_str, local_id_map = make_cluster_str(
        tweets_df,
        trees,
        incomplete_trees,
        tfidf_labels,
        cluster_id,
        qts,
        max_chars=50000,
    )
    return label_one_cluster(
        cluster_id, cluster_str, model="meta-llama/llama-3.3-70b-instruct"
    )


def process_all_clusters(max_workers=8 * 5):
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_cluster = {
            executor.submit(process_cluster, str(c)): str(c) for c in ["40", "25"]
        }

        for future in as_completed(future_to_cluster):
            cluster_id = future_to_cluster[future]
            try:
                results[cluster_id] = future.result()
            except Exception as e:
                print(f"Cluster {cluster_id} generated an exception: {e}")
                results[cluster_id] = make_error_result(cluster_id, "", str(e))

    return results


# Run it
start_time = time.time()
results = process_all_clusters()
print(f"Processing clusters took {time.time() - start_time:.2f}s")
display(results)

# %%
message = ONTOLOGY_LABEL_CLUSTER_PROMPT.format(
    ontology=ontology,
    tweet_texts=cluster_str,
    previous_ontology="",
    example_answer=ONTOLOGY_LABEL_CLUSTER_EXAMPLE,
)
start_time = time.time()
text_results = query_llm(
    message=message,
    model="anthropic/claude-3.5-haiku-20241022:beta",
)
print(f"LLM call for {cluster_id} took {time.time() - start_time:.2f}s")
print(text_results)
# %%


answer = extract_special_tokens(text_results, tokens=["ANSWER"])
parsed_answer = parse_extracted_data(answer)["ANSWER"]
validation_result = validate_ontology_results(parsed_answer, ontology)
display(answer)
display(parsed_answer)
display(validation_result)
# %%
ontology_items
# %%


async def fetch_openrouter_key():
    async with httpx.AsyncClient() as client:
        r = await client.get(
            "https://openrouter.ai/api/v1/auth/key",
            headers={"Authorization": f'Bearer {os.environ["OPENROUTER_API_KEY"]}'},
        )
        return r.json()


# Now this will work in interactive environments
key_response = asyncio.run(fetch_openrouter_key())
display(key_response)
# %%
supabase = create_client(url, key)


def download_archive(username: str):
    supabase = create_client(url, key)
    bucket = supabase.storage.from_("archives")
    bytes_data = bucket.download(f"{username}/archive.json")
    # Decode bytes to string and parse JSON
    json_str = bytes_data.decode("utf-8")
    return json.loads(json_str)


archive = download_archive("octopusyarn")
# %%


# processed_tweets_df, trees, incomplete_trees, qts = process_tweets(
#     supabase, archive, username, include_date=False
# )

account_id = archive["account"][0]["account"]["accountId"]
patched_tweets = patch_tweets_with_note_tweets(
    archive.get("note-tweet", []), archive["tweets"]
)
tweets_df = create_tweets_df(patched_tweets, username, account_id)

# %%
# Get quoted tweets
quote_map = get_quoted_tweet_ids(supabase, tweets_df.tweet_id.to_list())
quoted_tweet_ids = list(quote_map.values())
quoted_tweets, liked_quoted_tweets = get_all_quoted_tweets(supabase, quoted_tweet_ids)
qts = {
    "quoted_tweets": quoted_tweets,
    "liked_quoted_tweets": liked_quoted_tweets,
    "quote_map": quote_map,
}

# add quoted_tweets to tweets_df, id as a column, and the quoted-tweets as rows
tweets_df["quoted_tweet_id"] = tweets_df["tweet_id"].map(quote_map)
quoted_tweets_df = pd.DataFrame(quoted_tweets.values())
quoted_tweets_df["quoted_tweet_id"] = quoted_tweets_df["tweet_id"].map(quote_map)
tweets_df = pd.concat([tweets_df, quoted_tweets_df], ignore_index=True)

# Add quoted text to embedding text
tweets_df["emb_text"] = tweets_df["full_text"].apply(
    lambda x: f"[current] {clean_tweet_text(x)}"
)

# Add quoted tweet text if available
for tweet_id, quoted_id in quote_map.items():
    if quoted_id in quoted_tweets:
        quoted_text = clean_tweet_text(quoted_tweets[quoted_id]["full_text"])
    elif quoted_id in liked_quoted_tweets:
        quoted_text = clean_tweet_text(liked_quoted_tweets[quoted_id]["full_text"])
    else:
        quoted_text = ""
    idx = tweets_df.index[tweets_df["tweet_id"] == tweet_id][0]
    tweets_df.at[idx, "emb_text"] = (
        f"[quoted] {quoted_text}\n{tweets_df.at[idx, 'emb_text']}"
    )

# Get conversation data
reply_ids = tweets_df[tweets_df.reply_to_tweet_id.notna()].tweet_id.to_list()
conv_map = get_all_conversation_ids(supabase, reply_ids)
tweets_df["conversation_id"] = tweets_df["reply_to_tweet_id"].map(conv_map)

conversation_tweets = get_all_conversation_tweets(supabase, conv_map)
trees = build_conversation_trees(conversation_tweets)

n_replies_w_conv_id = tweets_df["conversation_id"].notna().sum()
n_replies = tweets_df["reply_to_tweet_id"].notna().sum()
print(f"Found {n_replies_w_conv_id}/{n_replies} replies with conversation IDs")

replies_w_no_conv_id = tweets_df[
    tweets_df["reply_to_tweet_id"].notna() & tweets_df["conversation_id"].isna()
]

found_tweets, found_liked = get_incomplete_reply_chains(
    replies_w_no_conv_id, supabase, batch_size=100
)

found_and_old = {**found_tweets, **df_to_tweet_dict(replies_w_no_conv_id)}


# %%
# %%
import seaborn as sns


def plot_parameter_coverage(results_df: pd.DataFrame, value_col: str = "n_clusters"):
    """
    Plot a heatmap showing parameter combinations colored by value_col.
    Range is determined by min and max values in the data.
    For score, clips extreme values to focus on range around 0.
    """
    # Get ranges from data
    min_cluster_size_range = (
        int(results_df["min_cluster_size"].min()),
        int(results_df["min_cluster_size"].max()),
    )
    min_samples_range = (
        int(results_df["min_samples"].min()),
        int(results_df["min_samples"].max()),
    )

    # Create full parameter grid
    min_cluster_sizes = range(min_cluster_size_range[0], min_cluster_size_range[1] + 1)
    min_samples = range(min_samples_range[0], min_samples_range[1] + 1)

    # Initialize value matrix with NaN
    values = np.full((len(min_cluster_sizes), len(min_samples)), np.nan)

    # Fill in values where we have data
    for _, row in results_df.iterrows():
        i = int(row["min_cluster_size"]) - min_cluster_size_range[0]
        j = int(row["min_samples"]) - min_samples_range[0]
        values[i, j] = row[value_col]

    vmin, vmax = None, None

    # Create heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        values,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=list(min_samples),
        yticklabels=list(min_cluster_sizes),
        vmin=vmin,
        vmax=vmax,
        annot_kws={"color": "black"},
    )
    plt.title(f"Parameter Search Results - {value_col}")
    plt.xlabel("min_samples")
    plt.ylabel("min_cluster_size")
    plt.tight_layout()
    plt.show()

    # Print coverage statistics
    total_combinations = values.size
    attempted_combinations = np.sum(~np.isnan(values))
    print(f"\nCoverage statistics:")
    print(f"Range min_cluster_size: {min_cluster_size_range}")
    print(f"Range min_samples: {min_samples_range}")
    print(f"Total possible combinations: {total_combinations}")
    print(f"Attempted combinations: {attempted_combinations}")
    print(
        f"Coverage percentage: {(attempted_combinations/total_combinations)*100:.1f}%"
    )


# %%
# load results from /Users/frsc/Documents/Projects/community-archive-personal/src/clustering_modal_test_data
usernames = [
    "mechanical_monk",
    "defenderofbasic",
    "iaimforgoat",
    "nosilverv",
    "bierlingm",
    "johnsonmxe",
]
username = "exgenesis"
username = "bierlingm"
result = json.load(
    open(
        f"/Users/frsc/Documents/Projects/community-archive-personal/src/clustering_modal_test_data/{username}_clustering_params.json"
    )
)
results_df = pd.DataFrame(result["results_df"])
display(results_df)
#
plot_parameter_coverage(results_df, value_col="n_clusters")
plot_parameter_coverage(results_df, value_col="persistence")
plot_parameter_coverage(results_df, value_col="noise_ratio")
plot_parameter_coverage(results_df, value_col="score")
print(f"params min_cluster_size: {result['min_cluster_size']}")
print(f"params min_samples: {result['min_samples']}")
print(f"params n_clusters: {result['n_clusters']}")
print(f"params noise_ratio: {result['noise_ratio']}")
print(f"params persistence: {result['persistence']}")
print(f"params max_cluster_size: {result['max_cluster_size']}")
print(f"params score: {result['score']}")
# %%
