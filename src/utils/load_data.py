# %%
from pathlib import Path
import pandas as pd
import pickle
import modal
import io
import json
import numpy as np
from modal import Function
from typing import Optional, Dict, Any

ANALYSIS_PHASES = [
    "process",
    "embed",
    "cluster",
    "label",
]


# Function to load user data using Modal
def load_user_data(
    username,
    force_recompute="none",
    required_files=[
        "labeled_cluster_hierarchy.parquet",
        "clustered_tweets_df.parquet",
        "incomplete_trees.pkl",
        "trees.pkl",
        "cluster_ontology_items.json",
        "local_tweet_id_maps.json",
        "group_results.json",
        "qts.pkl",
    ],
):
    """Load user data from local files first, then Modal volume if needed."""
    print(f"Loading data for {username} with force_recompute={force_recompute}")
    if force_recompute not in ANALYSIS_PHASES:
        force_recompute = "none"

    # Check local data directory
    if force_recompute == "none":
        print(f"Loading cached analysis from local files for {username}")
        data_dir = Path("./data") / username

        required_files = {f: data_dir / f for f in required_files}
        print(f"Required files: {required_files.keys()}")
        # Try loading from local files first
        if all(path.exists() for path in required_files.values()):
            print(
                f"Loading cached analysis from local files for {username} with force_recompute={force_recompute}"
            )
            try:
                data = {}
                # Load parquet files
                for volume_name, local_path in required_files.items():
                    if volume_name.endswith(".parquet"):
                        df = pd.read_parquet(local_path)
                        data[volume_name] = df
                    elif volume_name.endswith(".pkl"):
                        with open(local_path, "rb") as f:
                            data[volume_name] = pickle.load(f)
                    elif volume_name.endswith(".json"):
                        with open(local_path, "r") as f:
                            data[volume_name] = json.load(f)
                    elif volume_name.endswith(".npy"):
                        data[volume_name] = np.load(local_path)

                # Convert created_at to datetime
                data["clustered_tweets_df.parquet"]["created_at"] = pd.to_datetime(
                    data["clustered_tweets_df.parquet"]["created_at"]
                )
                print(data.keys())
                return data

            except Exception as e:
                print(f"Error loading local files: {e}")
                print("Falling back to Modal volume...")

        # If local files don't exist or couldn't be loaded, try Modal volume
        print(f"Trying Modal volume for {username}")
        volume = modal.Volume.lookup("twitter-archive-data")
        volume_data = load_from_volume(username, volume)

        # if volume_data is None:
        #     print(f"No volume data found for {username}. Running orchestrator...")
        #     modal.Function.lookup(
        #         app_name="twitter-archive-analysis", tag="orchestrator"
        #     ).remote(username, "all")
        #     volume_data = load_from_volume(username, volume)
        if volume_data is None:
            print(f"No volume data found for {username} in modal storage.")
            return None

        if volume_data is not None:
            # Save volume data locally for next time
            data_dir.mkdir(parents=True, exist_ok=True)
            print(f"Saving volume data to {data_dir}")

            # Save parquet files
            for filename, df in volume_data.items():
                if filename.endswith(".parquet"):
                    local_name = required_files[filename]
                    df.to_parquet(local_name)

            # Save pickle files
            for filename, obj in volume_data.items():
                if filename.endswith(".pkl"):
                    local_name = required_files[filename]
                    with open(local_name, "wb") as f:
                        pickle.dump(obj, f)

            # Save JSON files
            for filename, obj in volume_data.items():
                if filename.endswith(".json"):
                    local_name = required_files[filename]
                    with open(local_name, "w") as f:
                        json.dump(obj, f)

            return volume_data


def load_from_volume(
    username,
    volume,
    required_files=[
        f"labeled_cluster_hierarchy.parquet",
        f"clustered_tweets_df.parquet",
        f"incomplete_trees.pkl",
        f"trees.pkl",
        f"cluster_ontology_items.json",
        f"local_tweet_id_maps.json",
        f"group_results.json",
        f"qts.pkl",
    ],
):
    """Load user data from Modal volume."""
    print(f"Loading data from Modal volume for {username}")

    # Get list of all files in volume
    volume_files = [p.path for p in volume.listdir("/", recursive=True)]

    # Check if all required files exist
    if not all(f"{username}/{f}" in volume_files for f in required_files):
        print(f"Missing required files in volume for {username}")
        missing_files = [
            f for f in required_files if f"{username}/{f}" not in volume_files
        ]
        print(f"Missing files: {missing_files}")
        return None

    try:
        # Load parquet files
        data = {}
        for filename in [fn for fn in required_files if fn.endswith(".parquet")]:
            data_stream = io.BytesIO()
            for chunk in volume.read_file(f"{username}/{filename}"):
                data_stream.write(chunk)
            data_stream.seek(0)
            df = pd.read_parquet(data_stream)
            data[filename] = df

        # Load pickle files
        for filename in [fn for fn in required_files if fn.endswith(".pkl")]:
            data_stream = io.BytesIO()
            for chunk in volume.read_file(f"{username}/{filename}"):
                data_stream.write(chunk)
            data_stream.seek(0)
            obj = pickle.load(data_stream)
            data[filename] = obj

        # Load JSON files
        for filename in [fn for fn in required_files if fn.endswith(".json")]:
            data_stream = io.BytesIO()
            for chunk in volume.read_file(f"{username}/{filename}"):
                data_stream.write(chunk)
            data_stream.seek(0)
            obj = json.loads(data_stream.getvalue().decode("utf-8"))
            data[filename] = obj

        # Load numpy files
        for filename in [fn for fn in required_files if fn.endswith(".npy")]:
            data_stream = io.BytesIO()
            for chunk in volume.read_file(f"{username}/{filename}"):
                data_stream.write(chunk)
            data_stream.seek(0)
            data[filename] = np.load(data_stream)

        # Convert created_at to datetime
        if "clustered_tweets_df.parquet" in data:
            data["clustered_tweets_df.parquet"]["created_at"] = pd.to_datetime(
                data["clustered_tweets_df.parquet"]["created_at"]
            )
        if "tweets_df.parquet" in data:
            data["tweets_df.parquet"]["created_at"] = pd.to_datetime(
                data["tweets_df.parquet"]["created_at"]
            )
        return data

    except Exception as e:
        print(f"Error loading from volume: {e}")
        return None


# # load_user_data("exgenesis")
# # %%
# volume = modal.Volume.lookup("twitter-archive-data")


# # %%
# "exgenesis" in [p.path for p in volume.listdir("/")]


# # %%
# import io

# data_stream = io.BytesIO()
# for chunk in volume.read_file("exgenesis/labeled_cluster_hierarchy.parquet"):
#     data_stream.write(chunk)
# data_stream.seek(0)

# df = pd.read_parquet(data_stream)
# df
# %%
