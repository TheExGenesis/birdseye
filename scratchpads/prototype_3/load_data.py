# %%
from pathlib import Path
import pandas as pd
import pickle
import modal
import io


# Function to load user data using Modal
def load_user_data(username):
    """Load user data from local files first, then Modal volume if needed."""
    # Check local data directory
    data_dir = Path("./data") / username
    required_files = {
        "labeled_cluster_hierarchy.parquet": data_dir
        / "labeled_cluster_hierarchy.parquet",
        "convo_tweets_tweets_df.parquet": data_dir / "clustered_tweets_df.parquet",
        "convo_tweets_incomplete_trees.pkl": data_dir / "incomplete_trees.pkl",
        "convo_tweets_trees.pkl": data_dir / "trees.pkl",
    }

    # Try loading from local files first
    if all(path.exists() for path in required_files.values()):
        print(f"Loading cached analysis from local files for {username}")
        try:
            data = {}
            # Load parquet files
            for volume_name, local_path in required_files.items():
                if volume_name.endswith(".parquet"):
                    df = pd.read_parquet(local_path)
                    data[volume_name] = df
                else:
                    with open(local_path, "rb") as f:
                        data[volume_name] = pickle.load(f)

            # Convert created_at to datetime
            data["convo_tweets_tweets_df.parquet"]["created_at"] = pd.to_datetime(
                data["convo_tweets_tweets_df.parquet"]["created_at"]
            )

            return data

        except Exception as e:
            print(f"Error loading local files: {e}")
            print("Falling back to Modal volume...")

    # If local files don't exist or couldn't be loaded, try Modal volume
    print(f"Trying Modal volume for {username}")
    volume = modal.Volume.lookup("twitter-archive-data")
    volume_data = load_from_volume(username, volume)

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

        return volume_data

    # If both local and volume fail, call Modal function
    print(f"Fetching analysis via Modal for {username} (could be a while)")
    result = modal.Function.lookup(
        "twitter-archive-analysis", "get_or_create_analysis"
    ).remote(username)

    if result["status"] == "cached":
        print(f"Using cached Modal analysis for {username}")
    else:
        print(f"Generated new Modal analysis for {username}")

    # Try loading from volume after function returns
    volume = modal.Volume.lookup("twitter-archive-data")
    volume_data = load_from_volume(username, volume)

    if volume_data is None:
        raise RuntimeError(f"Failed to load data from volume after analysis completion")

    # Save files locally
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving analysis results to {data_dir}")

    for filename, content in volume_data.items():
        local_name = required_files[filename]
        if filename.endswith(".parquet"):
            content.to_parquet(local_name)
        else:
            with open(local_name, "wb") as f:
                pickle.dump(content, f)

    return volume_data


def load_from_volume(username, volume):
    """Load user data from Modal volume."""
    print(f"Loading data from Modal volume for {username}")

    required_files = [
        f"labeled_cluster_hierarchy.parquet",
        f"convo_tweets_tweets_df.parquet",
        f"convo_tweets_incomplete_trees.pkl",
        f"convo_tweets_trees.pkl",
    ]

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

        # Convert created_at to datetime
        data[f"convo_tweets_tweets_df.parquet"]["created_at"] = pd.to_datetime(
            data[f"convo_tweets_tweets_df.parquet"]["created_at"]
        )

        return data

    except Exception as e:
        print(f"Error loading from volume: {e}")
        return None


# load_user_data("exgenesis")
# %%
volume = modal.Volume.lookup("twitter-archive-data")


# %%
"exgenesis" in [p.path for p in volume.listdir("/")]


# %%
import io

data_stream = io.BytesIO()
for chunk in volume.read_file("exgenesis/labeled_cluster_hierarchy.parquet"):
    data_stream.write(chunk)
data_stream.seek(0)

df = pd.read_parquet(data_stream)
df
# %%
