from pathlib import Path
import pyarrow as pa


def get_user_paths(username: str) -> dict:
    """Get all paths for a given username"""
    data_dir = Path("/twitter-archive-data")
    user_dir = data_dir / username

    return {
        "data_dir": data_dir,
        "user_dir": user_dir,
        "archive": user_dir / "archive.json",
        "tweets_df": user_dir / "tweets_df.parquet",
        "embeddings": user_dir / "embeddings.npy",
        "reduced_embeddings": user_dir / "reduced_embeddings.npy",
        "clusters": user_dir / "clusters.json",
        "hierarchy": user_dir / "cluster_hierarchy.parquet",
        "soft_clusters": user_dir / "soft_clusters.npy",
        "cluster_labels_arr": user_dir / "cluster_labels.npy",
        "cluster_probs": user_dir / "cluster_probs.npy",
        "trees": user_dir / "trees.pkl",
        "incomplete_trees": user_dir / "incomplete_trees.pkl",
        "clustered_tweets_df": user_dir / "clustered_tweets_df.parquet",
        "cluster_ontology_items": user_dir / "cluster_ontology_items.json",
        "labeled_hierarchy": user_dir / "labeled_cluster_hierarchy.parquet",
    }


TWEETS_DF_SCHEMA = pa.schema(
    [
        ("tweet_id", pa.string()),
        ("username", pa.string()),
        ("accountId", pa.string()),
        (
            "created_at",
            pa.string(),
        ),  # Consider using pa.timestamp() if you want to store as timestamp
        ("full_text", pa.string()),
        ("retweet_count", pa.int64()),
        ("favorite_count", pa.int64()),
        ("reply_to_tweet_id", pa.string()),
        ("reply_to_user_id", pa.string()),
        ("reply_to_username", pa.string()),
        ("archive_upload_id", pa.int64()),
        ("conversation_id", pa.string()),
        ("emb_text", pa.string()),
        # ("cluster", pa.string()),
        # ("cluster_prob", pa.float64()),
        # ("most_prob_cluster", pa.string()),
        # ("most_prob_cluster_prob", pa.float64()),
    ]
)

CLUSTERED_TWEETS_DF_SCHEMA = pa.schema(
    [
        ("tweet_id", pa.string()),
        ("username", pa.string()),
        ("accountId", pa.string()),
        ("created_at", pa.string()),
        ("full_text", pa.string()),
        ("retweet_count", pa.int64()),
        ("favorite_count", pa.int64()),
        ("reply_to_tweet_id", pa.string()),
        ("reply_to_user_id", pa.string()),
        ("reply_to_username", pa.string()),
        ("archive_upload_id", pa.int64()),
        ("conversation_id", pa.string()),
        ("cluster", pa.string()),
        ("cluster_prob", pa.float64()),
        ("emb_text", pa.string()),
        # ("most_prob_cluster", pa.string()),
        # ("most_prob_cluster_prob", pa.float64()),
    ]
)

SUPABASE_URL = "https://fabxmporizzqflnftavs.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
