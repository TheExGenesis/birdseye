import streamlit as st
import pandas as pd
from supabase import create_client, Client
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional

# Supabase config
url: str = "https://fabxmporizzqflnftavs.supabase.co"
key: str = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
)


@st.cache_data(ttl=36000)
def fetch_users() -> List[Dict[str, Any]]:
    """Fetch all users from database"""
    logging.info("Executing fetch_users")
    supabase: Client = create_client(url, key)
    result = supabase.table("account").select("account_id", "username").execute()
    return result.data


@st.cache_data(ttl=36000)
def fetch_users_with_profiles(account_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """Fetch user profiles for given account IDs"""
    logging.info(f"Fetching profiles for {len(account_ids)} users")
    supabase: Client = create_client(url, key)

    # Fetch account data
    account_result = (
        supabase.table("account")
        .select(
            """
        account_id, username, account_display_name,
        num_tweets, num_following, num_followers, num_likes
    """
        )
        .in_("account_id", account_ids)
        .execute()
    )

    # Fetch profile data
    profile_result = (
        supabase.table("account")
        .select(
            "account_id, username, profile!inner(bio, location, website,avatar_media_url, archive_upload_id)"
        )
        .in_("account_id", account_ids)
        .execute()
    )

    # Process and merge results
    single_profile_users = {
        user["account_id"]: {
            **user,
            "profile": (
                max(user["profile"], key=lambda x: x["archive_upload_id"])
                if isinstance(user["profile"], list)
                else user["profile"]
            ),
        }
        for user in profile_result.data
        if user["account_id"] in account_ids
    }

    return {
        account["account_id"]: {
            **account,
            "profile": single_profile_users.get(account["account_id"], {}).get(
                "profile"
            ),
        }
        for account in account_result.data
    }


@st.cache_data(ttl=36000)
def fetch_tweets_with_images(tweet_ids: List[str]) -> Dict[str, List[str]]:
    """Fetch photo media_urls for given tweet IDs in batches of 500

    Args:
        tweet_ids: List of tweet IDs to fetch images for

    Returns:
        Dict mapping tweet_id to list of photo media_urls
    """
    supabase: Client = create_client(url, key)
    tweets_to_images = {}

    # Process in batches of 500
    for i in range(0, len(tweet_ids), 500):
        batch = tweet_ids[i : i + 500]

        result = (
            supabase.table("tweet_media")
            .select("tweet_id, media_url")
            .in_("tweet_id", batch)
            .eq("media_type", "photo")
            .execute()
        )

        # Group media_urls by tweet_id
        for row in result.data:
            tweet_id = row["tweet_id"]
            if tweet_id not in tweets_to_images:
                tweets_to_images[tweet_id] = []
            tweets_to_images[tweet_id].append(row["media_url"])

    return tweets_to_images


def get_descendant_clusters(
    cluster_id: str, clusters_by_parent: Dict[str, List[Dict[str, Any]]]
) -> List[str]:
    """Get all descendant cluster IDs for a given cluster"""
    descendants: List[str] = []
    children = clusters_by_parent.get(str(cluster_id), [])
    for child in children:
        child_id = str(child["cluster_id"])
        descendants.append(child_id)
        descendants.extend(get_descendant_clusters(child_id, clusters_by_parent))
    return descendants


@st.cache_data(ttl=36000)
def build_clusters_by_parent(
    hierarchy_df: pd.DataFrame,
) -> Dict[str, List[Dict[str, Any]]]:
    """Build dictionary of clusters organized by parent"""
    clusters_by_parent: Dict[str, List[Dict[str, Any]]] = {}
    for _, row in hierarchy_df.iterrows():
        parent = str(row["parent"]) if pd.notnull(row["parent"]) else None
        # if parent doesn't have a name, make parent -1
        parent_row = hierarchy_df[hierarchy_df["cluster_id"] == parent]
        if len(parent_row) == 0 or pd.isna(parent_row["name"].iloc[0]):
            parent = "-1"

        clusters_by_parent.setdefault(parent, []).append(row.to_dict())
    return clusters_by_parent


@st.cache_data(ttl=36000)
def get_cluster_tweets(
    tweets_df: pd.DataFrame,
    cluster_id: str,
    clusters_by_parent: Dict[str, List[Dict[str, Any]]],
) -> pd.DataFrame:
    """Get tweets for a cluster and its descendants"""
    cluster_ids = [cluster_id] + get_descendant_clusters(cluster_id, clusters_by_parent)
    return tweets_df[tweets_df["cluster"].isin(cluster_ids)]


@st.cache_data(ttl=36000)
def compute_all_cluster_stats(
    tweets_df: pd.DataFrame,
    hierarchy_df: pd.DataFrame,
    clusters_by_parent: Dict[str, List[Dict[str, Any]]],
) -> Dict[str, Dict[str, Any]]:
    """Pre-compute stats for all clusters more efficiently"""
    # First, build a mapping of all descendants for each cluster
    descendant_map: Dict[str, List[str]] = {}
    for _, row in hierarchy_df.iterrows():
        cluster_id = row["cluster_id"]
        if cluster_id not in descendant_map:
            descendant_map[cluster_id] = get_descendant_clusters(
                cluster_id, clusters_by_parent
            )

    # Create cluster_id to tweet mapping once
    cluster_tweet_groups = tweets_df.groupby("cluster")

    stats: Dict[str, Dict[str, Any]] = {}
    for _, row in hierarchy_df.iterrows():
        cluster_id = row["cluster_id"]
        all_cluster_ids = [cluster_id] + descendant_map[cluster_id]

        # Get tweets for all relevant clusters at once
        cluster_tweets = pd.concat(
            [
                cluster_tweet_groups.get_group(cid)
                for cid in all_cluster_ids
                if cid in cluster_tweet_groups.groups
            ]
        )

        stats[cluster_id] = {
            "cluster_row": row,
            "cluster_id": cluster_id,
            "cluster_name": row["name"],
            "num_tweets": len(cluster_tweets),
            "total_likes": cluster_tweets["favorite_count"].sum(),
            "median_likes": cluster_tweets["favorite_count"].median(),
            "median_date": cluster_tweets["created_at"].median(),
        }
    return stats


@st.cache_data(ttl=36000)
def prepare_cluster_stats(
    hierarchy_df: pd.DataFrame, tweets_df: pd.DataFrame, cluster_stats: Dict[str, Any]
) -> pd.DataFrame:
    """Build cluster stats dataframe with timeline data"""
    # Convert cluster_id to string and merge stats
    hierarchy_df = hierarchy_df.copy()
    hierarchy_df["cluster_id"] = hierarchy_df["cluster_id"].astype(str)

    # Create stats dataframe with medians instead of averages
    stats_records = [
        {
            "cluster_id": str(k),
            "num_tweets": v["num_tweets"],
            "total_likes": v["total_likes"],
            "median_likes": v["median_likes"],  # New field
            "median_date": v["median_date"],  # New field
        }
        for k, v in cluster_stats.items()
    ]

    cluster_stats_df = hierarchy_df.merge(pd.DataFrame(stats_records), on="cluster_id")

    # Calculate tweets per month timeline
    tweets_by_month = tweets_df.copy()[
        tweets_df.username.str.lower() == st.session_state["selected_user"].lower()
    ]
    tweets_by_month["month"] = (
        tweets_by_month["created_at"].dt.to_period("M").dt.to_timestamp()
    )

    # Get counts per cluster/month
    monthly_counts = (
        tweets_by_month.groupby(["cluster", "month"])
        .size()
        .reset_index(name="tweet_count")
    )

    # Fill in missing months
    all_months = pd.date_range(
        tweets_by_month["month"].min(), tweets_by_month["month"].max(), freq="MS"
    )
    all_combos = pd.DataFrame(
        [(c, m) for c in monthly_counts["cluster"].unique() for m in all_months],
        columns=["cluster", "month"],
    )

    monthly_counts = (
        monthly_counts.merge(all_combos, on=["cluster", "month"], how="right")
        .fillna(0)
        .sort_values("month")
    )

    # Convert to dict of timelines
    timelines = monthly_counts.groupby("cluster")["tweet_count"].apply(list).to_dict()
    cluster_stats_df["tweets_per_month"] = cluster_stats_df["cluster_id"].map(timelines)

    return cluster_stats_df[cluster_stats_df["level"] == 0]


def normalize_tweet(
    tweet_data: dict, tweet_id: str = None, child_tweet: dict = None
) -> dict:
    """Normalize tweet data to consistent format. Uses child tweet data as fallback for missing fields.

    Args:
        tweet_data: Raw tweet data from either trees or dataframe
        tweet_id: Optional tweet_id if not in tweet_data
        child_tweet: Optional child tweet to use for fallback data

    Returns:
        Normalized tweet dict with consistent property names
    """
    normalized = {}

    # ID - might be in data or passed as key
    normalized["tweet_id"] = tweet_data.get("tweet_id", tweet_id)
    if not normalized["tweet_id"] and child_tweet:
        normalized["tweet_id"] = child_tweet.get("reply_to_tweet_id")

    # Account info - could be account_id or accountId
    normalized["account_id"] = tweet_data.get("accountId", tweet_data.get("account_id"))
    if "accountId" not in normalized and child_tweet:
        normalized["account_id"] = child_tweet.get("reply_to_user_id")

    # Username - fallback to reply_to_username from child
    normalized["username"] = tweet_data.get("username", "")
    if not normalized["username"] and child_tweet:
        normalized["username"] = child_tweet.get("reply_to_username", "")

    # Created at - ensure timestamp, fallback to child's created_at
    created_at = tweet_data.get("created_at")
    if created_at is None and child_tweet:
        child_created = pd.to_datetime(child_tweet.get("created_at"))
        created_at = child_created - pd.Timedelta(minutes=1)

    if isinstance(created_at, str):
        normalized["created_at"] = pd.to_datetime(created_at)
    else:
        normalized["created_at"] = created_at

    # Text content
    normalized["full_text"] = tweet_data["full_text"]

    # Engagement metrics - might be strings or ints
    normalized["retweet_count"] = int(tweet_data.get("retweet_count", 0))
    normalized["favorite_count"] = int(tweet_data.get("favorite_count", 0))

    # Reply info
    normalized["reply_to_tweet_id"] = tweet_data.get("reply_to_tweet_id")
    normalized["reply_to_user_id"] = tweet_data.get("reply_to_user_id")

    # Handle None case for reply_to_username
    reply_username = tweet_data.get("reply_to_username")
    normalized["reply_to_username"] = reply_username.lower() if reply_username else ""

    # Other fields
    normalized["archive_upload_id"] = tweet_data.get("archive_upload_id")
    normalized["conversation_id"] = tweet_data.get("conversation_id")
    if not normalized["conversation_id"] and child_tweet:
        normalized["conversation_id"] = child_tweet.get("conversation_id")

    return normalized


import re


def unlink_markdown(text: str) -> str:
    """Convert markdown links to underlined text without links.
    [text](url) becomes <u>text</u>"""
    return re.sub(r"\[([^\]]+)\]\([^)]+\)", r"<u>\1</u>", text)


def get_default_user() -> Dict[str, Any]:
    """Return a default placeholder user dictionary with empty/zero values"""
    return {
        "account_id": "",
        "username": "",
        "account_display_name": "Unknown User",
        "bio": "",
        "location": "",
        "website": "",
        "num_tweets": 0,
        "num_following": 0,
        "num_followers": 0,
        "num_likes": 0,
        "profile": {"avatar_media_url": "", "archive_upload_id": 0},
    }
