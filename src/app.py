import streamlit as st
import pandas as pd
import logging
from pathlib import Path
import toolz as tz
from components.ontology_view import render_ontology_items, render_yearly_summaries
from utils.state import init_session_state, update_state
from utils.data import (
    fetch_users,
    fetch_users_with_profiles,
    build_clusters_by_parent,
    get_descendant_clusters,
    compute_all_cluster_stats,
    prepare_cluster_stats,
)
from components.profile import render_profile
from components.filters import render_tweet_filters
from components.tweet_view import (
    render_cluster_stats,
    render_timeline_chart,
    render_engaged_users,
    render_tweet_threads,
)
from utils.load_data import ANALYSIS_PHASES, load_user_data
from typing import Dict, Any, Tuple
import numpy as np
import time
from modal import Function, Volume
from typing import Optional
import hashlib

# Setup
st.set_page_config(layout="wide")
logging.basicConfig(level=logging.INFO)

# Initialize session state
init_session_state()

MODE = "prod"  # or prod
# Load CSS
with open("styles.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Get params from URL
params = st.query_params
url_cluster_id = params.get("cluster_id", [None])[0]
url_username = params.get("username", None)  # Get first element of the list
print(f"url_username: {url_username}")

# Get clicked references from URL if they exist
url_clicked_refs = params.get("clicked_refs", "")
if url_clicked_refs:
    st.session_state["clicked_reference_tweets"] = set(url_clicked_refs.split(","))
elif "clicked_reference_tweets" not in st.session_state:
    st.session_state["clicked_reference_tweets"] = set()

# Initialize session state for selected cluster if not already set
if url_cluster_id and "selected_cluster_id" not in st.session_state:
    st.session_state["selected_cluster_id"] = url_cluster_id


def centered_container(content_callable):
    """Wrapper to create a centered container with 3/5 width"""
    left_spacer, content_col, right_spacer = st.columns([1, 3, 1])
    with content_col:
        return content_callable()


import modal


# Replace the hardcoded usernames list with this function
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_usernames_from_volume():
    """Get list of usernames from twitter-archive-data volume"""
    try:
        volume = modal.Volume.lookup("twitter-archive-data")
        # Get directory listing and extract usernames
        entries = volume.listdir("/")
        print(f"entries: {entries}")
        # Filter for directories and extract names
        usernames = [
            entry.path  # Use path instead of name
            for entry in entries
            if entry.type == 2  # Directory type is 2
            and not entry.path.startswith(".")  # Skip hidden directories
        ]
        return sorted(usernames)
    except Exception as e:
        st.error(f"Error accessing volume: {e}")
        return []


# Replace the hardcoded usernames list with a call to this function
usernames = get_usernames_from_volume()


def on_user_change():
    """Reset relevant query params when user changes"""
    st.query_params.update(
        {
            "username": st.session_state.user_select,
        }
    )
    if "clicked_refs" in st.query_params:
        del st.query_params["clicked_refs"]
    if "cluster_id" in st.query_params:
        del st.query_params["cluster_id"]
    # Reset session state
    st.session_state["selected_cluster_id"] = None
    st.session_state["clicked_reference_tweets"] = set()


@st.cache_data(ttl=36000)
def cached_load_user_data(username, force_recompute="none"):
    """Load and cache user data"""
    lowercase_username = username.lower()
    data = load_user_data(lowercase_username, force_recompute=force_recompute)
    print(data.keys())
    tweets_df = data["clustered_tweets_df.parquet"][
        [
            col
            for col in data["clustered_tweets_df.parquet"].columns
            if col not in ["emb_text"]
        ]
    ]
    if "accountId" in tweets_df.columns:
        tweets_df["account_id"] = tweets_df["accountId"].astype(str)
        tweets_df = tweets_df.drop(columns=["accountId"])

    return (
        tweets_df,
        data["labeled_cluster_hierarchy.parquet"],
        data["trees.pkl"],
        data["incomplete_trees.pkl"],
        data["cluster_ontology_items.json"],
        data["local_tweet_id_maps.json"],
        data["group_results.json"],
        data["qts.pkl"],
    )


# After loading group_results, add helper functions
def find_group_for_cluster(cluster_id: str, groups: list) -> dict:
    """Find the group containing the given cluster ID"""
    for group in groups:
        if any(member["id"] == cluster_id for member in group["members"]):
            return group
    return None


def get_related_clusters(cluster_id: str, group: dict) -> list:
    """Get other clusters in the same group"""
    if not group:
        return []
    return [member for member in group["members"] if member["id"] != cluster_id]


def start_user_analysis(username: str) -> Optional[str]:
    """Start the analysis process and return the function call ID"""
    try:
        f = Function.lookup(app_name="twitter-archive-analysis", tag="orchestrator")
        # Check for existing running jobs first
        running_jobs = f.get_stats().function_calls
        for job in running_jobs:
            if job.args and job.args[0] == username and job.status == "running":
                return job.object_id  # Return existing job ID

        function_call = f.spawn(username, "all")
        return function_call.object_id
    except Exception as e:
        print(f"Error starting analysis: {e}")
        return None


def check_analysis_status(call_id: str) -> Dict[str, Any]:
    """Check the status of an ongoing analysis"""
    try:
        function_call = Function.from_id(call_id)
        stats = function_call.get_current_stats()
        return {
            "status": "running" if stats.backlog > 0 else "completed",
            "backlog": stats.backlog,
            "runners": stats.num_total_runners,
        }
    except Exception as e:
        print(f"Error checking status: {e}")
        return {"status": "error", "error": str(e)}


def render_loading_state(username: str, analysis_id: Optional[str] = None):
    """Render loading state with profile if available"""
    with centered_container(lambda: st.container()):
        st.title("🐦 Birdseye")
        st.markdown("A Community Archive tool for exploring your tweet history")
        st.button(
            "← Back to User Selection",
            on_click=back_to_user_selection,
            help="Select a different user",
        )

    # Try to get basic profile info even while loading
    account_ids = []  # We'll need to implement a way to get the user's account ID
    user_profiles = fetch_users_with_profiles(account_ids)
    user_profile = next(
        (
            v
            for v in user_profiles.values()
            if v["username"].lower() == username.lower()
        ),
        None,
    )

    if user_profile:
        centered_container(lambda: render_profile(user_profile, "..."))

    with centered_container(lambda: st.container()):
        if analysis_id:
            status = check_analysis_status(analysis_id)
            if status["status"] == "running":
                st.warning(
                    f"""🔄 Computing {username}'s Birdseye view... 
                    This may take several minutes. Feel free to leave and come back - 
                    your analysis will continue in the background.
                    
                    Current status: {status["backlog"]} tasks remaining"""
                )
            elif status["status"] == "error":
                st.error(f"Error during analysis: {status.get('error')}")
        else:
            st.info(
                f"""⏳ Starting analysis for @{username}... 
                This process typically takes 20-40 minutes for a full archive.
                You can leave this page and come back later - your analysis will
                continue in the background."""
            )


# Add user selection screen
def render_user_selection():
    """Render the user selection screen"""
    with centered_container(lambda: st.container()):
        st.title("🐦 Birdseye")
        st.markdown("A Community Archive tool for exploring your tweet history")

        st.subheader("Select a User")
        selected = st.selectbox(
            "Choose a user to explore",
            options=sorted(usernames),
            format_func=lambda x: f"@{x}",
            key="user_select",
            on_change=on_user_change,
        )

        if st.button("Explore", use_container_width=True):
            st.query_params["username"] = selected
            st.rerun()


if not url_username:
    render_user_selection()
    st.stop()

username = url_username
st.session_state["selected_user"] = username

st.session_state["selected_clusters"] = []

# Check if analysis is already running
analysis_id = st.session_state.get("analysis_id")

try:
    data = load_user_data(username, force_recompute="none")
    if data is None:
        if not analysis_id:
            # Start new analysis
            # analysis_id = start_user_analysis(username)
            if analysis_id:
                st.session_state["analysis_id"] = analysis_id

        # Show loading state
        render_loading_state(username, analysis_id)

        # Rerun to check status
        time.sleep(30)  # Wait a bit before rerunning
        st.rerun()
    else:
        # Clear analysis ID if we have data
        if "analysis_id" in st.session_state:
            del st.session_state["analysis_id"]

        # Continue with existing data processing...
        tweets_df = data["clustered_tweets_df.parquet"]
        hierarchy_df = data["labeled_cluster_hierarchy.parquet"]
        trees = data["trees.pkl"]
        incomplete_trees = data["incomplete_trees.pkl"]
        ontology_items = data["cluster_ontology_items.json"]
        local_tweet_id_maps = data["local_tweet_id_maps.json"]
        group_results = data["group_results.json"]
        qts = data["qts.pkl"]

        # Preprocess data
        tweets_df["favorite_count"] = tweets_df["favorite_count"].fillna(0).astype(int)
        tweets_df.drop_duplicates(subset=["tweet_id"], inplace=True)

        hierarchy_df = hierarchy_df[hierarchy_df["name"].notna()].drop_duplicates(
            subset=["cluster_id"]
        )

        # Build cluster hierarchy
        clusters_by_parent = build_clusters_by_parent(hierarchy_df)

        # Pre-compute cluster stats
        if tweets_df.shape[0] > 0:
            cluster_stats = compute_all_cluster_stats(
                tweets_df, hierarchy_df, clusters_by_parent
            )

        tree_tweets = []
        for k, v in trees.items():
            tree_tweets.extend(
                [
                    {k: p for k, p in t.items() if k not in ["fts"]}
                    for t in v["tweets"].values()
                ]
            )

        tree_tweets_df = pd.DataFrame(tree_tweets)
        # st.write(tree_tweets_df)
        # st.write(tweets_df)

        account_ids = (
            pd.concat(
                [
                    tree_tweets_df["account_id"].astype(str),
                ]
            )
            .unique()
            .tolist()
        )
        # st.write(cluster_thread_tweets)
        # st.dataframe(pd.DataFrame(cluster_thread_tweets.values()))

        user_profiles = fetch_users_with_profiles(account_ids)
        user_profiles_df = pd.DataFrame(
            [
                {**up, "avatar_media_url": up["profile"]["avatar_media_url"]}
                for up in user_profiles.values()
                if up["profile"]
            ]
        )
        # patch missing usernames in tweets_df
        tweets_df["username"] = tweets_df["account_id"].map(
            user_profiles_df.set_index("account_id")["username"]
        )
        # st.write(user_profiles_df)
        # Render Profile Section
        user_profile = user_profiles.get(
            next(
                (
                    k
                    for k, v in user_profiles.items()
                    if v["username"].lower()
                    == st.session_state["selected_user"].lower()
                ),
                None,
            )
        )

        # Replace existing cluster stats code with:
        cluster_stats_df = prepare_cluster_stats(hierarchy_df, tweets_df, cluster_stats)

        # Add select column to cluster_stats_df
        cluster_stats_df = cluster_stats_df.copy()
        cluster_stats_df["select"] = False

        # st.write(st.session_state)
        # Create placeholder for main content
        # st.write(user_profiles_df)
        # st.write(tweets_df.drop_duplicates(subset=["tweet_id"]))
        main_content = st.empty()
        with main_content.container():
            with centered_container(lambda: st.container()):
                st.title("🐦 Birdseye")
                st.markdown("A Community Archive tool for exploring your tweet history")
                st.button(
                    "← Back to User Selection",
                    on_click=back_to_user_selection,
                    help="Select a different user",
                )

            if user_profile:

                centered_container(
                    lambda: render_profile(
                        user_profile, len(hierarchy_df[hierarchy_df["level"] == 0])
                    )
                )

                with centered_container(lambda: st.expander("Advanced Options")):
                    filter_low_quality = st.checkbox(
                        "Hide low quality clusters",
                        value=True,
                        help="Filter out clusters marked as low quality by the AI",
                    )

                # Update column_config based on filter setting
                column_config = {
                    "cluster_id": st.column_config.TextColumn("Cluster ID"),
                    "num_tweets": st.column_config.NumberColumn("Number of Tweets"),
                    "median_likes": st.column_config.NumberColumn(
                        "Median Likes", format="%.2f"
                    ),
                    "total_likes": st.column_config.NumberColumn("Total Likes"),
                    "median_date": st.column_config.DateColumn("Median Date"),
                    "tweets_per_month": st.column_config.LineChartColumn(
                        "Tweets per Month"
                    ),
                }

                # Add low_quality_cluster column if filter is disabled
                if not filter_low_quality:
                    column_config["low_quality_cluster"] = st.column_config.TextColumn(
                        "Low Quality",
                        help="1 indicates cluster was marked as low quality by AI",
                    )

                # Filter hierarchy_df if checkbox is checked
                display_df = cluster_stats_df.sort_values(
                    by="median_date", ascending=False
                )
                if filter_low_quality:
                    display_df = display_df[display_df["low_quality_cluster"] != "1"]

                column_order = [
                    "name",
                    "num_tweets",
                    "median_likes",
                    "total_likes",
                    "median_date",
                ]
                column_order.append("tweets_per_month")
                if not filter_low_quality:
                    column_order.append("low_quality_cluster")

                # Create columns for layout with center alignment
                with centered_container(lambda: st.container()):
                    st.info(
                        "⚡ To select a cluster, click the box to the left of the 'name' column."
                    )
                    event = st.dataframe(
                        display_df,  # Use filtered dataframe
                        column_config=column_config,
                        hide_index=True,
                        use_container_width=True,  # Still true but only for the column width
                        column_order=column_order,
                        key="cluster_id",
                        selection_mode="single-row",
                        on_select="rerun",
                    )
                    if len(event.selection["rows"]) > 0:
                        if (
                            st.session_state["selected_cluster_id"]
                            != display_df.iloc[event.selection["rows"][0]]["cluster_id"]
                        ):
                            print(
                                f"selected row selected_cluster_id: {st.session_state['selected_cluster_id']}"
                            )
                            st.session_state["selected_cluster_id"] = display_df.iloc[
                                event.selection["rows"][0]
                            ]["cluster_id"]
                            # Clear clicked references when cluster changes
                            st.session_state["clicked_reference_tweets"] = set()
                            # Update query params - remove clicked_refs and set new cluster_id
                            st.query_params["cluster_id"] = st.session_state[
                                "selected_cluster_id"
                            ]
                            if "clicked_refs" in st.query_params:
                                del st.query_params["clicked_refs"]
                            st.rerun()
                    else:
                        print(
                            f"empty row selection selected_cluster_id: {st.session_state['selected_cluster_id']}"
                        )
                        st.session_state["clicked_reference_tweets"] = set()
                        if "clicked_refs" in st.query_params:
                            del st.query_params["clicked_refs"]
                        # st.rerun()

        if st.session_state["selected_cluster_id"]:
            cluster_row = hierarchy_df[
                hierarchy_df["cluster_id"] == st.session_state["selected_cluster_id"]
            ].iloc[0]
            st.session_state["cluster_ids"] = [
                st.session_state["selected_cluster_id"]
            ] + get_descendant_clusters(
                st.session_state["selected_cluster_id"], clusters_by_parent
            )
            cluster_tweets = tweets_df[
                tweets_df["cluster"].isin(st.session_state["cluster_ids"])
            ]
            # Create columns for side-by-side layout
            with centered_container(lambda: st.container()):
                stats_col, engaged_col, related_col = st.columns([2, 1, 1])

                with stats_col:
                    render_cluster_stats(cluster_row, cluster_tweets)

                with engaged_col:
                    render_engaged_users(
                        st.session_state["selected_user"], cluster_tweets
                    )

                with related_col:
                    # Find related clusters
                    current_group = find_group_for_cluster(
                        st.session_state["selected_cluster_id"], group_results["groups"]
                    )
                    if current_group:
                        st.subheader(f"Related Clusters")
                        related = get_related_clusters(
                            st.session_state["selected_cluster_id"], current_group
                        )
                        for cluster in related:
                            if st.button(
                                hierarchy_df[
                                    hierarchy_df["cluster_id"] == cluster["id"]
                                ].iloc[0]["name"],
                                key=f"related_cluster_{cluster['id']}",
                                use_container_width=True,
                            ):
                                # Just update session state and let the URL handler at the bottom handle the query params
                                st.session_state["selected_cluster_id"] = cluster["id"]
                                st.session_state["clicked_reference_tweets"] = set()
                                st.rerun()

        if st.session_state["selected_cluster_id"]:

            yearly_summaries = ontology_items[st.session_state["selected_cluster_id"]][
                "ontology_items"
            ].get("yearly_summaries", [])
            with centered_container(lambda: st.container()):
                if yearly_summaries:
                    render_yearly_summaries(yearly_summaries)
                render_timeline_chart(cluster_tweets, tweets_df)
            if st.session_state["selected_cluster_id"]:
                render_ontology_items(
                    ontology_items[st.session_state["selected_cluster_id"]][
                        "ontology_items"
                    ],
                    local_tweet_id_maps[st.session_state["selected_cluster_id"]],
                )
            st.markdown("## Threads and Tweets:")
            render_tweet_filters()
            st.info(
                "↔️ Scroll horizontally to see more tweet threads. Only the longest thread starting at each root is displayed."
            )

            render_tweet_threads(
                tweets_df,
                st.session_state["cluster_ids"],
                user_profiles,
                trees,
                incomplete_trees,
                username=st.session_state["selected_user"],
                user_profiles_df=user_profiles_df,
                qts=qts,
            )

except Exception as e:
    st.error(f"Error loading data: {e}")
    render_loading_state(username)

# Move this URL handling section to the very end of the file
if st.session_state.get("clicked_reference_tweets"):
    clicked_refs_str = ",".join(sorted(st.session_state["clicked_reference_tweets"]))
    new_params = {
        "cluster_id": st.session_state.get("selected_cluster_id"),
        "clicked_refs": clicked_refs_str,
    }
else:
    new_params = {
        "cluster_id": st.session_state.get("selected_cluster_id"),
    }
# Only update if params actually changed
current_params = dict(st.query_params)
if current_params != new_params:
    st.query_params.update(new_params)


def back_to_user_selection():
    st.session_state["selected_user"] = None
    st.query_params["username"] = ""
    if "clicked_refs" in st.query_params:
        del st.query_params["clicked_refs"]
    if "cluster_id" in st.query_params:
        del st.query_params["cluster_id"]
    st.rerun()
