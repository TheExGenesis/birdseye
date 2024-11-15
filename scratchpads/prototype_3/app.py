"""
Data examples

hierarchy_df
    cluster_id parent  level                                               name                                            summary  bad_group_flag
0            0     -1      0                                      No name found                                   No summary found            True
1            1     -1      0  Vibes & Life Trajectories: An AI-Powered Manif...  A collection of personal insights, AI-generate...           False
2           10    1-1      0                          Nice Network Interactions  A collection of tweets featuring multiple user...           False
3           11    1-2      0                            PORTAL Project Insights  A collection of reflective tweets exploring th...           False
4           12     -1      0                            Crew Formation Protocol  Exploring the dynamics of building an innovati...           False
..         ...    ...    ...                                                ...                                                ...             ...
102        1-5     -1      1  Emergent Intellectual Ecosystems: Collaborativ...  These interconnected clusters represent experi...           False
103        1-6     -1      1  Consciousness Cartography: Subjective Experien...  These interconnected clusters represent a nuan...           False
104        1-7     -1      1  Intellectual Metamorphosis: Navigating Persona...  These interconnected narratives explore the pr...           False
105        1-8     -1      1  Meta-Cognitive Engineering: Computational Appr...  These clusters represent an emerging interdisc...           False
106        1-9     -1      1  Cognitive Frontiers: Consciousness, Complexity...  These interconnected clusters explore the dyna...           False

[107 rows x 6 columns]
tweets_df
                  tweet_id   username  accountId                created_at  ... conversation_id                                           emb_text  cluster  cluster_prob
0      1846181666466705543  exgenesis  322603863 2024-10-15 13:29:44+00:00  ...             NaN  [root] wow crazy\n\nbtw do you know what the l...       -1      0.000000
1      1846180206291161443  exgenesis  322603863 2024-10-15 13:23:56+00:00  ...             NaN  [root] Didn’t know we were there already. Deet...       75      1.000000
2      1846180044869124178  exgenesis  322603863 2024-10-15 13:23:17+00:00  ...             NaN  [root] Didn’t know we were there already. Deet...       75      1.000000
3      1846180009595052452  exgenesis  322603863 2024-10-15 13:23:09+00:00  ...             NaN              [root] what??\n[context] \n[current]        75      1.000000
4      1846179619411578951  exgenesis  322603863 2024-10-15 13:21:36+00:00  ...             NaN                          [root] what??\n[current]        75      1.000000
...                    ...        ...        ...                       ...  ...             ...                                                ...      ...           ...
18022  1793673981348003895  exgenesis  322603863 2024-05-23 16:02:56+00:00  ...             NaN  [root] Finally, someone who understands me.\n[...       75      1.000000
18023  1793672647722643664  exgenesis  322603863 2024-05-23 15:57:38+00:00  ...             NaN  [root] You are just hands down one of my favor...       -1      0.000000
18024  1793669224671465918  exgenesis  322603863 2024-05-23 15:44:02+00:00  ...    1.793654e+18  [root] We're opening applications for Portal !...       33      0.963206
18025  1793663374473007168  exgenesis  322603863 2024-05-23 15:20:47+00:00  ...             NaN  [current] Coming together around the elephant....       63      1.000000
18026  1793662346952331610  exgenesis  322603863 2024-05-23 15:16:42+00:00  ...             NaN  [root] I'm hoping to go to this, and I'm also ...       -1      0.000000


> ./data/trees.pkl
> trees.keys()
dict_keys(['1649124924877832208', '1779899376799777042', '1781788938245443965', '1781829284501295614', '1782090660109303993', '1782131280634687497', '1782357666997702896', '1782360964731388286',...])

> trees['1649124924877832208'].keys()
dict_keys(['tweets', 'children', 'parents', 'root', 'paths'])

> trees['1649124924877832208']['paths']
{'1782440707652788517': ['1649124924877832208',
  '1649155393145241600',
  '1782438247483707681',
  '1782440707652788517']}

> trees['1649124924877832208']['parents']
{'1649155393145241600': '1649124924877832208',
 '1782438247483707681': '1649155393145241600',
 '1782440707652788517': '1782438247483707681'}
 
> trees['1649124924877832208']['tweets']
{'1649124924877832208': {'tweet_id': '1649124924877832208',
  'account_id': '1163927704049262592',
  'created_at': '2023-04-20T18:56:36+00:00',
  'full_text': "Who doesn't have a blog but you wish they had a blog?",
  'retweet_count': 0,
  'favorite_count': 7,
  'reply_to_tweet_id': None,
  'reply_to_user_id': None,
  'reply_to_username': None,
  'archive_upload_id': 160,
  'fts': "'blog':6,13 'doesn':2 'wish':9",
  'conversation_id': '1649124924877832208'},
 '1649155393145241600': {'tweet_id': '1649155393145241600',...},
 ...}

"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta
import logging
from supabase import create_client, Client

from pathlib import Path
from load_data import load_user_data

st.set_page_config(layout="wide")
# Add this near the top of the file
logging.basicConfig(level=logging.INFO)

url: str = "https://fabxmporizzqflnftavs.supabase.co"
key: str = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImZhYnhtcG9yaXp6cWZsbmZ0YXZzIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MjIyNDQ5MTIsImV4cCI6MjAzNzgyMDkxMn0.UIEJiUNkLsW28tBHmG-RQDW-I5JNlJLt62CSk9D_qG8"
)


def fetch_users():
    logging.info("Executing fetch_users")
    supabase = create_client(url, key)
    result = supabase.table("account").select("account_id", "username").execute()
    return result.data


# Create Modal stub for accessing the remote functions


# Function to load user data using Modal
@st.cache_data(ttl=3600)
def cached_load_user_data(username):
    lowercase_username = username.lower()
    data = load_user_data(lowercase_username)

    # Extract the dataframes and objects from the returned data
    hierarchy_df = data["labeled_cluster_hierarchy.parquet"]
    tweets_df = data["convo_tweets_tweets_df.parquet"]
    trees = data["convo_tweets_trees.pkl"]
    incomplete_trees = data["convo_tweets_incomplete_trees.pkl"]

    return tweets_df, hierarchy_df, trees, incomplete_trees


# Function to get descendant cluster IDs
def get_descendant_clusters(cluster_id):
    descendants = []
    children = clusters_by_parent.get(str(cluster_id), [])
    for child in children:
        child_id = str(child["cluster_id"])
        descendants.append(child_id)
        descendants.extend(get_descendant_clusters(child_id))
    return descendants


# Initialize session state for user selection
if "selected_user" not in st.session_state:
    st.session_state["selected_user"] = None

st.title("Tweet Cluster Explorer")


# @st.cache_data(ttl=3600)
def fetch_users_with_profiles(account_ids: list[str]):
    logging.info(f"Fetching profiles for {len(account_ids)} users")

    supabase = create_client(url, key)
    result = (
        supabase.table("account")
        .select(
            "account_id, username, profile!inner(avatar_media_url, archive_upload_id)"
        )
        .in_("account_id", account_ids)
        .execute()
    )
    single_profile_users = {
        user["account_id"]: {
            **user,
            "profile": max(user["profile"], key=lambda x: x["archive_upload_id"]),
        }
        for user in result.data
        if user["account_id"] in account_ids
    }

    return single_profile_users


# User selection screen
if not st.session_state["selected_user"]:
    st.write("### Select a user to explore their tweets")

    # Fetch available users
    users = fetch_users()
    usernames = [user["username"] for user in users]

    # User selection dropdown
    selected_username = st.selectbox(
        "Select a user", options=usernames, index=None, placeholder="Choose a user..."
    )

    if selected_username and st.button("Explore Tweets"):
        # Initialize session state
        if "selected_clusters" not in st.session_state:
            st.session_state["selected_clusters"] = []

        if "date_filter" not in st.session_state:
            st.session_state["date_filter"] = None

        st.session_state["selected_user"] = selected_username
        st.rerun()
# Main app UI
else:
    # Add a "Change User" button at the top
    if st.button("← Change User", type="secondary"):
        st.session_state["selected_user"] = None
        st.session_state["selected_clusters"] = []
        # Initialize session state

        st.rerun()

    # Load data for selected user
    try:
        tweets_df, hierarchy_df, trees, incomplete_trees = cached_load_user_data(
            st.session_state["selected_user"]
        )
        tweets_df["favorite_count"] = tweets_df["favorite_count"].fillna(0).astype(int)

        st.dataframe(tweets_df)
        st.dataframe(hierarchy_df)
        # Build clusters_by_parent dictionary
        # hierarchy_df["parent"] = hierarchy_df["parent"].map(
        #     lambda x: (
        #         f"1-{str(int(x))}"
        #         if (
        #             pd.notnull(x)
        #             and x > 0
        #             and (isinstance(x, float) or isinstance(x, int))
        #         )
        #         else x
        #     )
        # )
        # print(f"hierarchy_df")
        # print(hierarchy_df)
        # print(f"tweets_df")
        # print(tweets_df)

        print(f"parents {hierarchy_df.parent.unique()}")
        clusters_by_parent = {}
        for index, row in hierarchy_df.iterrows():
            parent = str(row["parent"]) if pd.notnull(row["parent"]) else None
            cluster_id = str(row["cluster_id"])
            clusters_by_parent.setdefault(parent, []).append(row)
        # print(f"clusters_by_parent {list(clusters_by_parent.keys())}")

        col1, col2 = st.columns(2)

        with col1:
            # Sorting options for clusters
            sort_options = {
                "Average Date": "avg_date",
                "Number of Tweets": "num_tweets",
                "Total Likes": "total_likes",
            }
            sort_by = st.selectbox(
                "Sort clusters by",
                options=list(sort_options.keys()),
                key="cluster_sort_by",
                index=0,
            )

            # Display breadcrumbs
            breadcrumb_cols = st.columns(
                [1] * (len(st.session_state["selected_clusters"]) + 1)
            )

            with breadcrumb_cols[0]:
                if st.button(
                    "Home",
                    key="home_breadcrumb",
                    type="secondary",
                    use_container_width=True,
                ):
                    st.session_state["selected_clusters"] = []
                    st.rerun()

            for idx, cluster_id in enumerate(st.session_state["selected_clusters"]):
                with breadcrumb_cols[idx + 1]:
                    cluster_row = hierarchy_df[
                        hierarchy_df["cluster_id"] == cluster_id
                    ].iloc[0]
                    cluster_name = cluster_row["name"]
                    if st.button(
                        cluster_name,
                        key=f"breadcrumb_{idx}",
                        type="secondary",
                        use_container_width=False,
                    ):
                        st.session_state["selected_clusters"] = st.session_state[
                            "selected_clusters"
                        ][: idx + 1]
                        st.rerun()

            # Get clusters at current level
            if st.session_state["selected_clusters"]:
                parent_cluster_id = st.session_state["selected_clusters"][-1]
                current_level_clusters = clusters_by_parent.get(parent_cluster_id, [])
            else:
                # Show level 1 clusters by default
                level_1_clusters = [
                    c for c in clusters_by_parent["-1"] if c["level"] == 1
                ]

                current_level_clusters = level_1_clusters

            # Compute additional stats for clusters
            cluster_stats = []
            for cluster_row in current_level_clusters:
                cluster_id = cluster_row["cluster_id"]
                cluster_name = cluster_row["name"]
                cluster_ids = [cluster_id] + get_descendant_clusters(cluster_id)
                cluster_tweets = tweets_df[tweets_df["cluster"].isin(cluster_ids)]

                num_tweets = len(cluster_tweets)
                total_likes = cluster_tweets["favorite_count"].sum()
                avg_date = cluster_tweets["created_at"].mean()
                cluster_stats.append(
                    {
                        "cluster_row": cluster_row,
                        "cluster_id": cluster_id,
                        "cluster_name": cluster_name,
                        "num_tweets": num_tweets,
                        "total_likes": total_likes,
                        "avg_date": avg_date,
                    }
                )

            # Sort clusters
            sort_key = sort_options[sort_by]
            if sort_key == "avg_date":
                cluster_stats = sorted(
                    cluster_stats, key=lambda x: x[sort_key], reverse=True
                )
            else:
                cluster_stats = sorted(
                    cluster_stats, key=lambda x: x[sort_key], reverse=True
                )

            # Display stats for the selected cluster
            if st.session_state["selected_clusters"]:
                selected_cluster_id = str(st.session_state["selected_clusters"][-1])
                cluster_row = hierarchy_df[
                    hierarchy_df["cluster_id"] == selected_cluster_id
                ].iloc[0]
                print(f"selected row {cluster_row}")
                cluster_name = cluster_row["name"]
                summary = cluster_row["summary"]
                # Get descendant cluster_ids
                cluster_ids = [selected_cluster_id] + get_descendant_clusters(
                    selected_cluster_id
                )
                print(f"cluster ids {cluster_ids}")

                cluster_tweets = tweets_df[
                    tweets_df["cluster"].astype(str).isin(cluster_ids)
                ]

                num_tweets = len(cluster_tweets)
                total_likes = cluster_tweets["favorite_count"].sum()
                avg_date = cluster_tweets["created_at"].mean()

                st.subheader(f"Cluster: {cluster_name}")
                st.write(
                    f"**Number of Tweets:** {num_tweets} | **Total Likes:** {total_likes} | **Average Date:** {avg_date.strftime('%Y-%m-%d') if pd.notnull(avg_date) else 'N/A'}"
                )
                st.write(f"**Summary:** {summary}")

                # Aggregate tweets over time by month
                cluster_tweets.loc[:, "month"] = (
                    cluster_tweets["created_at"].dt.to_period("M").dt.to_timestamp()
                )
                tweets_per_month = (
                    cluster_tweets.groupby("month")
                    .size()
                    .reset_index(name="tweet_count")
                )

                # Plot line chart with selection
                fig = go.Figure()
                fig.add_trace(
                    go.Scatter(
                        x=tweets_per_month["month"],
                        y=tweets_per_month["tweet_count"],
                        mode="lines+markers",
                    )
                )
                fig.update_layout(
                    title="Tweets over Time",
                    xaxis_title="Month",
                    yaxis_title="Tweet Count",
                )

                st.info(
                    "Drag horizontally on the graph to filter tweets in the right column."
                )

                selection = st.plotly_chart(
                    fig,
                    use_container_width=True,
                    key="word_occurrences",
                    selection_mode="box",
                    on_select="rerun",
                )

                # Handle selection
                if (
                    selection
                    and selection["selection"]
                    and selection["selection"]["points"]
                ):
                    selected_dates = [
                        pd.to_datetime(point["x"])
                        for point in selection["selection"]["points"]
                    ]
                    start_date = pd.to_datetime(min(selected_dates)).tz_localize("UTC")
                    end_date = pd.to_datetime(max(selected_dates)).tz_localize("UTC")
                    st.session_state["date_filter"] = (start_date, end_date)
                else:
                    st.session_state["date_filter"] = None

                # Display top engaged usernames
                reply_usernames = cluster_tweets["reply_to_username"].dropna()
                if not reply_usernames.empty:
                    top_reply_users = reply_usernames.value_counts().head(5)
                    st.write(
                        "**Top engaged users:** " + ", ".join(top_reply_users.index)
                    )
                else:
                    st.write("**Top engaged users:** N/A")

            for cluster in cluster_stats:
                cluster_row = cluster["cluster_row"]
                cluster_id = cluster["cluster_id"]
                cluster_name = cluster["cluster_name"]
                num_tweets = cluster["num_tweets"]
                total_likes = cluster["total_likes"]
                avg_date = cluster["avg_date"]
                summary = cluster_row["summary"]
                level = cluster_row["level"]  # Get the level

                with st.container():
                    st.write(f"### {cluster_name}")
                    stats_text = f"Number of Tweets: {num_tweets} | Total Likes: {total_likes} | Average Date: {avg_date.strftime('%Y-%m-%d') if pd.notnull(avg_date) else 'N/A'}"

                    # Add sub-cluster count for level 1 clusters
                    if level == 1:
                        sub_clusters = clusters_by_parent.get(cluster_id, [])
                        num_sub_clusters = len(sub_clusters)
                        stats_text += f" | Sub-clusters: {num_sub_clusters}"

                    st.write(stats_text)
                    if st.button(
                        f"Select '{cluster_name}'", key=f"cluster_{cluster_id}"
                    ):
                        st.session_state["selected_clusters"].append(cluster_id)
                        st.rerun()

            with st.expander("Level 0 Clusters"):
                level_0_clusters = [
                    c for c in clusters_by_parent["-1"] if c["level"] == 0
                ]
                # Compute and display stats for level 0 clusters
                level_0_stats = []
                for cluster_row in level_0_clusters:
                    cluster_id = cluster_row["cluster_id"]
                    cluster_name = cluster_row["name"]
                    cluster_ids = [cluster_id] + get_descendant_clusters(cluster_id)
                    cluster_tweets = tweets_df[tweets_df["cluster"].isin(cluster_ids)]

                    num_tweets = len(cluster_tweets)
                    total_likes = cluster_tweets["favorite_count"].sum()
                    avg_date = cluster_tweets["created_at"].mean()
                    level_0_stats.append(
                        {
                            "cluster_row": cluster_row,
                            "cluster_id": cluster_id,
                            "cluster_name": cluster_name,
                            "num_tweets": num_tweets,
                            "total_likes": total_likes,
                            "avg_date": avg_date,
                        }
                    )

                if not st.session_state["selected_clusters"]:
                    # Sort level 0 clusters
                    sort_key = sort_options[sort_by]
                    if sort_key == "avg_date":
                        level_0_stats = sorted(
                            level_0_stats, key=lambda x: x[sort_key], reverse=True
                        )
                    else:
                        level_0_stats = sorted(
                            level_0_stats, key=lambda x: x[sort_key], reverse=True
                        )

                    # Display level 0 clusters
                    for cluster in level_0_stats:
                        cluster_row = cluster["cluster_row"]
                        cluster_id = cluster["cluster_id"]
                        cluster_name = cluster["cluster_name"]
                        num_tweets = cluster["num_tweets"]
                        total_likes = cluster["total_likes"]
                        avg_date = cluster["avg_date"]
                        summary = cluster_row["summary"]

                        with st.container():
                            st.write(f"### {cluster_name}")
                            stats_text = f"Number of Tweets: {num_tweets} | Total Likes: {total_likes} | Average Date: {avg_date.strftime('%Y-%m-%d') if pd.notnull(avg_date) else 'N/A'}"
                            st.write(stats_text)
                            if st.button(
                                f"Select '{cluster_name}'", key=f"cluster_{cluster_id}"
                            ):
                                st.session_state["selected_clusters"].append(cluster_id)
                                st.rerun()

        with col2:
            st.header("Tweets")

            if st.session_state["selected_clusters"]:
                selected_cluster_id = str(st.session_state["selected_clusters"][-1])
                cluster_ids = [selected_cluster_id] + get_descendant_clusters(
                    selected_cluster_id
                )
                cluster_tweets = tweets_df[tweets_df["cluster"].isin(cluster_ids)]

                # Apply date filter if any
                if st.session_state["date_filter"]:
                    start_date, end_date = st.session_state["date_filter"]
                    start_date = pd.to_datetime(start_date)
                    end_date = pd.to_datetime(end_date)
                    cluster_tweets = cluster_tweets[
                        (cluster_tweets["created_at"] >= start_date)
                        & (cluster_tweets["created_at"] <= end_date)
                    ]

                sort_options = {
                    "Favorite Count": "favorite_count",
                    "Date": "created_at",
                    "Cluster Probability": "cluster_prob",
                }
                sort_by = st.selectbox(
                    "Sort tweets by", options=list(sort_options.keys()), index=0
                )
                ascending = st.checkbox("Ascending order", value=False)

                cluster_tweets = cluster_tweets.sort_values(
                    by=sort_options[sort_by], ascending=ascending
                )

                # Copy button for tweets
                if st.button("Copy Tweets"):
                    tweet_texts = cluster_tweets.apply(
                        lambda row: f"{row['username']} ({row['created_at']}): {row['full_text']}",
                        axis=1,
                    )
                    tweets_str = "\n\n".join(tweet_texts)
                    st.text_area("Tweets Text", tweets_str, height=200)

                # Display tweets with avatars
                st.markdown(
                    """
                    <style>
                    .tweet-container {
                        display: flex;
                        align-items: flex-start;
                        margin-bottom: 20px;
                    }
                    .tweet-avatar {
                        width: 48px;
                        height: 48px;
                        border-radius: 50%;
                        margin-right: 10px;
                    }
                    .tweet-content {
                        flex: 1;
                    }
                    .tweet-content a { color: inherit; text-decoration: none; }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

                # After loading other data, add this cache for user profiles
                # @st.cache_data(ttl=3600)
                def get_user_profiles():
                    logging.info("Getting user profiles")
                    # Get unique account IDs from tweets_df
                    account_ids = (
                        pd.concat(
                            [
                                tweets_df["accountId"].astype(str),
                                # tweets_df["reply_to_user_id"].dropna().astype(str),
                            ]
                        )
                        .unique()
                        .tolist()
                    )

                    # Fetch only needed profiles
                    profiles = fetch_users_with_profiles(account_ids)
                    # Convert to username-indexed dict for easy lookup
                    return profiles

                # Add after other data loading
                user_profiles = get_user_profiles()

                # Replace the tweet display section with this updated version
                for index, row in cluster_tweets.iterrows():
                    tweet_url = f"https://twitter.com/i/web/status/{row['tweet_id']}"

                    # Get profile for tweet author

                    profile = user_profiles.get(str(row["accountId"]), {})
                    avatar_url = (
                        profile.get("profile", {}).get("avatar_media_url")
                        if st.session_state["selected_user"] != "exgenesis"
                        else "https://pbs.twimg.com/profile_images/1842317384532541440/gbyCmXpi.jpg"
                    )
                    # Get profile for replied-to user if exists
                    reply_profile = (
                        user_profiles.get(row["reply_to_username"], {})
                        if pd.notnull(row["reply_to_username"])
                        else None
                    )
                    reply_avatar = (
                        reply_profile.get("profile", {}).get("avatar_media_url")
                        if reply_profile
                        else None
                    )

                    full_text = row["full_text"]
                    created_at = row["created_at"]
                    username = row["username"]
                    favorite_count = row["favorite_count"]
                    cluster_prob = row["cluster_prob"]

                    # Add reply info if present
                    reply_info = ""
                    if pd.notnull(row["reply_to_username"]):
                        reply_info = f"""
                        <div style="margin-bottom:5px;">
                            Replying to <b>@{row['reply_to_username']}</b>
                            {f'<img src="{reply_avatar}" style="width:24px;height:24px;border-radius:50%;margin-left:5px;vertical-align:middle">' if reply_avatar else ''}
                        </div>
                        """

                    st.markdown(
                        f"""
                        <div class="tweet-container">
                            <div class="tweet-avatar">
                                <img src="{avatar_url}" style="width:100%;height:100%;border-radius:50%;">
                            </div>
                            <div class="tweet-content">
                                <b>@{username}</b> - <a href="{tweet_url}" target="_blank">{created_at}</a>
                                <br>
                                {full_text}
                                <br>
                                <small>Favorite Count: {favorite_count}, Cluster Probability: {cluster_prob:.2f}</small>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown("---")
            else:
                st.write("Please select a cluster to view tweets.")

    except FileNotFoundError:
        st.error(
            f"No data found for user @{st.session_state['selected_user']}. Please select another user."
        )
        st.session_state["selected_user"] = None
        st.rerun()
