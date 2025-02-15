import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import base64
import os
from utils.state import get_sort_options
from utils.data import (
    fetch_tweets_with_images,
    normalize_tweet,
    unlink_markdown,
    get_default_user,
)


def get_base64_of_bin_file(bin_file):
    """Get base64 string of binary file"""
    if os.path.exists(bin_file):
        with open(bin_file, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode()
    return None


# Add placeholder CSS to the page
def init_placeholder_css():
    """Initialize placeholder CSS for tweet avatars"""
    placeholder_img = get_base64_of_bin_file("placeholder.jpg")

    css = """
    <style>
    .tweet-avatar {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        background-color: #ccc;  /* Fallback color if image is not available */
        overflow: hidden;
    }
    .tweet-avatar img {
        width: 100%;
        height: 100%;
        object-fit: cover;
    }
    </style>
    """

    if placeholder_img:
        css += f"""
        <style>
        .tweet-avatar {{
            background-image: url(data:image/jpg;base64,{placeholder_img});
            background-size: cover;
        }}
        </style>
        """

    st.html(css)


def render_cluster_stats(cluster_row, cluster_tweets):
    """Render stats for selected cluster"""
    num_tweets = len(cluster_tweets)
    total_likes = cluster_tweets["favorite_count"].sum()
    avg_date = cluster_tweets["created_at"].mean()

    st.subheader(f"{cluster_row['name']}")
    st.write(
        f"**Number of Tweets:** {num_tweets} | "
        f"**Total Likes:** {total_likes} | "
        f"**Average Date:** {avg_date.strftime('%Y-%m-%d') if pd.notnull(avg_date) else 'N/A'}"
    )
    st.write(
        f"**Summary:** {unlink_markdown(cluster_row['summary'])}",
        unsafe_allow_html=True,
    )


def render_timeline_chart(cluster_tweets, tweets_df):
    """Render interactive timeline chart with full timeline in background"""
    # Create columns for layout with center alignment

    st.info("📈 Drag horizontally on the graph to filter tweets in the right column.")

    # Prepare cluster timeline data
    cluster_tweets = cluster_tweets.copy()
    cluster_tweets.loc[:, "month"] = (
        cluster_tweets["created_at"].dt.tz_localize(None).dt.to_period("M")
    )

    # Create complete date range
    date_range = pd.period_range(
        start=cluster_tweets["month"].min(),
        end=cluster_tweets["month"].max(),
        freq="M",
    )

    # Get counts and reindex with full date range
    cluster_tweets_per_month = (
        cluster_tweets.groupby("month")
        .size()
        .reindex(date_range, fill_value=0)
        .reset_index(name="tweet_count")
    )

    # Convert period back to timestamp for plotting
    cluster_tweets_per_month["month"] = cluster_tweets_per_month[
        "index"
    ].dt.to_timestamp()

    # Do the same for full timeline
    full_timeline = tweets_df.copy()
    full_timeline.loc[:, "month"] = (
        full_timeline["created_at"].dt.tz_localize(None).dt.to_period("M")
    )

    # Get counts for non-cluster tweets with complete date range
    non_cluster_tweets = full_timeline[
        ~full_timeline["tweet_id"].isin(cluster_tweets["tweet_id"])
    ]
    non_cluster_per_month = (
        non_cluster_tweets.groupby("month")
        .size()
        .reindex(date_range, fill_value=0)
        .reset_index(name="tweet_count")
    )

    # Convert period back to timestamp for plotting
    non_cluster_per_month["month"] = non_cluster_per_month["index"].dt.to_timestamp()

    fig = go.Figure()

    # Add cluster bars first (will be at bottom of stack)
    fig.add_trace(
        go.Bar(
            x=cluster_tweets_per_month["month"],
            y=cluster_tweets_per_month["tweet_count"],
            name="Cluster Tweets",
            marker_color="rgba(55, 83, 200, 0.7)",  # Blue
        )
    )

    # Add other tweets on top, but visible=false by default
    fig.add_trace(
        go.Bar(
            x=non_cluster_per_month["month"],
            y=non_cluster_per_month["tweet_count"],
            name="Other Tweets",
            marker_color="rgba(200, 200, 200, 0.3)",  # Light gray
            visible=False,
        )
    )

    fig.update_layout(
        title="Tweets over Time",
        xaxis_title="Month",
        yaxis_title="Tweet Count",
        hovermode="x unified",
        bargap=0.1,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        barmode="stack",  # Enable stacking
        height=250,
    )

    selection = st.plotly_chart(
        fig,
        use_container_width=True,
        key="word_occurrences",
        selection_mode="box",
        on_select="rerun",
    )

    # Handle selection
    if selection and selection["selection"] and selection["selection"]["points"]:
        selected_dates = [
            pd.to_datetime(point["x"]) for point in selection["selection"]["points"]
        ]
        start_date = min(selected_dates).tz_localize("UTC")
        end_date = (
            max(selected_dates).tz_localize("UTC")
            + pd.offsets.MonthEnd(0)
            + pd.DateOffset(days=1)
        )
        st.session_state["date_filter"] = (start_date, end_date)
    else:
        st.session_state["date_filter"] = None


def render_engaged_users(username, cluster_tweets):
    """Render top engaged users section"""
    reply_usernames = cluster_tweets["reply_to_username"].dropna().str.lower()
    if not reply_usernames.empty:
        top_reply_users = (
            reply_usernames[reply_usernames != username.lower()].value_counts().head(5)
        )
        users_with_counts = [
            f"{user} ({count})" for user, count in top_reply_users.items()
        ]
        st.subheader("Most replied to: ")
        st.markdown("\n".join([f"- {user}" for user in users_with_counts]))
    else:
        st.subheader("Most replied to: N/A")


@st.cache_data(ttl=60, show_spinner=False)
def filter_and_sort_tweets(_tweets_df, cluster_ids, _sort_options, _filters):
    """Pre-compute filtered and sorted tweets, ensuring clicked tweets are included"""
    # First get clicked tweets if any exist
    clicked_tweets = []
    if "clicked_reference_tweets" in st.session_state:
        clicked_tweets = _tweets_df[
            _tweets_df["tweet_id"]
            .astype(str)
            .isin(st.session_state["clicked_reference_tweets"])
        ].copy()

    # Filter remaining tweets as normal
    filtered_df = _tweets_df[_tweets_df["cluster"].isin(cluster_ids)].copy()

    if _filters["filter_replies"]:
        print(_filters["selected_user"])
        # print(filtered_df["username"])
        filtered_df = filtered_df[
            (
                filtered_df["reply_to_username"].isna()
                # | (filtered_df["reply_to_username"] == filters["selected_user"])
            )
            & (filtered_df["username"].str.lower() == _filters["selected_user"].lower())
        ]

    if _filters["filter_retweets"]:
        filtered_df = filtered_df[~filtered_df["full_text"].str.startswith("RT ")]

    filtered_df = filtered_df[
        filtered_df["cluster_prob"] >= _filters["cluster_prob_threshold"]
    ]

    if _filters["date_filter"]:
        start_date, end_date = _filters["date_filter"]
        filtered_df = filtered_df[
            (filtered_df["created_at"] >= start_date)
            & (filtered_df["created_at"] <= end_date)
        ]

    # Combine clicked tweets with filtered tweets, dropping duplicates
    if len(clicked_tweets) > 0:
        filtered_df = pd.concat([clicked_tweets, filtered_df]).drop_duplicates(
            subset=["tweet_id"]
        )

    return filtered_df.sort_values(
        by=_sort_options["field"], ascending=_sort_options["ascending"]
    )


import re


def clean_tweet_text(text):
    """Clean tweet text for display"""
    text = re.sub(r"https://t\.co/\w+", "", text)
    return text.strip()


def render_quoted_tweet(quoted_tweet, quoted_tweet_profile):
    """Render a quoted tweet in a nested format"""
    quoted_url = f"https://twitter.com/i/web/status/{quoted_tweet['tweet_id']}"
    quoted_text = clean_tweet_text(quoted_tweet["full_text"])

    created_at = pd.to_datetime(quoted_tweet["created_at"])
    date_display = created_at.strftime("%b %d, %Y")
    date_full = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    return f"""<div class="quoted-tweet" style="border-left: 3px solid #ccc; margin: 5px 0; padding: 8px; background: rgba(0,0,0,0.02);">
        <b>@{quoted_tweet_profile["username"]}</b> - <a href="{quoted_url}" target="_blank" title="{date_full}">{date_display}</a>
        <br>{quoted_tweet["full_text"]}<br><small>♡ {quoted_tweet["favorite_count"]}</small></div>"""


def render_tweet(
    tweet_row,
    user_profile,
    tweets_with_images=None,
    quoted_tweet=None,
    quoted_tweet_profile=None,
):
    """Render a single tweet with optional quote tweet"""
    tweet_url = f"https://twitter.com/i/web/status/{tweet_row['tweet_id']}"

    created_at = pd.to_datetime(tweet_row["created_at"])
    date_display = created_at.strftime("%b %d, %Y")
    date_full = created_at.strftime("%Y-%m-%d %H:%M:%S UTC")

    # Check if tweet is clicked/selected
    is_selected = str(tweet_row["tweet_id"]) in st.session_state.get(
        "clicked_reference_tweets", set()
    )
    highlight_style = (
        "background-color: rgba(20, 160, 255, 0.2); border-radius: 8px;"
        if is_selected
        else ""
    )

    avatar_url = (
        user_profile.get(
            "avatar_media_url",
            "https://pbs.twimg.com/profile_images/1785443372930412544/BeoXxhPZ.jpg",
        )
        if user_profile["username"] != "exgenesis"
        else "https://pbs.twimg.com/profile_images/1842317384532541440/gbyCmXpi.jpg"
    )
    if not avatar_url:
        avatar_url = (
            "https://pbs.twimg.com/profile_images/1785443372930412544/BeoXxhPZ.jpg"
        )

    # Add images if present
    images_html = ""
    if tweets_with_images and str(tweet_row["tweet_id"]) in tweets_with_images:
        images = tweets_with_images[str(tweet_row["tweet_id"])]
        images_html = '\n<div class="tweet-images">'
        for img in images:
            images_html += f'<img src="{img}" style="max-width:100%;margin:5px 0;">'
        images_html += "</div>\n<br>"

    # Add quoted tweet if present
    quoted_tweet_html = ""
    if quoted_tweet:
        quoted_tweet_html = (
            "\n" + render_quoted_tweet(quoted_tweet, quoted_tweet_profile) + "\n"
        )
    tweet_text = clean_tweet_text(tweet_row["full_text"])
    st.markdown(
        f"""
        <div class="tweet-container" style="{highlight_style}">
            <div class="tweet-avatar">
                {f'<img src="{avatar_url}">' if avatar_url else ''}
            </div>
            <div class="tweet-content">
                <b>@{tweet_row['username']}</b> - <a href="{tweet_url}" target="_blank" title="{date_full}">{date_display}</a>
                <br>{tweet_text}<br>{images_html}{quoted_tweet_html}<small>♡ {tweet_row['favorite_count']}</small>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")


def tweet_df_to_threads(tweets_df, trees, incomplete_trees=None):
    """Convert tweet dataframe to a dictionary of threads, ensuring clicked tweets are included"""
    tweets = {}
    thread_dict = {}
    processed_tweets = set()
    include_incomplete_threads = st.session_state["show_incomplete"]

    # First process clicked tweets if any exist
    clicked_tweets = st.session_state.get("clicked_reference_tweets", set())
    if clicked_tweets:
        # Look for clicked tweets in complete trees first
        for tweet_id in clicked_tweets:
            if tweet_id in processed_tweets:
                continue

            for tree_id, tree in trees.items():
                if tweet_id in tree["tweets"]:
                    # Find all paths containing this tweet
                    for path_id, path in tree["paths"].items():
                        if tweet_id in path:
                            thread_dict[path_id] = path
                            tweets.update(
                                {
                                    tid: normalize_tweet(tree["tweets"][tid])
                                    for tid in path
                                }
                            )
                            processed_tweets.update(path)
                    break

            # If not found in complete trees, check incomplete trees
            if tweet_id not in processed_tweets and incomplete_trees:
                for tree_id, tree in incomplete_trees.items():
                    if tweet_id in tree["tweets"]:
                        for path_id, path in tree["paths"].items():
                            if tweet_id in path:
                                thread_dict[f"incomplete_{path_id}"] = path
                                tweets.update(
                                    {
                                        tid: normalize_tweet(
                                            tree["tweets"][tid],
                                            child_tweet=(
                                                tree["tweets"][path[i + 1]]
                                                if i < len(path) - 1
                                                else None
                                            ),
                                        )
                                        for i, tid in enumerate(path)
                                    }
                                )
                                processed_tweets.update(path)
                        break

    # Then process remaining tweets from the filtered dataframe
    for tweet_id in tweets_df["tweet_id"]:
        if tweet_id in processed_tweets:
            continue

        # Check complete trees
        for tree_id, tree in trees.items():
            if str(tweet_id) in tree["tweets"]:
                for path_id, path in tree["paths"].items():
                    if str(tweet_id) in path:
                        thread_dict[path_id] = path
                        tweets.update(
                            {tid: normalize_tweet(tree["tweets"][tid]) for tid in path}
                        )
                        processed_tweets.update(path)
                break

        # Check incomplete trees if enabled
        if include_incomplete_threads and tweet_id not in processed_tweets:
            for tree_id, tree in incomplete_trees.items():
                if str(tweet_id) in tree["tweets"]:
                    for path_id, path in tree["paths"].items():
                        if str(tweet_id) in path:
                            thread_dict[f"incomplete_{path_id}"] = path
                            tweets.update(
                                {
                                    tid: normalize_tweet(
                                        tree["tweets"][tid],
                                        child_tweet=(
                                            tree["tweets"][path[i + 1]]
                                            if i < len(path) - 1
                                            else None
                                        ),
                                    )
                                    for i, tid in enumerate(path)
                                }
                            )
                            processed_tweets.update(path)
                    break

    return thread_dict, tweets


def count_consecutive_self_tweets(thread_data, thread_tweets):
    """Count longest streak of consecutive tweets by the original author"""
    if not thread_data:
        return 0

    root_tweet = thread_tweets.get(thread_data[0], {})
    root_user = str(root_tweet.get("username", "")).lower().strip() or "unknown"

    max_streak = current_streak = 0

    for tweet_id in thread_data:
        tweet = thread_tweets.get(tweet_id, {})
        tweet_user = str(tweet.get("username", "")).lower().strip() or "unknown"

        if tweet_user == root_user:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
        else:
            current_streak = 0

    return max_streak


def order_threads_for_display(
    thread_dict, thread_tweets, sort_options, clicked_tweets=None
):
    """Order threads, prioritizing those containing clicked tweets and sorting by sort_field.
    Returns:
    - main_threads: ordered dict {root_id: path}
    - other_threads: ordered dict {root_id: [{path_id: path}]}
    """
    # Group threads by root tweet
    threads_by_root = {}
    for path_id, thread_data in thread_dict.items():
        root_id = thread_data[0]
        if root_id not in threads_by_root:
            threads_by_root[root_id] = []
        threads_by_root[root_id].append((path_id, thread_data))

    # Score each root thread based on clicked tweets
    root_scores = {}
    if clicked_tweets:
        for root_id, threads in threads_by_root.items():
            score = 0
            for _, thread in threads:
                score += sum(1 for tweet_id in thread if tweet_id in clicked_tweets)
            root_scores[root_id] = score

    # Get sort value for each root tweet
    root_sort_values = {}
    for root_id in threads_by_root:
        root_tweet = thread_tweets.get(root_id, {})
        sort_value = root_tweet.get(sort_options["field"], 0)
        # Handle timestamps specially
        if isinstance(sort_value, pd.Timestamp):
            sort_value = sort_value.timestamp()
        root_sort_values[root_id] = sort_value

    # Sort roots by score (descending), sort value, then consecutive self-tweets
    sorted_roots = sorted(
        threads_by_root.keys(),
        key=lambda r: (
            -root_scores.get(r, 0),
            root_sort_values[r] if sort_options["ascending"] else -root_sort_values[r],
            -max(
                count_consecutive_self_tweets(t[1], thread_tweets)
                for t in threads_by_root[r]
            ),
        ),
    )

    # Process threads in priority order
    main_threads = {}
    other_threads = {}

    for root_id in sorted_roots:
        threads = threads_by_root[root_id]
        # Sort threads for this root by consecutive self-tweets
        sorted_threads = sorted(
            threads,
            key=lambda x: count_consecutive_self_tweets(x[1], thread_tweets),
            reverse=True,
        )

        # First thread becomes main thread
        main_path_id, main_thread = sorted_threads[0]
        main_threads[root_id] = main_thread

        # Rest become other threads
        if len(sorted_threads) > 1:
            other_threads[root_id] = [
                {path_id: thread_data} for path_id, thread_data in sorted_threads[1:]
            ]

    return main_threads, other_threads


def render_tweet_threads(
    tweets_df,
    cluster_ids,
    user_profiles,
    trees,
    incomplete_trees,
    username,
    user_profiles_df,
    qts,
):
    """Render tweets organized in threads with horizontal scrolling"""
    # Initialize placeholder CSS
    init_placeholder_css()

    st.markdown(
        """
        <style>
        .thread-container {
            max-width: 100vw;
            overflow-x: auto;
            padding-bottom: 1rem;
        }
        .thread-columns {
            display: flex;
            flex-wrap: nowrap;
            gap: 1rem;
            padding: 0 1rem;
        }
        .thread-column {
            flex: 0 0 400px;
            background: rgba(255,255,255,0.05);
            border-radius: 8px;
            padding: 1rem;
        }
        
        .st-key-thread_container .stHorizontalBlock > div {
            flex: 0 0 400px !important;
            width: 400px !important;
            min-width: 400px !important;
            max-width: 400px !important;
            height: 1600px !important;
            overflow-y: auto !important;
            padding: 1rem !important;
            border-radius: 12px !important;  /* Add rounded corners */
        }
    
        /* Alternate background colors for thread columns */
        .st-key-thread_container .stHorizontalBlock > div:nth-child(odd) {
            background-color: rgba(255, 255, 255, 0.05);
        }
        
        .st-key-thread_container .stHorizontalBlock > div:nth-child(even) {
            background-color: rgba(0, 0, 0, 0);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.container():
        st.markdown(
            '<div class="thread-container"><div class="thread-columns">',
            unsafe_allow_html=True,
        )

        # Get filtered tweets
        sort_options = {
            "field": get_sort_options()["tweets"][st.session_state["tweet_sort_by"]],
            "ascending": st.session_state["tweet_sort_ascending"],
        }
        print(f"sort_options: {sort_options}")
        filters = {
            "filter_replies": st.session_state["filter_replies"],
            "filter_retweets": st.session_state["filter_retweets"],
            "cluster_prob_threshold": st.session_state["cluster_prob_threshold"],
            "date_filter": st.session_state["date_filter"],
            "selected_user": st.session_state["selected_user"],
        }

        filtered_tweets = filter_and_sort_tweets(
            tweets_df,
            cluster_ids,
            sort_options,
            filters,
        )

        # Process both complete and incomplete trees
        thread_dict, thread_tweets = tweet_df_to_threads(
            filtered_tweets,
            trees,
            incomplete_trees,
        )

        # Pass clicked tweets to order_threads_for_display
        clicked_tweets = st.session_state.get("clicked_reference_tweets", set())
        main_threads, other_threads = order_threads_for_display(
            thread_dict,
            thread_tweets,
            sort_options,
            clicked_tweets,
        )

        # Initialize lazy loading state per cluster
        current_cluster = str(cluster_ids[0]) if cluster_ids else "global"

        # Check if loaded_chunks exists and is a dict, else initialize
        if "loaded_chunks" not in st.session_state or not isinstance(
            st.session_state.loaded_chunks, dict
        ):
            st.session_state.loaded_chunks = {}

        # Now handle current cluster
        if current_cluster not in st.session_state.loaded_chunks:
            st.session_state.loaded_chunks[current_cluster] = (
                1  # Start with first chunk
            )

        # Split threads into chunks of 20 for lazy loading
        chunk_size = 20
        all_threads = list(main_threads.items())
        total_chunks = (len(all_threads) + chunk_size - 1) // chunk_size

        # Only process visible chunks using cluster-specific count
        loaded_chunks = st.session_state.loaded_chunks[current_cluster]
        visible_threads = all_threads[: loaded_chunks * chunk_size]

        # Fetch images only for visible tweets
        visible_tweet_ids = [
            tweet_id for _, thread in visible_threads for tweet_id in thread
        ]
        tweets_with_images = fetch_tweets_with_images(visible_tweet_ids)

        print(
            f"clicked_reference_tweets: {st.session_state['clicked_reference_tweets']}, {len(set(st.session_state['clicked_reference_tweets']) & set(thread_tweets.keys()))}"
        )

        # Create container for horizontally scrolling threads
        with st.container(key="thread_container"):
            # Calculate total columns needed (visible threads + load more button if needed)
            should_show_load_more = (
                st.session_state.loaded_chunks[current_cluster] < total_chunks
            )
            num_columns = len(visible_threads) + (1 if should_show_load_more else 0)
            cols = st.columns(max(num_columns, 1))  # Ensure at least 1 column

            # Render each visible thread in its own column
            for col_idx, (root_id, thread_data) in enumerate(visible_threads):
                with cols[col_idx]:
                    # Add visual indicator for incomplete threads
                    if any(
                        path_id.startswith("incomplete_")
                        for path_id in thread_dict
                        if thread_data[0] in thread_dict[path_id]
                    ):
                        st.warning("⚠️ Incomplete thread")

                    # Get tweets in order of the path
                    for tweet_id in thread_data:
                        if tweet_id in thread_tweets:
                            tweet_data = thread_tweets[tweet_id]

                            # Try to get user profile, create dummy if not found
                            matching_profiles = user_profiles_df[
                                user_profiles_df["account_id"]
                                == tweet_data["account_id"]
                            ]
                            if len(matching_profiles) > 0:
                                user_profile = matching_profiles.iloc[0]
                            else:
                                user_profile = pd.Series(
                                    {
                                        "username": tweet_data.get(
                                            "username", "unknown_user"
                                        ),
                                        "account_id": tweet_data["account_id"],
                                        "avatar_media_url": None,
                                    }
                                )

                            # Convert tweet data to match the format expected by render_tweet
                            formatted_tweet = pd.Series(
                                {
                                    "tweet_id": tweet_data["tweet_id"],
                                    "username": user_profile.get(
                                        "username", "unknown_user"
                                    ),
                                    "account_id": tweet_data.get("account_id", ""),
                                    "full_text": tweet_data.get("full_text", ""),
                                    "created_at": pd.to_datetime(
                                        tweet_data.get("created_at")
                                    ),
                                    "favorite_count": tweet_data.get(
                                        "favorite_count", 0
                                    ),
                                    "reply_to_username": tweet_data.get(
                                        "in_reply_to_username"
                                    ),
                                    "cluster_prob": 1,  # These are definitely part of the thread
                                    "avatar_media_url": user_profile.get(
                                        "avatar_media_url"
                                    ),
                                }
                            )
                            quoted_tweet_id = qts["quote_map"].get(tweet_id, None)
                            quoted_tweet = qts["quoted_tweets"].get(
                                quoted_tweet_id, None
                            )
                            if quoted_tweet:
                                quoted_profile_df = user_profiles_df[
                                    user_profiles_df["account_id"]
                                    == quoted_tweet["account_id"]
                                ]
                                quoted_tweet_profile = (
                                    quoted_profile_df.iloc[0]
                                    if len(quoted_profile_df) > 0
                                    else get_default_user()
                                )
                            else:
                                quoted_tweet_profile = None

                            render_tweet(
                                formatted_tweet,
                                user_profile,
                                tweets_with_images,
                                quoted_tweet,
                                quoted_tweet_profile,
                            )

            # Add load more button in last column if needed
            if should_show_load_more:
                with cols[-1]:
                    st.write("")  # Vertical spacer
                    if st.button("⏩ Load more", key=f"load_more_{current_cluster}"):
                        # Only increment for current cluster
                        st.session_state.loaded_chunks[current_cluster] += 1
                        # Use experimental_rerun to preserve component state
                        st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)
