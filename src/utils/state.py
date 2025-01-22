import streamlit as st

DEFAULT_STATE = {
    "selected_user": None,
    "selected_cluster_id": None,
    "selected_clusters": [],
    "date_filter": None,
    "recompute_from": None,
    "show_bad_clusters": False,
    "filter_replies": True,
    "filter_retweets": True,
    "cluster_sort_by": "Average Date",
    "cluster_prob_threshold": 0.3,
    "tweet_sort_by": "Favorite Count",
    "tweet_sort_ascending": False,
    "show_orphan_clusters": True,
    "show_incomplete": False,
}


def init_session_state():
    """Initialize session state with defaults"""
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data
def update_state(key: str, value: any, rerun: bool = True):
    """Update session state without triggering unnecessary reruns"""
    current_value = st.session_state.get(key)

    # Only update and rerun if value actually changed
    if current_value != value:
        st.session_state[key] = value

        # Handle dependent state updates without triggering additional reruns
        # if key == "selected_user":
        #     st.session_state["selected_clusters"] = []
        #     st.session_state["date_filter"] = None

        if rerun:
            st.rerun()


def get_sort_options():
    """Return mapping of display names to field names"""
    return {
        "clusters": {
            "Average Date": "avg_date",
            "Number of Tweets": "num_tweets",
            "Total Likes": "total_likes",
        },
        "tweets": {
            "Favorite Count": "favorite_count",
            "Date": "created_at",
            "Cluster Probability": "cluster_prob",
        },
    }
