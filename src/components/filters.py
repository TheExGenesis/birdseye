import streamlit as st
from utils.state import get_sort_options


def render_cluster_filters():
    """Render filter controls for clusters"""
    with st.expander("Advanced Options"):
        # Cluster sorting
        sort_options = get_sort_options()["clusters"]
        prev_sort = st.session_state["cluster_sort_by"]
        st.session_state["cluster_sort_by"] = st.selectbox(
            "Sort clusters by",
            options=list(sort_options.keys()),
            key="cluster_sort_select",
            index=list(sort_options.keys()).index(st.session_state["cluster_sort_by"]),
        )

        # Show bad clusters toggle
        st.session_state["show_bad_clusters"] = st.checkbox(
            "Show potentially irrelevant clusters",
            value=st.session_state["show_bad_clusters"],
        )


def render_tweet_filters():
    """Render filter controls for tweets"""
    with st.expander("Advanced Options"):
        # Basic filters
        st.session_state["filter_replies"] = st.checkbox(
            "Hide Replies", value=st.session_state["filter_replies"]
        )
        st.session_state["filter_retweets"] = st.checkbox(
            "Hide Retweets", value=st.session_state["filter_retweets"]
        )

        # Cluster probability threshold
        # st.session_state["cluster_prob_threshold"] = st.slider(
        #     "Minimum Cluster Probability",
        #     min_value=0.0,
        #     max_value=1.0,
        #     value=st.session_state["cluster_prob_threshold"],
        #     step=0.05,
        #     help="Only show tweets with cluster probability above this threshold",
        # )

        # Tweet sorting
        st.session_state["show_incomplete"] = not st.checkbox(
            "Hide incomplete threads",
            value=(not st.session_state["show_incomplete"]),
            help="Some conversation threads may be incomplete because not all tweets were captured in the archive",
        )
        sort_options = get_sort_options()["tweets"]
        st.session_state["tweet_sort_by"] = st.selectbox(
            "Sort tweets by",
            options=list(sort_options.keys()),
            index=list(sort_options.keys()).index(st.session_state["tweet_sort_by"]),
        )
        ascending = st.checkbox("Ascending order")
        if ascending != st.session_state.get("tweet_sort_ascending", False):
            st.session_state["tweet_sort_ascending"] = ascending
