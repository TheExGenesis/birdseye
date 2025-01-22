from typing import Any
import pandas as pd
import streamlit as st

from utils.state import get_sort_options, update_state


def render_breadcrumbs(hierarchy_df):
    """Render navigation breadcrumbs"""
    breadcrumb_cols = st.columns([1] * (len(st.session_state["selected_clusters"]) + 1))

    with breadcrumb_cols[0]:
        if st.button(
            "Home", key="home_breadcrumb", type="secondary", use_container_width=True
        ):
            update_state("selected_clusters", [], rerun=True)

    for idx, cluster_id in enumerate(st.session_state["selected_clusters"]):
        with breadcrumb_cols[idx + 1]:
            cluster_row = hierarchy_df[hierarchy_df["cluster_id"] == cluster_id].iloc[0]
            if st.button(
                cluster_row["name"],
                key=f"breadcrumb_{idx}",
                type="secondary",
                use_container_width=False,
            ):
                update_state(
                    "selected_clusters",
                    st.session_state["selected_clusters"][: idx + 1],
                    rerun=True,
                )


def render_cluster_row(cluster, level=0, orphan=False):
    """Render a single cluster row with stats"""
    cluster_row = cluster["cluster_row"]
    cluster_id = cluster["cluster_id"]
    cluster_name = cluster["cluster_name"]

    # Pre-compute stats text
    stats_text = (
        f"📝 {cluster['num_tweets']} tweets | "
        f"❤️ {cluster['total_likes']} likes | "
        f"📅 {cluster['avg_date'].strftime('%Y-%m-%d') if pd.notnull(cluster['avg_date']) else 'N/A'}"
    )

    cols = st.columns([2, 2, 1])
    with cols[0]:
        st.markdown(
            f'<div class="cluster-title"><span class="cluster-indent">{level * "&nbsp;&nbsp;"}</span>{cluster_name}</div>',
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            f'<div class="cluster-stats">{stats_text}</div>', unsafe_allow_html=True
        )
    with cols[2]:
        if st.button("Select", key=f"{orphan and 'orphan_' or ''}cluster_{cluster_id}"):
            new_clusters = st.session_state["selected_clusters"] + [cluster_id]
            update_state("selected_clusters", new_clusters, rerun=True)


@st.cache_data(ttl=60)
def prepare_cluster_display_data(
    current_clusters, cluster_stats, sort_key, reverse=True
):
    """Pre-compute sorted cluster stats for display"""
    current_stats = [cluster_stats[c["cluster_id"]] for c in current_clusters]
    return sorted(current_stats, key=lambda x: x[sort_key], reverse=reverse)


def render_cluster_list(clusters_by_parent, cluster_stats):
    """Render the hierarchical cluster list more efficiently"""
    # Create container for all cluster content
    cluster_container = st.container()

    with cluster_container:
        sort_options = get_sort_options()["clusters"]
        sort_key = sort_options[st.session_state["cluster_sort_by"]]
        reverse = sort_key != "name"

        # Get current level clusters
        if st.session_state["selected_clusters"]:
            parent_cluster_id = st.session_state["selected_clusters"][-1]
            current_clusters = clusters_by_parent.get(parent_cluster_id, [])
        else:
            current_clusters = [
                c
                for c in clusters_by_parent["-1"]
                if st.session_state["show_bad_clusters"]
                or not c["low_quality_cluster"] == "1"
            ]

        # Pre-compute display data
        current_stats = prepare_cluster_display_data(
            current_clusters, cluster_stats, sort_key, reverse
        )

        # Render clusters in single container
        for cluster in current_stats:
            render_cluster_row(cluster)

            child_clusters = clusters_by_parent.get(str(cluster["cluster_id"]), [])
            if child_clusters:
                with st.expander("Show/Hide Child Clusters", expanded=True):
                    child_stats = prepare_cluster_display_data(
                        child_clusters, cluster_stats, sort_key, reverse
                    )
                    for child in child_stats:
                        render_cluster_row(child, level=1)


def render_orphan_clusters(clusters_by_parent, hierarchy_df, tweets_df, cluster_stats):
    """Render orphaned level 0 clusters"""
    orphan_clusters = [
        c
        for c in clusters_by_parent["-1"]
        if c["level"] == 0
        and c["parent"] == "-1"
        and (
            st.session_state["show_bad_clusters"] or not c["low_quality_cluster"] == "1"
        )
    ]

    with st.expander(f"Orphan Clusters ({len(orphan_clusters)})"):
        show_only_orphans = st.checkbox("Show only orphan clusters", value=True)

        clusters_to_show = (
            orphan_clusters
            if show_only_orphans
            else [c for c in clusters_by_parent["-1"] if c["level"] == 0]
        )

        orphan_stats = [cluster_stats[c["cluster_id"]] for c in clusters_to_show]
        sort_key = get_sort_options()["clusters"][st.session_state["cluster_sort_by"]]
        orphan_stats.sort(key=lambda x: x[sort_key], reverse=True)

        for cluster in orphan_stats:
            render_cluster_row(cluster, orphan=True)
