import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render_timeline(tweets_df):
    """Render timeline chart of tweets"""
    timeline_df = tweets_df.copy()[tweets_df["username"].notna()]
    timeline_df["month"] = (
        pd.to_datetime(timeline_df["created_at"]).dt.tz_localize(None).dt.to_period("M")
    )
    tweets_by_month = timeline_df.groupby("month").size().reset_index(name="count")
    tweets_by_month["month"] = tweets_by_month["month"].dt.to_timestamp()

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=tweets_by_month["month"],
            y=tweets_by_month["count"],
            name="Tweets per Month",
            marker_color="#1DA1F2",
        )
    )

    fig.update_layout(
        height=120,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(
            showgrid=False, showline=True, linecolor="#eee", showticklabels=False
        ),
        yaxis=dict(
            # showgrid=True,
            # gridcolor="#eee",
            showline=True,
            linecolor="#eee",
            showticklabels=False,
        ),
        showlegend=False,
        bargap=0.1,
    )

    return fig


def render_profile(user_profile, total_clusters):
    """Render user profile with graceful fallback for missing data"""
    avatar_url = (
        user_profile.get("profile", {}).get("avatar_media_url")
        if st.session_state["selected_user"] != "exgenesis"
        else "https://pbs.twimg.com/profile_images/1842317384532541440/gbyCmXpi.jpg"
    )

    st.markdown(
        f"""<div class="profile-header">
            <div class="profile-avatar-container">
                <img src="{avatar_url}" class="profile-avatar">
            </div>
            <div class="profile-info">
                <h2>@{user_profile['username']}</h2>
                <div>{user_profile.get('account_display_name', '')}</div>
                <div>{user_profile.get('bio', '')}</div>
                <div class="cluster-stats">
                    <span>📊 {total_clusters} clusters</span>
                </div>
                <div class="profile-stats">
                    <div class="stat-item">
                        <span class="stat-value">{user_profile.get('num_tweets', 0):,}</span>
                        <span class="stat-label">Tweets</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{user_profile.get('num_following', 0):,}</span>
                        <span class="stat-label">Following</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{user_profile.get('num_followers', 0):,}</span>
                        <span class="stat-label">Followers</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-value">{user_profile.get('num_likes', 0):,}</span>
                        <span class="stat-label">Likes</span>
                    </div>
                </div>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    # Add custom CSS to handle responsiveness
    st.markdown(
        """
        <style>
        .profile-header {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            align-items: flex-start;
        }
        .profile-avatar-container {
            flex: 0 0 auto;
            width: 150px;
            min-width: 100px;
        }
        .profile-avatar {
            width: 100%;
            aspect-ratio: 1;
            border-radius: 50%;
            object-fit: cover;
        }
        .profile-info {
            flex: 1;
            min-width: 200px;
        }
        .profile-stats {
            display: flex;
            flex-wrap: wrap;
            gap: 1rem;
            margin-top: 0.5rem;
        }
        .stat-item {
            display: flex;
            flex-direction: column;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


# with profile_cols[1]:
#     # Display a timeline visualization showing tweet activity over time
#     # This helps users understand posting patterns and engagement levels
#     st.markdown("### Tweet Timeline")
#     st.markdown("Shows global tweet frequency per month")
#     fig = render_timeline(tweets_df)
#     st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
