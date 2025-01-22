import streamlit as st
import pandas as pd
import time
import re
from utils.data import unlink_markdown


def render_section(
    title,
    items,
    local_tweet_id_map,
    item_key="name",
    desc_key="description",
):
    """Render a section of ontology items as a card"""
    if not items:
        return

    # Add CSS for hover tooltip and buttons
    st.markdown(
        """
        <style>
        .tooltip {
            position: relative;
            display: inline-block;
        }
        
        .tooltip .tooltiptext {
            visibility: hidden;
            background-color: #f0f2f6;
            color: #31333F;
            text-align: left;
            padding: 8px;
            border-radius: 4px;
            border: 1px solid #ddd;
            position: absolute;
            z-index: 1;
            bottom: 100%;
            left: 0;
            opacity: 0;
            transition: opacity 0.3s;
        }
        
        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }


        /* Scope button styles to ontology sections */
        [data-testid="stHorizontalBlock"] .stButton > button {
            background: none!important;
            border: none!important;
            padding: 0!important;
            margin: 0!important;
            font-size: inherit!important;
            font-family: inherit!important;
            color: inherit!important;
            display: inline!important;
            text-align: left!important;
            width: auto!important;
            height: auto!important;
            line-height: inherit!important;
            min-height: 0px!important;
        }
        
        [data-testid="stHorizontalBlock"] .stButton > button:hover {
            background: none!important;
            color: inherit!important;
            border: none!important;
            text-decoration: underline;
        }
        
        [data-testid="stHorizontalBlock"] .stButton > button:focus {
            box-shadow: none!important;
            color: inherit!important;
            border: none!important;
        }

        /* Scope container styles to ontology sections */
        [data-testid="stHorizontalBlock"] .stButton {
            display: inline-block!important;
            margin: 0!important;
            padding: 0!important;
            line-height: inherit!important;
            min-height: 0px!important;
        }

        [data-testid="stHorizontalBlock"] .element-container,
        [data-testid="stHorizontalBlock"] .stElementContainer {
            margin: 0!important;
            padding: 0!important;
            line-height: inherit!important;
            min-height: 0px!important;
        }

        [data-testid="stHorizontalBlock"] .stButton button p {
            margin: 0!important;
            padding: 0!important;
            line-height: inherit!important;     
        }

        /* Ontology item styling */
        .ontology-item-content {
            background-color: rgba(255, 255, 255, 0.05);
            border-radius: 8px;
            padding: 8px 12px;
            margin-bottom: 8px;
        }

        /* Keep buttons inline with text */
        .ontology-item-content + .stButton {
            margin-top: -40px !important;
            margin-left: 8px !important;
            display: inline-block !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create card-style container
    with st.container():
        st.markdown(f"<h3 style='opacity: 0.35'>{title}</h3>", unsafe_allow_html=True)
        for item in items:
            name = item.get(item_key, "")
            desc = item.get(desc_key, "")
            tweet_refs = [
                r for r in item.get("tweet_references", []) if r in local_tweet_id_map
            ]

            # Format tweet references
            if tweet_refs:
                button_key = f"ref_button_{title}_{name}_{hash(str(tweet_refs))}"
                st.markdown(
                    f'<div class="ontology-item-content"><u>{name}</u>: {desc}</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"🔗 {len(tweet_refs)} references", key=button_key):
                    print(
                        f"clicked_reference_tweets: {st.session_state['clicked_reference_tweets']}"
                    )
                    st.session_state["clicked_reference_tweets"] = [
                        local_tweet_id_map[tweet_id] for tweet_id in tweet_refs
                    ]
            else:
                st.markdown(
                    f'<div class="ontology-item-content"><u>{name}</u>: {desc}</div>',
                    unsafe_allow_html=True,
                )


def render_social_relationships(relationships):
    """Render social relationships section"""
    if not relationships:
        return

    st.write("**Social Relationships**")
    for rel in relationships:
        username = rel.get("username", "")
        int_type = rel.get("interaction_type", "")
        tweet_refs = rel.get("tweet_references", [])
        ref_text = f" (🔗 {len(tweet_refs)} references)" if tweet_refs else ""
        st.markdown(f"- **@{username}**: {int_type}{ref_text}")
    st.markdown("---")


def render_yearly_summaries(yearly_summaries):
    """Render yearly summaries in tabs"""
    if not yearly_summaries:
        return

    st.subheader("Yearly Summaries")
    tabs = st.tabs([summary["period"] for summary in yearly_summaries])
    for tab, summary in zip(tabs, yearly_summaries):
        with tab:
            st.markdown(unlink_markdown(summary["summary"]), unsafe_allow_html=True)
    st.markdown("---")


def render_ontology_items(ontology_items, local_tweet_id_map):
    """Render ontology items with horizontally scrolling sections"""

    if not ontology_items:
        return

    # Create sections list
    sections = [
        ("Entities", ontology_items.get("entities", []), "name", "description"),
        (
            "Beliefs & Values",
            ontology_items.get("beliefs_and_values", []),
            "belief",
            "description",
        ),
        ("Goals", ontology_items.get("goals", []), "goal", "description"),
        (
            "Social Relationships",
            ontology_items.get("social_relationships", []),
            "username",
            "interaction_type",
        ),
        (
            "Moods & Emotional Tones",
            ontology_items.get("moods_and_emotional_tones", []),
            "mood",
            "description",
        ),
        (
            "Key Concepts",
            ontology_items.get("key_concepts_and_ideas", []),
            "concept",
            "description",
        ),
    ]

    # Filter out empty sections
    active_sections = [
        (title, items, key, desc) for title, items, key, desc in sections if items
    ]

    if active_sections:
        st.markdown("## Ontology:")
        with st.expander("🔍 View Detailed Analysis"):
            st.info(
                "↔️ We gathered some key features to help understand the topic all at once. You can click 🔗 references to see the tweets that inform them."
            )
            st.markdown(
                """
                <style>
                /* Target only the container with our specific key */
                .st-key-ontology_sections .stHorizontalBlock {
                    display: flex !important;
                    flex-wrap: nowrap !important;
                    overflow-x: auto !important;
                    gap: 1rem;
                    padding: 1rem;
                }
                
                .st-key-ontology_sections .stHorizontalBlock > div {
                    flex: 0 0 300px !important;
                    width: 300px !important;
                    min-width: 300px !important;
                    max-width: 300px !important;
                }
                </style>
                """,
                unsafe_allow_html=True,
            )

            # Create container with specific key
            with st.container(key="ontology_sections"):
                cols = st.columns(len(active_sections))

                # Render each section in its own column
                for col, (title, items, key, desc) in zip(cols, active_sections):
                    with col:
                        with st.container():
                            render_section(title, items, local_tweet_id_map, key, desc)
