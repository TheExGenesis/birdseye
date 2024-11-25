# %%
ontology = {
    "schema_info": "id should be unique singlie letters across all item types, starting from A. e.g.: entities will have ids A,B,C and beliefs_and_values will have ids D,E,F etc. tweet_references should be tweet ids, digits only.",
    "low_quality_cluster": {
        "schema_info": "Set to 1 if you think this grouping is ambiguous or low signal, stuff like small talk, short reactions, urls, disjoint topics, mostly @tags, suspended account messages, etc. 0 otherwise.",
        "value": "'0' or '1'",
    },
    "entities": [
        {
            "schema_info": "Significant entities: organizations, locations, people, etc. Do not mention the primary user, or users interacted with.",
            "id": "string",
            "name": "string",
            "description": "string",
            "tweet_references": ["string"],
        }
    ],
    "beliefs_and_values": [
        {
            "id": "string",
            "belief": "string",
            "description": "string",
            "tweet_references": ["string"],
        }
    ],
    "goals": [
        {
            "schema_info": "Explicit or inferred, concrete practical goals.",
            "id": "string",
            "goal": "string",
            "description": "string",
            "tweet_references": ["string"],
        }
    ],
    "social_relationships": [
        {
            "schema_info": "Key users interacted with, or inferred important social ties.",
            "id": "string",
            "username": "string",
            "interaction_type": "string",
            "tweet_references": ["string"],
        }
    ],
    "moods_and_emotional_tones": [
        {
            "id": "string",
            "mood": "string",
            "description": "string",
            "tweet_references": ["string"],
        }
    ],
    "key_concepts_and_ideas": [
        {
            "id": "string",
            "concept": "string",
            "description": "string",
            "tweet_references": ["string"],
        }
    ],
    "cluster_summary": {
        "schema_info": "Name should be '📕 A punchy handle for this cluster' with a representative emoji, no subtitles! Summary should be like a lighthearted wikipedia article, with hyperlinks ([name](id)) to objects in the ontology and tweet ids.",
        "name": "string",
        "summary": "string",
    },
    "yearly_summaries": [
        {
            "schema_info": "A lighthearted wikipedia paragraph with hyperlinks summarizing the events of each year. Cover all years.",
            "period": "year YYYY or interval YYYY-YYYY",
            "summary": "string",
        }
    ],
}


LABEL_CLUSTER_PROMPT = """
Please generate a concise and descriptive name and summary paragraph for the following cluster of tweets:

<TWEETS>
{tweet_texts}
</TWEETS>

The name should be straightforward and reflect the overall theme or topic of the tweets, as the user would write it. Begin the name with a representative emoji.
The summary should be concise and capture common themes, in the style of the user. Don't quote tweets directly.
If you can't find a name or summary, set the name to "No name found" and the summary to "No summary found".

Respond in JSON format, delimited by <ANSWER>:
<ANSWER>
{{
    "name": "📕 A punchy handle for this cluster",
    "summary": "A summary of the cluster themes",
    "bad_group_flag": 0 or 1. Set to 1 if you think this grouping is ambiguous or low signal - stuff like small talk, short reactions, urls, etc."
}}
</ANSWER>
"""

LABEL_GROUP_PROMPT = """
Please generate a concise and descriptive name and summary paragraph for the following group of related clusters:

<CLUSTERS>
{group_str}
</CLUSTERS>

The name should be straightforward and reflect the overall theme or topic of the tweets, as the user would write it. Begin the name with a representative emoji.
The summary should be concise and capture common themes, in the style of the user. Don't quote tweets directly.
If you can't find a name or summary, set the name to "No name found" and the summary to "No summary found".



Respond in JSON format, delimited by <ANSWER>:
<ANSWER>
{{
    "name": "📚 A title for this group of clusters",
    "summary": "A summary of how these clusters relate",
    "bad_group_flag": 0 or 1. Set to 1 if you think this grouping is ambiguous, low signal, or doesn't make sense."
}}
</ANSWER>
"""


ONTOLOGY_LABEL_CLUSTER_PROMPT = """
Please generate a concise and descriptive name and summary paragraph for the following sequence of tweets:

{previous_ontology}

<TWEETS>
{tweet_texts}
</TWEETS>

Instructions:
- Respond in JSON format, excluding schema_info, delimited by <ANSWER>
- Aim for 5 entries for each type, unless data is scarse. 
- Include every key in the schema. If no entries, set to empty list. 
- To select ontology items, all else equal, select the most significant and recent.

<ANSWER>
{ontology}
</ANSWER>
"""

group_ontology = {
    "schema_info": "Name should be '📚 A title for this group of clusters' with a representative emoji, no sub-titles! Summary should be like a lighthearted wikipedia article, with hyperlinks ([name](id)) to clusters if they can be connected to the group, even if they are not members.",
    "groups": [
        {
            "name": "📚 Group A",
            "reasoning": "there seems to be a cluster with theme A because...",
            "members": [{"name": "name", "id": "1"}],
            "summary": "string",
        }
    ],
}

ONTOLOGY_GROUP_EXAMPLES = {
    "schema_info": "Groups of thematically related clusters with fun names and emojis",
    "groups": [
        {
            "name": "🛠️ The Builder's Corner",
            "reasoning": "Clusters 27 and 37 both focus on technical development, tools, and the builder community",
            "members": [
                {"name": "Tech Community Interactions", "id": "27"},
                {"name": "Browser Extension Odyssey", "id": "37"},
            ],
            "summary": "A delightful collection of technical adventures, where our protagonist navigates the waters of [browser extension development](G1) while engaging with the tech community through [Thlpr](A) and other platforms. Features appearances from friendly faces like [Nosilverv](B) and dreams of better tools, including the elusive [version control for ideas](G2).",
        }
    ],
    "ungrouped": [
        {
            "name": "Anime Odyssey: A Journey of Narrative and Self-Discovery",
            "id": "6",
            "reasoning": "Stands alone as a distinct theme about media consumption and personal growth through anime",
        }
    ],
    "is_error": False,
}


ONTOLOGY_GROUP_PROMPT = """
Please group these clusters into a groups categories that tightly make sense together and give them punchy lighthearted names and an emoji. Especially if they seem to refer to the same projects, entities, specific themes. Some topics won't be in groups, but check twice and be exhaustive. 

<CLUSTERS>
{clusters_str}
</CLUSTERS>




Reply with a json delimited by <ANSWER>, using double quotes for property names.
<EXAMPLES>
{examples}
</EXAMPLES>

<ANSWER>
{ontology}
</ANSWER>
"""
