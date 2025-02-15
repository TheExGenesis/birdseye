# %%

from dataclasses import dataclass
from typing import List, Dict, Optional


@dataclass
class TweetReference:
    tweet_id: str


@dataclass
class Entity:
    id: str
    name: str
    description: str
    tweet_references: List[str]


@dataclass
class BeliefValue:
    id: str
    belief: str
    description: str
    tweet_references: List[str]


@dataclass
class Goal:
    id: str
    goal: str
    description: str
    tweet_references: List[str]


@dataclass
class SocialRelationship:
    id: str
    username: str
    interaction_type: str
    tweet_references: List[str]


@dataclass
class MoodTone:
    id: str
    mood: str
    description: str
    tweet_references: List[str]


@dataclass
class KeyConcept:
    id: str
    concept: str
    description: str
    tweet_references: List[str]


@dataclass
class YearlySummary:
    period: str
    summary: str


@dataclass
class ClusterSummary:
    name: str
    summary: str


@dataclass
class OntologyItems:
    entities: List[Entity]
    beliefs_and_values: List[BeliefValue]
    goals: List[Goal]
    social_relationships: List[SocialRelationship]
    moods_and_emotional_tones: List[MoodTone]
    key_concepts_and_ideas: List[KeyConcept]
    yearly_summaries: List[YearlySummary]


@dataclass
class ClusterOntology:
    cluster_id: str
    is_error: bool
    message: str
    ontology_items: OntologyItems
    cluster_summary: ClusterSummary
    low_quality_cluster: str


ClusterOntologyDict = Dict[str, ClusterOntology]

ontology = {
    "schema_info": "id should be unique singlie letters across all item types, starting from A. e.g.: entities will have ids A,B,C and beliefs_and_values will have ids D,E,F etc. tweet_references should be tweet ids, digits only.",
    "low_quality_cluster": {
        "schema_info": "Set to 1 if you think this grouping is ambiguous or low signal, stuff like small talk, short reactions, urls, disjoint topics, mostly @tags, suspended account messages, etc. 0 otherwise.",
        "value": "'0' or '1'",
    },
    "entities": [
        {
            "schema_info": "Significant entities: organizations, locations, people, events, etc. Do not mention the primary user, or users interacted with.",
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
            "schema_info": "Explicit or inferred, concrete practical stated goals.",
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
            "schema_info": "Key concepts or ideas, explicitly named that are recurrent in the tweets.",
            "id": "string",
            "concept": "string",
            "description": "string",
            "tweet_references": ["string"],
        }
    ],
    "yearly_summaries": [
        {
            "schema_info": "A lighthearted wikipedia paragraph with hyperlinks summarizing the events of each year. Cover all years present in the tweetsand no others.",
            "period": "year YYYY or interval YYYY-YYYY",
            "summary": "string",
        }
    ],
    "cluster_summary": {
        "schema_info": "Name should be '📕 A punchy handle for this cluster' with a representative emoji, no subtitles! Summary should be like a lighthearted wikipedia article, with hyperlinks ([name](id)) to objects in the ontology and tweet ids.",
        "name": "string",
        "summary": "string",
    },
}

ONTOLOGY_LABEL_CLUSTER_EXAMPLE = """
{
    "low_quality_cluster": {"value": "0"},
    "entities": [ 
        {
            "id": "E1",
            "name": "Community Garden",
            "description": "A shared urban garden space maintained by local residents",
            "tweet_references": ["5", "10", "12", "21"]
        },
        {
            "id": "E2", 
            "name": "Garden Club",
            "description": "Local group focused on sustainable gardening practices and education",
            "tweet_references": ["10", "13", "19"]
        },
        {
            "id": "E3",
            "name": "Seed Library",
            "description": "Community-managed collection of heirloom and local plant seeds",
            "tweet_references": ["25", "26", "27"]
        }
    ],
    "beliefs_and_values": [
        {
            "id": "B1",
            "belief": "Sustainability",
            "description": "Importance of environmentally conscious gardening methods",
            "tweet_references": ["15", "16"]
        },
        {
            "id": "B2",
            "belief": "Knowledge Sharing",
            "description": "Value of sharing gardening expertise within the community",
            "tweet_references": ["10", "12", "13"]
        },
        {
            "id": "B3",
            "belief": "Food Security",
            "description": "Growing food locally to enhance community resilience",
            "tweet_references": ["30", "31", "32"]
        }
    ],
    "goals": [
        {
            "id": "G1",
            "goal": "Expand Garden Space",
            "description": "Add new plots to accommodate growing community interest",
            "tweet_references": ["10", "12", "13"]
        },
        {
            "id": "G2",
            "goal": "Educational Programs",
            "description": "Develop workshops for teaching gardening skills",
            "tweet_references": ["19", "20", "21"]
        },
        {
            "id": "G3",
            "goal": "Seed Bank Growth",
            "description": "Expand the [Seed Library](E3) collection and preservation efforts",
            "tweet_references": ["25", "28", "29"]
        }
    ],
    "social_relationships": [
        {
            "id": "S1",
            "username": "garden_guru",
            "interaction_type": "Project Leader",
            "tweet_references": ["4", "5", "9", "10", "11"]
        },
        {
            "id": "S2",
            "username": "plant_friend",
            "interaction_type": "Regular Contributor",
            "tweet_references": ["3", "5", "12"]
        },
        {
            "id": "S3",
            "username": "seed_saver",
            "interaction_type": "Seed Library Coordinator",
            "tweet_references": ["25", "26", "27", "28"]
        }
    ],
    "moods_and_emotional_tones": [
        {
            "id": "M1",
            "mood": "Excited",
            "description": "Enthusiasm about garden growth and community involvement",
            "tweet_references": ["6", "10", "14"]
        },
        {
            "id": "M2",
            "mood": "Collaborative",
            "description": "Emphasis on working together and mutual support",
            "tweet_references": ["6", "15", "16"]
        },
        {
            "id": "M3",
            "mood": "Determined",
            "description": "Commitment to achieving garden expansion goals",
            "tweet_references": ["33", "34", "35"]
        }
    ],
    "key_concepts_and_ideas": [
        {
            "id": "K1",
            "concept": "Composting",
            "description": "Methods for creating and maintaining healthy soil",
            "tweet_references": ["10", "14"]
        },
        {
            "id": "K2",
            "concept": "Crop Rotation",
            "description": "Planning seasonal planting cycles for soil health",
            "tweet_references": ["21", "22", "23", "24"]
        },
        {
            "id": "K3",
            "concept": "Seed Saving",
            "description": "Techniques for preserving and storing viable seeds",
            "tweet_references": ["25", "26", "27"]
        }
    ],
    "yearly_summaries": [
        {
            "period": "2021",
            "summary": "Initial planning and establishment of the [Community Garden](E1) led by [garden_guru](S1), with focus on [sustainability](B1) principles."
        },
        {
            "period": "2022",
            "summary": "Launch of educational workshops by [plant_friend](S2) and implementation of [composting](K1) systems. The [Garden Club](E2) membership doubles."
        },
        {
            "period": "2023",
            "summary": "Establishment of the [Seed Library](E3) by [seed_saver](S3), introducing [seed saving](K3) programs and expanding [food security](B3) initiatives."
        },
        {
            "period": "2024",
            "summary": "Major expansion of garden plots, integration of [crop rotation](K2) systems, and development of comprehensive educational curriculum combining all key initiatives."
        }
    ],
    "cluster_summary": {
        "name": "🌱 Community Garden Growth Project",
        "summary": "A thriving community initiative led by [garden_guru](S1) to expand the [Garden Club](E2) through new plots and educational programs. The project emphasizes [sustainability](B1) and [food security](B3), featuring the innovative [Seed Library](E3) managed by [seed_saver](S3). Regular workshops cover [composting](K1) and [crop rotation](K2) techniques, while fostering a [collaborative](M2) community spirit."
    },
    
}"""


ONTOLOGY_LABEL_CLUSTER_PROMPT = """
Please generate a concise and descriptive name and summary paragraph for the following sequence of tweets:


<ONTOLOGY>
{ontology}
</ONTOLOGY>

EXAMPLE ANSWER:
<ANSWER>
{example_answer}
</ANSWER>


{previous_ontology}

<TWEETS>
{tweet_texts}
</TWEETS>

Instructions:
- Respond in JSON format, excluding schema_info, delimited by <ANSWER>. Ask no clarification questions.
- Aim for 3 entries for each type. 
- Include every key in the schema. If no entries, set to empty list. 
- To select ontology items, all else equal, select the most significant and recent.
- For each ontology item, aim for ~3 tweet references. Do not repeat the same item across categories.
- The given sequence may have nothing to do with the example.



"""

group_ontology = {
    "groups": [
        {
            "schema_info": "Name should be '📚 A title for this group of clusters' with a representative emoji, no sub-titles! Summary should be like a lighthearted wikipedia article, with hyperlinks ([name](id)) to clusters if they can be connected to the group, even if they are not members.",
            "name": "📚 Group A",
            "reasoning": "there seems to be a cluster with theme A because...",
            "members": [{"name": "name", "id": "1"}],
            "summary": "string",
        }
    ],
    "overall_summary": "string. A comprehensive summary of the themes in the user's archive. It should be like a lighthearted wikipedia article, with hyperlinks ([name](id)) to clusters.",
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
            "summary": "A delightful collection of technical adventures, where our protagonist navigates the waters of [browser extension development](G1) while engaging with the tech community through [Thlpr](27-A) and other platforms. Features appearances from friendly faces like [Nosilverv](27-B) and dreams of better tools, including the elusive [version control for ideas](37-G2).",
        },
        {
            "name": "🎲 Misc",
            "reasoning": "These clusters don't fit anywhere else",
            "members": [
                {"name": "Cooking", "id": "26"},
                {"name": "Anime", "id": "36"},
            ],
            "summary": "Diverse clusters that don't fit anywhere else.",
        },
    ],
    "overall_summary": {
        "schema_info": "A summary of the overall themes in the user's arc",
        "string": "A summary of the overall themes of the user's tweets",
    },
}


ONTOLOGY_GROUP_PROMPT = """
Please group these clusters into groups that tightly make sense together and give them punchy lighthearted names and an emoji. Especially if they seem to refer to the same projects, entities, specific themes. Include every cluster. Make a Misc group for clusters that don't fit anywhere else.


<ANSWER>
{ontology}
</ANSWER>

<CLUSTERS>
{clusters_str}
</CLUSTERS>

Reply with a json delimited by <ANSWER>, using double quotes for property names.
<EXAMPLES>
{examples}
</EXAMPLES>

Instructions:
- Aim for 10 groups.
- The inputs may have nothing to do with the example.
- Do not ask clarifying questions. If the answer would be long, abbreviate member names.


"""

# %%
