import pytest
import pandas as pd
import numpy as np
import time
from unittest.mock import patch, MagicMock
from ..lib.ontological_label_lib import (
    tfidf_label_clusters,
    validate_ontology_results,
    label_with_ontology,
    make_cluster_str,
    label_one_cluster,
)

# Mock data
MOCK_TWEETS_DF = pd.DataFrame(
    {
        "emb_text": [
            "this is a test tweet about python",
            "another test tweet about coding",
            "python is great for coding",
            "random tweet about something else",
            "more python coding tweets",
        ],
        "cluster": ["0", "0", "0", "1", "1"],
        "tweet_id": ["1", "2", "3", "4", "5"],
        "full_text": [
            "this is a test tweet about python",
            "another test tweet about coding",
            "python is great for coding",
            "random tweet about something else",
            "more python coding tweets",
        ],
        "created_at": ["2024-01-01"] * 5,
        "favorite_count": [1, 2, 3, 4, 5],
        "retweet_count": [1, 2, 3, 4, 5],
    }
)

MOCK_TREES = {
    "1": {"tweet_id": "1", "replies": []},
    "2": {"tweet_id": "2", "replies": ["3"]},
    "3": {"tweet_id": "3", "replies": []},
}

MOCK_INCOMPLETE_TREES = {}

MOCK_QTS = {}

MOCK_TFIDF_LABELS = {"0": ["python", "coding"], "1": ["random", "tweet"]}

MOCK_ONTOLOGY = {
    "cluster_summary": {"name": str, "summary": str},
    "low_quality_cluster": {"value": str},
}


def test_make_cluster_str():
    """Test cluster string generation"""
    cluster_str = make_cluster_str(
        MOCK_TWEETS_DF,
        MOCK_TREES,
        MOCK_INCOMPLETE_TREES,
        MOCK_TFIDF_LABELS,
        "0",
        MOCK_QTS,
    )

    # Check that cluster string contains key information
    assert "TF-IDF terms: python, coding" in cluster_str
    assert "this is a test tweet about python" in cluster_str
    assert "another test tweet about coding" in cluster_str

    # Check thread formatting
    assert "Thread:" in cluster_str
    assert "tweet_id: 2" in cluster_str
    assert "tweet_id: 3" in cluster_str


def test_tfidf_label_clusters():
    """Test TF-IDF based cluster labeling"""
    # Test basic functionality
    labels = tfidf_label_clusters(MOCK_TWEETS_DF, n_top_terms=2)
    assert "0" in labels
    assert "1" in labels
    assert len(labels["0"]) == 2
    assert len(labels["1"]) == 2

    # Test with excluded words
    labels = tfidf_label_clusters(MOCK_TWEETS_DF, n_top_terms=2, exclude_words=["test"])
    assert "test" not in labels["0"]


def test_validate_ontology_results():
    """Test ontology validation"""
    # Test valid results
    valid_results = {
        "cluster_summary": {"name": "Test Cluster", "summary": "Test Summary"},
        "low_quality_cluster": {"value": "0"},
    }
    result = validate_ontology_results(valid_results, MOCK_ONTOLOGY)
    assert result["valid"]

    # Test missing required fields
    invalid_results = {"cluster_summary": {"name": "Test Cluster"}}
    result = validate_ontology_results(invalid_results, MOCK_ONTOLOGY)
    assert not result["valid"]
    assert "cluster_summary.summary" in result["info"]


def test_label_one_cluster_rate_limit():
    """Test cluster labeling with rate limit handling"""
    cluster_id = "test_cluster"
    cluster_str = "Test cluster content"

    # Mock rate limit error first, then success
    responses = [
        Exception("Error code: 429 - {'type': 'rate_limit_error'}"),
        MagicMock(
            content=[
                MagicMock(
                    text="""
        <ANSWER>
        {
            "cluster_summary": {
                "name": "Test Cluster",
                "summary": "Test Summary"
            },
            "low_quality_cluster": {
                "value": "0"
            }
        }
        </ANSWER>
        """
                )
            ]
        ),
    ]

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = responses

    with patch("anthropic.Anthropic", return_value=mock_client):
        start_time = time.time()
        result = label_one_cluster(cluster_id, cluster_str)
        duration = time.time() - start_time

        # Should have waited at least 60 seconds after rate limit
        assert duration >= 60
        assert not result["is_error"]
        assert result["cluster_summary"]["name"] == "Test Cluster"

        # Check that it was called twice (once failing, once succeeding)
        assert mock_client.messages.create.call_count == 2


def test_label_one_cluster_success():
    """Test successful cluster labeling"""
    cluster_id = "test_cluster"
    cluster_str = "Test cluster content"

    mock_message = MagicMock()
    mock_message.content = [
        MagicMock(
            text="""
    <ANSWER>
    {
        "cluster_summary": {
            "name": "Test Cluster",
            "summary": "Test Summary"
        },
        "low_quality_cluster": {
            "value": "0"
        }
    }
    </ANSWER>
    """
        )
    ]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with patch("anthropic.Anthropic", return_value=mock_client):
        result = label_one_cluster(cluster_id, cluster_str)
        assert not result["is_error"]
        assert result["cluster_summary"]["name"] == "Test Cluster"
        assert mock_client.messages.create.call_count == 1
