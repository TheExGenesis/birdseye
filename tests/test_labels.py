import json
import pytest

# Mock data with various edge cases
MOCK_LABEL_DATA = [
    {"accountLabel": {"label": "test1", "description": "normal label"}},
    {
        "accountLabel": {
            "label": "test2\x0b",  # Control char
            "description": "label with control char",
        }
    },
    {"accountLabel": {}},  # Empty label
]


def test_basic_label_parsing():
    """Test basic label parsing works"""
    data_str = json.dumps(MOCK_LABEL_DATA)
    parsed = json.loads(data_str)
    assert len(parsed) == 3
    assert parsed[0]["accountLabel"]["label"] == "test1"


def test_control_char_handling():
    """Test handling of control characters in labels"""
    data_str = json.dumps(MOCK_LABEL_DATA)
    # Should raise JSONDecodeError with default settings
    with pytest.raises(json.JSONDecodeError):
        json.loads(data_str, strict=True)

    # Should work with strict=False
    parsed = json.loads(data_str, strict=False)
    assert parsed[1]["accountLabel"]["label"].endswith("\x0b")


def test_empty_label():
    """Test handling of empty labels"""
    data_str = json.dumps(MOCK_LABEL_DATA)
    parsed = json.loads(data_str)
    assert parsed[2]["accountLabel"] == {}
