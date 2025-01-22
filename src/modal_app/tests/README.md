# Testing Guide

## Setup

```bash
cd /Users/frsc/Documents/Projects/community-archive-personal
pip install -r requirements-test.txt
```

## Running Tests

```bash
# Run all tests
python -m pytest src/modal_app/tests

# Run specific test file
python -m pytest src/modal_app/tests/test_cluster_labeling.py

# Run with verbose output
python -m pytest src/modal_app/tests -v
```

## Test Structure

- `test_cluster_labeling.py`: Tests for cluster labeling pipeline
  - Tests TF-IDF based labeling
  - Tests cluster string generation
  - Tests ontology validation
  - Tests LLM labeling with mocks

## Adding Tests

1. Add mock data to the appropriate test file
2. Add test functions prefixed with `test_`
3. Use descriptive test names and docstrings
4. Mock external dependencies (e.g. Anthropic API)
