# Testing Guide

## Setup
```bash
pip install pytest
```

## Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_labels.py

# Run with verbose output
pytest -v
```

## Test Structure
- `test_labels.py`: Tests for label parsing and handling
  - Tests basic JSON parsing
  - Tests control character handling
  - Tests empty label cases

## Adding Tests
1. Add mock data to the MOCK_LABEL_DATA list
2. Add test functions prefixed with `test_`
3. Use descriptive test names and docstrings 