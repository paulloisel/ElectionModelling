# Tests

This directory contains unit tests and integration tests for the Election Modeling project.

## Test Files

### Unit Tests

- **`test_census_pipeline.py`** - Tests for the ACS Feature Reduction Pipeline
  - Tests variable filtering functionality
  - Tests correlation-based feature reduction
  - Tests pipeline initialization and metadata loading
  - Uses mocked data for consistent testing

- **`test_census_downloader.py`** - Tests for census data downloading
  - Tests Census API integration
  - Tests data fetching and processing
  - Tests error handling for API failures

- **`test_downloader.py`** - Tests for general downloader utilities
  - Tests file downloading functionality
  - Tests URL validation and error handling
  - Tests data processing utilities

### Integration Tests

- **`test_wa_congressional.py`** - Integration test for Washington congressional analysis
  - Tests complete workflow from data loading to analysis
  - Tests multi-year data processing
  - Tests variable selection and reduction pipeline
  - Uses simulated data for testing

## How to Run Tests

### Run all tests:
```bash
# From project root
python3 -m pytest tests/

# Or with more verbose output
python3 -m pytest tests/ -v
```

### Run specific test file:
```bash
python3 -m pytest tests/test_census_pipeline.py
python3 -m pytest tests/test_wa_congressional.py
```

### Run with coverage:
```bash
python3 -m pytest tests/ --cov=src --cov-report=html
```

## Test Structure

Tests follow standard Python testing conventions:
- Use `pytest` framework
- Test functions start with `test_`
- Use descriptive test names
- Include setup and teardown as needed
- Mock external dependencies (APIs, file systems)

## Test Data

- Tests use simulated/mocked data to ensure consistency
- No external API calls during testing
- Test data is generated programmatically
- Results are deterministic and repeatable

## Continuous Integration

Tests are designed to run in CI/CD environments:
- No external dependencies
- Fast execution
- Clear pass/fail results
- Comprehensive coverage of core functionality

## Notes

- Tests focus on functionality, not performance
- External APIs are mocked to avoid rate limits
- File I/O is tested with temporary files
- Error conditions are thoroughly tested
