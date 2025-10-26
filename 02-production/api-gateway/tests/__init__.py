"""
API Gateway Tests

This package contains tests for the API Gateway service including
unit tests, integration tests, and performance tests.
"""

# Test configuration
TEST_CONFIG = {
    'timeout': 5000,
    'retries': 3,
    'parallel': True
}

__all__ = ['TEST_CONFIG']