import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply the eval marker to every test collected under tests/eval/."""
    for item in items:
        if "tests/eval" in item.path.as_posix():
            item.add_marker(pytest.mark.eval)
