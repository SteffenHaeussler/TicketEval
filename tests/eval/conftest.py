import pytest

# execution.md's M2-T8 requires the cross-cutting workflow suite to be "collected by
# default". It is network-free and runs in seconds, so it stays in the commit gate
# rather than joining the eval-marked set the rest of tests/eval/ lives in.
DEFAULT_COLLECTED = frozenset({"test_runner_workflow.py"})


def pytest_collection_modifyitems(items):
    """Auto-apply the eval marker to tests/eval/, except the default-gate suites."""
    for item in items:
        if "tests/eval" not in item.path.as_posix():
            continue
        if item.path.name in DEFAULT_COLLECTED:
            continue
        item.add_marker(pytest.mark.eval)
