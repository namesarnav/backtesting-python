"""Phase 0 checkpoint: confirms the test suite collects and runs."""


def test_repo_imports():
    import engine  # noqa: F401
    import strategies  # noqa: F401
    import metrics  # noqa: F401
    import viz  # noqa: F401
