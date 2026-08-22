import pytest

from stock_advisor.config.settings import settings


@pytest.fixture(autouse=True)
def _isolate_sqlite_cache(tmp_path, monkeypatch):
    """Give every test its own empty SQLite file instead of the real data/advisor.sqlite.

    Without this, the price cache and news cache (both keyed in the same on-disk DB) can
    leak state between tests, or between a real interactive run of the app and a later
    test run: e.g. a ticker fetched for real during manual testing stays "fresh" in the
    shared file and silently short-circuits a test that expects a mocked provider to be
    called. Pointing settings.db_path at a fresh temp file per test removes that coupling.
    """
    monkeypatch.setattr(settings, "db_path", tmp_path / "test_advisor.sqlite")
