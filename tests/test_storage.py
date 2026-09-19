from pathlib import Path

import pytest

from engineering_agent.storage import JsonStore, StorageError


def test_corrupt_json_record_raises_storage_error(tmp_path):
    store = JsonStore(tmp_path, "runs")
    path = store._path("run-corrupt")

    path.write_text(
        '{"broken": ',
        encoding="utf-8",
    )

    with pytest.raises(StorageError):
        store.load("run-corrupt")


def test_missing_record_returns_none(tmp_path):
    store = JsonStore(tmp_path, "runs")

    assert store.load("missing") is None
