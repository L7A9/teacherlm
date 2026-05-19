from __future__ import annotations

import importlib


def test_app_import_does_not_require_storage_vector_or_queue_clients() -> None:
    app_module = importlib.import_module("main")

    assert app_module.app is not None
