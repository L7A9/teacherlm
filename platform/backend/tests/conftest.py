from __future__ import annotations

import sys
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"


def _activate_backend_imports() -> None:
    for path in (CORE_DIR, BACKEND_DIR):
        path_text = str(path)
        if path_text in sys.path:
            sys.path.remove(path_text)
        sys.path.insert(0, path_text)

    cached_config = sys.modules.get("config")
    if cached_config is not None:
        cached_file = Path(getattr(cached_config, "__file__", "")).resolve()
        if cached_file != BACKEND_DIR / "config.py":
            del sys.modules["config"]


_activate_backend_imports()


def pytest_collect_file(file_path: Path, parent):  # noqa: ANN001
    if file_path.suffix == ".py" and file_path.parent == TESTS_DIR:
        _activate_backend_imports()
    return None
