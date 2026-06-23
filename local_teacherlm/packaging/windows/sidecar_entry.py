from __future__ import annotations

import multiprocessing
import os

import uvicorn

from local_api.main import app


def main() -> None:
    multiprocessing.freeze_support()
    os.environ.setdefault("DEBUG", "false")
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8765,
        log_level="info",
        access_log=False,
    )


if __name__ == "__main__":
    main()
