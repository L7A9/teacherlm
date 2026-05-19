from __future__ import annotations

import re

from config import Settings


def test_dev_cors_regex_allows_localhost_and_lan_frontend_origins() -> None:
    regex = Settings().cors_origin_regex
    assert regex is not None
    pattern = re.compile(regex)

    assert pattern.fullmatch("http://localhost:3000")
    assert pattern.fullmatch("http://127.0.0.1:3000")
    assert pattern.fullmatch("http://0.0.0.0:3000")
    assert pattern.fullmatch("http://192.168.1.20:3000")
    assert not pattern.fullmatch("https://example.com")
