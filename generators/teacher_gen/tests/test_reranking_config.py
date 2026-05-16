from __future__ import annotations

import sys
import unittest
from pathlib import Path


TEACHER_GEN_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
for path in (TEACHER_GEN_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from config import Settings  # noqa: E402


class TeacherGenRerankingConfigTests(unittest.TestCase):
    def test_local_reranking_is_not_configured_in_generator(self) -> None:
        settings = Settings()

        self.assertFalse(hasattr(settings, "reranker_enabled"))
        self.assertFalse(hasattr(settings, "hyde_enabled"))


if __name__ == "__main__":
    unittest.main()
