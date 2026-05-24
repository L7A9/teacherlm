from __future__ import annotations

import sys
import uuid
import unittest
from datetime import datetime, timezone
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from schemas.file import UploadedFileRead  # noqa: E402


class FileSchemaTests(unittest.TestCase):
    def test_learning_pipeline_statuses_are_valid_file_statuses(self) -> None:
        for status in ["extracting_concepts", "building_course"]:
            parsed = UploadedFileRead.model_validate(
                {
                    "id": uuid.uuid4(),
                    "conversation_id": uuid.uuid4(),
                    "filename": "lecture.pdf",
                    "file_id": "uploads/lecture.pdf",
                    "status": status,
                    "chunk_count": 0,
                    "created_at": datetime.now(timezone.utc),
                }
            )

            self.assertEqual(parsed.status, status)


if __name__ == "__main__":
    unittest.main()
