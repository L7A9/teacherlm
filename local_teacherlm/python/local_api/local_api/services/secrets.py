from __future__ import annotations

import os

from cryptography.fernet import Fernet

from local_api.config import get_settings


class SecretBox:
    def __init__(self) -> None:
        self.settings = get_settings()
        self._fernet: Fernet | None = None

    def encrypt(self, value: str | None) -> str | None:
        if not value:
            return None
        return self._get_fernet().encrypt(value.encode("utf-8")).decode("utf-8")

    def decrypt(self, value: str | None) -> str | None:
        if not value:
            return None
        return self._get_fernet().decrypt(value.encode("utf-8")).decode("utf-8")

    def _get_fernet(self) -> Fernet:
        if self._fernet is not None:
            return self._fernet
        path = self.settings.secret_key_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            key = path.read_bytes()
        else:
            key = Fernet.generate_key()
            path.write_bytes(key)
            if os.name != "nt":
                path.chmod(0o600)
        self._fernet = Fernet(key)
        return self._fernet


_secret_box: SecretBox | None = None


def get_secret_box() -> SecretBox:
    global _secret_box
    if _secret_box is None:
        _secret_box = SecretBox()
    return _secret_box

