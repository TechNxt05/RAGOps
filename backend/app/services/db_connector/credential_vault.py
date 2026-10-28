"""
Credential Vault — Encrypted-at-Rest External Database Credential Storage.
Uses Fernet envelope encryption derived from application SECRET_KEY.
Ensures zero plaintext database credentials are saved to PostgreSQL or emitted to OTel / OpenSmith traces.
"""

import os
import re
import base64
import hashlib
from typing import Any, Dict, Union
from cryptography.fernet import Fernet


def _get_fernet_key() -> bytes:
    from app.core.config import settings
    key_hash = hashlib.sha256(settings.secret_key.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(key_hash)


class CredentialVault:
    def __init__(self):
        key = _get_fernet_key()
        self.cipher = Fernet(key)

    def encrypt(self, plaintext: str) -> str:
        """Encrypt plaintext credential / password."""
        from app.core.config import settings
        if not settings.secret_key_configured:
            raise RuntimeError("SECRET_KEY must be set before database credentials can be stored.")
        if not plaintext:
            return ""
        return self.cipher.encrypt(plaintext.encode("utf-8")).decode("utf-8")

    def decrypt(self, ciphertext: str) -> str:
        """Decrypt encrypted credential string."""
        if not ciphertext:
            return ""
        return self.cipher.decrypt(ciphertext.encode("utf-8")).decode("utf-8")

    @staticmethod
    def redact_secrets(data: Union[str, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        Redact passwords, connection URIs, and secret tokens from strings or dicts
        before sending to OpenTelemetry spans, OpenSmith traces, or application logs.
        """
        if isinstance(data, str):
            # Redact URI passwords (e.g. postgresql://user:password@host:port/db -> postgresql://user:***@host:port/db)
            uri_pattern = re.compile(r'(://[^:]+:)([^@]+)(@)')
            redacted = uri_pattern.sub(r'\1***\3', data)
            # Redact password parameter key-value pairs
            kv_pattern = re.compile(r'(password|pass|pwd|secret|auth)\s*[:=]\s*["\']?[^"\'\s,]+["\']?', re.IGNORECASE)
            redacted = kv_pattern.sub(r'\1=***', redacted)
            return redacted
        elif isinstance(data, dict):
            redacted_dict = {}
            for k, v in data.items():
                if k.lower() in {"password", "pass", "pwd", "secret", "encrypted_credentials", "token"}:
                    redacted_dict[k] = "***REDACTED***"
                elif isinstance(v, (str, dict)):
                    redacted_dict[k] = CredentialVault.redact_secrets(v)
                else:
                    redacted_dict[k] = v
            return redacted_dict
        return data


credential_vault = CredentialVault()
