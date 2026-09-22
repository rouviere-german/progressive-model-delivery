"""Ed25519 signing for release manifests.

The training pipeline holds the private key; routers and shadow workers hold only a trust store of
public keys. A key is identified by the first 16 hex characters of the SHA-256 of its raw public
bytes, so a manifest names the key that signed it and verification looks the key up instead of
trusting whatever key the manifest might carry.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from progressive_model_delivery.core.fileio import atomic_write_bytes

__all__ = ["SigningError", "SigningKey", "TrustStore", "VerifyKey"]


class SigningError(ValueError):
    """A key could not be parsed or a signature is malformed."""


def _key_id(raw_public: bytes) -> str:
    return hashlib.sha256(raw_public).hexdigest()[:16]


@dataclass(frozen=True, slots=True)
class VerifyKey:
    raw: bytes

    def __post_init__(self) -> None:
        if len(self.raw) != 32:
            raise SigningError("an Ed25519 public key has 32 bytes")

    @property
    def key_id(self) -> str:
        return _key_id(self.raw)

    @classmethod
    def from_b64(cls, text: str) -> VerifyKey:
        try:
            return cls(base64.b64decode(text.strip(), validate=True))
        except (binascii.Error, ValueError) as exc:
            raise SigningError("public key is not valid base64") from exc

    def to_b64(self) -> str:
        return base64.b64encode(self.raw).decode("ascii")

    def verify(self, message: bytes, signature: bytes) -> bool:
        try:
            Ed25519PublicKey.from_public_bytes(self.raw).verify(signature, message)
        except InvalidSignature:
            return False
        return True


class SigningKey:
    """An Ed25519 private key. It is never logged and ``repr`` shows only the key id."""

    __slots__ = ("_key",)

    def __init__(self, key: Ed25519PrivateKey) -> None:
        self._key = key

    @classmethod
    def generate(cls) -> SigningKey:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def from_pem(cls, data: bytes) -> SigningKey:
        try:
            key = serialization.load_pem_private_key(data, password=None)
        except (ValueError, TypeError) as exc:
            raise SigningError("signing key is not an unencrypted PKCS#8 PEM") from exc
        if not isinstance(key, Ed25519PrivateKey):
            raise SigningError("signing key is not an Ed25519 key")
        return cls(key)

    @classmethod
    def load(cls, path: Path) -> SigningKey:
        return cls.from_pem(path.read_bytes())

    def save(self, path: Path) -> None:
        """Write the PEM with owner-only permissions (where the OS supports them)."""
        atomic_write_bytes(path, self.to_pem())
        path.chmod(0o600)

    def to_pem(self) -> bytes:
        return self._key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    def public(self) -> VerifyKey:
        return VerifyKey(
            self._key.public_key().public_bytes(
                serialization.Encoding.Raw, serialization.PublicFormat.Raw
            )
        )

    @property
    def key_id(self) -> str:
        return self.public().key_id

    def sign(self, message: bytes) -> bytes:
        return self._key.sign(message)

    def __repr__(self) -> str:
        return f"SigningKey(key_id={self.key_id!r})"


class TrustStore:
    """The public keys a verifier accepts, by key id."""

    def __init__(self, keys: Iterable[VerifyKey] = ()) -> None:
        self._keys: dict[str, VerifyKey] = {key.key_id: key for key in keys}

    @classmethod
    def from_b64(cls, values: Iterable[str]) -> TrustStore:
        return cls(VerifyKey.from_b64(value) for value in values if value.strip())

    @property
    def key_ids(self) -> Mapping[str, VerifyKey]:
        return dict(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def verify(self, key_id: str, message: bytes, signature_b64: str) -> bool:
        """True only for a well-formed signature by a trusted key over exactly ``message``."""
        key = self._keys.get(key_id)
        if key is None:
            return False
        try:
            signature = base64.b64decode(signature_b64, validate=True)
        except (binascii.Error, ValueError):
            return False
        if len(signature) != 64:
            return False
        return key.verify(message, signature)
