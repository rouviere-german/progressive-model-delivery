from __future__ import annotations

import base64
from pathlib import Path

import pytest

from progressive_model_delivery.registry.signing import (
    SigningError,
    SigningKey,
    TrustStore,
    VerifyKey,
)

pytestmark = pytest.mark.unit


def test_sign_and_verify_round_trip(tmp_path: Path) -> None:
    key = SigningKey.generate()
    path = tmp_path / "release.pem"
    key.save(path)
    loaded = SigningKey.load(path)
    assert loaded.key_id == key.key_id
    message = b"manifest bytes"
    signature = base64.b64encode(loaded.sign(message)).decode()
    trust = TrustStore.from_b64([key.public().to_b64(), ""])
    assert len(trust) == 1
    assert trust.verify(key.key_id, message, signature)
    assert not trust.verify(key.key_id, message + b"!", signature)
    assert "PRIVATE" not in repr(key)
    assert key.key_id in repr(key)


def test_unknown_keys_and_malformed_signatures_fail_closed() -> None:
    trusted, other = SigningKey.generate(), SigningKey.generate()
    trust = TrustStore([trusted.public()])
    message = b"m"
    forged = base64.b64encode(other.sign(message)).decode()
    assert not trust.verify(other.key_id, message, forged)  # key not trusted
    assert not trust.verify(trusted.key_id, message, forged)  # wrong key's signature
    assert not trust.verify(trusted.key_id, message, "not base64!")
    assert not trust.verify(trusted.key_id, message, base64.b64encode(b"short").decode())


def test_malformed_keys_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(SigningError):
        VerifyKey(b"too short")
    with pytest.raises(SigningError):
        VerifyKey.from_b64("***")
    with pytest.raises(SigningError):
        SigningKey.from_pem(b"-----BEGIN PRIVATE KEY-----\nnope\n-----END PRIVATE KEY-----\n")
