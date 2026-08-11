"""Verified content-addressed artifact storage."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any, Iterator

from evolsa.core.hashing import sha256_bytes
from evolsa.core.serde import canonical_json


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class ContentStoreError(RuntimeError):
    """Base class for content store failures."""


class UnsafeContentPathError(ContentStoreError, ValueError):
    """Raised when an identifier could escape or redirect the store root."""


class ContentHashMismatchError(ContentStoreError):
    """Raised when bytes do not match their claimed content address."""


class ContentAddressedStore:
    """Store immutable byte blobs under SHA-256 content addresses.

    Every read re-hashes the blob. Paths are derived only from strictly
    validated lowercase digests and checked after symlink resolution.
    """

    algorithm = "sha256"

    def __init__(self, root: str | os.PathLike[str]) -> None:
        raw_root = Path(root).expanduser()
        raw_root.mkdir(parents=True, exist_ok=True)
        self._root = raw_root.resolve(strict=True)
        if not self._root.is_dir():
            raise NotADirectoryError(self._root)
        self._blob_root = self._root / self.algorithm
        self._blob_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._require_within_root(self._blob_root)

    @property
    def root(self) -> Path:
        return self._root

    def put_bytes(
        self,
        content: bytes | bytearray | memoryview,
        *,
        expected_digest: str | None = None,
    ) -> str:
        data = bytes(content)
        digest = sha256_bytes(data)
        if expected_digest is not None:
            self._validate_digest(expected_digest)
            if digest != expected_digest:
                raise ContentHashMismatchError(
                    f"content hashes to {digest}, not expected {expected_digest}"
                )
        destination = self.path_for(digest)
        destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._require_within_root(destination.parent)

        if destination.exists() or destination.is_symlink():
            if self.get_bytes(digest) != data:
                raise ContentHashMismatchError(f"existing blob is corrupt: {digest}")
            return digest

        temporary_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=destination.parent, prefix=".pending-", delete=False
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(data)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.chmod(temporary_name, 0o600)
            try:
                os.link(temporary_name, destination)
            except FileExistsError:
                # A concurrent writer won. Its bytes still have to verify.
                pass
            if self.get_bytes(digest) != data:
                raise ContentHashMismatchError(f"stored blob is corrupt: {digest}")
        finally:
            if temporary_name is not None:
                try:
                    os.unlink(temporary_name)
                except FileNotFoundError:
                    pass
        return digest

    def put_text(self, content: str, *, encoding: str = "utf-8") -> str:
        if not isinstance(content, str):
            raise TypeError("content must be str")
        return self.put_bytes(content.encode(encoding))

    def put_json(self, content: Any) -> str:
        return self.put_text(canonical_json(content))

    def get_bytes(self, digest: str) -> bytes:
        path = self.path_for(digest)
        try:
            if path.is_symlink():
                raise UnsafeContentPathError(f"blob cannot be a symlink: {digest}")
            if path.exists() and not path.is_file():
                raise UnsafeContentPathError(f"blob path is not a regular file: {digest}")
            data = path.read_bytes()
        except FileNotFoundError:
            raise KeyError(digest) from None
        actual = sha256_bytes(data)
        if actual != digest:
            raise ContentHashMismatchError(
                f"blob {digest} failed verification (actual {actual})"
            )
        return data

    def get_text(self, digest: str, *, encoding: str = "utf-8") -> str:
        return self.get_bytes(digest).decode(encoding)

    def exists(self, digest: str) -> bool:
        path = self.path_for(digest)
        if not path.exists() and not path.is_symlink():
            return False
        self.get_bytes(digest)
        return True

    def verify(self, digest: str) -> bool:
        self.get_bytes(digest)
        return True

    def verify_all(self) -> tuple[str, ...]:
        verified: list[str] = []
        if not self._blob_root.exists():
            return ()
        for prefix in sorted(self._blob_root.iterdir()):
            if prefix.is_symlink() or not prefix.is_dir() or not re.fullmatch(r"[0-9a-f]{2}", prefix.name):
                raise UnsafeContentPathError(f"unexpected content-store path: {prefix}")
            for blob in sorted(prefix.iterdir()):
                digest = prefix.name + blob.name
                self._validate_digest(digest)
                if blob != self.path_for(digest):
                    raise UnsafeContentPathError(f"unexpected content-store path: {blob}")
                if blob.is_symlink() or not blob.is_file():
                    raise UnsafeContentPathError(f"blob path is not a regular file: {blob}")
                self.get_bytes(digest)
                verified.append(digest)
        return tuple(verified)

    def iter_digests(self) -> Iterator[str]:
        yield from self.verify_all()

    def path_for(self, digest: str) -> Path:
        self._validate_digest(digest)
        candidate = self._blob_root / digest[:2] / digest[2:]
        self._require_within_root(candidate)
        return candidate

    @staticmethod
    def _validate_digest(digest: str) -> None:
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise UnsafeContentPathError("content address must be 64 lowercase hex characters")

    def _require_within_root(self, path: Path) -> None:
        resolved = path.resolve(strict=False)
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise UnsafeContentPathError(f"path escapes content store: {path}") from None
