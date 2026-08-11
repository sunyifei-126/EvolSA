"""Fail-closed authorization for evolvable operator passports."""

from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass, replace
from typing import Collection, Mapping, Sequence

from evolsa.core import (
    EVOLVABLE_SCOPES,
    PROTECTED_SCOPES,
    EffectScope,
    OperatorLifecycle,
    OperatorPassport,
)


_KEY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SIGNATURE_ALGORITHM = "hmac-sha256"
_SIGNING_DOMAIN = b"evolsa.operator-passport/v1\x00"


class TrustKernelError(RuntimeError):
    """Base class for passport trust failures."""


class PassportIntegrityError(TrustKernelError):
    """Raised when a passport hash or signature is absent or invalid."""


class EffectAuthorizationError(TrustKernelError):
    """Raised when an operator requests an undeclared or protected effect."""


class ReleaseAuthorizationError(TrustKernelError):
    """Raised when a passport is not authorized for human-visible release."""


@dataclass(frozen=True, slots=True)
class TrustDecision:
    operator_id: str
    content_hash: str
    signer_id: str
    authorized_effects: tuple[EffectScope, ...]
    release_authorized: bool = False
    release_actor: str = ""


def sign_passport(passport: OperatorPassport, key_id: str, secret: bytes) -> OperatorPassport:
    """Return a copy signed by a named HMAC trust root.

    HMAC is suitable for a local trusted kernel where signer and verifier share
    a secret. A remote community registry should use an asymmetric signature
    service and retain this verifier behind that service boundary.
    """

    if not isinstance(passport, OperatorPassport):
        raise TypeError("passport must be OperatorPassport")
    if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
        raise ValueError("invalid key_id")
    if not isinstance(secret, bytes) or len(secret) < 16:
        raise ValueError("HMAC secret must contain at least 16 bytes")
    signature = _signature_for(passport.content_hash, key_id, secret)
    return replace(passport, signature=signature)


def _signature_for(content_hash: str, key_id: str, secret: bytes) -> str:
    message = _SIGNING_DOMAIN + content_hash.encode("ascii")
    digest = hmac.new(secret, message, hashlib.sha256).hexdigest()
    return f"{_SIGNATURE_ALGORITHM}:{key_id}:{digest}"


class TrustKernel:
    """Verify integrity, effects, protected scopes, and release authority.

    All capabilities are implicit-deny: an execution effect must be explicitly
    listed in ``expected_effects``; protected effects are never authorizable;
    and publication requires a stable, evidenced passport signed by a release
    key plus an allowlisted human/service principal.
    """

    def __init__(
        self,
        trusted_signers: Mapping[str, bytes],
        *,
        release_signers: Collection[str] = (),
        release_principals: Collection[str] = (),
        additional_protected_scopes: Collection[EffectScope] = (),
    ) -> None:
        if not trusted_signers:
            raise ValueError("at least one trusted signer is required")
        keys: dict[str, bytes] = {}
        for key_id, secret in trusted_signers.items():
            if not isinstance(key_id, str) or not _KEY_ID.fullmatch(key_id):
                raise ValueError(f"invalid trusted signer id: {key_id!r}")
            if not isinstance(secret, bytes) or len(secret) < 16:
                raise ValueError(f"trusted signer {key_id!r} requires at least 16 secret bytes")
            keys[key_id] = secret
        unknown_release_keys = set(release_signers) - set(keys)
        if unknown_release_keys:
            raise ValueError(f"release signers are not trusted: {sorted(unknown_release_keys)!r}")
        if any(not isinstance(principal, str) or not principal for principal in release_principals):
            raise ValueError("release principals must be non-empty strings")
        try:
            extra_protected = {EffectScope(scope) for scope in additional_protected_scopes}
        except ValueError as error:
            raise ValueError("unknown protected scope") from error

        self._trusted_signers = keys
        self._release_signers = frozenset(release_signers)
        self._release_principals = frozenset(release_principals)
        self._protected_scopes = frozenset(PROTECTED_SCOPES | extra_protected)

    @property
    def protected_scopes(self) -> frozenset[EffectScope]:
        return self._protected_scopes

    def verify_passport(
        self,
        passport: OperatorPassport,
        *,
        expected_hash: str | None = None,
    ) -> TrustDecision:
        """Verify schema, effect declarations, content hash, and signature."""

        self._validate_contract(passport)
        content_hash = passport.content_hash
        if not _DIGEST.fullmatch(content_hash):
            raise PassportIntegrityError("passport produced an invalid content hash")
        if expected_hash is not None:
            if not isinstance(expected_hash, str) or not _DIGEST.fullmatch(expected_hash):
                raise PassportIntegrityError("expected passport hash is malformed")
            if not hmac.compare_digest(content_hash, expected_hash):
                raise PassportIntegrityError("passport content hash does not match registry hash")
        signer_id = self._verify_signature(passport, content_hash)
        return TrustDecision(
            operator_id=passport.operator_id,
            content_hash=content_hash,
            signer_id=signer_id,
            authorized_effects=tuple(passport.expected_effects),
        )

    def authorize_execution(
        self,
        passport: OperatorPassport,
        requested_effects: Sequence[EffectScope],
        *,
        expected_hash: str | None = None,
    ) -> TrustDecision:
        """Authorize only explicitly expected, non-forbidden, evolvable effects."""

        decision = self.verify_passport(passport, expected_hash=expected_hash)
        try:
            requested = tuple(EffectScope(scope) for scope in requested_effects)
        except ValueError as error:
            raise EffectAuthorizationError("unknown requested effect scope") from error
        if len(set(requested)) != len(requested):
            raise EffectAuthorizationError("requested effects must be unique")
        expected = set(passport.expected_effects)
        forbidden = set(passport.forbidden_effects)
        for scope in requested:
            if scope in self._protected_scopes:
                raise EffectAuthorizationError(f"protected effect cannot be authorized: {scope.value}")
            if scope in forbidden:
                raise EffectAuthorizationError(f"explicitly forbidden effect requested: {scope.value}")
            if scope not in expected:
                raise EffectAuthorizationError(f"undeclared effect requested: {scope.value}")
        return replace(decision, authorized_effects=requested)

    def authorize_release(
        self,
        passport: OperatorPassport,
        actor: str,
        *,
        expected_hash: str | None = None,
    ) -> TrustDecision:
        """Authorize publication only with signature and human/service permission."""

        decision = self.verify_passport(passport, expected_hash=expected_hash)
        if passport.lifecycle is not OperatorLifecycle.STABLE:
            raise ReleaseAuthorizationError("only stable operators may be released")
        if not passport.evidence_ids:
            raise ReleaseAuthorizationError("release requires at least one evidence id")
        if decision.signer_id not in self._release_signers:
            raise ReleaseAuthorizationError("passport signer lacks release authority")
        if not isinstance(actor, str) or actor not in self._release_principals:
            raise ReleaseAuthorizationError("release actor lacks permission")
        return replace(decision, release_authorized=True, release_actor=actor)

    def _validate_contract(self, passport: OperatorPassport) -> None:
        if not isinstance(passport, OperatorPassport):
            raise PassportIntegrityError("unrecognized passport type")
        if passport.api_version != "evolsa/v1":
            raise PassportIntegrityError(f"unsupported passport api_version: {passport.api_version}")
        expected = tuple(passport.expected_effects)
        forbidden = tuple(passport.forbidden_effects)
        if any(not isinstance(scope, EffectScope) for scope in (*expected, *forbidden)):
            raise EffectAuthorizationError("effect declarations contain an unknown scope")
        if len(set(expected)) != len(expected) or len(set(forbidden)) != len(forbidden):
            raise EffectAuthorizationError("effect declarations must be unique")
        if set(expected) & set(forbidden):
            raise EffectAuthorizationError("an effect cannot be expected and forbidden")
        protected = set(expected) & set(self._protected_scopes)
        if protected:
            labels = ", ".join(sorted(scope.value for scope in protected))
            raise EffectAuthorizationError(f"passport declares protected effects: {labels}")
        non_evolvable = set(expected) - set(EVOLVABLE_SCOPES)
        if non_evolvable:
            labels = ", ".join(sorted(scope.value for scope in non_evolvable))
            raise EffectAuthorizationError(f"passport declares non-evolvable effects: {labels}")

    def _verify_signature(self, passport: OperatorPassport, content_hash: str) -> str:
        if not isinstance(passport.signature, str):
            raise PassportIntegrityError("passport signature is missing or malformed")
        parts = passport.signature.split(":")
        if len(parts) != 3 or parts[0] != _SIGNATURE_ALGORITHM:
            raise PassportIntegrityError("passport signature is missing or malformed")
        _, key_id, supplied_digest = parts
        if not _KEY_ID.fullmatch(key_id) or not _DIGEST.fullmatch(supplied_digest):
            raise PassportIntegrityError("passport signature is malformed")
        secret = self._trusted_signers.get(key_id)
        if secret is None:
            raise PassportIntegrityError("passport signer is not trusted")
        expected_signature = _signature_for(content_hash, key_id, secret)
        if not hmac.compare_digest(passport.signature, expected_signature):
            raise PassportIntegrityError("passport signature verification failed")
        return key_id
