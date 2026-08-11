"""Conservative subprocess execution for local EvolSA experiments.

This module is intentionally named ``SubprocessSandbox``: it is process-level
resource control, not container, VM, network, syscall, or filesystem isolation.
Run untrusted code in a separately configured container/VM boundary.
"""

from __future__ import annotations

import math
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from evolsa.core import Budget, ExecutionStatus, Observation

try:  # pragma: no cover - exercised on the POSIX CI targets.
    import resource
except ImportError:  # pragma: no cover
    resource = None  # type: ignore[assignment]


_ENVIRONMENT_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SHELL_EXECUTABLES = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "csh", "tcsh", "fish", "pwsh", "powershell", "cmd.exe"}
)


class SandboxViolation(ValueError):
    """Raised before executing a command outside the configured boundary."""


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    default_timeout_seconds: float = 60.0
    max_output_bytes: int = 1_000_000
    max_memory_bytes: int | None = 1_073_741_824
    max_open_files: int | None = 64
    max_processes: int | None = 16

    def __post_init__(self) -> None:
        if not math.isfinite(self.default_timeout_seconds) or self.default_timeout_seconds < 0:
            raise ValueError("default_timeout_seconds must be finite and non-negative")
        if self.max_output_bytes < 1:
            raise ValueError("max_output_bytes must be positive")
        for value, label in (
            (self.max_memory_bytes, "max_memory_bytes"),
            (self.max_open_files, "max_open_files"),
            (self.max_processes, "max_processes"),
        ):
            if value is not None and value < 1:
                raise ValueError(f"{label} must be positive when configured")


class SubprocessSandbox:
    """Execute argv without a shell in a minimized environment.

    ``workspace`` constrains the child working directory, but does not prevent
    the child from reading other paths allowed to the current OS user. Network
    access is likewise *not* isolated. These limitations are carried in every
    returned :class:`Observation` so callers cannot mistake this for a
    container security boundary.
    """

    isolation_kind = "subprocess"
    provides_container_isolation = False
    provides_filesystem_isolation = False
    provides_network_isolation = False

    def __init__(
        self,
        workspace: str | os.PathLike[str] | None = None,
        *,
        limits: SandboxLimits | None = None,
        allowed_environment: Iterable[str] = (),
    ) -> None:
        self._temporary_workspace: tempfile.TemporaryDirectory[str] | None = None
        if workspace is None:
            self._temporary_workspace = tempfile.TemporaryDirectory(prefix="evolsa-sandbox-")
            workspace_path = Path(self._temporary_workspace.name)
        else:
            workspace_path = Path(workspace).expanduser()
            workspace_path.mkdir(parents=True, exist_ok=True)
        self._workspace = workspace_path.resolve(strict=True)
        if not self._workspace.is_dir():
            raise NotADirectoryError(self._workspace)
        self._limits = limits or SandboxLimits()
        allowed = frozenset(allowed_environment)
        if any(not isinstance(key, str) or not _ENVIRONMENT_KEY.fullmatch(key) for key in allowed):
            raise ValueError("allowed environment names must be portable identifiers")
        self._allowed_environment = allowed

    @property
    def workspace(self) -> Path:
        return self._workspace

    @property
    def limits(self) -> SandboxLimits:
        return self._limits

    def run(
        self,
        command: Sequence[str],
        *,
        budget: Budget | None = None,
        timeout_seconds: float | None = None,
        cwd: str | os.PathLike[str] | None = None,
        input_text: str | None = None,
        env: Mapping[str, str] | None = None,
        shell: bool = False,
    ) -> Observation:
        if shell:
            raise SandboxViolation("shell execution is disabled")
        if input_text is not None and not isinstance(input_text, str):
            raise SandboxViolation("input_text must be a string when supplied")
        argv = self._validate_command(command)
        working_directory = self._resolve_cwd(cwd)
        child_environment = self._build_environment(env)
        timeout = self._effective_timeout(budget, timeout_seconds)
        started = time.monotonic()
        metadata = self._metadata(argv, working_directory, timeout)

        stdin = subprocess.PIPE if input_text is not None else subprocess.DEVNULL
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            try:
                process = subprocess.Popen(
                    argv,
                    stdin=stdin,
                    stdout=stdout_file,
                    stderr=stderr_file,
                    cwd=working_directory,
                    env=child_environment,
                    shell=False,
                    start_new_session=os.name == "posix",
                    preexec_fn=self._resource_limiter(timeout) if os.name == "posix" else None,
                )
            except (FileNotFoundError, PermissionError, OSError, subprocess.SubprocessError) as error:
                runtime = time.monotonic() - started
                return Observation(
                    status=ExecutionStatus.INVALID,
                    logs=str(error),
                    error_signature=f"spawn:{type(error).__name__}",
                    runtime_seconds=runtime,
                    metadata={**metadata, "returncode": None},
                )

            timed_out = False
            try:
                process.communicate(
                    input=None if input_text is None else input_text.encode("utf-8"),
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                self._terminate(process)
                process.communicate()

            runtime = time.monotonic() - started
            output, output_truncated = self._read_output(stdout_file)
            logs, logs_truncated = self._read_output(stderr_file)
            result_metadata = {
                **metadata,
                "returncode": process.returncode,
                "output_truncated": output_truncated,
                "logs_truncated": logs_truncated,
            }
            if timed_out:
                return Observation(
                    status=ExecutionStatus.TIMEOUT,
                    output=output,
                    logs=logs,
                    error_signature="timeout",
                    runtime_seconds=runtime,
                    metadata=result_metadata,
                )
            if process.returncode == 0:
                return Observation(
                    status=ExecutionStatus.SUCCESS,
                    output=output,
                    logs=logs,
                    runtime_seconds=runtime,
                    metadata=result_metadata,
                )
            signature = (
                f"signal:{-process.returncode}"
                if process.returncode is not None and process.returncode < 0
                else f"exit:{process.returncode}"
            )
            return Observation(
                status=ExecutionStatus.ERROR,
                output=output,
                logs=logs,
                error_signature=signature,
                runtime_seconds=runtime,
                metadata=result_metadata,
            )

    def close(self) -> None:
        if self._temporary_workspace is not None:
            self._temporary_workspace.cleanup()
            self._temporary_workspace = None

    def __enter__(self) -> SubprocessSandbox:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def _validate_command(command: Sequence[str]) -> tuple[str, ...]:
        if isinstance(command, (str, bytes)):
            raise SandboxViolation("command must be an argv sequence; shell strings are forbidden")
        if not isinstance(command, Sequence) or not command:
            raise SandboxViolation("command must be a non-empty argv sequence")
        argv = tuple(command)
        if any(not isinstance(argument, str) or not argument or "\x00" in argument for argument in argv):
            raise SandboxViolation("argv entries must be non-empty strings without NUL")
        executable = Path(argv[0]).name.lower()
        if executable in _SHELL_EXECUTABLES:
            raise SandboxViolation("direct shell executables are disabled")
        if executable == "env" and len(argv) > 1 and Path(argv[1]).name.lower() in _SHELL_EXECUTABLES:
            raise SandboxViolation("direct shell executables are disabled")
        return argv

    def _resolve_cwd(self, cwd: str | os.PathLike[str] | None) -> Path:
        candidate = self._workspace if cwd is None else Path(cwd).expanduser()
        if not candidate.is_absolute():
            candidate = self._workspace / candidate
        try:
            resolved = candidate.resolve(strict=True)
        except (FileNotFoundError, NotADirectoryError, OSError) as error:
            raise SandboxViolation("working directory does not resolve to a directory") from error
        try:
            resolved.relative_to(self._workspace)
        except ValueError:
            raise SandboxViolation("working directory escapes sandbox workspace") from None
        if not resolved.is_dir():
            raise SandboxViolation("working directory must be a directory")
        return resolved

    def _build_environment(self, supplied: Mapping[str, str] | None) -> dict[str, str]:
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"}
        if os.name == "nt" and "SYSTEMROOT" in os.environ:  # pragma: no cover
            environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
        if supplied is None:
            return environment
        for key, value in supplied.items():
            if key not in self._allowed_environment:
                raise SandboxViolation(f"environment variable is not allowlisted: {key}")
            if not isinstance(value, str) or "\x00" in value:
                raise SandboxViolation(f"invalid environment value for: {key}")
            environment[key] = value
        return environment

    def _effective_timeout(self, budget: Budget | None, requested: float | None) -> float:
        candidates = [self._limits.default_timeout_seconds]
        if requested is not None:
            if not math.isfinite(requested) or requested < 0:
                raise ValueError("timeout_seconds must be finite and non-negative")
            candidates.append(requested)
        if budget is not None:
            if not isinstance(budget, Budget):
                raise TypeError("budget must be Budget")
            candidates.append(budget.max_exec_seconds)
        return min(candidates)

    def _resource_limiter(self, timeout: float):
        if resource is None:  # pragma: no cover
            return None
        limits = self._limits

        def apply_limits() -> None:
            os.umask(0o077)
            self._set_limit(resource.RLIMIT_CORE, 0)
            self._set_limit(resource.RLIMIT_FSIZE, limits.max_output_bytes)
            # macOS exposes RLIMIT_AS but rejects useful finite values in a
            # pre-exec child on supported Python builds. Apply it only on
            # platforms where the kernel enforces it reliably; the metadata
            # below reports the actual guarantee rather than the request.
            if (
                limits.max_memory_bytes is not None
                and sys.platform.startswith("linux")
                and hasattr(resource, "RLIMIT_AS")
            ):
                self._set_limit(resource.RLIMIT_AS, limits.max_memory_bytes)
            if limits.max_open_files is not None:
                self._set_limit(resource.RLIMIT_NOFILE, limits.max_open_files)
            if limits.max_processes is not None and hasattr(resource, "RLIMIT_NPROC"):
                self._set_limit(resource.RLIMIT_NPROC, limits.max_processes)
            if hasattr(resource, "RLIMIT_CPU"):
                self._set_limit(resource.RLIMIT_CPU, max(1, math.ceil(timeout)))

        return apply_limits

    @staticmethod
    def _set_limit(kind: int, requested: int) -> None:
        if resource is None:  # pragma: no cover
            return
        _, hard = resource.getrlimit(kind)
        effective = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
        resource.setrlimit(kind, (effective, effective))

    @staticmethod
    def _terminate(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:  # pragma: no cover
                process.kill()
        except ProcessLookupError:
            pass

    def _read_output(self, stream) -> tuple[str, bool]:
        stream.seek(0)
        data = stream.read(self._limits.max_output_bytes + 1)
        truncated = len(data) > self._limits.max_output_bytes
        return data[: self._limits.max_output_bytes].decode("utf-8", errors="replace"), truncated

    def _metadata(self, argv: tuple[str, ...], cwd: Path, timeout: float) -> Mapping[str, object]:
        return {
            "isolation": self.isolation_kind,
            "container_isolation": self.provides_container_isolation,
            "filesystem_isolation": self.provides_filesystem_isolation,
            "network_isolation": self.provides_network_isolation,
            "memory_limit_enforced": bool(
                self._limits.max_memory_bytes is not None
                and sys.platform.startswith("linux")
                and resource is not None
                and hasattr(resource, "RLIMIT_AS")
            ),
            "resource_limits": {
                "max_output_bytes": self._limits.max_output_bytes,
                "max_memory_bytes": self._limits.max_memory_bytes,
                "max_open_files": self._limits.max_open_files,
                "max_processes": self._limits.max_processes,
            },
            "security_notice": "process resource limits only; not container isolation",
            "executable": argv[0],
            "cwd": str(cwd),
            "timeout_seconds": timeout,
            "environment_keys": tuple(sorted({"PATH", "LANG", "LC_ALL"} | self._allowed_environment)),
        }
