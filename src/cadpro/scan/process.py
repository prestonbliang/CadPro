"""Bounded and cancellable external-process execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import threading
import time
from typing import Callable, Sequence, TypeVar


class ProcessCancelled(RuntimeError):
    """Raised after a user cancellation terminates a child process."""


class ExternalProcessError(RuntimeError):
    """Raised when a native reconstruction tool returns a failure status."""

    def __init__(self, message: str, *, return_code: int | None, tail: str) -> None:
        super().__init__(message)
        self.return_code = return_code
        self.tail = tail


class CancellationToken:
    def __init__(self, probe: Callable[[], bool] | None = None) -> None:
        self._event = threading.Event()
        self._probe = probe

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set() or bool(self._probe and self._probe())

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise ProcessCancelled("The reconstruction job was cancelled.")


@dataclass(frozen=True)
class ProcessResult:
    arguments: tuple[str, ...]
    return_code: int
    duration_seconds: float
    log_path: Path
    output_tail: str


@dataclass(frozen=True)
class CapturedProcessResult:
    process: ProcessResult
    stdout: bytes


def run_process(
    arguments: Sequence[str | Path],
    *,
    working_directory: str | Path,
    log_path: str | Path,
    timeout_seconds: float,
    cancellation: CancellationToken,
) -> ProcessResult:
    """Run one trusted argument array with disk-backed output and hard bounds."""

    if not arguments:
        raise ValueError("arguments cannot be empty")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    normalized = tuple(str(argument) for argument in arguments)
    if any("\x00" in argument for argument in normalized):
        raise ValueError("process arguments cannot contain NUL bytes")
    cwd = Path(working_directory).resolve(strict=True)
    destination = Path(log_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cancellation.raise_if_cancelled()
    started = time.monotonic()
    with destination.open("a", encoding="utf-8", errors="replace") as log:
        log.write("$ " + " ".join(_display_argument(value) for value in normalized) + "\n")
        log.flush()
        try:
            process = subprocess.Popen(
                normalized,
                cwd=cwd,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                shell=False,
                text=True,
            )
        except OSError as error:
            raise ExternalProcessError(
                f"Could not start {Path(normalized[0]).name}: {type(error).__name__}.",
                return_code=None,
                tail="",
            ) from error
        while process.poll() is None:
            if cancellation.cancelled:
                _stop_process(process)
                raise ProcessCancelled(f"Cancelled {Path(normalized[0]).name}.")
            if time.monotonic() - started > timeout_seconds:
                _stop_process(process)
                tail = _read_tail(destination)
                raise ExternalProcessError(
                    f"{Path(normalized[0]).name} exceeded the {timeout_seconds:g}-second timeout.",
                    return_code=process.returncode,
                    tail=tail,
                )
            time.sleep(0.1)
        return_code = int(process.returncode or 0)
    duration = time.monotonic() - started
    tail = _read_tail(destination)
    if return_code != 0:
        raise ExternalProcessError(
            f"{Path(normalized[0]).name} exited with code {return_code}.",
            return_code=return_code,
            tail=tail,
        )
    return ProcessResult(normalized, return_code, duration, destination, tail)


def run_process_capture(
    arguments: Sequence[str | Path],
    *,
    working_directory: str | Path,
    log_path: str | Path,
    timeout_seconds: float,
    cancellation: CancellationToken,
    maximum_stdout_bytes: int = 2 * 1024 * 1024,
) -> CapturedProcessResult:
    """Capture a bounded machine-readable stdout while logging stderr to disk."""

    if maximum_stdout_bytes <= 0:
        raise ValueError("maximum_stdout_bytes must be positive")
    destination = Path(log_path)
    capture_path = destination.with_suffix(destination.suffix + ".stdout")
    normalized = tuple(str(argument) for argument in arguments)
    if not normalized or any("\x00" in argument for argument in normalized):
        raise ValueError("process arguments must be a non-empty NUL-free sequence")
    cwd = Path(working_directory).resolve(strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cancellation.raise_if_cancelled()
    started = time.monotonic()
    try:
        with (
            destination.open("a", encoding="utf-8", errors="replace") as log,
            capture_path.open("wb") as capture,
        ):
            log.write("$ " + " ".join(_display_argument(value) for value in normalized) + "\n")
            log.flush()
            try:
                process = subprocess.Popen(
                    normalized,
                    cwd=cwd,
                    stdin=subprocess.DEVNULL,
                    stdout=capture,
                    stderr=log,
                    shell=False,
                )
            except OSError as error:
                raise ExternalProcessError(
                    f"Could not start {Path(normalized[0]).name}: {type(error).__name__}.",
                    return_code=None,
                    tail="",
                ) from error
            while process.poll() is None:
                if cancellation.cancelled:
                    _stop_process(process)
                    raise ProcessCancelled(f"Cancelled {Path(normalized[0]).name}.")
                if time.monotonic() - started > timeout_seconds:
                    _stop_process(process)
                    raise ExternalProcessError(
                        f"{Path(normalized[0]).name} exceeded the "
                        f"{timeout_seconds:g}-second timeout.",
                        return_code=process.returncode,
                        tail=_read_tail(destination),
                    )
                if capture.tell() > maximum_stdout_bytes:
                    _stop_process(process)
                    raise ExternalProcessError(
                        f"{Path(normalized[0]).name} exceeded its output limit.",
                        return_code=process.returncode,
                        tail=_read_tail(destination),
                    )
                time.sleep(0.1)
            return_code = int(process.returncode or 0)
        duration = time.monotonic() - started
        tail = _read_tail(destination)
        payload = capture_path.read_bytes()
        if len(payload) > maximum_stdout_bytes:
            raise ExternalProcessError(
                f"{Path(normalized[0]).name} exceeded its output limit.",
                return_code=return_code,
                tail=tail,
            )
        if return_code != 0:
            raise ExternalProcessError(
                f"{Path(normalized[0]).name} exited with code {return_code}.",
                return_code=return_code,
                tail=tail,
            )
        result = ProcessResult(normalized, return_code, duration, destination, tail)
        return CapturedProcessResult(result, payload)
    finally:
        capture_path.unlink(missing_ok=True)


_StreamType = TypeVar("_StreamType", str, bytes)


def _stop_process(process: subprocess.Popen[_StreamType]) -> None:
    try:
        process.terminate()
        process.wait(timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        try:
            process.kill()
            process.wait(timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            pass


def _read_tail(path: Path, maximum_bytes: int = 16 * 1024) -> str:
    try:
        size = path.stat().st_size
        with path.open("rb") as stream:
            stream.seek(max(0, size - maximum_bytes))
            return stream.read(maximum_bytes).decode("utf-8", errors="replace")
    except OSError:
        return ""


def _display_argument(value: str) -> str:
    # This is logging only; subprocess receives the original argument array.
    if any(character.isspace() for character in value):
        return '"' + value.replace('"', "'") + '"'
    return value
