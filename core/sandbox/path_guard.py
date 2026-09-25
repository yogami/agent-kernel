"""Filesystem path confinement guard blocking directory traversal and escapes outside allowed workspace roots."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import threading
from typing import Any, Iterator


class PathSecurityViolationError(PermissionError):
    """Raised when a tool attempts to access or manipulate paths outside allowed roots."""
    pass


SENSITIVE_SYSTEM_PREFIXES = (
    "/etc",
    "/root",
    "/var/run",
    "/proc",
    "/sys",
    os.path.expanduser("~/.ssh"),
    os.path.expanduser("~/.aws"),
    os.path.expanduser("~/.gnupg"),
)

_tls = threading.local()


def validate_path_confinement(
    target_path: str | Path,
    allowed_roots: list[str | Path],
) -> Path:
    """Ensure target_path resolves strictly within at least one allowed root.

    Raises:
        PathSecurityViolationError: If target_path escapes all allowed roots or touches sensitive system locations.
    """
    path_str = str(target_path).strip()

    # Immediate rejection of obvious traversal payloads
    if ".." in path_str.replace("\\", "/").split("/"):
        escaped = True
        for root in allowed_roots:
            try:
                resolved_root = Path(root).expanduser().resolve()
                resolved_target = (resolved_root / path_str).resolve()
                if resolved_target.is_relative_to(resolved_root):
                    escaped = False
                    break
            except Exception:
                continue

        if escaped:
            raise PathSecurityViolationError(
                f"Security Violation: Path traversal detected in '{path_str}'. Resolved path escapes allowed boundaries."
            )

    try:
        p_obj = Path(path_str).expanduser()
    except Exception as err:
        raise PathSecurityViolationError(
            f"Security Violation: Invalid target path '{path_str}': {err}"
        ) from err

    # Check sensitive system paths
    resolved_target_str = str(p_obj.resolve())
    for sensitive in SENSITIVE_SYSTEM_PREFIXES:
        if resolved_target_str == sensitive or resolved_target_str.startswith(sensitive + "/"):
            raise PathSecurityViolationError(
                f"Security Violation: Direct access to protected system path '{resolved_target_str}' is forbidden."
            )

    # If allowed roots are specified, enforce confinement within at least one root
    if allowed_roots:
        is_contained = False
        resolved_allowed: list[Path] = []
        for r in allowed_roots:
            try:
                rr = Path(r).expanduser().resolve()
                resolved_allowed.append(rr)
                if p_obj.is_absolute():
                    if p_obj.resolve().is_relative_to(rr):
                        is_contained = True
                        break
                else:
                    if (rr / p_obj).resolve().is_relative_to(rr):
                        is_contained = True
                        break
            except Exception:
                continue

        if not is_contained:
            roots_str = ", ".join(str(r) for r in resolved_allowed)
            raise PathSecurityViolationError(
                f"Security Violation: Path '{resolved_target_str}' is outside permitted workspace roots: [{roots_str}]."
            )

    return p_obj.resolve()


def inspect_arguments_for_path_violations(
    arguments: dict[str, Any],
    allowed_roots: list[str | Path] | None = None,
) -> None:
    """Scan tool argument dictionary for path traversal payloads and unauthorized paths."""
    path_keys = {
        "path",
        "file_path",
        "filepath",
        "filename",
        "dir",
        "directory",
        "dest",
        "source",
        "src",
        "target",
        "output_path",
    }

    effective_roots = allowed_roots or []

    for key, val in arguments.items():
        if isinstance(val, str):
            val_lower = val.lower().strip()
            is_path_key = key.lower() in path_keys
            looks_like_path = (
                val.startswith(("/", "./", "../", "~"))
                or "/etc/" in val_lower
                or ".ssh" in val_lower
                or ".." in val
            )

            if is_path_key or looks_like_path:
                validate_path_confinement(val, effective_roots)


@contextmanager
def filesystem_path_confinement(allowed_roots: list[str | Path]) -> Iterator[None]:
    """Context manager setting active thread-local allowed roots."""
    prev_roots = getattr(_tls, "allowed_roots", None)
    _tls.allowed_roots = allowed_roots
    try:
        yield
    finally:
        _tls.allowed_roots = prev_roots
