"""Command safety and shell injection defense guard."""

from __future__ import annotations

import re
import shlex
from typing import Any


class CommandSecurityViolationError(PermissionError):
    """Raised when a tool argument attempts shell command injection or unauthorized execution."""
    pass


# Disallowed metacharacters and tokens indicating command chaining or execution escapes
DANGEROUS_PATTERNS = [
    (r";", "Command chaining semicolon (;)"),
    (r"&&", "Conditional AND chaining (&&)"),
    (r"\|\|", "Conditional OR chaining (||)"),
    (r"\|", "Pipe redirection (|)"),
    (r"`", "Backtick command substitution (`)"),
    (r"\$\(", "Subshell command substitution ($())"),
    (r">\s*/dev/tcp", "Direct socket redirection (> /dev/tcp)"),
    (r">\s*/dev/udp", "Direct socket redirection (> /dev/udp)"),
    (r"\brm\s+-rf\b", "Destructive recursive deletion (rm -rf)"),
    (r"\bsudo\b", "Privilege escalation attempt (sudo)"),
    (r"\beval\b", "Arbitrary code evaluation (eval)"),
]


def validate_command_safety(
    command: str,
    allow_shell: bool = False,
    allowed_binaries: list[str] | None = None,
) -> None:
    """Validate a command string against injection patterns and binary allowlists.

    Raises:
        CommandSecurityViolationError: If shell metacharacters or unauthorized binaries are detected.
    """
    cmd_str = command.strip()
    if not cmd_str:
        return

    if not allow_shell:
        for pattern, desc in DANGEROUS_PATTERNS:
            if re.search(pattern, cmd_str, flags=re.IGNORECASE):
                raise CommandSecurityViolationError(
                    f"Security Violation: Command injection pattern detected: {desc} in '{cmd_str}'."
                )

    if allowed_binaries is not None:
        try:
            tokens = shlex.split(cmd_str)
        except ValueError as err:
            raise CommandSecurityViolationError(
                f"Security Violation: Malformed command string: {err}"
            ) from err

        if not tokens:
            return

        bin_name = tokens[0].split("/")[-1]
        allowed_names = {b.split("/")[-1] for b in allowed_binaries}
        if bin_name not in allowed_names:
            raise CommandSecurityViolationError(
                f"Security Violation: Binary '{bin_name}' is not in allowed list: {sorted(allowed_names)}."
            )


def inspect_arguments_for_command_injection(
    arguments: dict[str, Any],
    allow_shell: bool = False,
    allowed_binaries: list[str] | None = None,
) -> None:
    """Scan tool argument dictionary for command injection attempts."""
    cmd_keys = {"command", "cmd", "script", "shell_cmd", "exec_cmd", "query_cmd"}

    for key, val in arguments.items():
        if isinstance(val, str):
            is_cmd_key = key.lower() in cmd_keys
            # Also check if any string value contains explicit injection patterns
            has_chaining = any(tok in val for tok in [";", "&&", "||", "|", "`", "$("])

            if is_cmd_key or has_chaining:
                validate_command_safety(
                    command=val,
                    allow_shell=allow_shell,
                    allowed_binaries=allowed_binaries,
                )
