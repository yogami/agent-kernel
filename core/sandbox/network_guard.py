"""Network egress firewall intercepting unauthorized socket connections via thread-local gating."""

from __future__ import annotations

from contextlib import contextmanager
import socket
import threading
from typing import Any, Iterator


class NetworkSecurityViolationError(PermissionError):
    """Raised when a sandboxed tool attempts unauthorized network socket connections."""
    pass


_tls = threading.local()
_SYSTEM_SOCKET_CONNECT = socket.socket.connect
_SYSTEM_CREATE_CONNECTION = getattr(socket, "create_connection", None)
_HOOKS_INSTALLED = False


def _guarded_connect(self: socket.socket, *args: Any, **kwargs: Any) -> Any:
    if getattr(_tls, "block_network", False):
        raise NetworkSecurityViolationError(
            "Security Violation: Outbound network egress is blocked for this tool runtime."
        )
    return _SYSTEM_SOCKET_CONNECT(self, *args, **kwargs)


def _guarded_create_connection(*args: Any, **kwargs: Any) -> Any:
    if getattr(_tls, "block_network", False):
        raise NetworkSecurityViolationError(
            "Security Violation: Outbound network connection is blocked for this tool runtime."
        )
    if _SYSTEM_CREATE_CONNECTION is not None:
        return _SYSTEM_CREATE_CONNECTION(*args, **kwargs)


def _ensure_hooks_installed() -> None:
    global _HOOKS_INSTALLED
    if not _HOOKS_INSTALLED:
        socket.socket.connect = _guarded_connect  # type: ignore[assignment]
        if _SYSTEM_CREATE_CONNECTION is not None:
            socket.create_connection = _guarded_create_connection  # type: ignore[assignment]
        _HOOKS_INSTALLED = True


@contextmanager
def network_egress_firewall(allow_network: bool = False) -> Iterator[None]:
    """Trap and block socket connection attempts on current thread if allow_network is False."""
    _ensure_hooks_installed()
    prev = getattr(_tls, "block_network", False)
    _tls.block_network = not allow_network
    try:
        yield
    finally:
        _tls.block_network = prev
