"""Tenant Security Policy Manager.

Manages multi-tenant configuration for network egress allowlists, tool capability
permissions, path confinement, and causal verification strictness.
"""

from __future__ import annotations

import threading
from typing import Any

from domain.models import TenantSecurityPolicy, ToolCapability


class TenantPolicyStore:
    """Thread-safe store for tenant security policies with sensible defaults."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._policies: dict[str, TenantSecurityPolicy] = {}

    def get_policy(self, tenant_id: str) -> TenantSecurityPolicy:
        """Retrieve policy for tenant, or initialize default secure policy."""
        with self._lock:
            if tenant_id not in self._policies:
                default_policy = TenantSecurityPolicy(
                    tenant_id=tenant_id,
                    allowed_egress_domains=["*"],
                    allowed_tool_capabilities=ToolCapability(
                        requires_network=True,
                        requires_filesystem=True,
                        timeout_seconds=10.0,
                        max_memory_mb=512,
                        allowed_paths=[],
                        allow_subprocesses=False,
                        allow_shell=False,
                    ),
                    quarantine_retention_days=30,
                    command_injection_guard_enabled=True,
                    path_confinement_enabled=True,
                    causal_verification_strictness="STRICT",
                    rate_limit_rpm=120,
                )
                self._policies[tenant_id] = default_policy
            return self._policies[tenant_id]

    def set_policy(self, policy: TenantSecurityPolicy) -> None:
        """Store or update security policy for a tenant."""
        with self._lock:
            self._policies[policy.tenant_id] = policy

    def list_policies(self) -> list[TenantSecurityPolicy]:
        """List all configured tenant policies."""
        with self._lock:
            return list(self._policies.values())

    def is_domain_allowed(self, tenant_id: str, domain: str) -> bool:
        """Check if an outbound network domain is permitted under tenant policy."""
        policy = self.get_policy(tenant_id)
        if not policy.allowed_egress_domains:
            return False
        if "*" in policy.allowed_egress_domains:
            return True
        clean_domain = domain.lower().strip()
        for allowed in policy.allowed_egress_domains:
            clean_allowed = allowed.lower().strip()
            if clean_domain == clean_allowed:
                return True
            if clean_allowed.startswith("*.") and clean_domain.endswith(clean_allowed[1:]):
                return True
        return False
