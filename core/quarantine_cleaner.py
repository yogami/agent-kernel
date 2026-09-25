"""Quarantine garbage collection and retention policy enforcement."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from domain.models import AdmissionStatus, QuarantineRetentionPolicy
from domain.ports import FactStorePort


class QuarantineCleaner:
    """Automated maintenance worker for purging expired records from the quarantine table."""

    def __init__(
        self,
        fact_store: FactStorePort,
        default_policy: QuarantineRetentionPolicy | None = None,
    ) -> None:
        self.fact_store = fact_store
        self.default_policy = default_policy or QuarantineRetentionPolicy()

    def run_purge_cycle(
        self,
        policy: QuarantineRetentionPolicy | None = None,
        tenant_id: str = "default_tenant",
    ) -> dict[str, Any]:
        """Execute a single garbage collection cycle scoped by tenant."""
        active_policy = policy or self.default_policy
        cutoff_time = datetime.now(timezone.utc) - timedelta(days=active_policy.max_age_days)

        purged_count = self.fact_store.purge_quarantine_records(
            older_than=cutoff_time,
            statuses=active_policy.statuses_to_purge,
            tenant_id=tenant_id,
        )

        return {
            "tenant_id": tenant_id,
            "purged_count": purged_count,
            "cutoff_time": cutoff_time.isoformat(),
            "statuses_purged": [s.value for s in active_policy.statuses_to_purge],
            "max_age_days": active_policy.max_age_days,
        }
