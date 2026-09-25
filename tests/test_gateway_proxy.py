"""Unit and integration tests for Enterprise Gateway proxy endpoints (v1)."""

from __future__ import annotations

import json
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient

from api.app import app
from infrastructure.audit_logger import audit_logger


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, headers={"X-API-Key": "dev-secret-key"})


def test_v1_memory_admit_valid_fact(client: TestClient) -> None:
    """Verify novel, non-contradictory fact is admitted through the v1 proxy."""
    tenant = f"tenant_valid_{uuid.uuid4().hex[:8]}"
    payload = {
        "tenant_id": tenant,
        "session_id": "session_1",
        "subject": "patient_001",
        "predicate": "allergy",
        "object": "penicillin",
        "confidence": 0.95,
    }
    resp = client.post("/v1/memory/admit", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["admitted"] is True
    assert data["status"] == "PROMOTED"
    assert data["quarantined"] is False
    assert "candidate_id" in data


def test_v1_memory_admit_contradiction_quarantined(client: TestClient) -> None:
    """Verify contradictory fact is quarantined and an audit event is recorded."""
    tenant = f"tenant_contra_{uuid.uuid4().hex[:8]}"
    # First admit baseline fact
    first_resp = client.post(
        "/v1/memory/admit",
        json={
            "tenant_id": tenant,
            "session_id": "session_contra",
            "subject": "patient_002",
            "predicate": "blood_pressure",
            "object": "normotensive",
            "confidence": 0.95,
        },
    )
    assert first_resp.status_code == 200
    assert first_resp.json()["admitted"] is True

    # Now attempt contradictory fact
    second_resp = client.post(
        "/v1/memory/admit",
        json={
            "tenant_id": tenant,
            "session_id": "session_contra",
            "subject": "patient_002",
            "predicate": "blood_pressure",
            "object": "hypertensive",
            "confidence": 0.95,
        },
    )
    assert second_resp.status_code == 200
    contra_data = second_resp.json()
    assert contra_data["admitted"] is False
    assert contra_data["status"] == "QUARANTINED"
    assert contra_data["quarantined"] is True

    # Check audit log
    events = audit_logger.get_events(tenant_id=tenant, event_type="memory.contradiction_quarantined")
    assert len(events) >= 1
    assert events[0].severity.value == "WARNING"
    assert events[0].outcome == "QUARANTINE"


def test_v1_causal_verify_valid_and_falsified(client: TestClient) -> None:
    """Verify pre-flight causal verification on valid vs corrupted SCMs."""
    tenant = f"tenant_causal_{uuid.uuid4().hex[:8]}"
    # 1. Valid SCM: cluster_autoscaling
    valid_resp = client.post(
        "/v1/causal/verify",
        json={
            "tenant_id": tenant,
            "scm_id": "cluster_autoscaling",
            "proposed_intervention": "replica_count",
            "target_outcome": "p99_latency",
        },
    )
    assert valid_resp.status_code == 200
    valid_data = valid_resp.json()
    assert valid_data["verified"] is True
    assert valid_data["status"] == "VERIFIED"
    assert len(valid_data["violations"]) == 0

    # 2. Corrupted SCM: network_latency_routing
    corrupt_resp = client.post(
        "/v1/causal/verify",
        json={
            "tenant_id": tenant,
            "scm_id": "network_latency_routing",
            "proposed_intervention": "routing_weight",
            "target_outcome": "rtt_latency",
        },
    )
    assert corrupt_resp.status_code == 200
    corrupt_data = corrupt_resp.json()
    assert corrupt_data["verified"] is False
    assert corrupt_data["status"] == "FALSIFIED"
    assert len(corrupt_data["violations"]) > 0

    # Verify audit event
    causal_events = audit_logger.get_events(tenant_id=tenant, event_type="causal.falsification_blocked")
    assert len(causal_events) >= 1
    assert causal_events[0].severity.value == "CRITICAL"


def test_v1_tools_execute_benign_and_security_guards(client: TestClient) -> None:
    """Verify tool execution proxy allows benign calls and intercepts exploits."""
    tenant = f"tenant_tool_sec_{uuid.uuid4().hex[:8]}"

    # 1. Benign execution: calculate
    benign_resp = client.post(
        "/v1/tools/execute",
        json={
            "tenant_id": tenant,
            "tool_name": "calculate",
            "arguments": {"expression": "25 * 4"},
        },
    )
    assert benign_resp.status_code == 200
    b_data = benign_resp.json()
    assert b_data["success"] is True
    assert b_data["output"]["result"] == 100
    assert b_data["security_blocked"] is False

    # 2. Command injection attempt
    injection_resp = client.post(
        "/v1/tools/execute",
        json={
            "tenant_id": tenant,
            "tool_name": "calculate",
            "arguments": {"expression": "10; cat /etc/passwd"},
        },
    )
    assert injection_resp.status_code == 200
    inj_data = injection_resp.json()
    assert inj_data["success"] is False
    assert inj_data["security_blocked"] is True
    assert "Command injection" in inj_data["error"]

    # 3. Path traversal attempt
    traversal_resp = client.post(
        "/v1/tools/execute",
        json={
            "tenant_id": tenant,
            "tool_name": "calculate",
            "arguments": {"expression": "../../../secret.key"},
        },
    )
    assert traversal_resp.status_code == 200
    trav_data = traversal_resp.json()
    assert trav_data["success"] is False
    assert trav_data["security_blocked"] is True
    assert "Path traversal" in trav_data["error"]


def test_v1_tenant_policy_crud(client: TestClient) -> None:
    """Verify tenant security policy read and update workflow."""
    tenant = "tenant_policy_demo"

    # Read default policy
    get_resp = client.get(f"/v1/policies/{tenant}")
    assert get_resp.status_code == 200
    default_pol = get_resp.json()
    assert default_pol["tenant_id"] == tenant
    assert default_pol["quarantine_retention_days"] == 30
    assert default_pol["rate_limit_rpm"] == 120

    # Update policy
    update_resp = client.put(
        f"/v1/policies/{tenant}",
        json={
            "quarantine_retention_days": 45,
            "rate_limit_rpm": 250,
            "allowed_egress_domains": ["api.hospital.org"],
        },
    )
    assert update_resp.status_code == 200
    updated_pol = update_resp.json()
    assert updated_pol["quarantine_retention_days"] == 45
    assert updated_pol["rate_limit_rpm"] == 250
    assert updated_pol["allowed_egress_domains"] == ["api.hospital.org"]

    # Re-verify through GET
    verify_resp = client.get(f"/v1/policies/{tenant}")
    assert verify_resp.json()["quarantine_retention_days"] == 45


def test_v1_audit_logs_cef_and_json_export(client: TestClient) -> None:
    """Verify audit log streaming in JSON and ArcSight CEF formats."""
    # JSON export
    json_resp = client.get("/v1/audit/logs?format=json&limit=10")
    assert json_resp.status_code == 200
    events = json_resp.json()
    assert isinstance(events, list)
    if events:
        assert "event_type" in events[0]
        assert "tenant_id" in events[0]

    # CEF export
    cef_resp = client.get("/v1/audit/logs?format=cef&limit=10")
    assert cef_resp.status_code == 200
    cef_text = cef_resp.text
    assert isinstance(cef_text, str)
    if cef_text.strip():
        first_line = cef_text.strip().split("\n")[0]
        assert first_line.startswith("CEF:0|AgentKernel|SecurityGateway|1.0|")
