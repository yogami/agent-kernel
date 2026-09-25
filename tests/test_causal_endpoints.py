import os
import pytest
from fastapi.testclient import TestClient
from api.app import app

AUTH_HEADERS = {"X-API-Key": os.getenv("AGENT_KERNEL_API_KEY", "dev-secret-key")}
client = TestClient(app)


def test_causal_simulation_requires_auth():
    payload = {
        "intervention": {"drug_prescription": "amoxicillin", "dosage_mg": 500},
        "patient_context": {"penicillin_allergy": True}
    }
    resp = client.post("/v1/ablation/causal/simulate", json=payload)
    assert resp.status_code in (401, 403)


def test_get_causal_graph():
    resp = client.get("/v1/ablation/causal/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert data["scm_name"] == "clinical_pharmacotherapy"
    assert len(data["nodes"]) >= 10
    assert len(data["edges"]) >= 8


def test_simulate_causal_intervention_allergic():
    payload = {
        "intervention": {"drug_prescription": "amoxicillin", "dosage_mg": 500},
        "patient_context": {"penicillin_allergy": True, "baseline_gfr": 75.0}
    }
    resp = client.post("/v1/ablation/causal/simulate", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_safe"] is False
    assert any("ANAPHYLAXIS" in w for w in data["warnings"])


def test_simulate_causal_intervention_hemorrhage():
    payload = {
        "intervention": {"drug_prescription": "warfarin", "dosage_mg": 5},
        "patient_context": {"active_bleeding": True}
    }
    resp = client.post("/v1/ablation/causal/simulate", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_safe"] is False
    assert any("HEMORRHAGE" in w for w in data["warnings"])


def test_simulate_causal_intervention_aki():
    payload = {
        "intervention": {"drug_prescription": "ramipril", "dosage_mg": 50},
        "patient_context": {"baseline_gfr": 20.0}
    }
    resp = client.post("/v1/ablation/causal/simulate", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["is_safe"] is False
    assert any("KIDNEY" in w for w in data["warnings"])


def test_counterfactual_attribution():
    payload = {
        "factual_evidence": {"drug_prescription": "amoxicillin", "penicillin_allergy": True},
        "hypothetical_action": {"drug_prescription": "ramipril"},
        "observed_bad_outcome": "anaphylaxis_reaction"
    }
    resp = client.post("/v1/ablation/causal/counterfactual", json=payload, headers=AUTH_HEADERS)
    assert resp.status_code == 200
    data = resp.json()
    assert data["outcome_prevented"] is True
    assert "Counterfactual Proof" in data["explanation"]
