import os
import pytest
from fastapi.testclient import TestClient
from api.app import app
from infrastructure.postgres_store import PostgresEpisodeStore

@pytest.fixture(scope="module")
def client():
    # If TEST_DATABASE_URL is set, we could override the dependency
    test_db_url = os.getenv("TEST_DATABASE_URL")
    if test_db_url:
        app.state.episode_store = PostgresEpisodeStore(test_db_url)
    with TestClient(app, headers={"X-API-Key": "dev-secret-key"}) as c:
        yield c

def test_e2e_basic_chat(client):
    response = client.post("/api/chat", json={
        "user_input": "Hello",
        "session_id": "e2e_session_real_1",
        "tenant_id": "default_tenant"
    })
    assert response.status_code == 200

def test_e2e_tool_call_deidentify(client):
    response = client.post("/api/chat", json={
        "user_input": "deidentify",
        "session_id": "e2e_session_real_2",
        "tenant_id": "default_tenant"
    })
    assert response.status_code == 200

def test_e2e_memory_trace_patient_001(client):
    response = client.post("/api/chat", json={
        "user_input": "patient_001",
        "session_id": "e2e_session_real_3",
        "tenant_id": "default_tenant"
    })
    assert response.status_code == 200

def test_e2e_multi_turn_session(client):
    res1 = client.post("/api/chat", json={
        "user_input": "First message",
        "session_id": "e2e_session_real_4",
    })
    assert res1.status_code == 200
    res2 = client.post("/api/chat", json={
        "user_input": "Second message",
        "session_id": "e2e_session_real_4",
    })
    assert res2.status_code == 200
