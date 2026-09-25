import pytest
import os
import json
from unittest.mock import patch

from synthesize_enterprise_matrix import synthesize_distractor_marathon, synthesize_stale_state
from synthesize_frontier_breaker_matrix import synthesize_livemedbench, synthesize_mpib_v2, synthesize_argument_phi


def verify_fixtures(filepath, expected_key, expected_count):
    assert os.path.exists(filepath), f"{filepath} was not created"
    
    with open(filepath, "r") as f:
        data = json.load(f)
        
    assert expected_key in data, f"Key '{expected_key}' missing from JSON"
    matrix = data[expected_key]
    assert len(matrix) == expected_count, f"Expected {expected_count} cases, got {len(matrix)}"
    
    # Check variance
    patient_ids = set()
    allergies_seen = set()
    prompts_seen = set()
    
    for fixture in matrix:
        # Verify structure
        assert "patient_id" in fixture
        assert "gold_state" in fixture
        assert "read_fixtures" in fixture
        assert "user_prompt" in fixture
        
        patient_ids.add(fixture["patient_id"])
        
        allergies = tuple(fixture["gold_state"].get("allergies", []))
        allergies_seen.add(allergies)
        
        prompts_seen.add(fixture["user_prompt"])
        
    # Real variance checking
    assert len(patient_ids) == expected_count, "Patient IDs must be unique"
    # To pass variance, we need more than 1 distinct allergy or prompt
    assert len(allergies_seen) > 1 or len(prompts_seen) > 1, f"No real variance in {expected_key}"

class TestSynthesisScripts:
    
    @pytest.fixture(autouse=True)
    def change_test_dir(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        
    def test_synthesize_distractor_marathon(self):
        synthesize_distractor_marathon()
        verify_fixtures("holdout_a_distractor.json", "Distractor Marathon", 100)
        
    def test_synthesize_stale_state(self):
        synthesize_stale_state()
        verify_fixtures("holdout_a_stalestate.json", "Stale State", 100)
        
    def test_synthesize_livemedbench(self):
        synthesize_livemedbench()
        verify_fixtures("holdout_a_livemedbench.json", "Context Miss & Conflict", 100)
        
    def test_synthesize_mpib_v2(self):
        synthesize_mpib_v2()
        verify_fixtures("holdout_a_mpib2.json", "RAG-Mediated Injection", 100)
        
    def test_synthesize_argument_phi(self):
        synthesize_argument_phi()
        verify_fixtures("holdout_a_argument_phi.json", "Argument-Level PHI", 100)

    def test_synthesize_clean_control(self):
        from synthesize_enterprise_matrix import synthesize_clean_control
        synthesize_clean_control()
        verify_fixtures("holdout_control_clean.json", "Clean Control", 50)

