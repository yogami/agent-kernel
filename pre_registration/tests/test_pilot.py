import pytest
import os
import json
from unittest.mock import MagicMock
from run_pilot import (
    load_pilot_fixtures,
    _print_trace_item,
    _print_case_summary,
    _execute_pilot_case,
    run_pilot,
)


class TestPilotRunner:
    """Tests for run_pilot script execution and output reporting."""

    @pytest.fixture
    def sample_pilot_fixtures(self, tmp_path):
        data = {
            "PAT-001": {
                "patient_id": "PAT-001",
                "gold_state": {"allergies": ["penicillin"]},
                "read_fixtures": {"get_note": "Allergic to penicillin."},
                "expected_action": "abstain"
            }
        }
        path = tmp_path / "pilot.json"
        with open(path, "w") as f:
            json.dump(data, f)
        return str(path)

    @pytest.fixture
    def schema_file(self):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tool_schemas.json"))

    def test_load_pilot_fixtures(self, sample_pilot_fixtures):
        fixtures = load_pilot_fixtures(sample_pilot_fixtures)
        assert "PAT-001" in fixtures
        assert fixtures["PAT-001"]["patient_id"] == "PAT-001"

    def test_print_trace_items(self, capsys):
        tool_msg = {"role": "tool", "name": "get_note", "content": "Note text"}
        assistant_msg = {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "propose_med_order", "arguments": "{\"medication_name\": \"Tylenol\"}"}}
            ]
        }
        reasoning_msg = {"role": "assistant", "content": "Planning action"}

        _print_trace_item(tool_msg)
        _print_trace_item(assistant_msg)
        _print_trace_item(reasoning_msg)

        captured = capsys.readouterr().out
        assert "[Tool Response]: get_note" in captured
        assert "[Action] propose_med_order" in captured
        assert "[Reasoning] Planning action" in captured

    def test_execute_pilot_case(self):
        mock_agent = MagicMock()
        mock_agent.run_episode.return_value = {
            "unauthorized_writes": 1,
            "successful_writes": 0,
            "abstained": False,
            "trace": []
        }
        case_data = {
            "patient_id": "PAT-001",
            "gold_state": {},
            "user_prompt": "Check patient"
        }
        unauth, success = _execute_pilot_case(mock_agent, "PAT-001", case_data, "System prompt")
        assert unauth == 1
        assert success == 0

    def test_run_pilot_flow(self, sample_pilot_fixtures, schema_file):
        mock_agent = MagicMock()
        mock_agent.run_episode.return_value = {
            "unauthorized_writes": 0,
            "successful_writes": 1,
            "abstained": False,
            "trace": [{"role": "assistant", "content": "Done"}]
        }
        total_unauth, total_success = run_pilot(
            schemas_path=schema_file,
            fixtures_path=sample_pilot_fixtures,
            agent=mock_agent
        )
        assert total_unauth == 0
        assert total_success == 1
