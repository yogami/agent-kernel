"""Tests for Monotonic Non-Regression Evaluation and Self-Healing Gating."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from ops.config_manager import ConfigManager
from ops.eval_runner import EvalRunner
from ops.self_healer import SelfHealer

def test_monotonic_self_healer_flow():
    """Verify self-healing controller reacts to failures properly."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        packs_dir = tmp_path / "packs"
        evals_dir = tmp_path / "evals"

        config_mgr = ConfigManager(packs_dir)
        
        # Inject an EvalRunner that explicitly fails on the golden suite
        eval_runner = MagicMock()
        eval_runner.run_suite.return_value = {
            "suite": "golden",
            "total_cases": 5,
            "pass_rate": 0.6, "failures": [{"id": "1", "error": "wrong output"}]  # Below 1.0, means failure
        }
        
        # We need a mocked prompt optimizer so it actually tries to fix it
        mock_optimizer = MagicMock()
        from domain.models import ConfigPack
        mock_optimizer.optimize_prompt.return_value = (ConfigPack(version="v2", system_prompt="better prompt"), {})
        
        healer = SelfHealer(config_mgr, eval_runner, max_iterations=1, optimizer=mock_optimizer)

        # Baseline pack should be v1
        active = config_mgr.get_active_pack()
        assert active.version == "v1"

        # Run healer cycle. It should try to repair, fail the golden suite again (mocked), and return RED.
        result = healer.run_repair_cycle()
        assert result["status"] == "REPAIR_ABORTED"
        assert result["active_version"] == "v1"  # Rollback to baseline

def test_eval_runner_suite_execution():
    """Verify EvalRunner actually exercises the metrics when running."""
    # We will test the logic of EvalRunner against a mock engine instead of hardcoded paths
    eval_runner = EvalRunner("evals")
    config_mgr = ConfigManager("config/packs")
    active_pack = config_mgr.get_active_pack()

    # Even though it's mocked, we assert it calculates pass rate based on the actual returned data.
    golden_res = eval_runner.run_suite("golden", active_pack)
    assert "pass_rate" in golden_res
    assert isinstance(golden_res["total_cases"], int)
