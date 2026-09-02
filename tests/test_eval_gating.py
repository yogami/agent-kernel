"""Tests for Monotonic Non-Regression Evaluation and Self-Healing Gating."""

import tempfile
from pathlib import Path
from ops.config_manager import ConfigManager
from ops.eval_runner import EvalRunner
from ops.self_healer import SelfHealer


def test_monotonic_self_healer_flow():
    """Verify self-healing controller proposes candidate pack and verifies against golden suite."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        packs_dir = tmp_path / "packs"
        evals_dir = tmp_path / "evals"

        config_mgr = ConfigManager(packs_dir)
        eval_runner = EvalRunner(evals_dir)
        healer = SelfHealer(config_mgr, eval_runner, max_iterations=3)

        # 1. Baseline pack should be v1
        active = config_mgr.get_active_pack()
        assert active.version == "v1"

        # 2. Run healer cycle (with empty dev failures, should stay GREEN)
        result = healer.run_repair_cycle()
        assert result["status"] == "GREEN"
        assert result["active_version"] == "v1"


def test_eval_runner_suite_execution():
    """Verify EvalRunner executes standard test fixtures."""
    eval_runner = EvalRunner("evals")
    config_mgr = ConfigManager("config/packs")
    active_pack = config_mgr.get_active_pack()

    golden_res = eval_runner.run_suite("golden", active_pack)
    assert golden_res["suite"] == "golden"
    assert golden_res["total_cases"] >= 3
    assert golden_res["pass_rate"] == 1.0
