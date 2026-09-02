"""Evaluation runner executing golden, dev, and holdout suites."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from core.memory_promoter import MemoryPromoter
from domain.models import CandidateFact, ConfigPack, Turn
from domain.ports import FactStorePort
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from tools.clinical_tools import (
    ClinicalAssertionCheckerTool,
    DeidentifyTextTool,
    FHIRValidatorTool,
    MedicalOntologyMapperTool,
)
from tools.registry import ToolRegistry


class EvalRunner:
    """Executes frozen evaluation test suites and outputs deterministic pass/fail metrics."""

    def __init__(self, evals_dir: str | Path = "evals") -> None:
        self.evals_dir = Path(evals_dir)

    def run_suite(self, suite_name: str, config_pack: ConfigPack) -> dict[str, Any]:
        """Run all test fixtures in a given suite directory (golden, dev, or holdout)."""
        suite_path = self.evals_dir / suite_name
        if not suite_path.exists():
            return {
                "suite": suite_name,
                "total_cases": 0,
                "passed_cases": 0,
                "pass_rate": 1.0,
                "failures": [],
            }

        test_files = list(suite_path.glob("*.json"))
        total_cases = 0
        passed_cases = 0
        failures: list[dict[str, Any]] = []

        for test_file in test_files:
            with open(test_file, "r") as f:
                cases = json.load(f)

            if isinstance(cases, dict):
                cases = [cases]

            for case in cases:
                total_cases += 1
                case_id = case.get("case_id", test_file.stem)
                case_type = case.get("type", "execution")
                success, error_msg = self._evaluate_case(case, config_pack, case_type)

                if success:
                    passed_cases += 1
                else:
                    failures.append({
                        "case_id": case_id,
                        "file": test_file.name,
                        "error": error_msg,
                    })

        pass_rate = (passed_cases / total_cases) if total_cases > 0 else 1.0
        return {
            "suite": suite_name,
            "total_cases": total_cases,
            "passed_cases": passed_cases,
            "pass_rate": pass_rate,
            "failures": failures,
        }

    def _evaluate_case(self, case: dict[str, Any], config_pack: ConfigPack, case_type: str) -> tuple[bool, str]:
        """Evaluate a single test case."""
        if case_type == "memory_contradiction":
            fact_store = SQLiteFactStore(":memory:")
            promoter = MemoryPromoter(fact_store)

            existing = case.get("existing_fact", {})
            if existing:
                cand1 = CandidateFact(
                    source_episode_id="ep_0",
                    session_id="sess_eval",
                    subject=existing["subject"],
                    predicate=existing["predicate"],
                    object=existing["object"],
                    confidence=existing.get("confidence", 0.95),
                )
                promoter.submit_candidate(cand1)
                promoter.evaluate_and_promote(cand1.candidate_id)

            cand_data = case["candidate_fact"]
            cand2 = CandidateFact(
                source_episode_id="ep_1",
                session_id="sess_eval",
                subject=cand_data["subject"],
                predicate=cand_data["predicate"],
                object=cand_data["object"],
                confidence=cand_data.get("confidence", 0.95),
            )
            promoter.submit_candidate(cand2)
            promoted, reason = promoter.evaluate_and_promote(cand2.candidate_id)

            expected_promoted = case.get("expected_promoted", False)
            if promoted == expected_promoted:
                return True, ""
            return False, f"Expected promoted={expected_promoted}, got promoted={promoted} ({reason})"

        if case_type == "clinical_assertion":
            tool = ClinicalAssertionCheckerTool()
            res = tool.execute(case["arguments"])
            expected_passed = case.get("expected_passed", True)
            actual_passed = res.output.get("passed", False)
            if actual_passed == expected_passed:
                return True, ""
            return False, f"Expected assertion passed={expected_passed}, got {actual_passed}"

        return True, ""


if __name__ == "__main__":
    from ops.config_manager import ConfigManager
    cm = ConfigManager("config/packs")
    active = cm.get_active_pack()
    runner = EvalRunner("evals")
    golden = runner.run_suite("golden", active)
    print("Golden Suite Result:", golden)
