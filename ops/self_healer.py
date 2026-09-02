"""Self-Healing LLM-Ops Controller with Monotonic Non-Regression Gating."""

from __future__ import annotations

from typing import Any
from domain.models import ConfigPack
from ops.config_manager import ConfigManager
from ops.eval_runner import EvalRunner


class SelfHealer:
    """Diagnoses failure traces, generates candidate config packs, and enforces non-regression gating."""

    def __init__(self, config_manager: ConfigManager, eval_runner: EvalRunner, max_iterations: int = 5) -> None:
        self.config_manager = config_manager
        self.eval_runner = eval_runner
        self.max_iterations = max_iterations

    def run_repair_cycle(self) -> dict[str, Any]:
        """Execute automated diagnosis, proposal, evaluation, and monotonic commit."""
        active_pack = self.config_manager.get_active_pack()

        # 1. Run baseline golden and dev evaluations
        golden_baseline = self.eval_runner.run_suite("golden", active_pack)
        dev_results = self.eval_runner.run_suite("dev", active_pack)

        if not dev_results["failures"]:
            return {
                "status": "GREEN",
                "message": "All dev test cases passing. No repair needed.",
                "active_version": active_pack.version,
                "golden_pass_rate": golden_baseline["pass_rate"],
                "dev_pass_rate": dev_results["pass_rate"],
            }

        # 2. Iterate up to max_iterations
        for iteration in range(1, self.max_iterations + 1):
            next_version = f"v{int(active_pack.version.replace('v', '')) + iteration}"

            # Diagnose failure and synthesize candidate patch
            patched_prompt, patched_rules = self._synthesize_patch(
                active_pack, dev_results["failures"]
            )

            candidate_pack = self.config_manager.save_candidate_pack(
                version=next_version,
                system_prompt=patched_prompt,
                rules=patched_rules,
                tool_schemas=active_pack.tool_schemas,
            )

            # 3. Evaluate candidate against Golden (100% required) and Holdout
            cand_golden = self.eval_runner.run_suite("golden", candidate_pack)
            cand_dev = self.eval_runner.run_suite("dev", candidate_pack)
            cand_holdout = self.eval_runner.run_suite("holdout", candidate_pack)

            # Monotonic Non-Regression Gate Rule:
            # Golden must be 1.0 (zero regression), and Dev score must improve
            if (
                cand_golden["pass_rate"] >= 1.0
                and cand_dev["pass_rate"] > dev_results["pass_rate"]
                and cand_holdout["pass_rate"] >= 1.0
            ):
                # Commit candidate pack
                self.config_manager.activate_pack(next_version)
                return {
                    "status": "REPAIRED_AND_COMMITTED",
                    "committed_version": next_version,
                    "iteration": iteration,
                    "golden_pass_rate": cand_golden["pass_rate"],
                    "dev_pass_rate": cand_dev["pass_rate"],
                    "holdout_pass_rate": cand_holdout["pass_rate"],
                }

        # Abort and roll back to active pack if gating fails
        return {
            "status": "REPAIR_ABORTED",
            "message": f"Candidate patches regressed golden suite or failed to improve dev after {self.max_iterations} iterations. Retained {active_pack.version}.",
            "active_version": active_pack.version,
            "dev_pass_rate": dev_results["pass_rate"],
        }

    def _synthesize_patch(self, active_pack: ConfigPack, failures: list[dict[str, Any]]) -> tuple[str, list[str]]:
        """Diagnose root-cause errors and construct updated rule list."""
        updated_rules = list(active_pack.rules)

        for failure in failures:
            err = failure.get("error", "")
            if "Contradiction" in err or "allergy" in err.lower():
                rule = "Strictly check existing patient facts before asserting or updating allergy and medication records."
                if rule not in updated_rules:
                    updated_rules.append(rule)
            elif "FHIR" in err or "schema" in err.lower():
                rule = "Ensure all clinical entities map to closed Pydantic schemas and valid HL7 FHIR (R4) structures."
                if rule not in updated_rules:
                    updated_rules.append(rule)

        prompt = active_pack.system_prompt + " Maintain strict verification and zero memory pollution."
        return prompt, updated_rules
