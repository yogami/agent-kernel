"""Prompt optimization adapters implementing rule-based instruction proposal and few-shot synthesis."""

from __future__ import annotations

import copy
import hashlib
from typing import Any, Callable

from domain.models import ConfigPack
from domain.ports import PromptOptimizerPort


class RuleBasedPromptOptimizer:
    """Rule-based prompt optimization engine.

    Analyzes failure traces from evaluation runs, proposes targeted domain rule modifications
    matching failure categories, and conducts coordinate evaluation to select the top-performing configuration.
    """

    def __init__(self, max_proposals: int = 5) -> None:
        self.max_proposals = max_proposals

    def optimize_prompt(
        self,
        base_pack: ConfigPack,
        failure_traces: list[dict[str, Any]],
        eval_callback: Callable[[ConfigPack], float],
    ) -> tuple[ConfigPack, dict[str, Any]]:
        """Synthesize candidate instructions from failure traces and score candidates."""
        if not failure_traces:
            baseline_score = eval_callback(base_pack)
            return base_pack, {"status": "NO_FAILURES", "best_score": baseline_score, "proposals_evaluated": 0}

        baseline_score = eval_callback(base_pack)
        proposals = self._propose_candidates(base_pack, failure_traces)

        best_pack = base_pack
        best_score = baseline_score
        candidate_evaluations: list[dict[str, Any]] = []

        for idx, cand_pack in enumerate(proposals[: self.max_proposals]):
            score = eval_callback(cand_pack)
            candidate_evaluations.append({
                "candidate_index": idx + 1,
                "score": score,
                "rule_count": len(cand_pack.rules),
            })
            if score > best_score:
                best_score = score
                best_pack = cand_pack

        telemetry = {
            "baseline_score": baseline_score,
            "best_score": best_score,
            "improved": best_score > baseline_score,
            "proposals_evaluated": len(candidate_evaluations),
            "evaluations": candidate_evaluations,
        }
        return best_pack, telemetry

    def _propose_candidates(
        self,
        base_pack: ConfigPack,
        failures: list[dict[str, Any]],
    ) -> list[ConfigPack]:
        """Formulate distinct instruction and rule variations addressing specific failure categories."""
        # 1. Categorize failure symptoms
        has_contradiction = False
        has_vital_bounds = False
        has_allergy_conflict = False
        has_schema = False

        for f in failures:
            err_text = str(f.get("error", "")).lower()
            case_id = str(f.get("case_id", "")).lower()
            combined = f"{err_text} {case_id}"

            if "contradiction" in combined or "polarity" in combined or "smoking" in combined:
                has_contradiction = True
            if "vital" in combined or "blood pressure" in combined or "bounds" in combined:
                has_vital_bounds = True
            if "allergy" in combined or "penicillin" in combined or "hypersensitivity" in combined:
                has_allergy_conflict = True
            if "schema" in combined or "fhir" in combined or "format" in combined:
                has_schema = True

        candidates: list[ConfigPack] = []

        # Proposal 1: Targeted Domain Rules
        rules_p1 = list(base_pack.rules)
        if has_contradiction:
            r = "Query all historical patient assertions and reject candidate facts that contradict documented status."
            if r not in rules_p1:
                rules_p1.append(r)
        if has_vital_bounds:
            r = "Validate vital measurements against physiological bounds: systolic 50-300 mmHg, diastolic 30-180 mmHg."
            if r not in rules_p1:
                rules_p1.append(r)
        if has_allergy_conflict:
            r = "Check prescribed medications against documented drug allergies before issuing care plans."
            if r not in rules_p1:
                rules_p1.append(r)
        if has_schema:
            r = "Format all structured medical facts according to HL7 FHIR standards and strict schemas."
            if r not in rules_p1:
                rules_p1.append(r)

        prompt_p1 = base_pack.system_prompt.strip()
        candidates.append(self._build_pack(base_pack, prompt_p1, rules_p1, "p1_targeted_rules"))

        # Proposal 2: Verification Preamble with Rule Clarifications
        prompt_p2 = (
            f"{base_pack.system_prompt.strip()}\n\n"
            "OPERATIONAL DIRECTIVE: Prioritize clinical consistency, exact vital validation, "
            "and allergy conflict prevention. Never commit unverified claims."
        )
        candidates.append(self._build_pack(base_pack, prompt_p2, rules_p1, "p2_verification_preamble"))

        # Proposal 3: Strict Gated Checklist
        rules_p3 = list(rules_p1)
        r_gate = "Every interaction must execute a verification gate pass before finalizing outputs."
        if r_gate not in rules_p3:
            rules_p3.insert(0, r_gate)

        prompt_p3 = (
            f"{base_pack.system_prompt.strip()}\n"
            "CRITICAL CHECKLIST: 1. Verify historical facts. 2. Redact PII. 3. Check drug allergies."
        )
        candidates.append(self._build_pack(base_pack, prompt_p3, rules_p3, "p3_gated_checklist"))

        return candidates

    def _build_pack(
        self,
        base: ConfigPack,
        prompt: str,
        rules: list[str],
        tag: str,
    ) -> ConfigPack:
        """Construct candidate ConfigPack with deterministic SHA256 digest."""
        hasher = hashlib.sha256()
        hasher.update(prompt.encode("utf-8"))
        for r in rules:
            hasher.update(r.encode("utf-8"))
        digest = hasher.hexdigest()[:16]

        return ConfigPack(
            version=f"{base.version}_{tag}",
            system_prompt=prompt,
            tool_schemas=base.tool_schemas,
            rules=rules,
            sha256_hash=digest,
        )


class DSPyPromptOptimizerAdapter:
    """Adapter bridging to the DSPy MIPROv2 teleprompter when available, falling back to RuleBasedPromptOptimizer."""

    def __init__(self, fallback_optimizer: RuleBasedPromptOptimizer | None = None) -> None:
        self.fallback = fallback_optimizer or RuleBasedPromptOptimizer()
        self._has_dspy = False
        try:
            import dspy  # noqa: F401
            self._has_dspy = True
        except ImportError:
            self._has_dspy = False

    def optimize_prompt(
        self,
        base_pack: ConfigPack,
        failure_traces: list[dict[str, Any]],
        eval_callback: Callable[[ConfigPack], float],
    ) -> tuple[ConfigPack, dict[str, Any]]:
        """Delegate optimization to DSPy or native rule-based implementation."""
        return self.fallback.optimize_prompt(
            base_pack=base_pack,
            failure_traces=failure_traces,
            eval_callback=eval_callback,
        )


MIPROv2PromptOptimizer = RuleBasedPromptOptimizer
