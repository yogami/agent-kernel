"""Natural Language Inference (NLI) adapters for contradiction and entailment classification."""

from __future__ import annotations

import json
from typing import Any
from domain.models import NLILabel, NLIResult
from domain.ports import LLMProviderPort, NLIProviderPort


class MockNLIAdapter(NLIProviderPort):
    """Deterministic NLI classifier identifying semantic contradictions and entailments."""

    def __init__(self) -> None:
        self.scripted_pairs: dict[tuple[str, str], NLIResult] = {}
        # Predefined clinical and factual antonym/conflict pairs
        self.conflict_keywords: list[tuple[str, str]] = [
            ("hypertensive", "normotensive"),
            ("high blood pressure", "normal blood pressure"),
            ("malignant", "benign"),
            ("carcinoma", "clear margin"),
            ("smokes", "denies tobacco"),
            ("smoker", "non-smoker"),
            ("penicillin", "no known allergies"),
            ("allergy", "no allergy"),
            ("active infection", "infection resolved"),
            ("positive", "negative"),
            ("diabetic", "non-diabetic"),
            ("pregnant", "not pregnant"),
            ("tachycardia", "bradycardia"),
        ]

    def script_pair(self, premise: str, hypothesis: str, result: NLIResult) -> None:
        """Register a scripted NLI result for an exact premise and hypothesis pair."""
        key = (premise.strip().lower(), hypothesis.strip().lower())
        self.scripted_pairs[key] = result

    def _evaluate(self, premise: str, hypothesis: str) -> NLIResult:
        p_clean = premise.strip().lower()
        h_clean = hypothesis.strip().lower()

        # Check scripted table first
        if (p_clean, h_clean) in self.scripted_pairs:
            return self.scripted_pairs[(p_clean, h_clean)]

        # Check bidirectional conflict pairs
        for word1, word2 in self.conflict_keywords:
            if (word1 in p_clean and word2 in h_clean) or (word2 in p_clean and word1 in h_clean):
                return NLIResult(
                    label=NLILabel.CONTRADICTION,
                    contradiction_score=0.96,
                    entailment_score=0.02,
                    neutral_score=0.02,
                    detail=f"Semantic contradiction detected between '{word1}' and '{word2}'.",
                )

        # Check exact or near-identical statements (Entailment)
        if p_clean == h_clean:
            return NLIResult(
                label=NLILabel.ENTAILMENT,
                contradiction_score=0.01,
                entailment_score=0.98,
                neutral_score=0.01,
                detail="Statements are identical.",
            )

        # Default to neutral
        return NLIResult(
            label=NLILabel.NEUTRAL,
            contradiction_score=0.10,
            entailment_score=0.15,
            neutral_score=0.75,
            detail="No direct contradiction or entailment found.",
        )

    def classify_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify logical relationship between premise and hypothesis synchronously."""
        return self._evaluate(premise, hypothesis)

    async def classify_pair_async(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify logical relationship between premise and hypothesis asynchronously."""
        return self._evaluate(premise, hypothesis)


class LLMBasedNLIAdapter(NLIProviderPort):
    """Production cross-encoder evaluator using LLM structured output for NLI classification."""

    def __init__(self, llm: LLMProviderPort) -> None:
        self.llm = llm

    def _build_prompt(self, premise: str, hypothesis: str) -> list[dict[str, Any]]:
        instructions = (
            "You are a rigorous Natural Language Inference (NLI) classifier for clinical and factual knowledge. "
            "Determine the logical relationship of the Hypothesis relative to the Premise. "
            "Labels: 'contradiction', 'entailment', 'neutral'.\n\n"
            "Return JSON matching this exact structure:\n"
            "{\n"
            '  "label": "contradiction" | "entailment" | "neutral",\n'
            '  "contradiction_score": <float 0.0 to 1.0>,\n'
            '  "entailment_score": <float 0.0 to 1.0>,\n'
            '  "neutral_score": <float 0.0 to 1.0>,\n'
            '  "detail": "<brief reason>"\n'
            "}"
        )
        user_content = f"Premise: {premise}\nHypothesis: {hypothesis}"
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, text: str) -> NLIResult:
        try:
            clean = text.strip()
            if clean.startswith("```json"):
                clean = clean[7:]
            if clean.startswith("```"):
                clean = clean[3:]
            if clean.endswith("```"):
                clean = clean[:-3]
            clean = clean.strip()
            data = json.loads(clean)
            label_str = data.get("label", "neutral").lower()
            return NLIResult(
                label=NLILabel(label_str) if label_str in [l.value for l in NLILabel] else NLILabel.NEUTRAL,
                contradiction_score=float(data.get("contradiction_score", 0.0)),
                entailment_score=float(data.get("entailment_score", 0.0)),
                neutral_score=float(data.get("neutral_score", 0.0)),
                detail=data.get("detail", ""),
            )
        except Exception as exc:
            return NLIResult(
                label=NLILabel.NEUTRAL,
                contradiction_score=0.0,
                entailment_score=0.0,
                neutral_score=1.0,
                detail=f"Parsing error: {exc}",
            )

    def classify_pair(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify logical relationship synchronously without asyncio deadlocks."""
        messages = self._build_prompt(premise, hypothesis)
        if hasattr(self.llm, "scripted_responses") or self.llm.__class__.__name__ == "MockLLMAdapter":
            resp = self.llm.generate(messages=messages, temperature=0.0)
            return self._parse_response(resp.get("content", ""))

        # Directly use a synchronous httpx client to avoid loop issues
        import httpx
        import os
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                import time
                time.sleep(4.5)
                resp = httpx.post(
                    "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
                    json={"model": "gemini-3.5-flash-lite", "messages": messages, "temperature": 0.0},
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    timeout=30.0
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                return self._parse_response(content)
            except Exception as e:
                return NLIResult(label=NLILabel.NEUTRAL, contradiction_score=0, entailment_score=0, neutral_score=1, detail=f"Sync NLI Error: {e}")
        
        # Fallback to the provider's generate (which may hang if async)
        resp = self.llm.generate(messages=messages, temperature=0.0)
        content = resp.get("content", "")
        return self._parse_response(content)

    async def classify_pair_async(self, premise: str, hypothesis: str) -> NLIResult:
        """Classify logical relationship asynchronously."""
        messages = self._build_prompt(premise, hypothesis)
        resp = await self.llm.generate_async(messages=messages, temperature=0.0)
        content = resp.get("content", "")
        return self._parse_response(content)
