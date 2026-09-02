"""Synthetic patient record generator for longitudinal multi-encounter evaluations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class SyntheaRecordGenerator:
    """Generates synthetic longitudinal patient encounters with planted temporal conflicts."""

    @staticmethod
    def generate_benchmark_records(count: int = 100) -> list[dict[str, Any]]:
        """Generate N multi-encounter patient histories."""
        records: list[dict[str, Any]] = []

        for i in range(1, count + 1):
            patient_id = f"PAT-{1000 + i}"
            has_conflict = (i % 4 == 0)  # 25% of cases contain planted temporal contradictions

            encounters = [
                {
                    "encounter_id": f"ENC-{patient_id}-1",
                    "date": "2025-01-15",
                    "note_text": f"Patient: Herr Schmidt ({patient_id}), geb. 12.04.1978. "
                                f"Clinic: Charité Berlin. Anamnese: Patient denies drug allergies. "
                                f"Blood pressure: 125/82 mmHg. Diagnosen: Arterielle Hypertonie (I10). "
                                f"Plan: Continue ramipril 5mg.",
                    "planted_facts": [
                        {"subject": f"{patient_id}_allergies", "predicate": "status", "object": "no known drug allergies", "confidence": 0.95},
                        {"subject": f"{patient_id}_hypertension", "predicate": "status", "object": "active", "confidence": 0.95},
                    ],
                },
                {
                    "encounter_id": f"ENC-{patient_id}-2",
                    "date": "2025-06-20",
                    "note_text": (
                        f"Follow-up visit for Herr Schmidt ({patient_id}) at Klinikum Mitte. "
                        f"Patient reports mild skin rash after taking amoxicillin last month. "
                        f"Suspected beta-lactam hypersensitivity. BP: 128/84 mmHg."
                        if has_conflict else
                        f"Routine checkup for Herr Schmidt ({patient_id}). BP 122/80 mmHg. No acute complaints. Stable."
                    ),
                    "planted_facts": [
                        (
                            {"subject": f"{patient_id}_allergies", "predicate": "has_allergy", "object": "penicillin/amoxicillin rash", "confidence": 0.92}
                            if has_conflict else
                            {"subject": f"{patient_id}_status", "predicate": "condition", "object": "stable", "confidence": 0.95}
                        )
                    ],
                    "is_conflict_encounter": has_conflict,
                },
            ]

            records.append({
                "patient_id": patient_id,
                "has_conflict": has_conflict,
                "encounters": encounters,
            })

        return records

    @classmethod
    def write_datasets(cls, output_dir: str | Path = "evals/datasets") -> None:
        """Export generated datasets and test fixture files."""
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        records = cls.generate_benchmark_records(100)

        # Write raw dataset
        with open(out_path / "100_clinical_cases.json", "w") as f:
            json.dump(records, f, indent=2)

        # Write golden facts and conflicts
        with open(out_path / "gold_facts.jsonl", "w") as f_gold, open(out_path / "conflicts.jsonl", "w") as f_conf:
            for rec in records:
                for enc in rec["encounters"]:
                    for fact in enc["planted_facts"]:
                        f_gold.write(json.dumps(fact) + "\n")
                if rec["has_conflict"]:
                    f_conf.write(json.dumps({
                        "patient_id": rec["patient_id"],
                        "encounters": [e["encounter_id"] for e in rec["encounters"]],
                        "conflict_type": "allergy_status_change",
                    }) + "\n")


if __name__ == "__main__":
    SyntheaRecordGenerator.write_datasets()
    print("Generated 100 benchmark clinical records in evals/datasets/")
