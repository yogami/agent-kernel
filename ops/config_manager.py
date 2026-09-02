"""Configuration Pack Manager with version snapshots and SHA-256 integrity checks."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from domain.models import ConfigPack


class ConfigManager:
    """Manages immutable versioned configuration packs."""

    def __init__(self, base_dir: str | Path = "config/packs") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_bootstrap_pack()

    def _compute_hash(self, system_prompt: str, tool_schemas: list[dict[str, Any]], rules: list[str]) -> str:
        """Calculate SHA-256 hash of prompt configuration for immutability check."""
        serialized = json.dumps({
            "system_prompt": system_prompt,
            "tool_schemas": tool_schemas,
            "rules": sorted(rules),
        }, sort_keys=True)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _ensure_bootstrap_pack(self) -> None:
        """Create initial v1 pack if repository is empty."""
        v1_dir = self.base_dir / "v1"
        active_pointer = self.base_dir / "active_pack.json"

        if not v1_dir.exists():
            v1_dir.mkdir(parents=True, exist_ok=True)
            default_prompt = (
                "You are an autonomous AI systems agent specialized in clinical data harmonization, "
                "entity extraction, and safe tool execution. Always use typed tools to verify assertions, "
                "scrub PII, and generate HL7 FHIR standard resources."
            )
            default_rules = [
                "Always redact patient names and hospital names using deidentify_clinical_text.",
                "Verify clinical vital signs and contraindications using check_clinical_assertions.",
                "Ensure extracted resources strictly adhere to HL7 FHIR (R4) JSON format.",
            ]
            sha = self._compute_hash(default_prompt, [], default_rules)
            v1_pack = ConfigPack(
                version="v1",
                system_prompt=default_prompt,
                tool_schemas=[],
                rules=default_rules,
                sha256_hash=sha,
            )

            with open(v1_dir / "pack.json", "w") as f:
                json.dump(v1_pack.model_dump(mode="json"), f, indent=2)

            if not active_pointer.exists():
                with open(active_pointer, "w") as f:
                    json.dump({"active_version": "v1"}, f, indent=2)

    def get_active_pack(self) -> ConfigPack:
        """Load the currently active production configuration pack."""
        active_pointer = self.base_dir / "active_pack.json"
        if not active_pointer.exists():
            self._ensure_bootstrap_pack()

        with open(active_pointer, "r") as f:
            data = json.load(f)
            version = data.get("active_version", "v1")

        return self.get_pack(version)

    def get_pack(self, version: str) -> ConfigPack:
        """Fetch a specific versioned configuration pack."""
        pack_file = self.base_dir / version / "pack.json"
        if not pack_file.exists():
            raise FileNotFoundError(f"Configuration pack '{version}' does not exist at {pack_file}.")

        with open(pack_file, "r") as f:
            data = json.load(f)
            return ConfigPack(**data)

    def save_candidate_pack(self, version: str, system_prompt: str, rules: list[str], tool_schemas: list[dict[str, Any]] | None = None) -> ConfigPack:
        """Save a new proposed version snapshot."""
        schemas = tool_schemas or []
        sha = self._compute_hash(system_prompt, schemas, rules)
        pack = ConfigPack(
            version=version,
            system_prompt=system_prompt,
            tool_schemas=schemas,
            rules=rules,
            sha256_hash=sha,
        )

        version_dir = self.base_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)
        with open(version_dir / "pack.json", "w") as f:
            json.dump(pack.model_dump(mode="json"), f, indent=2)

        return pack

    def activate_pack(self, version: str) -> None:
        """Set active pointer to specified validated version."""
        pack_file = self.base_dir / version / "pack.json"
        if not pack_file.exists():
            raise FileNotFoundError(f"Cannot activate non-existent pack '{version}'.")

        active_pointer = self.base_dir / "active_pack.json"
        with open(active_pointer, "w") as f:
            json.dump({"active_version": version}, f, indent=2)
