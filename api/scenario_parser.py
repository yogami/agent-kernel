"""
Scenario Ingestion Parser for Agent Kernel.
Supports Markdown (.md), Plain Text (.txt), JSON (.json), and PDF (.pdf).
All functions maintain Cyclomatic Complexity <= 3.
"""

from __future__ import annotations

import io
import json
import re
from typing import Any, Dict, List, Optional


def _is_valid_item(item: str) -> bool:
    """Checks if an item from CSV string is non-empty and non-null."""
    clean = item.strip()
    return bool(clean and clean.lower() != "none")


def _filter_items(raw_items: List[str]) -> List[str]:
    """Filters valid items from raw tokens."""
    return [s.strip() for s in raw_items if _is_valid_item(s)]


def _clean_csv_list(text: Optional[str]) -> List[str]:
    """Splits comma-separated text into a clean list of strings."""
    if not text:
        return []
    return _filter_items(text.split(","))


def _parse_field_line(line: str, result: Dict[str, str]) -> None:
    """Parses a single key-value line formatted like '- Key: Value'."""
    clean = re.sub(r"^[\s\-*#]+", "", line).strip()
    if ":" not in clean:
        return
    key, val = clean.split(":", 1)
    result[key.strip().lower().replace(" ", "_")] = val.strip()


def _parse_block_fields(block: str) -> Dict[str, str]:
    """Extracts key-value fields from a scenario block."""
    result: Dict[str, str] = {}
    for line in block.splitlines():
        _parse_field_line(line, result)
    return result


def _resolve_first_field(data: Dict[str, str], keys: List[str], default: str) -> str:
    """Finds the first present key value in data dictionary."""
    for key in keys:
        val = data.get(key)
        if val:
            return val
    return default


def _build_gold_state(data: Dict[str, str]) -> Dict[str, Any]:
    """Constructs the structured ground truth patient state."""
    med_text = _resolve_first_field(data, ["meds", "medications"], "")
    prob_text = _resolve_first_field(data, ["problems", "conditions"], "")
    return {
        "allergies": _clean_csv_list(data.get("allergies")),
        "meds": _clean_csv_list(med_text),
        "problems": _clean_csv_list(prob_text),
        "constraints": _clean_csv_list(data.get("constraints")),
    }


def _build_read_fixtures(data: Dict[str, str]) -> Dict[str, Any]:
    """Builds the mock read endpoints returned to the agent tools."""
    note = _resolve_first_field(data, ["note", "clinical_note", "situation"], "Clinical record on file.")
    return {
        "get_note": note,
        "get_meds": _clean_csv_list(data.get("meds")),
        "get_allergies": _clean_csv_list(data.get("allergies")),
        "get_labs": "Routine metabolic panels normal.",
    }


def _action_from_explicit(explicit: str) -> str:
    """Extracts expected action string from explicit input."""
    return "propose" if "propose" in explicit.lower() else "abstain"


def _has_safety_conflict(data: Dict[str, str]) -> bool:
    """Checks whether the case contains explicit allergy or constraint blocks."""
    return bool(data.get("allergies") or data.get("constraints"))


def _determine_expected_action(data: Dict[str, str]) -> str:
    """Determines whether the scenario expects abstain or propose."""
    explicit = _resolve_first_field(data, ["expected", "expected_action"], "")
    if explicit:
        return _action_from_explicit(explicit)
    if _has_safety_conflict(data):
        return "abstain"
    return "propose"


def _build_single_scenario(data: Dict[str, str], idx: int) -> Optional[Dict[str, Any]]:
    """Assembles a validated canonical scenario dict from parsed fields."""
    if not data:
        return None
    patient_id = _resolve_first_field(data, ["patient_id", "id"], f"UPLOAD-{idx:03d}")
    threat = _resolve_first_field(data, ["scenario", "threat", "title"], "Custom Scenario")
    prompt = _resolve_first_field(
        data,
        ["user_prompt", "prompt"],
        f"Review chart for {patient_id} and take appropriate action.",
    )
    return {
        "patient_id": patient_id,
        "threat": threat,
        "expected_action": _determine_expected_action(data),
        "user_prompt": prompt,
        "gold_state": _build_gold_state(data),
        "read_fixtures": _build_read_fixtures(data),
    }


def _split_into_blocks(text: str) -> List[str]:
    """Splits markdown or text into distinct scenario chunks."""
    chunks = re.split(r"(?m)(?:^---+\s*$|^#+\s*Scenario:?)", text)
    return [chunk.strip() for chunk in chunks if chunk.strip()]


def _parse_single_block(block: str, idx: int) -> Optional[Dict[str, Any]]:
    """Parses a text block into a single scenario case."""
    fields = _parse_block_fields(block)
    return _build_single_scenario(fields, idx)


def _get_blocks(text: str) -> List[str]:
    """Ensures at least one block exists from input text."""
    blocks = _split_into_blocks(text)
    if blocks:
        return blocks
    return [text]


def _collect_valid_cases(parsed: List[Optional[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extracts non-None parsed scenario items."""
    return [c for c in parsed if c is not None]


def parse_markdown_scenarios(text: str) -> List[Dict[str, Any]]:
    """Parses markdown or formatted text into evaluation cases."""
    blocks = _get_blocks(text)
    parsed = [_parse_single_block(b, i) for i, b in enumerate(blocks, start=1)]
    return _collect_valid_cases(parsed)


def _parse_json_cases(content_bytes: bytes) -> List[Dict[str, Any]]:
    """Loads JSON scenario array."""
    data = json.loads(content_bytes.decode("utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def _extract_page_text(page: Any) -> str:
    """Extracts text from a single PDF page."""
    return page.extract_text() or ""


def _extract_pdf_text(content_bytes: bytes) -> str:
    """Extracts text content from PDF bytes using pypdf."""
    try:
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(content_bytes))
        pages = [_extract_page_text(page) for page in reader.pages]
        return "\n".join(pages)
    except Exception as err:
        return f"Error extracting PDF: {err}"


def parse_uploaded_file(filename: str, content_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Main dispatcher for uploaded files.
    Supports .json, .md, .txt, and .pdf.
    """
    ext = filename.lower().split(".")[-1]
    if ext == "json":
        return _parse_json_cases(content_bytes)
    if ext == "pdf":
        return parse_markdown_scenarios(_extract_pdf_text(content_bytes))
    text = content_bytes.decode("utf-8", errors="replace")
    return parse_markdown_scenarios(text)
