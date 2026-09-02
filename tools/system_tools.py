"""Utility system tools for calculations and temporal operations."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Any
from pydantic import BaseModel, Field

from domain.models import ToolResult
from domain.ports import ToolPort


class CalculatorInput(BaseModel):
    expression: str = Field(..., description="Mathematical expression to safely evaluate, e.g. '120 * 0.85'.")


class CalculatorTool(ToolPort):
    """Deterministic mathematical calculation tool."""

    name: str = "calculate"
    description: str = "Perform safe arithmetic and math calculations."

    ALLOWED_NAMES = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "pow": pow,
        "sqrt": math.sqrt,
    }

    @property
    def schema(self) -> dict[str, Any]:
        return CalculatorInput.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = CalculatorInput(**arguments)
        expr = inp.expression.strip()

        # Sanitize expression against dangerous builtins
        if any(bad in expr for bad in ["__", "import", "eval", "exec", "open", "sys", "os"]):
            return ToolResult(
                tool_id="calc",
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message="Prohibited characters or tokens in math expression.",
            )

        try:
            # Safely evaluate with restricted globals/locals
            result = eval(expr, {"__builtins__": {}}, self.ALLOWED_NAMES)
            return ToolResult(
                tool_id="calc",
                tool_name=self.name,
                output={"expression": expr, "result": result},
            )
        except Exception as exc:
            return ToolResult(
                tool_id="calc",
                tool_name=self.name,
                output=None,
                is_error=True,
                error_message=f"Evaluation error: {exc}",
            )


class DateValidatorInput(BaseModel):
    date_str: str = Field(..., description="Date string to parse and validate, e.g. '2026-09-02'.")


class DateValidatorTool(ToolPort):
    """Validates date formats and calculates temporal distance."""

    name: str = "validate_date"
    description: str = "Parse and validate dates and compute recency relative to current time."

    @property
    def schema(self) -> dict[str, Any]:
        return DateValidatorInput.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = DateValidatorInput(**arguments)
        for fmt in ["%Y-%m-%d", "%d.%m.%Y", "%m/%d/%Y", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                dt = datetime.strptime(inp.date_str, fmt).replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                diff_days = (now - dt).days
                return ToolResult(
                    tool_id="date_val",
                    tool_name=self.name,
                    output={
                        "valid": True,
                        "iso_date": dt.isoformat(),
                        "age_days": diff_days,
                        "is_future": diff_days < 0,
                    },
                )
            except ValueError:
                continue

        return ToolResult(
            tool_id="date_val",
            tool_name=self.name,
            output={"valid": False, "error": f"Unable to parse date '{inp.date_str}' in known formats."},
            is_error=True,
            error_message="Invalid date format.",
        )
