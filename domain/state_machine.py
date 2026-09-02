"""Deterministic Finite State Machine definitions, state transitions, and typed errors."""

from __future__ import annotations

from enum import Enum
from typing import Any


class KernelState(str, Enum):
    """Discrete states of the inner agent execution loop."""
    COMPOSE_CONTEXT = "COMPOSE_CONTEXT"
    MODEL_CALL = "MODEL_CALL"
    VALIDATE_TOOL_CALL = "VALIDATE_TOOL_CALL"
    EXECUTE_TOOL = "EXECUTE_TOOL"
    VALIDATE_OUTPUT = "VALIDATE_OUTPUT"
    PERSIST_EPISODE = "PERSIST_EPISODE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class AgentKernelException(Exception):
    """Base exception for all agent kernel runtime errors."""
    def __init__(self, message: str, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class BudgetExceededError(AgentKernelException):
    """Raised when turn count, token budget, or dollar cost limits are exceeded."""
    pass


class LoopDetectedError(AgentKernelException):
    """Raised when recursive or identical tool call signatures are detected."""
    pass


class ToolExecutionError(AgentKernelException):
    """Raised when a tool handler fails during execution."""
    pass


class GuardrailViolationError(AgentKernelException):
    """Raised when output firewall or clinical assertion check fails."""
    pass


class ContradictionDetectedError(AgentKernelException):
    """Raised when candidate fact conflicts with an active verified fact."""
    pass


class SchemaValidationError(AgentKernelException):
    """Raised when input parameters or output formats fail Pydantic validation."""
    pass


# Allowed state transitions in the deterministic FSM
VALID_TRANSITIONS: dict[KernelState, set[KernelState]] = {
    KernelState.COMPOSE_CONTEXT: {KernelState.MODEL_CALL, KernelState.FAILED},
    KernelState.MODEL_CALL: {
        KernelState.VALIDATE_TOOL_CALL,
        KernelState.VALIDATE_OUTPUT,
        KernelState.FAILED,
    },
    KernelState.VALIDATE_TOOL_CALL: {
        KernelState.EXECUTE_TOOL,
        KernelState.MODEL_CALL,
        KernelState.FAILED,
    },
    KernelState.EXECUTE_TOOL: {KernelState.MODEL_CALL, KernelState.FAILED},
    KernelState.VALIDATE_OUTPUT: {KernelState.PERSIST_EPISODE, KernelState.FAILED},
    KernelState.PERSIST_EPISODE: {KernelState.COMPLETED, KernelState.FAILED},
    KernelState.COMPLETED: set(),
    KernelState.FAILED: set(),
}


def can_transition(current_state: KernelState, target_state: KernelState) -> bool:
    """Check if a state transition is valid according to FSM rules."""
    return target_state in VALID_TRANSITIONS.get(current_state, set())
