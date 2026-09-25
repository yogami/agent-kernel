from enum import Enum


class ToolName(str, Enum):
    PROPOSE_MED_ORDER = "propose_med_order"
    EXPORT_SUMMARY = "export_summary"
    GET_NOTE = "get_note"
    GET_MEDS = "get_meds"
    GET_ALLERGIES = "get_allergies"
    GET_LABS = "get_labs"
    GET_PAGINATED_LABS = "get_paginated_labs"
    GET_GUIDELINE = "get_guideline"


class ToolStatus(str, Enum):
    AUTHORIZED = "authorized"
    REJECTED = "rejected"
    ERROR = "error"


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ObservationSource(str, Enum):
    TOOL_OBSERVATION = "tool_observation"


DEFAULT_MODEL = "openai/gpt-4o-mini"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TURNS = 15

