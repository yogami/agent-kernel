from dataclasses import asdict
from datetime import datetime
from enum import Enum
from typing import Any

def to_dict(obj: Any) -> Any:
    if isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif hasattr(obj, "__dataclass_fields__"):
        return to_dict(asdict(obj))
    else:
        return obj
