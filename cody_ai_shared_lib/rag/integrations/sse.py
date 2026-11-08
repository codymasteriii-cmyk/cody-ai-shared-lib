"""Server-Sent Events (SSE) formatter (§10.9, AC-RAG-54).

Pure string formatting helper with no web framework dependencies.
"""
from __future__ import annotations

import json
from typing import Any
from pydantic import BaseModel


def format_event(event: Any, data: Any = None) -> str:
    """Format an event and payload into SSE format: 'event: <name>\\ndata: <json>\\n\\n'.

    Accepts:
      - format_event("delta", {"text": "hello"})
      - format_event(RAGStreamEvent(event="delta", data={"text": "hello"}))
      - format_event({"event": "delta", "data": {"text": "hello"}})
    """
    event_name: str
    event_data: Any

    if data is not None:
        event_name = str(event)
        event_data = data
    elif hasattr(event, "event") and hasattr(event, "data"):
        event_name = str(event.event)
        event_data = event.data
    elif isinstance(event, dict) and "event" in event and "data" in event:
        event_name = str(event["event"])
        event_data = event["data"]
    elif isinstance(event, (list, tuple)) and len(event) == 2:
        event_name = str(event[0])
        event_data = event[1]
    else:
        raise ValueError(f"Cannot format SSE event from {type(event).__name__}")

    if isinstance(event_data, BaseModel):
        json_str = event_data.model_dump_json()
    else:
        json_str = json.dumps(event_data, default=str)

    return f"event: {event_name}\ndata: {json_str}\n\n"
