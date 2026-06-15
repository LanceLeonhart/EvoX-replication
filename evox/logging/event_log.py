"""EventLog: append-only JSONL event stream for a run.

One file captures the whole narrative of a run: run start/end, every inner
iteration, every window summary, and every strategy switch. JSONL keeps it
greppable and easy to post-process (see ``scripts/summarize_runs.py``).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional


class EventLog:
    def __init__(self, path: Optional[str] = None) -> None:
        self.path = path
        self._events: List[Dict[str, Any]] = []
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            # truncate any previous file at this path
            open(path, "w").close()

    def log(self, event_type: str, **fields: Any) -> Dict[str, Any]:
        event = {"ts": time.time(), "event": event_type}
        event.update(fields)
        self._events.append(event)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(event, default=_json_default) + "\n")
        return event

    def events(self) -> List[Dict[str, Any]]:
        return list(self._events)

    def of_type(self, event_type: str) -> List[Dict[str, Any]]:
        return [e for e in self._events if e["event"] == event_type]


def _json_default(o: Any) -> Any:
    if hasattr(o, "to_dict"):
        return o.to_dict()
    return str(o)
