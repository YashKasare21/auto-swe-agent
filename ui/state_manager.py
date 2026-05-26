"""Persistent state manager for Streamlit live monitoring.

The agent writes a JSON state file after each iteration; the Streamlit
app reads it on a polling loop for real-time display.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

STATE_FILE = Path(__file__).parent / ".agent_state.json"


class AgentStateManager:
    def __init__(self, state_file: Path = STATE_FILE):
        self.state_file = state_file

    def save_state(self, state: Dict[str, Any]) -> None:
        state["last_updated"] = time.time()
        self.state_file.write_text(json.dumps(state, default=str, indent=2))

    def load_state(self) -> Optional[Dict[str, Any]]:
        if not self.state_file.exists():
            return None
        try:
            return json.loads(self.state_file.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def is_running(self) -> bool:
        state = self.load_state()
        if not state:
            return False
        last_update = state.get("last_updated", 0)
        stale = time.time() - last_update > 30
        return not stale and state.get("status") not in (
            "completed",
            "complete",
            "error",
        )

    def clear(self) -> None:
        if self.state_file.exists():
            self.state_file.unlink()
