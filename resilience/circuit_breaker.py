"""Circuit breaker for LLM API calls."""

from __future__ import annotations

import time
from typing import Dict


class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures: Dict[str, int] = {}
        self.last_failure_time: Dict[str, float] = {}
        self.states: Dict[str, str] = {}  # "closed" | "open" | "half-open"

    def record_success(self, model_name: str) -> None:
        self.failures[model_name] = 0
        self.states[model_name] = "closed"

    def record_failure(self, model_name: str) -> None:
        self.failures[model_name] = self.failures.get(model_name, 0) + 1
        self.last_failure_time[model_name] = time.time()
        if self.failures[model_name] >= self.failure_threshold:
            self.states[model_name] = "open"
            print(
                f"[CIRCUIT] {model_name} circuit OPENED after {self.failures[model_name]} failures"
            )

    def can_call(self, model_name: str) -> bool:
        state = self.states.get(model_name, "closed")
        if state == "closed":
            return True
        if state == "open":
            elapsed = time.time() - self.last_failure_time.get(model_name, 0)
            if elapsed >= self.recovery_timeout:
                self.states[model_name] = "half-open"
                print(
                    f"[CIRCUIT] {model_name} circuit HALF-OPEN (recovery timeout elapsed)"
                )
                return True
            return False
        return True  # half-open: allow one probe call

    def get_status(self) -> Dict[str, Dict]:
        models = set(list(self.failures.keys()) + list(self.states.keys()))
        return {
            m: {
                "state": self.states.get(m, "closed"),
                "failures": self.failures.get(m, 0),
            }
            for m in models
        }

    def reset(self) -> None:
        self.failures.clear()
        self.last_failure_time.clear()
        self.states.clear()
