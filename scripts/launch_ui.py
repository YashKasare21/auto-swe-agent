#!/usr/bin/env python3
"""Launch the Streamlit UI dashboard for auto-swe-agent."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "ui/app.py",
            "--server.port",
            "8501",
        ]
    )


if __name__ == "__main__":
    main()
