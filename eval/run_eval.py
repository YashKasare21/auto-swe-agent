"""Eval harness for auto-swe-agent. Runs golden test cases and reports pass/fail."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

AGENT_DIR = Path(__file__).parent.parent
VENV_PYTHON = AGENT_DIR / "venv" / "bin" / "python"
VENV_PYTEST = AGENT_DIR / "venv" / "bin" / "pytest"

PASSWORD_PDF_ISSUE = """\
Bug/Feature: Support password-protected PDFs

Currently docstream raises a raw PyMuPDF error when a password-protected
PDF is passed. It should raise a clean ExtractionError with a helpful message.

Required changes in docstream/core/extractor_v2.py — extract_structured():
1. Add optional password parameter to extract_structured()
2. Detect encrypted PDFs using fitz.Document.is_encrypted
3. Attempt decryption if password is provided via doc.authenticate(password)
4. Raise ExtractionError("PDF is password protected. Pass password= to extract()")
   if PDF is encrypted and no password given
5. Raise ExtractionError("Incorrect password for PDF.") if authenticate() returns False

API change: blocks = docstream.extract("protected.pdf", password="secret")

Write a pytest test in tests/test_password_pdf.py that mocks an encrypted PDF
and verifies the correct ExtractionError is raised when no password is provided.
"""


@dataclass
class EvalCase:
    case_id: str
    issue: str
    workspace: str
    validation_cmd: str
    reset_cmds: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    case_id: str
    passed: bool
    iterations_used: int
    model_used: str
    time_taken: float
    error: str = ""
    tests_passed: Optional[bool] = None
    verification_attempts: int = 0
    branch_name: Optional[str] = None
    commit_hash: Optional[str] = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    most_used_model: Optional[str] = None
    circuit_events: int = 0
    circuits_open: int = 0
    semantic_search_calls: int = 0


CASES = [
    EvalCase(
        case_id="add-endpoint-bug",
        issue="The /add endpoint in main.py returns string concatenation instead of integer addition. Fix it so pytest tests pass.",
        workspace=str(Path("~/Desktop/Projects/auto-swe-agent").expanduser()),
        validation_cmd=f"{VENV_PYTEST} test_main.py -q",
        reset_cmds=[
            "WRITE:~/Desktop/Projects/auto-swe-agent/main.py",
        ],
    ),
    EvalCase(
        case_id="docstream-password-pdf",
        issue=PASSWORD_PDF_ISSUE,
        workspace=str(Path("~/Desktop/Projects/docstream").expanduser()),
        validation_cmd=f"{VENV_PYTEST} tests/test_password_pdf.py -q",
        reset_cmds=[
            "git -C ~/Desktop/Projects/docstream checkout docstream/core/extractor_v2.py docstream/__init__.py",
            "rm -f ~/Desktop/Projects/docstream/tests/test_password_pdf.py",
        ],
    ),
]


def _run(cmd: str | list, cwd: str | None = None, timeout: int = 30) -> tuple[int, str]:
    result = subprocess.run(
        cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
        timeout=timeout, cwd=cwd
    )
    return result.returncode, (result.stdout + result.stderr).strip()


BUGGY_MAIN = """from fastapi import FastAPI
app = FastAPI()

@app.get("/add")
def add(a: str, b: str):
    return {"result": a + b}
"""


def _reset_case(case: EvalCase) -> None:
    for cmd in case.reset_cmds:
        if cmd.startswith("WRITE:"):
            _, path = cmd.split(":", 1)
            path = Path(path.strip()).expanduser()
            path.write_text(BUGGY_MAIN)
            print(f"  [RESET] Wrote {path}:")
            print("  " + "\n  ".join(BUGGY_MAIN.splitlines()))
            # Verify it's valid Python
            rc, out = _run([str(VENV_PYTHON), "-c",
                            f"import ast; ast.parse(open('{path}').read()); print('syntax OK')"])
            print(f"  [RESET] Syntax check: {out} (exit {rc})")
        else:
            print(f"  [RESET] Running: {cmd}")
            code, out = _run(cmd)
            print(f"  [RESET] Exit {code}: {out[:200] if out else '(no output)'}")


def _pre_validation_check(case: EvalCase) -> None:
    """Print workspace state before running pytest."""
    ws = Path(case.workspace)
    print(f"  [PRE-VALIDATION] Files in {ws}:")
    for p in sorted(ws.rglob("*.py")):
        rel = p.relative_to(ws)
        # Skip venv and __pycache__
        if any(part in ("venv", ".venv", "__pycache__") for part in rel.parts):
            continue
        print(f"    {rel} (exists={p.exists()})")
    # Specific file checks
    if case.case_id == "add-endpoint-bug":
        mp = ws / "main.py"
        print(f"  [PRE-VALIDATION] main.py exists={mp.exists()}")
        if mp.exists():
            print("  [PRE-VALIDATION] main.py content:")
            print("  " + "\n  ".join(mp.read_text().splitlines()))
    elif case.case_id == "docstream-password-pdf":
        tp = ws / "tests" / "test_password_pdf.py"
        print(f"  [PRE-VALIDATION] test_password_pdf.py exists={tp.exists()}")


def _run_agent(case: EvalCase, timeout: int = 120) -> tuple[int, str]:
    result = subprocess.run(
        [str(VENV_PYTHON), str(AGENT_DIR / "agent.py"), case.issue, "--workspace", case.workspace],
        capture_output=True, text=True, timeout=timeout, cwd=str(AGENT_DIR),
        env={**os.environ},
    )
    return result.returncode, (result.stdout + result.stderr).strip()


def _extract_meta(output: str) -> tuple:
    """Parse all fields from the [SUMMARY] line printed by agent main()."""
    iterations = output.count("--- [NODE] PLANNER")
    model = "unknown"
    for line in output.splitlines():
        if "--- [NODE] PLANNER | model=" in line:
            model = line.split("model=")[-1].strip().rstrip("-").strip()

    tests_passed: Optional[bool] = None
    verification_attempts = 0
    branch_name: Optional[str] = None
    commit_hash: Optional[str] = None
    total_cost_usd: float = 0.0
    total_tokens: int = 0
    most_used_model: Optional[str] = None
    circuit_events: int = 0
    circuits_open: int = 0
    semantic_search_calls: int = 0

    for line in output.splitlines():
        if "[SUMMARY]" in line:
            try:
                def _get(key: str) -> str:
                    parts = line.split(f"{key}=")
                    if len(parts) < 2:
                        return "0"
                    return parts[1].split(" |")[0].strip()

                tp_str = _get("tests_passed")
                tests_passed = True if tp_str == "True" else (False if tp_str == "False" else None)
                verification_attempts = int(_get("verification_attempts"))
                bn = _get("branch_name")
                branch_name = None if bn == "None" else bn
                ch = _get("commit_hash")
                commit_hash = None if ch == "None" else ch
                total_cost_usd = float(_get("total_cost_usd"))
                total_tokens = int(_get("total_tokens"))
                mum = _get("most_used_model")
                most_used_model = None if mum == "None" else mum
                circuit_events = int(_get("circuit_events"))
                circuits_open = int(_get("circuits_open"))
                semantic_search_calls = int(_get("semantic_search_calls"))
            except (IndexError, ValueError):
                pass
            break

    return (iterations, model, tests_passed, verification_attempts,
            branch_name, commit_hash, total_cost_usd, total_tokens,
            most_used_model, circuit_events, circuits_open,
            semantic_search_calls)


def run_eval(cases: list[EvalCase] = CASES) -> list[EvalResult]:
    results: list[EvalResult] = []

    for case in cases:
        print(f"\n{'='*60}")
        print(f"Running: {case.case_id}")
        print(f"{'='*60}")

        print("  Resetting to buggy state...")
        _reset_case(case)

        start = time.time()
        error = ""
        agent_output = ""
        try:
            code, agent_output = _run_agent(case)
            if code != 0:
                # Extract the actual exception from the output
                for line in reversed(agent_output.splitlines()):
                    if line.strip() and not line.startswith(" "):
                        error = line.strip()
                        break
                error = error or f"Agent exited {code}"
        except subprocess.TimeoutExpired:
            error = "Agent timed out (120s)"
        elapsed = round(time.time() - start, 1)
        print(f"  Agent output (last 5 lines):")
        for line in agent_output.splitlines()[-5:]:
            print(f"    {line}")

        meta = _extract_meta(agent_output)
        (iterations, model, agent_tests_passed, agent_verification_attempts,
         branch_name, commit_hash, total_cost_usd, total_tokens,
         most_used_model, circuit_events, circuits_open,
         semantic_search_calls) = meta[:12]

        # Run validation
        passed = False
        if not error:
            _pre_validation_check(case)
            val_code, val_out = _run(case.validation_cmd, cwd=case.workspace, timeout=30)
            passed = val_code == 0
            if not passed:
                error = val_out[:200]
            print(f"  Validation: {'PASS' if passed else 'FAIL'}")
            print(f"  {val_out[:300]}")
        else:
            print(f"  Skipping validation — agent error: {error}")

        results.append(EvalResult(
            case_id=case.case_id,
            passed=passed,
            iterations_used=iterations,
            model_used=model,
            time_taken=elapsed,
            error=error,
            tests_passed=agent_tests_passed,
            verification_attempts=agent_verification_attempts,
            branch_name=branch_name,
            commit_hash=commit_hash,
            total_cost_usd=total_cost_usd,
            total_tokens=total_tokens,
            most_used_model=most_used_model,
            circuit_events=circuit_events,
            circuits_open=circuits_open,
            semantic_search_calls=semantic_search_calls,
        ))

        if case != cases[-1]:
            print("  [RATE LIMIT] Waiting 30s before next case...")
            time.sleep(30)

    return results


def print_report(results: list[EvalResult]) -> None:
    print(f"\n{'='*80}")
    print("EVAL REPORT")
    print(f"{'='*80}")
    print(f"| {'Case':<28} | {'Result':<6} | {'Iter':>4} | {'Verify':>6} | {'Cost':>8} | "
          f"{'Tokens':>7} | {'Time':>6} | {'CE':>3} | {'SS':>3} |")
    print(f"|{'-'*30}|{'-'*8}|{'-'*6}|{'-'*8}|{'-'*10}|{'-'*9}|{'-'*8}|{'-'*5}|{'-'*5}|")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        tp = "✓" if r.tests_passed else ("✗" if r.tests_passed is False else "-")
        cost_flag = " !" if r.total_cost_usd > 5.0 else ""
        ce_flag = " !" if r.circuits_open > 0 else ""
        print(f"| {r.case_id:<28} | {status:<6} | {r.iterations_used:>4} | {tp:>4}/{r.verification_attempts:<1} | "
              f"${r.total_cost_usd:>6.4f}{cost_flag} | {r.total_tokens:>7} | {r.time_taken:>5.1f}s | "
              f"{r.circuit_events:>3}{ce_flag} | {r.semantic_search_calls:>3} |")
    passed = sum(r.passed for r in results)
    total_cost = sum(r.total_cost_usd for r in results)
    total_ce = sum(r.circuit_events for r in results)
    total_ss = sum(r.semantic_search_calls for r in results)
    print(f"\n{passed}/{len(results)} passed | Total cost: ${total_cost:.4f} | "
          f"Total circuit events: {total_ce} | Total semantic searches: {total_ss}")

    # Print circuit summary for any case with events
    for r in results:
        if r.circuit_events > 0:
            print(f"  {r.case_id}: {r.circuit_events} circuit events "
                  f"({r.circuits_open} circuits still open at end)")


def save_results(results: list[EvalResult]) -> Path:
    out_dir = Path(__file__).parent
    out_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"results_{ts}.json"
    out_path.write_text(json.dumps([r.__dict__ for r in results], indent=2))
    print(f"\nResults saved to {out_path}")
    return out_path


if __name__ == "__main__":
    results = run_eval()
    print_report(results)
    save_results(results)
