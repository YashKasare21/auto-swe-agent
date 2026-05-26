"""Deploy auto-swe-agent Streamlit dashboard to Hugging Face Spaces.

Usage:
    python scripts/deploy_hf.py \
        --token hf_xxxxxxxxxx \
        --repo-id your-username/auto-swe-agent

This script assembles a minimal deployment directory, initialises git,
and pushes the contents to the target Hugging Face Space (Docker SDK).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEPLOY_SOURCES = [
    "agents/",
    "tools/",
    "tracking/",
    "resilience/",
    "indexing/",
    "ui/",
    "observability/",
    "swe_bench/",
    "eval/",
    "docstream/",
    "main.py",
    "agent.py",
]


def deploy_hf(token: str, repo_id: str, branch: str = "main") -> None:
    work_dir = Path(tempfile.mkdtemp(prefix="deploy_hf_"))
    deploy_dir = work_dir / "deploy"
    deploy_dir.mkdir(parents=True, exist_ok=True)

    print(f"[deploy] Assembling deployment in {deploy_dir}")

    for src in DEPLOY_SOURCES:
        src_path = PROJECT_ROOT / src
        dst_path = deploy_dir / src
        if src_path.is_dir():
            shutil.copytree(
                src_path,
                dst_path,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        elif src_path.is_file():
            shutil.copy2(src_path, dst_path)

    hf_readme = PROJECT_ROOT / "hf-space" / "README.md"
    if hf_readme.exists():
        shutil.copy2(hf_readme, deploy_dir / "README.md")
        print("[deploy] Copied hf-space/README.md")

    hf_dockerfile = PROJECT_ROOT / "hf-space" / "Dockerfile"
    if hf_dockerfile.exists():
        shutil.copy2(hf_dockerfile, deploy_dir / "Dockerfile")
        print("[deploy] Copied hf-space/Dockerfile")

    requirements_dst = deploy_dir / "requirements.txt"

    req_src = PROJECT_ROOT / "requirements.txt"
    if req_src.exists():
        shutil.copy2(req_src, requirements_dst)

    print("[deploy] Initialising git repository")

    subprocess.run(["git", "init"], cwd=deploy_dir, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "deploy@auto-swe-agent"],
        cwd=deploy_dir,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "auto-swe-agent-deploy"],
        cwd=deploy_dir,
        check=True,
        capture_output=True,
    )

    subprocess.run(
        ["git", "add", "-A"], cwd=deploy_dir, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "--allow-empty", "-m", "Deploy auto-swe-agent dashboard"],
        cwd=deploy_dir,
        check=True,
        capture_output=True,
    )

    hf_url = f"https://huggingface.co/spaces/{repo_id}"

    remote_url = f"https://{token}@huggingface.co/spaces/{repo_id}"
    subprocess.run(
        ["git", "remote", "add", "origin", remote_url],
        cwd=deploy_dir,
        check=True,
        capture_output=True,
    )

    print(f"[deploy] Pushing to {hf_url}")

    try:
        result = subprocess.run(
            ["git", "push", "--force", "origin", f"HEAD:{branch}"],
            cwd=deploy_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            fallback_branch = "master" if branch == "main" else "main"
            print(f"[deploy] Push to '{branch}' failed — trying '{fallback_branch}'")
            result = subprocess.run(
                ["git", "push", "--force", "origin", f"HEAD:{fallback_branch}"],
                cwd=deploy_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )
            if result.returncode != 0:
                print(f"[deploy] Push error:\n{result.stderr}")
                sys.exit(1)
    except subprocess.TimeoutExpired:
        print("[deploy] Push timed out after 120s")
        sys.exit(1)

    print(f"[deploy] Successfully deployed to {hf_url}")
    print(f"[deploy] Dashboard will be available at {hf_url} once the Space builds")

    shutil.rmtree(work_dir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy auto-swe-agent dashboard to Hugging Face Spaces"
    )
    parser.add_argument("--token", required=True, help="Hugging Face API token")
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target Space repo ID, e.g. 'username/auto-swe-agent'",
    )
    parser.add_argument("--branch", default="main", help="Space branch (default: main)")
    args = parser.parse_args()

    deploy_hf(token=args.token, repo_id=args.repo_id, branch=args.branch)


if __name__ == "__main__":
    main()
