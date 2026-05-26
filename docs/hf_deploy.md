# Hugging Face Spaces Deployment Guide

## Overview

The auto-swe-agent Streamlit dashboard can be deployed as a **Docker-backed Hugging Face Space**. This provides a publicly accessible live monitoring interface for the agent's execution state, cost analytics, and circuit breaker health.

## Prerequisites

- A [Hugging Face](https://huggingface.co) account
- A Hugging Face **write token** (create one at `huggingface.co/settings/tokens`)
- Python 3.11+ locally

## Creating the Space on Hugging Face

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click **Create new Space**
3. Configure:
   - **Space Name**: `auto-swe-agent`
   - **License**: MIT
   - **Space SDK**: **Docker**
   - **Docker Template**: **Blank**
   - **Space Hardware**: CPU basic (free tier is sufficient)
4. Click **Create Space**

> **Note**: The Space title, emoji, and colour theme are defined in `hf-space/README.md` — you do not need to set them in the UI.

## Setting API Keys (Secrets)

The dashboard requires API keys to function. Add them as **Space secrets**:

1. Go to your Space → **Settings** → **Repository Secrets**
2. Add the following secrets:

| Secret Name | Description |
|---|---|
| `GEMINI_API_KEY` | Google Gemini API key (primary model provider) |
| `GROQ_API_KEY` | Groq API key (open-source model provider) |
| `LANGFUSE_PUBLIC_KEY` | Langfuse public key (optional, for observability) |
| `LANGFUSE_SECRET_KEY` | Langfuse secret key (optional) |
| `LANGFUSE_HOST` | Langfuse host URL (optional, default: `https://cloud.langfuse.com`) |

Secrets are automatically injected as environment variables by Hugging Face Spaces.

## Automated Deployment

The project includes an automated deployment script at `scripts/deploy_hf.py`.

```bash
python scripts/deploy_hf.py \
    --token hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxx \
    --repo-id your-username/auto-swe-agent
```

### What the Script Does

1. Creates a temporary working directory (`deploy_hf_temp`)
2. Copies the following source components:
   - `agents/`, `tools/`, `tracking/`, `resilience/`, `indexing/`, `ui/`
   - `observability/`, `swe_bench/`, `eval/`, `docstream/`
   - `main.py`, `agent.py`
3. Copies `hf-space/README.md` and `hf-space/Dockerfile` as the Space root
4. Copies `requirements.txt`
5. Initialises a git repository, commits, and force-pushes to the target Space
6. Falls back from `main` to `master` branch if the initial push fails

### Manual Deployment (Alternative)

If you prefer to deploy manually:

```bash
# Clone the Space
git clone https://huggingface.co/spaces/your-username/auto-swe-agent
cd auto-swe-agent

# Copy deployment files
cp /path/to/auto-swe-agent/hf-space/Dockerfile ./
cp /path/to/auto-swe-agent/hf-space/README.md ./
cp /path/to/auto-swe-agent/requirements.txt ./

# Copy source directories
cp -r /path/to/auto-swe-agent/agents ./
cp -r /path/to/auto-swe-agent/ui ./
# ... repeat for all required directories listed above

# Commit and push
git add -A
git commit -m "Deploy auto-swe-agent dashboard"
git push
```

## Deployed File Structure

The Space expects this layout:

```
├── Dockerfile              # From hf-space/Dockerfile
├── README.md               # From hf-space/README.md (HF frontmatter)
├── requirements.txt        # Python dependencies
├── agents/                 # Multi-agent orchestration
├── ui/                     # Streamlit dashboard
├── tools/                  # Agent tool implementations
├── indexing/               # Semantic code search
├── tracking/               # Cost tracking
├── resilience/             # Circuit breaker + retry
├── observability/          # Langfuse telemetry
├── swe_bench/              # SWE-bench harness
├── eval/                   # Eval harness
├── docstream/              # PDF extraction
├── main.py                 # FastAPI test fixture
└── agent.py                # LangGraph graph builder
```

## Runtime Configuration

The dashboard runs on **port 8501** inside the container. The Dockerfile starts Streamlit with:

```dockerfile
CMD ["streamlit", "run", "ui/app.py", "--server.port=8501",
     "--server.address=0.0.0.0", "--server.headless=true"]
```

### Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `GEMINI_API_KEY` | Yes | — | Google Gemini API key |
| `GROQ_API_KEY` | Yes | — | Groq API key |
| `LANGFUSE_PUBLIC_KEY` | No | — | Langfuse observability key |
| `LANGFUSE_SECRET_KEY` | No | — | Langfuse observability secret |
| `LANGFUSE_HOST` | No | `https://cloud.langfuse.com` | Langfuse host URL |

## Troubleshooting

| Problem | Likely Cause | Solution |
|---|---|---|
| Space build fails | Missing `requirements.txt` | Ensure `requirements.txt` is present at the Space root |
| `ModuleNotFoundError` | Missing source directory | Check all required directories are copied |
| Dashboard shows blank page | API keys not set | Add secrets in Space settings |
| Push rejected | Branch name mismatch | Use `--branch master` flag or try manual push |
| Build timeout | Large dependency download | Space CPU basic tier rebuilds on every push; use `--no-cache-dir` in pip |

## Viewing the Dashboard

Once the Space has finished building (typically 2–5 minutes), the dashboard is available at:

```
https://huggingface.co/spaces/your-username/auto-swe-agent
```

Build logs can be monitored from the Space page under the **Builder** tab.
