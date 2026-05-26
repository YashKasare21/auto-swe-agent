FROM python:3.11-slim

# Install git (required for git_workflow node)
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Default: keep container alive for exec_run calls
CMD ["sleep", "infinity"]
