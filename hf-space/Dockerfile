FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agents/ agents/
COPY tools/ tools/
COPY tracking/ tracking/
COPY resilience/ resilience/
COPY indexing/ indexing/
COPY ui/ ui/
COPY observability/ observability/
COPY swe_bench/ swe_bench/
COPY eval/ eval/
COPY docstream/ docstream/
COPY main.py agent.py ./

EXPOSE 8501

CMD ["streamlit", "run", "ui/app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
