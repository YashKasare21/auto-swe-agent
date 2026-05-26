.PHONY: install install-dev test lint format run ui swe-bench report clean

# ── Installation ──────────────────────────────────────────────────────────────

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements.txt -r requirements-dev.txt

# ── Testing ───────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v --tb=short

test-coverage:
	pytest tests/ -v --tb=short --cov=. --cov-report=term-missing

# ── Linting & Formatting ──────────────────────────────────────────────────────

lint:
	black --check .
	isort --check-only .
	mypy agent.py agents/ tools/

format:
	black .
	isort .

# ── Running ───────────────────────────────────────────────────────────────────

run:
	python agent.py

ui:
	streamlit run ui/app.py

swe-bench:
	python -m swe_bench.run_swe_bench --num-tasks 10

eval:
	python eval/run_eval.py

# ── Reporting ─────────────────────────────────────────────────────────────────

report:
	python scripts/swe_bench_report.py

cost-report:
	python scripts/cost_report.py

# ── Maintenance ───────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	rm -rf .coverage htmlcov/
