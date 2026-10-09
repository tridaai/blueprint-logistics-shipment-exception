.PHONY: demo test evals serve install install-pip

install:  ## Install the locked set with uv (primary workflow)
	uv sync --extra dev

install-pip:  ## Fallback install: pip with the pinned/frozen set
	pip install -e ".[dev]" -c constraints.txt

demo:  ## Traced demo: one sample shipment, end to end (offline)
	uv run shipment-agent-demo

test:  ## Run the pytest suite
	uv run pytest -q

evals:  ## Run the golden-dataset evals (gate: type accuracy >= 90%)
	uv run python evals/run_evals.py

serve:  ## Serve the API + web UI on http://localhost:8000
	uv run uvicorn shipment_agent.api:app --port 8000
