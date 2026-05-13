.PHONY: install install-dev backend frontend data-prep test lint clean help

PYTHON := python3
PIP := pip3

help:
	@echo "LEXAR — Available commands:"
	@echo "  make install      Install all dependencies"
	@echo "  make install-dev  Install with dev extras"
	@echo "  make backend      Start the FastAPI backend (port 8000)"
	@echo "  make frontend     Start the Streamlit frontend (port 8501)"
	@echo "  make data-prep    Build chunk JSON files from source PDFs"
	@echo "  make test         Run test suite"
	@echo "  make lint         Run flake8 linter"
	@echo "  make clean        Remove caches and build artifacts"

install:
	$(PIP) install -r requirements.txt
	$(PIP) install -e . --no-deps

install-dev:
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

backend:
	uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload

frontend:
	streamlit run frontend/app.py --server.port 8501

data-prep:
	$(PYTHON) scripts/prepare_data.py

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

lint:
	flake8 lexar/ backend/ --max-line-length=127 --count --select=E9,F63,F7,F82 --show-source
	flake8 lexar/ backend/ --max-line-length=127 --count --exit-zero

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf dist/ build/ *.egg-info/ .eggs/ .pytest_cache/
	@echo "Cleaned."

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down
