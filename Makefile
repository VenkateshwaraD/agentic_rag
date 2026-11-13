.PHONY: dev-up dev-down install lint test test-unit corpus index eval run-api run-ui docker-build deploy-staging deploy-prod

PYTHON ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

install:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -e ".[dev]"
	-$(BIN)/python -m spacy download en_core_web_lg

dev-up:
	docker compose up -d postgres opensearch
	@echo "Waiting for services..." && sleep 8
	$(BIN)/copilot db init || true

dev-down:
	docker compose down

lint:
	$(BIN)/ruff check src tests eval
	$(BIN)/ruff format --check src tests eval
	-$(BIN)/mypy src

test: test-unit

test-unit:
	$(BIN)/pytest tests/unit -q

test-integration:
	$(BIN)/pytest tests/integration -q -m integration

corpus:
	$(BIN)/copilot corpus generate

index:
	$(BIN)/copilot index --rebuild

eval:
	$(BIN)/copilot eval run

run-api:
	$(BIN)/uvicorn app.main:app --host 0.0.0.0 --port 8000

run-ui:
	$(BIN)/streamlit run src/ui/app.py

docker-build:
	docker build -f deploy/docker/backend.Dockerfile -t agentic-rag-backend .
	docker build -f deploy/docker/frontend.Dockerfile -t agentic-rag-frontend .

deploy-staging:
	kubectl apply -k deploy/k8s/overlays/staging

deploy-prod:
	kubectl apply -k deploy/k8s/overlays/prod
