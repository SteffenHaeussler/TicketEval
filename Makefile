.PHONY: install install-hooks lint format-check format typecheck check test test-eval test-eval-ollama eval eval-ollama eval-dataset-check eval-report coverage smoke test-docker test-docker-tracing server server-docker up down logs stack-reset jaeger search-attributes worker llm-worker api doctor ticket status approve reject batch reset

N ?= 100
RUN_ID ?=
API_URL ?= http://localhost:8000
JAEGER_URL ?= http://localhost:16686
TEMPORAL_NAMESPACE ?= default

install:
	uv sync --all-groups
	$(MAKE) install-hooks

install-hooks:
	@hook_path="$$(git rev-parse --git-path hooks/pre-push)"; \
	printf '%s\n' \
		'#!/usr/bin/env bash' \
		'set -euo pipefail' \
		'HOOK_DIR="$$(cd "$$(dirname "$$0")" && pwd)"' \
		'exec uv run pre-commit hook-impl --config=.pre-commit-config.yaml --hook-type=pre-push --hook-dir "$$HOOK_DIR" -- "$$@"' \
		> "$$hook_path"; \
	chmod +x "$$hook_path"

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run pyright

check: format-check lint typecheck test

test:
	uv run pytest

test-eval:
	uv run pytest -m eval -o addopts=

test-eval-ollama:
	uv run pytest -m ollama -o addopts=

eval:
	uv run python scripts/eval.py run --profile primary-quality --agent tunable --reviewer both --allow-unverified

# Preflight already spends ~21 real generations before the first case is scored, so
# keep the scored set small. Case deadline and concurrency are derived from preflight.
eval-ollama:
	uv run python scripts/eval.py run --profile primary-quality --agent ollama --reviewer both --allow-unverified --limit 6

eval-dataset-check:
	uv run python scripts/eval.py dataset-check

# Reads an existing run's artifacts; needs neither Temporal nor Ollama.
eval-report:
	@test -n "$(RUN_ID)" || { echo "usage: make eval-report RUN_ID=run-..."; exit 2; }
	uv run python scripts/eval.py report --run-id $(RUN_ID)

coverage:
	uv run pytest --cov=ticketflow --cov-report=term-missing

## --- deployment smoke tests (against a running docker stack) ---

smoke:
	API_URL=$(API_URL) uv run pytest tests/test_smoke_stack.py -o addopts=

test-docker: up
	API_URL=$(API_URL) uv run pytest tests/test_smoke_stack.py -o addopts=
	docker compose down

test-docker-tracing:
	TICKETFLOW_TRACE_EXPORTER=otlp COMPOSE_PROFILES=tracing docker compose up --build -d
	API_URL=$(API_URL) uv run pytest tests/test_smoke_stack.py -o addopts=
	API_URL=$(API_URL) JAEGER_URL=$(JAEGER_URL) uv run pytest tests/test_tracing_stack.py -o addopts=
	COMPOSE_PROFILES=tracing docker compose down

## --- run the stack (one target per terminal) ---

server:
	temporal server start-dev

server-docker:
	docker compose up temporal temporal-init jaeger

## --- full stack in docker (server, workers, api in one command) ---

up:
	docker compose up --build -d

down:
	docker compose down

logs:
	docker compose logs -f

stack-reset:
	docker compose down -v

jaeger:
	docker compose up jaeger

search-attributes:
	temporal operator search-attribute create --namespace $(TEMPORAL_NAMESPACE) --name TicketStatus --type Keyword

worker:
	uv run python -m ticketflow.worker

llm-worker:
	MOCK_AGENT_LATENCY_MAX_S=3 uv run python -m ticketflow.llm_worker

api:
	uv run uvicorn ticketflow.api:app --reload

doctor:
	uv run python scripts/doctor.py

## --- drive a ticket through (usage: make ticket / make status ID=abc123) ---

ticket:
	@uv run python scripts/doctor.py --quiet --base-url $(API_URL)
	curl -s -X POST $(API_URL)/tickets \
	  -H 'Content-Type: application/json' \
	  -d '{"customer_email": "jo@example.com", "subject": "refund please", "body": "I was double charged."}'

status:
	curl -s $(API_URL)/tickets/$(ID)

approve:
	curl -s -X POST $(API_URL)/tickets/$(ID)/approval \
	  -H 'Content-Type: application/json' \
	  -d '{"approved": true, "approver": "make", "note": "approved via make"}'

reject:
	curl -s -X POST $(API_URL)/tickets/$(ID)/approval \
	  -H 'Content-Type: application/json' \
	  -d '{"approved": false, "approver": "make", "note": "rejected via make"}'

batch:
	uv run python scripts/batch.py --count $(N) --base-url $(API_URL)

reset:
	uv run python scripts/reset.py
