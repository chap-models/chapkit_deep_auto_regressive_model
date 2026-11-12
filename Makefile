.PHONY: help install lint mlproject test build-package compare compare-full build run run-ghcr serve-test test-contract test-api test-mlproject stop-test clean

IMAGE      ?= chapkit-auto-regressive:latest
GHCR_IMAGE ?= ghcr.io/chap-models/chapkit_auto_regressive:latest
TEST_NAME  ?= chapkit-auto-regressive-test
TEST_PORT  ?= 8131
TEST_URL   ?= http://localhost:$(TEST_PORT)
# Fast epoch count for the service tests; the code path is the same at any count.
AR_N_ITER  ?= 30
# chapkit test data: 100 training and 60 prediction periods x 5 locations, enough
# for the default contexts (36 months, 52 weeks); chapkit's defaults are shorter.
CONTRACT_ARGS ?= --rows 500 --predict-rows 300 --timeout 600 --verbose
# Checkouts of the models this one replaces, for the parity comparisons.
REF_WEEKLY  ?= ../auto_regressive_weekly_v2
REF_MONTHLY ?= ../auto_regressive_monthly_v2

help:
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@echo "  install         Install dependencies (uv sync)"
	@echo "  lint            Ruff format check + lint, mypy, pyright, and check MLproject is up to date"
	@echo "  mlproject       Regenerate MLproject from main.py"
	@echo "  test            Unit, library and train/predict pipeline tests"
	@echo "  build-package   Build the sdist and wheel (uv build)"
	@echo "  compare         Compare forecasts with the v2 models (fast epochs; needs REF_WEEKLY/REF_MONTHLY)"
	@echo "  compare-full    Same, at the shipped epoch counts (slow)"
	@echo "  build           Build docker image ($(IMAGE))"
	@echo "  run             Build and run the service on :8000"
	@echo "  run-ghcr        Pull and run the prebuilt GHCR image on :8000"
	@echo "  test-contract   chapkit's own end-to-end contract test against the image, monthly and weekly"
	@echo "  test-api        Train + predict over the API, compared with the v2 models"
	@echo "  test-mlproject  test-contract against chapkit mlproject run (the MLproject, not main.py)"
	@echo "  clean           Remove local image, caches and test outputs"

install:
	@uv sync

lint:
	@echo ">>> Ruff format check"
	@uv run ruff format --check .
	@echo ">>> Ruff lint"
	@uv run ruff check .
	@echo ">>> mypy"
	@uv run mypy
	@echo ">>> pyright"
	@uv run pyright
	@echo ">>> MLproject check"
	@uv run chapkit mlproject export --check 2>/dev/null

mlproject:
	@uv run chapkit mlproject export 2>/dev/null

test:
	@uv run pytest -q

build-package:
	@uv build

compare:
	@uv run python tools/compare_v2.py --period week --reference $(REF_WEEKLY)
	@uv run python tools/compare_v2.py --period month --reference $(REF_MONTHLY)

compare-full:
	@uv run python tools/compare_v2.py --period week --reference $(REF_WEEKLY) --full
	@uv run python tools/compare_v2.py --period month --reference $(REF_MONTHLY) --full

build:
	@echo ">>> Building $(IMAGE)"
	@docker build --build-arg GIT_REVISION=$(shell git rev-parse HEAD 2>/dev/null) -t $(IMAGE) .

run: build
	@docker run --rm -p 8000:8000 --name chapkit-auto-regressive $(IMAGE)

run-ghcr:
	@docker run --rm --pull always -p 8000:8000 --name chapkit-auto-regressive $(GHCR_IMAGE)

# Detached service (with the fast epoch count) that the API-level targets run against.
serve-test: build
	@docker rm -f $(TEST_NAME) >/dev/null 2>&1 || true
	@docker run -d -p $(TEST_PORT):8000 -e AR_N_ITER=$(AR_N_ITER) --name $(TEST_NAME) $(IMAGE) $(CMD) >/dev/null
	@echo ">>> Waiting for $(TEST_URL)/health"
	@for i in $$(seq 1 60); do \
		if curl -fsS $(TEST_URL)/health >/dev/null 2>&1; then echo "    healthy"; exit 0; fi; \
		sleep 2; \
	done; echo "    service did not become healthy"; docker logs $(TEST_NAME); exit 1

test-contract: serve-test
	@uv run chapkit test --url $(TEST_URL) --period-type monthly $(CONTRACT_ARGS) \
		&& uv run chapkit test --url $(TEST_URL) --period-type weekly $(CONTRACT_ARGS); \
		status=$$?; $(MAKE) -s stop-test; exit $$status

test-api: serve-test
	@uv run python tools/api_compare.py --url $(TEST_URL) --period week --reference $(REF_WEEKLY) \
		&& uv run python tools/api_compare.py --url $(TEST_URL) --period month --reference $(REF_MONTHLY); \
		status=$$?; $(MAKE) -s stop-test; exit $$status

test-mlproject:
	@$(MAKE) -s serve-test CMD="chapkit mlproject run . --host 0.0.0.0 --port 8000"
	@uv run chapkit test --url $(TEST_URL) --period-type monthly $(CONTRACT_ARGS) \
		&& uv run chapkit test --url $(TEST_URL) --period-type weekly $(CONTRACT_ARGS) \
		&& uv run python tools/api_compare.py --url $(TEST_URL) --period month --reference $(REF_MONTHLY); \
		status=$$?; $(MAKE) -s stop-test; exit $$status

stop-test:
	@docker rm -f $(TEST_NAME) >/dev/null 2>&1 || true

clean:
	@docker rmi -f $(IMAGE) 2>/dev/null || true
	@rm -rf dist data test_out .pytest_cache .ruff_cache .mypy_cache
	@find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

.DEFAULT_GOAL := help
