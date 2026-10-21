.PHONY: up down logs test test-integration lint eval sample-data

up:               ## Build and start the whole stack
	docker compose up --build -d

down:             ## Stop the stack (data volumes are kept)
	docker compose down

logs:             ## Follow api + worker logs
	docker compose logs -f api worker

test:             ## Unit tests (no network, no database)
	cd backend && uv run pytest -q

test-integration: ## Unit + integration tests (needs TEST_DATABASE_URL)
	cd backend && TEST_DATABASE_URL=$${TEST_DATABASE_URL:?set TEST_DATABASE_URL} uv run pytest -q

lint:             ## Lint/format check backend, type-check frontend
	cd backend && uv run ruff check . && uv run ruff format --check .
	cd frontend && npx tsc -b

eval:             ## Run the evaluation against the running stack
	cd backend && uv run python eval/run_eval.py

sample-data:      ## Regenerate sample documents
	cd backend && uv run python ../scripts/generate_sample_data.py
