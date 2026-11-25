.DEFAULT_GOAL := help
export PYTHONPATH := $(CURDIR)/rag_agents:$(CURDIR)

RAG_VENV := $(CURDIR)/rag_agents/.venv/bin
DBT_VENV := $(CURDIR)/dbt_project/.venv/bin
DBT_DIR  := dbt_project/bacen_analytics

.PHONY: help up down logs dbt-build ingest-copom eval test app clean

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'

# --- Infrastructure (Airflow 3 + PostgreSQL 18) ---

up: ## Start Airflow and PostgreSQL services
	docker compose up -d

down: ## Stop Airflow and PostgreSQL services
	docker compose down

logs: ## Tail Docker Compose logs
	docker compose logs -f

# --- Data Warehouse (dbt Core) ---

dbt-build: ## Run dbt models and tests in DAG order (blocks mart on staging test failure)
	cd $(DBT_DIR) && "$(DBT_VENV)/dbt" build

# --- RAG Pipeline & Evaluation ---

ingest-copom: ## Ingest COPOM PDFs into ChromaDB
	cd rag_agents && "$(RAG_VENV)/python" -m rag.ingest_copom

eval: ## Run the 15-query retrieval benchmark (HitRate@6 / MRR@6)
	"$(RAG_VENV)/python" eval/benchmark_retrieval.py

# --- Testing & Application ---

test: ## Run offline pytest suite
	"$(RAG_VENV)/pytest"

app: ## Launch the Streamlit copilot interface
	"$(RAG_VENV)/streamlit" run app.py

clean: ## Remove Python and dbt build caches (preserves .venv and ChromaDB)
	find . -type d \( -name "__pycache__" -o -name ".pytest_cache" \) -exec rm -rf {} +
	rm -rf $(DBT_DIR)/target $(DBT_DIR)/logs