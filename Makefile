# Makefile
WD := $(shell pwd)
ENV := $(shell readlink -f .env)
REQ := $(shell readlink -f requirements.txt)
PYTHON_VERSION="3.12"

envlink: ## Propagate all environments from the central .env and make sure requirementes are also propagated
	@touch "$(WD)/apps/webapp/.env" "$(WD)/apps/worker/.env" "$(WD)/ml/.env"
	@ln -sf "$(ENV)" "$(WD)/apps/webapp/.env"
	@ln -sf "$(ENV)" "$(WD)/apps/worker/.env"
	@ln -sf "$(ENV)" "$(WD)/ml/.env"
	@cp "$(REQ)" "$(WD)/ml/requirements.txt"
	@cp "$(REQ)" "$(WD)/apps/worker/requirements.txt"

venv: ## Create the venv or show how to activate
	@if [ ! -d ".venv" ]; then \
		echo "Creating virtual environment with Python $(PYTHON_VERSION)..."; \
		python$(PYTHON_VERSION) -m venv .venv; \
		echo "Installing requirements..."; \
		.venv/bin/pip install --upgrade pip; \
		.venv/bin/pip install -r requirements.txt; \
		echo "Virtual environment created and requirements installed."; \
	else \
		echo "Virtual environment already exists at .venv"; \
	fi
	@echo "source .venv/bin/activate"

lift: ## Start all Docker services in detached mode
	docker compose up -d

lift-minio: ## Start all Docker services including MinIO in detached mode
	docker compose --profile minio up -d

tensorboard: ## Start TensorBoard service in detached mode
	docker compose --profile tensorboard up -d
	@echo "TensorBoard running at: http://localhost:6006"

logging: ## Start logging infrastructure (Loki + Grafana)
	docker compose --profile logging up -d
	@if [ -f .env ]; then \
		export $$(grep -v '^#' .env | xargs); \
		echo "Grafana running at: http://localhost:$$GRAFANA_PORT (admin/admin)"; \
		echo "Loki API at: http://localhost:$$LOKI_PORT"; \
	else \
		echo "Grafana running at: http://localhost:3000 (admin/admin)"; \
		echo "Loki API at: http://localhost:3100"; \
	fi

run: ## Run applications - usage: make run webapp [simple] | make run backend
	@if [ "$(filter webapp,$(MAKECMDGOALS))" ]; then \
		if [ "$(filter simple,$(MAKECMDGOALS))" ]; then \
			echo "Starting simple webapp (Streamlit)..."; \
			cd apps/webapp-minimal && streamlit run app.py; \
		else \
			echo "Starting Next.js webapp..."; \
			cd apps/webapp && npm install && npm run dev; \
		fi \
	elif [ "$(filter backend,$(MAKECMDGOALS))" ]; then \
		if [ -f .env ]; then \
			export $$(grep -v '^#' .env | xargs); \
		fi; \
		if [ "$$BACKEND_MODE" = "fastapi" ]; then \
			echo "Starting FastAPI backend..."; \
			cd apps/backend/fastapi && python server.py; \
		elif [ "$$BACKEND_MODE" = "flask" ]; then \
			echo "Starting Flask backend..."; \
			cd apps/backend/flask && python server.py; \
		else \
			echo "Error: BACKEND_MODE environment variable must be set to 'fastapi' or 'flask'"; \
			echo "Example: BACKEND_MODE=fastapi make run backend"; \
			exit 1; \
		fi \
	else \
		echo "Usage: make run webapp [simple] | make run backend"; \
		echo "  make run webapp       - Start Next.js webapp"; \
		echo "  make run webapp simple - Start Streamlit webapp"; \
		echo "  make run backend      - Start backend server (requires BACKEND_MODE=fastapi|flask)"; \
	fi

webapp: ## Target for 'make run webapp'
	@:

simple: ## Target for 'make run webapp simple'
	@:

backend: ## Target for 'make run backend'
	@:

help: ## Show this help message
	@echo "Ultiplate Template - Make Commands"
	@echo "================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# ML Training Commands

train-quick-test: ## Quick training test (10% data, 1 epoch)
	python ml/models/train_trl.py \
		--model_provider deepseek-ocr \
		--dataset_type iam \
		--sample_ratio 0.1 \
		--num_train_epochs 1 \
		--per_device_train_batch_size 2

train-bootstrap: ## Full bootstrap training (DeepSeek-OCR on IAM)
	python ml/models/train_trl.py \
		--model_provider deepseek-ocr \
		--dataset_type iam \
		--num_train_epochs 3 \
		--per_device_train_batch_size 2 \
		--load_in_4bit

train-bootstrap-smol: ## Bootstrap training with SmolVLM-256M (fastest)
	python ml/models/train_trl.py \
		--model_provider smolvlm-256m \
		--dataset_type iam \
		--num_train_epochs 3 \
		--per_device_train_batch_size 4

train-personalize: ## Personalization training (requires MANIFEST_DIR and BOOTSTRAP_ADAPTER)
	@if [ -z "$(MANIFEST_DIR)" ] || [ -z "$(BOOTSTRAP_ADAPTER)" ]; then \
		echo "Error: Set MANIFEST_DIR and BOOTSTRAP_ADAPTER"; \
		echo "Usage: make train-personalize MANIFEST_DIR=./my_data BOOTSTRAP_ADAPTER=./checkpoints/adapter"; \
		exit 1; \
	fi
	python ml/models/train_trl.py \
		--model_provider deepseek-ocr \
		--dataset_type manifest \
		--manifest_data_dir $(MANIFEST_DIR) \
		--bootstrap_adapter_path $(BOOTSTRAP_ADAPTER) \
		--num_train_epochs 2 \
		--learning_rate 3e-4 \
		--augment \
		--augment_strength 0.7

benchmark: ## Benchmark models (requires MODELS)
	@if [ -z "$(MODELS)" ]; then \
		echo "Error: Set MODELS"; \
		echo "Usage: make benchmark MODELS='deepseek-ocr smolvlm-256m'"; \
		exit 1; \
	fi
	python ml/benchmark.py \
		--models $(MODELS) \
		--quantization 4bit \
		--output ml/LEADERBOARD.md

compare-experiments: ## Compare training experiments
	python ml/compare_experiments.py \
		--logs_dir ml/checkpoints/logs \
		--output ml/EXPERIMENT_COMPARISON.md \
		--json ml/experiments.json

test-augmentation: ## Test augmentation presets
	python ml/data/augmentation.py

config-test: ## Test config system
	python ml/config.py

.PHONY: train-quick-test train-bootstrap train-bootstrap-smol train-personalize benchmark compare-experiments test-augmentation config-test
