SHELL := /bin/bash

experiments:  ## Run the experiments
	pixi run bash ./kick-off.sh

serve:  ## Start the mlflow server
	pixi run mlflow server --backend-store-uri sqlite:///mlruns.db --default-artifact-root ./mlruns --port 5001

prune:  ## Hard-delete soft-deleted MLflow runs/experiments (stop the server first)
	pixi run mlflow gc --tracking-uri sqlite:///mlruns.db --backend-store-uri sqlite:///mlruns.db

clean_up:  ## Clean up the mlruns and mlruns.db
	rm -rf mlruns mlruns.db

.DEFAULT_GOAL := help

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'
