SEED ?= 42
PY   ?= python3

.PHONY: help install data validate test test-fast clean gate phase2 api \
        features train evaluate model-gate clean-models

help:
	@echo "RevenueOS"
	@echo ""
	@echo "  make install     install python dependencies"
	@echo "  make data        generate the synthetic dataset (SEED=$(SEED))"
	@echo "  make validate    run the data validation report (review gate)"
	@echo "  make test        run the full test suite"
	@echo "  make gate        data + validate + test  <- run this after any simulator change"
	@echo "  make clean       remove generated data"
	@echo ""
	@echo "  make data SEED=7        regenerate under a different seed"
	@echo ""
	@echo "  --- Phase 3/4 (model) ---"
	@echo "  make features    build train/val/test feature matrices + provenance"
	@echo "  make train       train baselines + XGBoost, calibrate, FREEZE model"
	@echo "  make evaluate    oracle eval + OPE + ablations + figures + report"
	@echo "  make model-gate  features -> train -> evaluate -> tests"
	@echo "  make data-small         quick 4k-session dataset for iteration"

install:
	$(PY) -m pip install -r requirements.txt

data:
	$(PY) -m ml.simulation.generate --seed $(SEED)

data-small:
	$(PY) -m ml.simulation.generate --seed $(SEED) --customers 800 --sessions 4000

validate:
	$(PY) -m ml.validation.report

test:
	$(PY) -m pytest tests/ -v

test-fast:
	$(PY) -m pytest tests/ -q

# Full Phase 2 review gate.
gate: data validate test-fast
	@echo ""
	@echo "Gate complete. Read evaluation/results/data_validation_report.md before Phase 3."

phase2: gate

features:
	$(PY) -m ml.features.build

train:
	$(PY) -m ml.models.train

evaluate:
	$(PY) -m ml.evaluation.run_all

# Phase 3/4 gate. Deliberately does NOT regenerate simulator data: the
# simulator is frozen at 1.1.0 and re-rolling it would invalidate the freeze.
model-gate: features train evaluate test-fast
	@echo ""
	@echo "Model gate complete. Read evaluation/results/model_report.md."

clean-models:
	rm -f ml/artifacts/*.pkl ml/artifacts/model_metadata.json
	rm -f data/processed/*.parquet
	rm -f evaluation/results/*.json evaluation/results/*.csv evaluation/results/model_report.md

api:
	$(PY) -m uvicorn backend.app.main:app --reload --port 8000

clean:
	rm -f data/generated/*.parquet data/generated/manifest.json
	rm -rf .pytest_cache __pycache__