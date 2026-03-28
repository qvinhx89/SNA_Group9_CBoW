# Makefile for SNA Twitch Influencer Project
# Cross-platform build and run automation

PYTHON = python
PIP = pip
VENV = venv

# Directories
SRC_DIR = src
DATA_DIR = data
OUTPUT_DIR = outputs
REPORTS_DIR = reports

.PHONY: all setup clean stage0 stage1 stage2 stage3 stage4 stage5 stage6 run_all test help

# Default target
all: help

# Help
help:
	@echo "SNA Twitch Influencer Project - Makefile"
	@echo ""
	@echo "Setup:"
	@echo "  make setup       - Create virtualenv and install dependencies"
	@echo "  make clean       - Remove generated files and outputs"
	@echo ""
	@echo "Pipeline stages:"
	@echo "  make stage0      - Data audit and preprocessing"
	@echo "  make stage1      - Centrality computation"
	@echo "  make stage2      - Community detection and k-shell"
	@echo "  make stage3      - SIS computation and typology"
	@echo "  make stage4      - IC calibration and single-seed validation"
	@echo "  make stage5      - Multi-seed IC benchmark"
	@echo "  make stage6      - ML detectability analysis"
	@echo ""
	@echo "Full pipeline:"
	@echo "  make run_all     - Run entire pipeline from raw data"
	@echo "  make test        - Run all tests"

# Environment setup
setup:
	@echo "Setting up Python environment..."
	$(PYTHON) -m venv $(VENV)
	$(VENV)/Scripts/pip install --upgrade pip
	$(VENV)/Scripts/pip install -r requirements.txt
	@echo "Setup complete. Activate with: source venv/Scripts/activate (Git Bash) or venv\\Scripts\\activate (CMD)"

# Clean outputs
clean:
	@echo "Cleaning generated files..."
	rm -rf $(OUTPUT_DIR)/stage*/*
	rm -rf logs/run_history/*
	rm -rf logs/timing/*
	@echo "Clean complete."

# Stage 0: Data audit and preprocessing
stage0:
	@echo "=== Stage 0: Data Audit and Preprocessing ==="
	$(PYTHON) $(SRC_DIR)/data/load_raw.py
	$(PYTHON) $(SRC_DIR)/data/preprocess_graph.py
	@echo "Stage 0 complete. Check $(OUTPUT_DIR)/stage0_data_quality/"

# Stage 1: Centrality computation
stage1: stage0
	@echo "=== Stage 1: Centrality Computation ==="
	$(PYTHON) $(SRC_DIR)/graph/centrality.py
	@echo "Stage 1 complete. Check $(OUTPUT_DIR)/stage1/"

# Stage 2: Community detection and k-shell
stage2: stage1
	@echo "=== Stage 2: Community Detection and K-Shell ==="
	$(PYTHON) $(SRC_DIR)/graph/community.py
	$(PYTHON) $(SRC_DIR)/graph/kshell.py
	@echo "Stage 2 complete. Check $(OUTPUT_DIR)/stage2/"

# Stage 3: SIS computation, typology, null model comparison
stage3: stage2
	@echo "=== Stage 3: SIS and Typology ==="
	$(PYTHON) $(SRC_DIR)/sis/compute_sis.py
	$(PYTHON) $(SRC_DIR)/sis/build_typology.py
	$(PYTHON) $(SRC_DIR)/sis/robustness.py
	$(PYTHON) $(SRC_DIR)/graph/null_model.py
	@echo "Stage 3 complete. Check $(OUTPUT_DIR)/stage3/"

# Stage 4: IC calibration and single-seed validation
stage4: stage3
	@echo "=== Stage 4: IC Calibration and Single-Seed Validation ==="
	$(PYTHON) $(SRC_DIR)/simulation/ic_calibration.py
	$(PYTHON) $(SRC_DIR)/simulation/run_single_seed_ic.py
	@echo "Stage 4 complete. Check $(OUTPUT_DIR)/stage4_single_seed/"

# Stage 5: Multi-seed IC benchmark
stage5: stage4
	@echo "=== Stage 5: Multi-Seed IC Benchmark ==="
	$(PYTHON) $(SRC_DIR)/simulation/seed_strategies.py
	$(PYTHON) $(SRC_DIR)/simulation/run_multi_seed_ic.py
	@echo "Stage 5 complete. Check $(OUTPUT_DIR)/stage5_multi_seed/"

# Stage 6: ML detectability analysis (full pipeline with RF and SHAP)
stage6: stage5
	@echo "=== Stage 6: ML Detectability Analysis ==="
	$(PYTHON) $(SRC_DIR)/ml/features_surface.py
	$(PYTHON) $(SRC_DIR)/ml/train_lr.py
	$(PYTHON) $(SRC_DIR)/ml/train_rf.py
	$(PYTHON) $(SRC_DIR)/ml/shap_analysis.py
	$(PYTHON) $(SRC_DIR)/ml/evaluate_metrics.py
	@echo "Stage 6 complete. Check $(OUTPUT_DIR)/stage6_ml/"

# Run entire pipeline
run_all: stage0 stage1 stage2 stage3 stage4 stage5 stage6
	@echo "=== Full Pipeline Complete ==="
	@echo "All outputs saved to $(OUTPUT_DIR)/"
	@echo "Figures saved to $(REPORTS_DIR)/figures/"
	@echo "Tables saved to $(REPORTS_DIR)/tables/"

# Run tests
test:
	@echo "Running tests..."
	$(PYTHON) -m pytest tests/ -v
	@echo "Tests complete."

# Generate final report
report: run_all
	@echo "Generating final report..."
	$(PYTHON) $(SRC_DIR)/evaluation/summary_tables.py
	@echo "Report artifacts ready in $(REPORTS_DIR)/"
