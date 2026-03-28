#!/bin/bash
# SNA Twitch Influencer Project - Full Pipeline Runner
# Cross-platform bash script (Linux/Mac/Windows Git Bash)

set -e  # Exit on error

echo "=================================================="
echo "SNA Twitch Influencer Project - Full Pipeline"
echo "=================================================="
echo ""

# Configuration
PYTHON=${PYTHON:-python}
SRC_DIR="src"
OUTPUT_DIR="outputs"
REPORTS_DIR="reports"
LOG_DIR="logs"

# Create directories if not exist
mkdir -p "$OUTPUT_DIR/stage0_data_quality"
mkdir -p "$OUTPUT_DIR/stage1"
mkdir -p "$OUTPUT_DIR/stage2"
mkdir -p "$OUTPUT_DIR/stage3"
mkdir -p "$OUTPUT_DIR/stage3_ic_calibration"
mkdir -p "$OUTPUT_DIR/stage4_single_seed"
mkdir -p "$OUTPUT_DIR/stage5_multi_seed"
mkdir -p "$OUTPUT_DIR/stage6_ml"
mkdir -p "$REPORTS_DIR/figures"
mkdir -p "$REPORTS_DIR/tables"
mkdir -p "$LOG_DIR/run_history"
mkdir -p "$LOG_DIR/timing"

# Record start time
START_TIME=$(date +%s)
RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/run_history/run_${RUN_ID}.log"

echo "Run ID: $RUN_ID"
echo "Log file: $LOG_FILE"
echo ""

# Function to run stage with timing
run_stage() {
    local stage_name=$1
    local stage_script=$2

    echo "=== $stage_name ===" | tee -a "$LOG_FILE"
    local stage_start=$(date +%s)

    $PYTHON "$stage_script" 2>&1 | tee -a "$LOG_FILE"

    local stage_end=$(date +%s)
    local stage_duration=$((stage_end - stage_start))
    echo "[$stage_name] completed in ${stage_duration}s" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
}

# ============================================
# STAGE 0: Data Audit and Preprocessing
# ============================================
echo "STAGE 0: Data Audit and Preprocessing" | tee -a "$LOG_FILE"
run_stage "Load Raw Data" "$SRC_DIR/data/load_raw.py"
run_stage "Preprocess Graph" "$SRC_DIR/data/preprocess_graph.py"

# ============================================
# STAGE 1: Centrality Computation
# ============================================
echo "STAGE 1: Centrality Computation" | tee -a "$LOG_FILE"
run_stage "Compute Centralities" "$SRC_DIR/graph/centrality.py"

# ============================================
# STAGE 2: Community Detection and K-Shell
# ============================================
echo "STAGE 2: Community Detection and K-Shell" | tee -a "$LOG_FILE"
run_stage "Community Detection" "$SRC_DIR/graph/community.py"
run_stage "K-Shell Decomposition" "$SRC_DIR/graph/kshell.py"

# ============================================
# STAGE 3: SIS, Typology, and Null Model
# ============================================
echo "STAGE 3: SIS and Typology" | tee -a "$LOG_FILE"
run_stage "Compute SIS" "$SRC_DIR/sis/compute_sis.py"
run_stage "Build Typology" "$SRC_DIR/sis/build_typology.py"
run_stage "Robustness Analysis" "$SRC_DIR/sis/robustness.py"
run_stage "Null Model Comparison" "$SRC_DIR/graph/null_model.py"

# ============================================
# STAGE 4: IC Calibration and Single-Seed
# ============================================
echo "STAGE 4: IC Calibration and Single-Seed" | tee -a "$LOG_FILE"
run_stage "IC Calibration" "$SRC_DIR/simulation/ic_calibration.py"
run_stage "Single-Seed IC" "$SRC_DIR/simulation/run_single_seed_ic.py"

# ============================================
# STAGE 5: Multi-Seed IC Benchmark
# ============================================
echo "STAGE 5: Multi-Seed IC Benchmark" | tee -a "$LOG_FILE"
run_stage "Seed Strategies" "$SRC_DIR/simulation/seed_strategies.py"
run_stage "Multi-Seed IC" "$SRC_DIR/simulation/run_multi_seed_ic.py"

# ============================================
# STAGE 6: ML Detectability (Full pipeline with RF and SHAP)
# ============================================
echo "STAGE 6: ML Detectability" | tee -a "$LOG_FILE"
run_stage "Feature Engineering" "$SRC_DIR/ml/features_surface.py"
run_stage "Train LR Models" "$SRC_DIR/ml/train_lr.py"
run_stage "Train RF Model" "$SRC_DIR/ml/train_rf.py"
run_stage "SHAP Analysis" "$SRC_DIR/ml/shap_analysis.py"
run_stage "Evaluate Metrics" "$SRC_DIR/ml/evaluate_metrics.py"

# ============================================
# Summary Tables and Final Report
# ============================================
echo "FINAL: Summary Tables" | tee -a "$LOG_FILE"
run_stage "Generate Summary" "$SRC_DIR/evaluation/summary_tables.py"

# ============================================
# Completion Summary
# ============================================
END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))

echo "==================================================" | tee -a "$LOG_FILE"
echo "PIPELINE COMPLETE" | tee -a "$LOG_FILE"
echo "==================================================" | tee -a "$LOG_FILE"
echo "Total runtime: ${TOTAL_DURATION}s ($(($TOTAL_DURATION / 60))m $(($TOTAL_DURATION % 60))s)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "Outputs:" | tee -a "$LOG_FILE"
echo "  - Stage outputs: $OUTPUT_DIR/" | tee -a "$LOG_FILE"
echo "  - Figures: $REPORTS_DIR/figures/" | tee -a "$LOG_FILE"
echo "  - Tables: $REPORTS_DIR/tables/" | tee -a "$LOG_FILE"
echo "  - Log: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Save timing summary
echo "{\"run_id\": \"$RUN_ID\", \"total_seconds\": $TOTAL_DURATION}" > "$LOG_DIR/timing/timing_${RUN_ID}.json"

echo "Done!"
