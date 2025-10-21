#!/bin/bash
# TF-IDF Baseline Training Script
# Trains TF-IDF baseline model with automatic hyperparameter tuning via grid search
# Usage: bash train_tfidf_baseline.sh --dataset your-username/game-reviews-sentiment

set -e  # Exit on error

# Default values
DATASET=""
GRIDSEARCH_SUBSET=0.1
FINAL_SUBSET=1.0
OUTPUT_BASE_DIR="model_phase/results"
N_JOBS=""
SKIP_GRIDSEARCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --gridsearch_subset)
            GRIDSEARCH_SUBSET="$2"
            shift 2
            ;;
        --final_subset)
            FINAL_SUBSET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --n_jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --skip_gridsearch)
            SKIP_GRIDSEARCH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash train_tfidf_baseline.sh --dataset DATASET [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET            HuggingFace dataset name (required)"
            echo "  --gridsearch_subset FLOAT    Subset for grid search (default: 0.1)"
            echo "  --final_subset FLOAT         Subset for final training (default: 1.0)"
            echo "  --output_dir DIR             Base output directory (default: model_phase/results)"
            echo "  --n_jobs N                   Number of CPU cores to use"
            echo "  --skip_gridsearch            Skip grid search and use provided hyperparameters"
            exit 1
            ;;
    esac
done

# Validate dataset
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required"
    echo "Usage: bash train_tfidf_baseline.sh --dataset your-username/game-reviews-sentiment"
    exit 1
fi

echo "============================================================"
echo "TF-IDF Baseline Model Training"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Grid Search Subset: $GRIDSEARCH_SUBSET"
echo "Final Training Subset: $FINAL_SUBSET"
echo "Output Directory: $OUTPUT_BASE_DIR"
if [ -n "$N_JOBS" ]; then
    echo "CPU Cores: $N_JOBS"
fi
echo ""

# Step 1: Grid Search (if not skipped)
if [ "$SKIP_GRIDSEARCH" = false ]; then
    echo "============================================================"
    echo "STEP 1/3: Running Grid Search"
    echo "============================================================"
    echo "Finding best hyperparameters on $GRIDSEARCH_SUBSET subset..."
    echo ""
    
    GRIDSEARCH_DIR="$OUTPUT_BASE_DIR/gridsearch"
    
    # Grid search parameters
    MAX_FEATURES_LIST=(5000 10000 20000)
    NGRAM_CONFIGS=("1 1" "1 2" "1 3")
    MAX_ITER_LIST=(500 1000 2000)
    
    # Results file
    RESULTS_FILE="$GRIDSEARCH_DIR/gridsearch_results.txt"
    mkdir -p "$GRIDSEARCH_DIR"
    echo "Grid Search Results - $(date)" > "$RESULTS_FILE"
    echo "Dataset: $DATASET" >> "$RESULTS_FILE"
    echo "Subset: $GRIDSEARCH_SUBSET" >> "$RESULTS_FILE"
    echo "==========================================" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    # Best results tracking
    BEST_F1=0
    BEST_CONFIG=""
    BEST_OUTPUT_DIR=""
    
    # Counter
    TOTAL_CONFIGS=$((${#MAX_FEATURES_LIST[@]} * ${#NGRAM_CONFIGS[@]} * ${#MAX_ITER_LIST[@]}))
    CURRENT=0
    
    echo "Total configurations to test: $TOTAL_CONFIGS"
    echo ""
    
    # Grid search loop
    for MAX_FEATURES in "${MAX_FEATURES_LIST[@]}"; do
        for NGRAM in "${NGRAM_CONFIGS[@]}"; do
            read NGRAM_MIN NGRAM_MAX <<< "$NGRAM"
            for MAX_ITER in "${MAX_ITER_LIST[@]}"; do
                CURRENT=$((CURRENT + 1))
                
                echo "=========================================="
                echo "Configuration $CURRENT/$TOTAL_CONFIGS"
                echo "=========================================="
                echo "max_features: $MAX_FEATURES"
                echo "ngram_range: ($NGRAM_MIN, $NGRAM_MAX)"
                echo "max_iter: $MAX_ITER"
                echo ""
                
                # Create unique output directory
                OUTPUT_DIR="$GRIDSEARCH_DIR/config_${CURRENT}_mf${MAX_FEATURES}_ng${NGRAM_MIN}${NGRAM_MAX}_mi${MAX_ITER}"
                
                # Build command (no HuggingFace upload during grid search)
                CMD="python model_phase/main_tfidf_baseline.py \
                    --dataset $DATASET \
                    --max_features $MAX_FEATURES \
                    --ngram_min $NGRAM_MIN \
                    --ngram_max $NGRAM_MAX \
                    --max_iter $MAX_ITER \
                    --subset $GRIDSEARCH_SUBSET \
                    --output_dir $OUTPUT_DIR \
                    --no_upload"
                
                # Add n_jobs if specified
                if [ -n "$N_JOBS" ]; then
                    CMD="$CMD --n_jobs $N_JOBS"
                fi
                
                # Run training
                echo "Running: $CMD"
                eval $CMD
                
                # Extract validation F1 score from results.json
                RESULTS_JSON="$OUTPUT_DIR/results.json"
                if [ -f "$RESULTS_JSON" ]; then
                    VAL_F1=$(python -c "import json; data=json.load(open('$RESULTS_JSON')); print(f\"{data.get('validation_f1', 0):.4f}\")")
                    VAL_ACC=$(python -c "import json; data=json.load(open('$RESULTS_JSON')); print(f\"{data.get('validation_accuracy', 0):.4f}\")")
                    TRAIN_TIME=$(python -c "import json; data=json.load(open('$RESULTS_JSON')); print(f\"{data.get('training_time', 0):.2f}\")")
                    
                    echo ""
                    echo "Results:"
                    echo "  Validation F1: $VAL_F1"
                    echo "  Validation Accuracy: $VAL_ACC"
                    echo "  Training Time: ${TRAIN_TIME}s"
                    echo ""
                    
                    # Log to results file
                    echo "Configuration $CURRENT:" >> "$RESULTS_FILE"
                    echo "  max_features: $MAX_FEATURES" >> "$RESULTS_FILE"
                    echo "  ngram_range: ($NGRAM_MIN, $NGRAM_MAX)" >> "$RESULTS_FILE"
                    echo "  max_iter: $MAX_ITER" >> "$RESULTS_FILE"
                    echo "  Validation F1: $VAL_F1" >> "$RESULTS_FILE"
                    echo "  Validation Accuracy: $VAL_ACC" >> "$RESULTS_FILE"
                    echo "  Training Time: ${TRAIN_TIME}s" >> "$RESULTS_FILE"
                    echo "  Output: $OUTPUT_DIR" >> "$RESULTS_FILE"
                    echo "" >> "$RESULTS_FILE"
                    
                    # Update best if this is better
                    VAL_F1_NUM=$(python -c "print(float('$VAL_F1'))")
                    BEST_F1_NUM=$(python -c "print(float('$BEST_F1'))")
                    IS_BETTER=$(python -c "print('yes' if $VAL_F1_NUM > $BEST_F1_NUM else 'no')")
                    
                    if [ "$IS_BETTER" = "yes" ]; then
                        BEST_F1=$VAL_F1
                        BEST_CONFIG="max_features=$MAX_FEATURES, ngram=($NGRAM_MIN,$NGRAM_MAX), max_iter=$MAX_ITER"
                        BEST_OUTPUT_DIR=$OUTPUT_DIR
                        echo "*** NEW BEST CONFIGURATION! ***"
                        echo ""
                    fi
                else
                    echo "Error: Results file not found at $RESULTS_JSON"
                    echo "Configuration $CURRENT: ERROR - Results file not found" >> "$RESULTS_FILE"
                    echo "" >> "$RESULTS_FILE"
                fi
                
                echo ""
            done
        done
    done
    
    # Save best config summary
    BEST_CONFIG_FILE="$GRIDSEARCH_DIR/best_config.txt"
    echo "Best Configuration Found" > "$BEST_CONFIG_FILE"
    echo "======================" >> "$BEST_CONFIG_FILE"
    echo "Configuration: $BEST_CONFIG" >> "$BEST_CONFIG_FILE"
    echo "Validation F1: $BEST_F1" >> "$BEST_CONFIG_FILE"
    echo "Model Directory: $BEST_OUTPUT_DIR" >> "$BEST_CONFIG_FILE"
    
    echo "All results saved to: $RESULTS_FILE"
    echo "Best configuration saved to: $BEST_CONFIG_FILE"
    
    # Build grid search command
    GS_CMD="bash model_phase/gridsearch_baseline.sh --dataset $DATASET --subset $GRIDSEARCH_SUBSET --output_dir $GRIDSEARCH_DIR"
    
    if [ -n "$N_JOBS" ]; then
        GS_CMD="$GS_CMD --n_jobs $N_JOBS"
    fi
    
    echo "Running: $GS_CMD"
    echo ""
    
    eval $GS_CMD
    
    echo ""
    echo "✓ Grid search complete!"
    echo ""
else
    echo "============================================================"
    echo "STEP 1/3: Skipping Grid Search"
    echo "============================================================"
    echo "Using provided hyperparameters..."
    echo ""
    GRIDSEARCH_DIR="$OUTPUT_BASE_DIR/gridsearch"
fi

# Step 2: Extract Best Configuration
echo "============================================================"
echo "STEP 2/3: Extracting Best Configuration"
echo "============================================================"

BEST_CONFIG_FILE="$GRIDSEARCH_DIR/best_config.txt"

if [ ! -f "$BEST_CONFIG_FILE" ]; then
    echo "Error: Best config file not found at $BEST_CONFIG_FILE"
    echo "Make sure grid search completed successfully."
    exit 1
fi

echo "Reading best configuration from: $BEST_CONFIG_FILE"
echo ""
cat "$BEST_CONFIG_FILE"
echo ""

# Extract hyperparameters from best config
# Parse the line: "Configuration: max_features=10000, ngram=(1,2), max_iter=1000"
CONFIG_LINE=$(grep "Configuration:" "$BEST_CONFIG_FILE")
MAX_FEATURES=$(echo "$CONFIG_LINE" | sed -n 's/.*max_features=\([0-9]*\).*/\1/p')
NGRAM_MIN=$(echo "$CONFIG_LINE" | sed -n 's/.*ngram=(\([0-9]*\),\([0-9]*\)).*/\1/p')
NGRAM_MAX=$(echo "$CONFIG_LINE" | sed -n 's/.*ngram=(\([0-9]*\),\([0-9]*\)).*/\2/p')
MAX_ITER=$(echo "$CONFIG_LINE" | sed -n 's/.*max_iter=\([0-9]*\).*/\1/p')

echo "Extracted hyperparameters:"
echo "  max_features: $MAX_FEATURES"
echo "  ngram_range: ($NGRAM_MIN, $NGRAM_MAX)"
echo "  max_iter: $MAX_ITER"
echo ""

# Step 3: Final Training with Best Configuration
echo "============================================================"
echo "STEP 3/3: Final Training with Best Configuration"
echo "============================================================"
echo "Training on $FINAL_SUBSET subset with best hyperparameters..."
echo "This model will be uploaded to HuggingFace Hub."
echo ""

# Build final training command
FINAL_CMD="python model_phase/main_tfidf_baseline.py \
    --dataset $DATASET \
    --max_features $MAX_FEATURES \
    --ngram_min $NGRAM_MIN \
    --ngram_max $NGRAM_MAX \
    --max_iter $MAX_ITER \
    --subset $FINAL_SUBSET"

if [ -n "$N_JOBS" ]; then
    FINAL_CMD="$FINAL_CMD --n_jobs $N_JOBS"
fi

echo "Running: $FINAL_CMD"
echo ""

eval $FINAL_CMD

# Step 4: Summary
echo ""
echo "============================================================"
echo "PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "Summary:"
echo "1. ✓ Grid search found best hyperparameters"
echo "2. ✓ Final model trained with optimal configuration"
echo "3. ✓ Results uploaded to HuggingFace Hub"
echo ""
echo "Best Configuration Used:"
echo "  max_features: $MAX_FEATURES"
echo "  ngram_range: ($NGRAM_MIN, $NGRAM_MAX)"
echo "  max_iter: $MAX_ITER"
echo ""
echo "Check your HuggingFace profile for the uploaded model!"
echo ""
