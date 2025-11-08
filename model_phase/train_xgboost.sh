#!/bin/bash
# XGBoost Training Script (Using Pre-computed Embeddings)
# Loads embeddings from checkpoint and trains XGBoost classifier
# Usage: bash train_xgboost.sh --checkpoint_dir /path/to/checkpoints

set -e  # Exit on error

# XGBoost default parameters (will be tuned in grid search)
readonly DEFAULT_N_ESTIMATORS=100
readonly DEFAULT_MAX_DEPTH=6
readonly DEFAULT_LEARNING_RATE=0.3
readonly DEFAULT_SUBSAMPLE=1.0
readonly DEFAULT_COLSAMPLE_BYTREE=1.0

# Grid search parameters
readonly N_ESTIMATORS_VALUES=(500 1000 1500)
readonly MAX_DEPTH_VALUES=(6 8 10)
readonly LEARNING_RATE_VALUES=(0.05 0.1 0.2)

# Load dataset from .env if available
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep HF_DATASET_NAME | xargs)
fi

# Default values
CHECKPOINT_DIR=""
DATASET="${HF_DATASET_NAME:-}"
GRIDSEARCH_SUBSET=0.1
FINAL_SUBSET=1.0
OUTPUT_BASE_DIR="model_phase/results"
USE_WANDB=true
SKIP_GRIDSEARCH=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint_dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
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
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --no_wandb)
            USE_WANDB=false
            shift
            ;;
        --skip_gridsearch)
            SKIP_GRIDSEARCH=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash train_xgboost.sh --checkpoint_dir /path/to/checkpoints [OPTIONS]"
            exit 1
            ;;
    esac
done

# Validate checkpoint directory
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: --checkpoint_dir is required"
    echo "Usage: bash train_xgboost.sh --checkpoint_dir /path/to/checkpoints"
    exit 1
fi

if [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Error: Checkpoint directory not found: $CHECKPOINT_DIR"
    exit 1
fi

# Check for required checkpoint files (bao gồm cả test embeddings)
REQUIRED_FILES=(
    "$CHECKPOINT_DIR/checkpoint_state.json"
    "$CHECKPOINT_DIR/train_embeddings_embeddings.npz"
    "$CHECKPOINT_DIR/val_embeddings_embeddings.npz"
    "$CHECKPOINT_DIR/test_embeddings_embeddings.npz"
)

echo "Kiểm tra các file checkpoint cần thiết..."
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: File checkpoint bắt buộc không tìm thấy: $file"
        exit 1
    fi
    echo "  ✓ $(basename $file)"
done

echo "============================================================"
echo "XGBoost Model Training (Using Pre-computed Embeddings)"
echo "============================================================"
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "Grid Search Subset: $GRIDSEARCH_SUBSET"
echo "Final Training Subset: $FINAL_SUBSET"
echo "WandB Logging: $USE_WANDB"
echo ""

# Check for GPU
echo "Checking for GPU availability..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "Warning: Could not check GPU availability"
echo ""

# Step 1: Grid Search (if not skipped)
if [ "$SKIP_GRIDSEARCH" = false ]; then
    echo "============================================================"
    echo "STEP 1/3: Running Grid Search on 10% Data"
    echo "============================================================"
    echo "Finding best hyperparameters..."
    echo ""
    
    GRIDSEARCH_DIR="$OUTPUT_BASE_DIR/gridsearch_xgboost"
    
    # Build grid search command
    GRIDSEARCH_CMD="python model_phase/main_xgboost_from_checkpoint.py \
        --grid_search \
        --checkpoint_dir $CHECKPOINT_DIR \
        --subset $GRIDSEARCH_SUBSET \
        --output_dir $GRIDSEARCH_DIR"
    
    # Convert arrays to space-separated strings
    N_EST_STR="${N_ESTIMATORS_VALUES[@]}"
    MAX_D_STR="${MAX_DEPTH_VALUES[@]}"
    LR_STR="${LEARNING_RATE_VALUES[@]}"
    
    GRIDSEARCH_CMD="$GRIDSEARCH_CMD \
        --n_estimators_values $N_EST_STR \
        --max_depth_values $MAX_D_STR \
        --learning_rate_values $LR_STR"
    
    echo "Running: $GRIDSEARCH_CMD"
    echo ""
    
    eval $GRIDSEARCH_CMD
    
    echo ""
    echo "✓ Grid search complete!"
    echo ""
else
    echo "============================================================"
    echo "STEP 1/3: Skipping Grid Search"
    echo "============================================================"
    echo "Using default hyperparameters..."
    echo ""
    GRIDSEARCH_DIR="$OUTPUT_BASE_DIR/gridsearch_xgboost"
fi

# Step 2: Extract Best Configuration
echo "============================================================"
echo "STEP 2/3: Extracting Best Configuration"
echo "============================================================"

BEST_CONFIG_FILE="$GRIDSEARCH_DIR/best_config.txt"

if [ ! -f "$BEST_CONFIG_FILE" ]; then
    echo "Warning: Best config file not found at $BEST_CONFIG_FILE"
    echo "Using default hyperparameters..."
    BEST_N_ESTIMATORS=$DEFAULT_N_ESTIMATORS
    BEST_MAX_DEPTH=$DEFAULT_MAX_DEPTH
    BEST_LEARNING_RATE=$DEFAULT_LEARNING_RATE
else
    echo "Reading best configuration from: $BEST_CONFIG_FILE"
    echo ""
    cat "$BEST_CONFIG_FILE"
    echo ""
    
    # Extract hyperparameters from best config
    BEST_N_ESTIMATORS=$(grep "n_estimators:" "$BEST_CONFIG_FILE" | awk '{print $2}')
    BEST_MAX_DEPTH=$(grep "max_depth:" "$BEST_CONFIG_FILE" | awk '{print $2}')
    BEST_LEARNING_RATE=$(grep "learning_rate:" "$BEST_CONFIG_FILE" | awk '{print $2}')
    
    # Validate extracted values
    if [ -z "$BEST_N_ESTIMATORS" ] || [ -z "$BEST_MAX_DEPTH" ] || [ -z "$BEST_LEARNING_RATE" ]; then
        echo "Warning: Could not extract all hyperparameters from best config"
        echo "Using default values..."
        BEST_N_ESTIMATORS=$DEFAULT_N_ESTIMATORS
        BEST_MAX_DEPTH=$DEFAULT_MAX_DEPTH
        BEST_LEARNING_RATE=$DEFAULT_LEARNING_RATE
    fi
fi

echo "Hyperparameters for final training:"
echo "  n_estimators: $BEST_N_ESTIMATORS"
echo "  max_depth: $BEST_MAX_DEPTH"
echo "  learning_rate: $BEST_LEARNING_RATE"
echo "  subsample: $DEFAULT_SUBSAMPLE (fixed)"
echo "  colsample_bytree: $DEFAULT_COLSAMPLE_BYTREE (fixed)"
echo ""

# Step 3: Final Training with Best Configuration
echo "============================================================"
echo "STEP 3/3: Final Training with Best Configuration"
echo "============================================================"
echo "Training on $FINAL_SUBSET subset with best hyperparameters..."
echo "This model will be uploaded to HuggingFace Hub."
echo ""

# Create experiment name for final training
FINAL_EXPERIMENT_NAME="xgboost_n${BEST_N_ESTIMATORS}_d${BEST_MAX_DEPTH}_lr${BEST_LEARNING_RATE}"

# Build final training command
FINAL_CMD="python model_phase/main_xgboost_from_checkpoint.py \
    --checkpoint_dir $CHECKPOINT_DIR \
    --n_estimators $BEST_N_ESTIMATORS \
    --max_depth $BEST_MAX_DEPTH \
    --learning_rate $BEST_LEARNING_RATE \
    --subsample $DEFAULT_SUBSAMPLE \
    --colsample_bytree $DEFAULT_COLSAMPLE_BYTREE \
    --subset $FINAL_SUBSET \
    --experiment_name $FINAL_EXPERIMENT_NAME"

if [ "$USE_WANDB" = true ]; then
    FINAL_CMD="$FINAL_CMD --use_wandb"
fi

# Thêm dataset name và HF repo để tự động upload
if [ -n "$DATASET" ]; then
    FINAL_CMD="$FINAL_CMD --dataset $DATASET"
    # Tự động upload lên HuggingFace với repo name = dataset name
    FINAL_CMD="$FINAL_CMD --hf_repo $DATASET"
    echo "📤 Sẽ upload kết quả lên HuggingFace Hub: $DATASET"
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
echo "3. ✓ Results saved to output directory"
echo ""
echo "Best Configuration Used:"
echo "  n_estimators: $BEST_N_ESTIMATORS"
echo "  max_depth: $BEST_MAX_DEPTH"
echo "  learning_rate: $BEST_LEARNING_RATE"
echo "  subsample: $DEFAULT_SUBSAMPLE"
echo "  colsample_bytree: $DEFAULT_COLSAMPLE_BYTREE"
echo ""
echo "Checkpoint Directory: $CHECKPOINT_DIR"
echo "Output Directory: $OUTPUT_BASE_DIR"
echo ""
