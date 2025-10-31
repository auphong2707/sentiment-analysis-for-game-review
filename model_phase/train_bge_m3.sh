#!/bin/bash
# BGE-M3 Training Script
# Trains BGE-M3 model with automatic hyperparameter tuning via grid search
# Usage: bash train_bge_m3.sh --dataset your-username/game-reviews-sentiment

set -e  # Exit on error

# BGE-M3 parameters (constants - not tuned)
readonly MAX_LENGTH=512
readonly BATCH_SIZE=64
readonly KERNEL="rbf"

# Grid search parameters (specific C-gamma pairs)
# Pairs: (1,0.25), (1,0.5), (10,'scale'), (10,0.25), (10,0.5)
readonly C_VALUES=(1 1 10 10 10)
readonly GAMMA_VALUES=(0.25 0.5 'scale' 0.25 0.5)

# Load dataset from .env if available
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep HF_DATASET_NAME | xargs)
fi

# Default values
DATASET="${HF_DATASET_NAME:-}"
GRIDSEARCH_SUBSET=0.1
FINAL_SUBSET=1.0
OUTPUT_BASE_DIR="model_phase/results"
USE_WANDB=true
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
            echo "Usage: bash train_bge_m3.sh --dataset DATASET [OPTIONS]"
            exit 1
            ;;
    esac
done

# Validate dataset
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required (not found in .env or command line)"
    echo "Usage: bash train_bge_m3.sh --dataset your-username/game-reviews-sentiment"
    exit 1
fi

echo "============================================================"
echo "BGE-M3 Model Training"
echo "============================================================"
echo "Dataset: $DATASET"
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
    echo "STEP 1/3: Running Efficient Grid Search"
    echo "============================================================"
    echo "Finding best hyperparameters on $GRIDSEARCH_SUBSET subset..."
    echo "(Embeddings will be loaded ONCE, then SVM trained multiple times)"
    echo ""
    
    GRIDSEARCH_DIR="$OUTPUT_BASE_DIR/gridsearch"
    
    # Build grid search command
    GRIDSEARCH_CMD="python model_phase/main_bge_m3.py \
        --grid_search \
        --dataset $DATASET \
        --max_length $MAX_LENGTH \
        --batch_size $BATCH_SIZE \
        --kernel $KERNEL \
        --subset $GRIDSEARCH_SUBSET \
        --output_dir $GRIDSEARCH_DIR"
    
    # Convert arrays to space-separated strings
    C_VALUES_STR="${C_VALUES[@]}"
    GAMMA_VALUES_STR="${GAMMA_VALUES[@]}"
    
    GRIDSEARCH_CMD="$GRIDSEARCH_CMD --C_values $C_VALUES_STR --gamma_values $GAMMA_VALUES_STR"
    
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
CONFIG_LINE=$(grep "Configuration:" "$BEST_CONFIG_FILE")
BEST_C=$(echo "$CONFIG_LINE" | sed -n 's/.*C=\([0-9.e+-]*\).*/\1/p')
BEST_GAMMA=$(echo "$CONFIG_LINE" | sed -n 's/.*gamma=\([a-z0-9.e+-]*\).*/\1/p')

# Validate extracted values
if [ -z "$BEST_C" ] || [ -z "$BEST_GAMMA" ]; then
    echo "Error: Could not extract hyperparameters from best config"
    echo "Config line: $CONFIG_LINE"
    exit 1
fi

echo "Extracted hyperparameters:"
echo "  C: $BEST_C"
echo "  gamma: $BEST_GAMMA"
echo "  kernel: $KERNEL (fixed)"
echo "  max_length: $MAX_LENGTH (fixed)"
echo "  batch_size: $BATCH_SIZE (fixed)"
echo ""

# Step 3: Final Training with Best Configuration
echo "============================================================"
echo "STEP 3/3: Final Training with Best Configuration"
echo "============================================================"
echo "Training on $FINAL_SUBSET subset with best hyperparameters..."
echo "This model will be uploaded to HuggingFace Hub."
echo ""

# Create experiment name for final training
FINAL_EXPERIMENT_NAME="bge_m3_official_${BEST_C}"

# Build final training command
FINAL_CMD="python model_phase/main_bge_m3.py \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --C $BEST_C \
    --gamma $BEST_GAMMA \
    --kernel $KERNEL \
    --subset $FINAL_SUBSET \
    --experiment_name $FINAL_EXPERIMENT_NAME"

if [ "$USE_WANDB" = true ]; then
    FINAL_CMD="$FINAL_CMD --use_wandb"
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
echo "  C: $BEST_C"
echo "  gamma: $BEST_GAMMA"
echo "  kernel: $KERNEL (fixed)"
echo "  max_length: $MAX_LENGTH (fixed)"
echo "  batch_size: $BATCH_SIZE (fixed)"
echo ""
echo "Check your HuggingFace profile for the uploaded model!"
echo ""
