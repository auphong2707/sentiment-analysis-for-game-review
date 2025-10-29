#!/bin/bash
# BGE-M3 Training Script
# Trains BGE-M3 model with automatic hyperparameter tuning via grid search
# Usage: bash train_bge_m3.sh --dataset your-username/game-reviews-sentiment

set -e  # Exit on error

# BGE-M3 parameters (constants - not tuned)
readonly MAX_LENGTH=512
readonly BATCH_SIZE=64
readonly KERNEL="rbf"

# Grid search parameters (tune C and gamma for RBF kernel)
readonly C_VALUES=(0.1 1 3 10)
readonly GAMMA_VALUES=(0.125 0.25 0.5)

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
    echo "STEP 1/3: Running Grid Search"
    echo "============================================================"
    echo "Finding best hyperparameters on $GRIDSEARCH_SUBSET subset..."
    echo ""
    
    GRIDSEARCH_DIR="$OUTPUT_BASE_DIR/gridsearch"
    
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
    TOTAL_CONFIGS=$((${#C_VALUES[@]} * ${#GAMMA_VALUES[@]}))
    CURRENT=0
    
    echo "Total configurations to test: $TOTAL_CONFIGS"
    echo "BGE-M3 Settings (fixed): max_length=$MAX_LENGTH, batch_size=$BATCH_SIZE, kernel=$KERNEL"
    echo ""
    
    # Grid search loop (C x gamma)
    for C_VALUE in "${C_VALUES[@]}"; do
        for GAMMA_VALUE in "${GAMMA_VALUES[@]}"; do
            CURRENT=$((CURRENT + 1))
            
            echo "=========================================="
            echo "Configuration $CURRENT/$TOTAL_CONFIGS"
            echo "=========================================="
            echo "C (regularization): $C_VALUE"
            echo "gamma: $GAMMA_VALUE"
            echo ""
        
            # Create unique output directory
            OUTPUT_DIR="$GRIDSEARCH_DIR/config_${CURRENT}_C${C_VALUE}_gamma${GAMMA_VALUE}"
            
            # Create experiment name for WandB
            EXPERIMENT_NAME="bge_m3_ex_${CURRENT}"
            
            # Build command (no HuggingFace upload during grid search)
            CMD="python model_phase/main_bge_m3.py \
                --dataset $DATASET \
                --max_length $MAX_LENGTH \
                --batch_size $BATCH_SIZE \
                --C $C_VALUE \
                --gamma $GAMMA_VALUE \
                --kernel $KERNEL \
                --subset $GRIDSEARCH_SUBSET \
                --output_dir $OUTPUT_DIR \
                --no_upload \
                --skip_test_eval \
                --experiment_name $EXPERIMENT_NAME"
        
            # Add wandb if specified
            if [ "$USE_WANDB" = true ]; then
                CMD="$CMD --use_wandb"
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
                echo "  C: $C_VALUE" >> "$RESULTS_FILE"
                echo "  gamma: $GAMMA_VALUE" >> "$RESULTS_FILE"
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
                    BEST_CONFIG="C=$C_VALUE, gamma=$GAMMA_VALUE"
                    BEST_OUTPUT_DIR=$OUTPUT_DIR
                    BEST_C=$C_VALUE
                    BEST_GAMMA=$GAMMA_VALUE
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
    
    # Save best config summary
    BEST_CONFIG_FILE="$GRIDSEARCH_DIR/best_config.txt"
    echo "Best Configuration Found" > "$BEST_CONFIG_FILE"
    echo "======================" >> "$BEST_CONFIG_FILE"
    echo "Configuration: $BEST_CONFIG" >> "$BEST_CONFIG_FILE"
    echo "Validation F1: $BEST_F1" >> "$BEST_CONFIG_FILE"
    echo "Model Directory: $BEST_OUTPUT_DIR" >> "$BEST_CONFIG_FILE"
    
    echo "All results saved to: $RESULTS_FILE"
    echo "Best configuration saved to: $BEST_CONFIG_FILE"
    
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
BEST_C=$(echo "$CONFIG_LINE" | sed -n 's/.*C=\([0-9.]*\).*/\1/p')
BEST_GAMMA=$(echo "$CONFIG_LINE" | sed -n 's/.*gamma=\([a-z0-9.]*\).*/\1/p')

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
echo ""ommand
FINAL_CMD="python model_phase/main_bge_m3.py \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --C $BEST_C \
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
echo "  max_length: $MAX_LENGTH (fixed)"
echo "  batch_size: $BATCH_SIZE (fixed)"
echo ""
echo "Check your HuggingFace profile for the uploaded model!"
echo ""
