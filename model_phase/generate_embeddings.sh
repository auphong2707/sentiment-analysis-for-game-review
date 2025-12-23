#!/bin/bash
# bilingual-embedding-base Embedding Generation Script
# Generates bilingual-embedding-base embeddings for downstream classifiers (e.g., XGBoost)
# Usage: bash generate_embeddings.sh --dataset your-username/game-reviews-sentiment

set -e  # Exit on error

# bilingual-embedding-base parameters (constants)
readonly MAX_LENGTH=256
readonly BATCH_SIZE=256

# Load dataset from .env if available
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep HF_DATASET_NAME | xargs)
fi

# Default values
DATASET="${HF_DATASET_NAME:-}"
SUBSET=1.0
OUTPUT_BASE_DIR="model_phase/results"
USE_WANDB=false
EXPERIMENT_NAME=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_BASE_DIR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash generate_embeddings.sh --dataset DATASET [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET          HuggingFace dataset name (required)"
            echo "  --subset FRACTION          Fraction of data to use (default: 1.0)"
            echo "  --output_dir DIR           Output directory (default: model_phase/results)"
            echo "  --batch_size SIZE          Batch size (default: 64)"
            echo "  --max_length LENGTH        Max sequence length (default: 256, ~4x faster than 512)"
            echo "  --use_wandb                Enable WandB logging"
            echo "  --experiment_name NAME     Custom experiment name"
            exit 1
            ;;
    esac
done

# Validate dataset
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required (not found in .env or command line)"
    echo "Usage: bash generate_embeddings.sh --dataset your-username/game-reviews-sentiment"
    exit 1
fi

echo "============================================================"
echo "bilingual-embedding-base Embedding Generation"
echo "============================================================"
echo "Dataset: $DATASET"
echo "Subset: $SUBSET"
echo "Batch Size: $BATCH_SIZE"
echo "Max Length: $MAX_LENGTH"
echo "WandB Logging: $USE_WANDB"
if [ -n "$EXPERIMENT_NAME" ]; then
    echo "Experiment Name: $EXPERIMENT_NAME"
fi
echo ""

# Check for GPU
echo "Checking for GPU availability..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "Warning: Could not check GPU availability"
echo ""

# Build embedding generation command
if [ -n "$EXPERIMENT_NAME" ]; then
    EMBEDDING_DIR="$OUTPUT_BASE_DIR/$EXPERIMENT_NAME"
else
    EMBEDDING_DIR="$OUTPUT_BASE_DIR/bilingual_embeddings"
fi

EMBEDDING_CMD="python model_phase/generate_bilingual_embeddings.py \
    --dataset $DATASET \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --subset $SUBSET \
    --output_dir $EMBEDDING_DIR"

if [ "$USE_WANDB" = true ]; then
    EMBEDDING_CMD="$EMBEDDING_CMD --use_wandb"
fi

if [ -n "$EXPERIMENT_NAME" ]; then
    EMBEDDING_CMD="$EMBEDDING_CMD --experiment_name $EXPERIMENT_NAME"
fi

echo "Running: $EMBEDDING_CMD"
echo ""

eval $EMBEDDING_CMD

echo ""
echo "============================================================"
echo "Embedding Generation Complete!"
echo "============================================================"
echo "Embeddings saved to: $EMBEDDING_DIR/checkpoints/"
echo ""
echo "Next steps:"
echo "  1. Train XGBoost classifier:"
echo "     bash model_phase/train_xgboost.sh --checkpoint_dir $EMBEDDING_DIR/checkpoints"
echo ""
echo "  2. Or run manually:"
echo "     python model_phase/main_xgboost.py --checkpoint_dir $EMBEDDING_DIR/checkpoints"
echo ""
