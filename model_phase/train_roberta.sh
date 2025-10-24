#!/bin/bash
# RoBERTa Training Script (Bash)
# Fine-tunes RoBERTa for sentiment analysis on game reviews
# Usage: bash model_phase/train_roberta.sh --dataset your-username/game-reviews-sentiment

set -e  # Exit on error

# Load dataset from .env if available
if [ -f .env ]; then
    export $(grep -v '^#' .env | grep HF_DATASET_NAME | xargs)
fi

# Default values
DATASET="${HF_DATASET_NAME:-}"
MODEL_NAME="roberta-base"
MAX_LENGTH=512
BATCH_SIZE=16
LEARNING_RATE=2e-5
NUM_EPOCHS=3
WARMUP_STEPS=0
WEIGHT_DECAY=0.01
SUBSET=1.0
OUTPUT_DIR=""
USE_WANDB=false
NO_UPLOAD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --max_length)
            MAX_LENGTH="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_epochs)
            NUM_EPOCHS="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --weight_decay)
            WEIGHT_DECAY="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --use_wandb)
            USE_WANDB=true
            shift
            ;;
        --no_upload)
            NO_UPLOAD=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: bash model_phase/train_roberta.sh --dataset DATASET [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dataset DATASET            HuggingFace dataset name (required)"
            echo "  --model_name MODEL           Pre-trained model name (default: roberta-base)"
            echo "  --max_length LENGTH          Maximum sequence length (default: 512)"
            echo "  --batch_size SIZE            Batch size for training (default: 16)"
            echo "  --learning_rate RATE         Learning rate (default: 2e-5)"
            echo "  --num_epochs EPOCHS          Number of training epochs (default: 3)"
            echo "  --warmup_steps STEPS         Warmup steps (default: 0)"
            echo "  --weight_decay DECAY         Weight decay (default: 0.01)"
            echo "  --subset FRACTION            Fraction of data to use (default: 1.0)"
            echo "  --output_dir DIR             Output directory (default: auto-generated)"
            echo "  --use_wandb                  Use WandB for experiment tracking"
            echo "  --no_upload                  Skip uploading to HuggingFace Hub"
            exit 1
            ;;
    esac
done

# Validate dataset
if [ -z "$DATASET" ]; then
    echo "Error: --dataset is required (not found in .env or command line)"
    echo "Usage: bash model_phase/train_roberta.sh --dataset your-username/game-reviews-sentiment"
    echo "Or set HF_DATASET_NAME in .env file"
    exit 1
fi

echo "============================================================"
echo "RoBERTa Fine-tuning for Sentiment Analysis"
echo "============================================================"
echo ""
echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Model: $MODEL_NAME"
echo "  Max Length: $MAX_LENGTH"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo "  Epochs: $NUM_EPOCHS"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Data Subset: $SUBSET"
if [ -n "$OUTPUT_DIR" ]; then
    echo "  Output Directory: $OUTPUT_DIR"
fi
echo ""

# Check for GPU
echo "Checking for GPU availability..."
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU Name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "Warning: Could not check GPU availability"
echo ""

# Build training command
CMD="python model_phase/main_roberta.py \
    --dataset $DATASET \
    --model_name $MODEL_NAME \
    --max_length $MAX_LENGTH \
    --batch_size $BATCH_SIZE \
    --learning_rate $LEARNING_RATE \
    --num_epochs $NUM_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --weight_decay $WEIGHT_DECAY \
    --subset $SUBSET"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output_dir $OUTPUT_DIR"
fi

if [ "$USE_WANDB" = true ]; then
    CMD="$CMD --use_wandb"
fi

if [ "$NO_UPLOAD" = true ]; then
    CMD="$CMD --no_upload"
fi

echo "============================================================"
echo "Starting Training"
echo "============================================================"
echo ""

# Run training
eval $CMD

echo ""
echo "============================================================"
echo "Training Complete!"
echo "============================================================"
echo ""
echo "Model trained successfully!"
if [ "$NO_UPLOAD" = false ]; then
    echo "Check your HuggingFace profile for the uploaded model!"
fi
echo ""
