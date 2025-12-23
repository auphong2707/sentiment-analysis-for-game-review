#!/bin/bash

# ============================================================================
# EmbeddingGemma-300m Embedding Generation Script
# ============================================================================
# This script generates embeddings using Google's embeddinggemma-300m model
# for game review sentiment classification.
#
# Features:
# - Uses Sentence Transformers library for optimal performance
# - Checkpoint support for resumable generation
# - GPU acceleration (requires accepting Google's license on HuggingFace)
# - Configurable batch size and max sequence length
#
# Usage:
#   ./generate_embeddings.sh --dataset <dataset_name> [options]
#
# Options:
#   --dataset         HuggingFace dataset name (required)
#   --batch_size      Batch size (default: 512, optimized for 16GB GPU)
#   --max_length      Max sequence length (default: 256, model supports up to 2048)
#   --subset          Data fraction to use (default: 1.0)
#   --output_dir      Output directory (default: auto-generated)
#   --use_wandb       Enable WandB logging
#   --experiment_name Custom experiment name
#   --resume_from     Resume from checkpoint directory
# ============================================================================

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DATASET=""
BATCH_SIZE=512
MAX_LENGTH=256
SUBSET=1.0
OUTPUT_DIR=""
USE_WANDB=false
EXPERIMENT_NAME=""
RESUME_FROM=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset)
            DATASET="$2"
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
        --experiment_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --resume_from)
            RESUME_FROM="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 --dataset <dataset_name> [options]"
            echo ""
            echo "Options:"
            echo "  --dataset         HuggingFace dataset name (required)"
            echo "  --batch_size      Batch size (default: 512)"
            echo "  --max_length      Max sequence length (default: 256, max: 2048)"
            echo "  --subset          Data fraction to use (default: 1.0)"
            echo "  --output_dir      Output directory"
            echo "  --use_wandb       Enable WandB logging"
            echo "  --experiment_name Custom experiment name"
            echo "  --resume_from     Resume from checkpoint directory"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [ -z "$DATASET" ]; then
    echo -e "${RED}Error: --dataset is required${NC}"
    echo "Use --help for usage information"
    exit 1
fi

# Print configuration
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}EmbeddingGemma-300m Embedding Generation${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo -e "${GREEN}Configuration:${NC}"
echo "  Model: google/embeddinggemma-300m"
echo "  Dataset: $DATASET"
echo "  Batch size: $BATCH_SIZE"
echo "  Max length: $MAX_LENGTH tokens"
echo "  Subset: $SUBSET"
echo "  Output dir: ${OUTPUT_DIR:-auto-generated}"
echo "  WandB: $USE_WANDB"
[ -n "$EXPERIMENT_NAME" ] && echo "  Experiment name: $EXPERIMENT_NAME"
[ -n "$RESUME_FROM" ] && echo "  Resume from: $RESUME_FROM"
echo ""

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU Information:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
else
    echo -e "${YELLOW}Warning: nvidia-smi not found. GPU may not be available.${NC}"
    echo -e "${YELLOW}         This will be VERY slow on CPU.${NC}"
    echo ""
fi

# Check if HuggingFace CLI is logged in
if command -v huggingface-cli &> /dev/null; then
    if ! huggingface-cli whoami &> /dev/null; then
        echo -e "${YELLOW}Warning: Not logged in to HuggingFace Hub${NC}"
        echo -e "${YELLOW}         EmbeddingGemma requires accepting Google's license${NC}"
        echo -e "${YELLOW}         Please run: huggingface-cli login${NC}"
        echo ""
    fi
fi

# Build Python command
PYTHON_CMD="python model_phase/generate_gemma_embeddings.py"
PYTHON_CMD="$PYTHON_CMD --dataset \"$DATASET\""
PYTHON_CMD="$PYTHON_CMD --batch_size $BATCH_SIZE"
PYTHON_CMD="$PYTHON_CMD --max_length $MAX_LENGTH"
PYTHON_CMD="$PYTHON_CMD --subset $SUBSET"

[ -n "$OUTPUT_DIR" ] && PYTHON_CMD="$PYTHON_CMD --output_dir \"$OUTPUT_DIR\""
[ "$USE_WANDB" = true ] && PYTHON_CMD="$PYTHON_CMD --use_wandb"
[ -n "$EXPERIMENT_NAME" ] && PYTHON_CMD="$PYTHON_CMD --experiment_name \"$EXPERIMENT_NAME\""
[ -n "$RESUME_FROM" ] && PYTHON_CMD="$PYTHON_CMD --resume_from \"$RESUME_FROM\""

# Execute
echo -e "${GREEN}Starting embedding generation...${NC}"
echo ""
eval $PYTHON_CMD

# Check if successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}✓ Embedding generation completed successfully!${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo ""
    echo -e "${BLUE}Next steps:${NC}"
    echo "  1. Train XGBoost classifier with generated embeddings:"
    echo "     ./model_phase/train_xgboost.sh --checkpoint_dir <checkpoint_dir>"
    echo ""
    echo "  2. Or use Python directly:"
    echo "     python model_phase/main_xgboost.py --checkpoint_dir <checkpoint_dir>"
    echo ""
else
    echo ""
    echo -e "${RED}============================================================================${NC}"
    echo -e "${RED}✗ Embedding generation failed${NC}"
    echo -e "${RED}============================================================================${NC}"
    exit 1
fi
