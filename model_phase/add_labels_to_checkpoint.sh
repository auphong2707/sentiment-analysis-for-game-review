#!/bin/bash

# Add labels to checkpoint files that only contain embeddings
# This script adds labels from HuggingFace dataset to existing checkpoint files

# Usage:
#   ./add_labels_to_checkpoint.sh --checkpoint_dir /path/to/checkpoints
#   ./add_labels_to_checkpoint.sh --checkpoint_dir /path/to/checkpoints --verify-only
#   ./add_labels_to_checkpoint.sh --checkpoint_dir /path/to/checkpoints --no-replace

set -e  # Exit on error

# Default values
CHECKPOINT_DIR=""
DATASET="auphong2707/game-reviews-sentiment"
SPLITS="train validation test"
VERIFY_ONLY=false
NO_REPLACE=false

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
        --splits)
            SPLITS="$2"
            shift 2
            ;;
        --verify-only)
            VERIFY_ONLY=true
            shift
            ;;
        --no-replace)
            NO_REPLACE=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$CHECKPOINT_DIR" ]; then
    echo "Error: --checkpoint_dir is required"
    echo ""
    echo "Usage:"
    echo "  $0 --checkpoint_dir /path/to/checkpoints [options]"
    echo ""
    echo "Options:"
    echo "  --dataset DATASET        HuggingFace dataset name (default: auphong2707/game-reviews-sentiment)"
    echo "  --splits SPLITS          Space-separated splits to process (default: train validation test)"
    echo "  --verify-only            Only verify checkpoints, do not add labels"
    echo "  --no-replace             Keep new files with _with_labels suffix, do not replace originals"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "========================================================================"
echo "Add Labels to Checkpoint"
echo "========================================================================"
echo "Checkpoint directory: $CHECKPOINT_DIR"
echo "Dataset: $DATASET"
echo "Splits: $SPLITS"
echo "Verify only: $VERIFY_ONLY"
echo "No replace: $NO_REPLACE"
echo "========================================================================"
echo ""

# Build command
CMD="python $SCRIPT_DIR/add_labels_to_checkpoint.py \
    --checkpoint_dir \"$CHECKPOINT_DIR\" \
    --dataset \"$DATASET\" \
    --splits $SPLITS"

if [ "$VERIFY_ONLY" = true ]; then
    CMD="$CMD --verify-only"
fi

if [ "$NO_REPLACE" = true ]; then
    CMD="$CMD --no-replace"
fi

# Run the script
eval $CMD

echo ""
echo "========================================================================"
echo "Done!"
echo "========================================================================"
