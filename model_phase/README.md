# Model Phase - Sentiment Analysis

Train machine learning models for sentiment analysis on game reviews.

## 🚀 Quick Start

### BGE-M3 + XGBoost (Recommended)
```bash
# Step 1: Generate embeddings (one-time, ~1-2 hours on GPU)
bash model_phase/generate_embeddings.sh --dataset username/game-reviews-sentiment

# Step 2: Train XGBoost (~20-40 mins)
bash model_phase/train_xgboost.sh --checkpoint_dir model_phase/results/bge_m3_embeddings/checkpoints
```

### RoBERTa
```bash
# Fine-tune RoBERTa with grid search
bash model_phase/train_roberta.sh --dataset username/game-reviews-sentiment
```

### LSTM Baseline
```bash
# Train LSTM with grid search
bash model_phase/train_LSTM_baseline.sh --dataset username/game-reviews-sentiment
```

---

## Models Overview

| Model | Training Time | Accuracy | Notes |
|-------|---------------|----------|-------|
| **BGE-M3 + XGBoost** | ~2-4 hours | 80-90% | Best performance, requires GPU for embeddings |
| **RoBERTa** | ~4-8 hours | 75-85% | End-to-end fine-tuning, GPU required |
| **LSTM** | ~1-2 hours | 70-80% | Baseline model, GPU optional |

---

## Configuration

Create `.env` in project root:
```env
HF_DATASET_NAME=username/game-reviews-sentiment
HF_TOKEN=your_huggingface_token
WANDB_API_KEY=your_wandb_api_key
```

---

## Command Reference

### Generate Embeddings
```bash
bash model_phase/generate_embeddings.sh \
    --dataset username/game-reviews-sentiment \
    --batch_size 64 \
    --max_length 512 \
    --subset 1.0 \
    --use_wandb
```

**Options:**
- `--dataset` - HuggingFace dataset (required)
- `--batch_size` - Batch size (default: 64)
- `--max_length` - Max tokens (default: 512)
- `--subset` - Data fraction 0-1 (default: 1.0)
- `--experiment_name` - Custom name
- `--use_wandb` - Enable W&B logging

---

### Train XGBoost
```bash
bash model_phase/train_xgboost.sh \
    --checkpoint_dir path/to/checkpoints \
    --gridsearch_subset 0.1 \
    --final_subset 1.0 \
    --use_wandb
```

**Options:**
- `--checkpoint_dir` - Path to embeddings (required)
- `--dataset` - Dataset name for metadata
- `--gridsearch_subset` - Grid search fraction (default: 0.1)
- `--final_subset` - Final training fraction (default: 1.0)
- `--use_wandb` - Enable W&B logging
- `--skip_gridsearch` - Skip grid search

**Grid Search Parameters:**
- `learning_rate`: 0.05, 0.1, 0.15
- `n_estimators`: 2000, 2500, 3000
- `max_depth`: 4, 6, 8
- `min_child_weight`: 1, 3, 5

---

### Train RoBERTa
```bash
bash model_phase/train_roberta.sh \
    --dataset username/game-reviews-sentiment \
    --gridsearch_subset 0.1 \
    --final_subset 1.0 \
    --use_wandb
```

**Options:**
- `--dataset` - HuggingFace dataset (required)
- `--gridsearch_subset` - Grid search fraction (default: 0.1)
- `--final_subset` - Final training fraction (default: 1.0)
- `--use_wandb` / `--no_wandb` - Toggle W&B
- `--skip_gridsearch` - Skip grid search
- `--resume_from_checkpoint` - Resume training
- `--no_checkpoints` - Disable checkpoints
- `--eval_steps` - Steps between evaluations (default: 1000)
- `--save_steps` - Steps between saves (default: 1000)

**Grid Search Parameters:**
- `learning_rate`: 5e-6, 1e-5, 2e-5, 5e-5
- Fixed: batch_size=32, epochs=5, max_length=256, eval_steps=1000, save_steps=1000

---

### Train LSTM
```bash
bash model_phase/train_LSTM_baseline.sh \
    --dataset username/game-reviews-sentiment \
    --gridsearch_subset 0.1 \
    --final_subset 1.0 \
    --n_jobs 4 \
    --use_wandb
```

**Options:**
- `--dataset` - HuggingFace dataset (required)
- `--gridsearch_subset` - Grid search fraction (default: 0.1)
- `--final_subset` - Final training fraction (default: 1.0)
- `--use_wandb` / `--no_wandb` - Toggle W&B (default: enabled)
- `--n_jobs` - CPU cores
- `--skip_gridsearch` - Skip grid search
- `--eval_steps` - Steps between evaluations (default: 500)
- `--save_steps` - Steps between saves (default: 500)

**Grid Search Parameters:**
- `learning_rate`: 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3
- Fixed: embed_dim=128, hidden_dim=128, batch_size=64, epochs=5, eval_steps=500, save_steps=500

---

## Output Structure

All scripts save results to `model_phase/results/` with auto-upload to HuggingFace Hub.

**BGE-M3 Embeddings:**
```
results/bge_m3_embeddings/checkpoints/
├── checkpoint_state.json
├── train_embeddings_embeddings.npz
├── validation_embeddings_embeddings.npz
└── test_embeddings_embeddings.npz
```

**XGBoost:**
```
results/xgboost_lr0.1_n2500_d6/
├── xgboost_model.json
├── label_encoder.pkl
├── config.json
├── results.json
├── raw_outputs_validation.jsonl
└── raw_outputs_test.jsonl
```

**RoBERTa/LSTM:**
```
results/roberta_official_2e-5/
├── pytorch_model.bin / model.h5
├── config.json
├── results.json
└── raw_outputs_*.jsonl
```

---

## Workflow Summary

All training scripts follow the same 3-step pattern:

1. **Grid Search** - Test hyperparameters on 10% data (optimizes F1-Macro)
2. **Extract Best Config** - Auto-parse optimal parameters
3. **Final Training** - Train on 100% data → Upload to HuggingFace

**Validation & Checkpointing:**
- RoBERTa: Validates every 1000 steps (~18-19 times/epoch), saves best 2 checkpoints
- LSTM: Validates every 500 steps (~18-19 times/epoch), saves best 2 checkpoints
- Both models keep only 2 best checkpoints based on validation F1 score

Skip grid search if already done: `--skip_gridsearch`

---

## Archive

Old or experimental code in `model_phase/archive/`.
