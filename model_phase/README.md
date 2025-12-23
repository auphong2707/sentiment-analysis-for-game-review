# Model Phase - Sentiment Analysis

This folder contains machine learning models for sentiment analysis on game reviews.

## 🚀 Quick Start (Recommended)

### BGE-M3 + XGBoost (Best Performance)

**Two-step pipeline for state-of-the-art results:**

**Step 1: Generate BGE-M3 Embeddings**
```bash
# Generate embeddings once, reuse for multiple classifiers
bash model_phase/generate_embeddings.sh --dataset username/game-reviews-sentiment
```

**Step 2: Train XGBoost Classifier**
```bash
# Train XGBoost with automatic hyperparameter tuning
bash model_phase/train_xgboost.sh --checkpoint_dir model_phase/results/bge_m3_embeddings/checkpoints
```

This pipeline:
1. Generates BGE-M3 embeddings with checkpoint support (resumable)
2. Runs grid search on 10% data to find best XGBoost hyperparameters
3. Trains final model on 100% data with optimal settings
4. Uploads final model to HuggingFace Hub

**Time**: ~2-4 hours (GPU) | **Result**: Production-ready model on HuggingFace

---

### TF-IDF Baseline (Fast & Simple)

**Run the complete automated pipeline:**

```bash
# Linux (uses dataset from .env by default)
bash model_phase/train_tfidf_baseline.sh --dataset username/game-reviews-sentiment
```

This single command will:
1. Run grid search on 10% data (finds best C regularization parameter)
2. Train final model on 100% data with optimal settings
3. Upload final model to HuggingFace Hub

**Time**: ~8-15 minutes | **Result**: Production-ready baseline model

---

## 📚 Documentation

This README contains everything you need. No other documentation files needed.

---

## 📖 Table of Contents

- [Quick Start](#-quick-start-recommended)
  - [BGE-M3 + XGBoost (Best Performance)](#bge-m3--xgboost-best-performance)
  - [TF-IDF Baseline (Fast & Simple)](#tf-idf-baseline-fast--simple)
- [Models](#models)
  - [BGE-M3 + XGBoost](#bge-m3--xgboost)
  - [TF-IDF + Logistic Regression](#baseline-tf-idf--logistic-regression)
- [BGE-M3 + XGBoost Workflow](#bge-m3--xgboost-workflow)
  - [Step 1: Generate Embeddings](#step-1-generate-embeddings)
  - [Step 2: Train XGBoost](#step-2-train-xgboost)
- [TF-IDF Complete Pipeline](#tfidf-complete-pipeline-automated)
- [Command-Line Arguments](#command-line-arguments)
- [Output Structure](#output)
- [Archive](#archive)

---

## Models

### BGE-M3 + XGBoost

**Files**: `generate_bge_m3_embeddings.py`, `main_xgboost.py`

**Core Idea**: Use state-of-the-art embeddings from BGE-M3 (a powerful multilingual text encoder) combined with XGBoost gradient boosting for classification.

**How it Works**:
- **Step 1**: BGE-M3 generates dense 1024-dimensional embeddings that capture semantic meaning
- **Step 2**: XGBoost trains a gradient-boosted tree classifier on these embeddings
- Embeddings are generated once and cached, then reused for multiple experiments

**Key Features**:
- ✅ State-of-the-art performance with deep semantic understanding
- ✅ Checkpoint support - resume from interruptions
- ✅ Separates embedding generation from training (efficient workflow)
- ✅ Handles class imbalance automatically
- ✅ Fast inference once embeddings are generated
- ⚠️ Requires GPU for efficient embedding generation
- ⚠️ Initial embedding generation takes time (~1-2 hours)

**Performance Expectations**:
- **Accuracy**: 80-90%
- **Embedding Generation**: 1-2 hours (GPU, one-time)
- **Training time**: 10-30 minutes (CPU)
- **Inference speed**: Very fast with cached embeddings
- **Model size**: ~50-100 MB

---

### Baseline: TF-IDF + Logistic Regression

**File**: `main_tfidf_baseline.py`

**Core Idea**: Sentiment is determined by the frequency of specific words.

**How it Works**:
- Uses TF-IDF (Term Frequency-Inverse Document Frequency) to weigh important words
- Applies Logistic Regression for multi-class classification (positive, mixed, negative)
- Ignores word order but captures word importance

**Key Features**:
- ✅ Fast training and prediction
- ✅ Interpretable (can see which words matter most for each class)
- ✅ Good baseline to measure against deep learning models
- ✅ No GPU required
- ✅ Small model size

---

## BGE-M3 + XGBoost Workflow

### Step 1: Generate Embeddings

Generate BGE-M3 embeddings once, then reuse them for multiple experiments.

**Basic Usage:**
```bash
bash model_phase/generate_embeddings.sh --dataset username/game-reviews-sentiment
```

**With Options:**
```bash
bash model_phase/generate_embeddings.sh \
    --dataset username/game-reviews-sentiment \
    --batch_size 64 \
    --max_length 512 \
    --subset 1.0 \
    --experiment_name my_embeddings \
    --use_wandb
```

**What It Does:**
- Loads BGE-M3 model (`BAAI/bge-m3`)
- Generates 1024-dimensional embeddings for train/val/test splits
- Saves embeddings as `.npz` files in checkpoint directory
- Supports incremental checkpointing (saves every 100 batches)
- Can resume from interruptions

**Output:**
```
model_phase/results/bge_m3_embeddings/checkpoints/
├── checkpoint_state.json
├── train_embeddings_embeddings.npz
├── validation_embeddings_embeddings.npz
└── test_embeddings_embeddings.npz
```

**Arguments:**
- `--dataset` - HuggingFace dataset name (required)
- `--batch_size` - Batch size for embedding generation (default: 64)
- `--max_length` - Max sequence length (default: 512)
- `--subset` - Fraction of data to use (default: 1.0)
- `--experiment_name` - Custom name for outputs
- `--use_wandb` - Enable WandB logging

---

### Step 2: Train XGBoost

Train XGBoost classifier using pre-computed embeddings.

**Basic Usage:**
```bash
bash model_phase/train_xgboost.sh --checkpoint_dir model_phase/results/bge_m3_embeddings/checkpoints
```

**With Options:**
```bash
bash model_phase/train_xgboost.sh \
    --checkpoint_dir model_phase/results/bge_m3_embeddings/checkpoints \
    --gridsearch_subset 0.1 \
    --final_subset 1.0 \
    --use_wandb
```

**What It Does:**
1. **Grid Search** (10-20 min on 10% data)
   - Tests combinations of hyperparameters:
     - `learning_rate`: 0.05, 0.1, 0.15
     - `n_estimators`: 2000, 2500, 3000
     - `max_depth`: 4, 6, 8
     - `min_child_weight`: 1, 3, 5
   - Optimizes for F1-Macro (balanced across classes)
   
2. **Extract Best Config** (<1 sec)
   - Automatically parses best hyperparameters
   
3. **Final Training** (10-30 min on 100% data)
   - Trains with best config
   - Applies balanced class weights
   - Early stopping with 200 rounds patience
   - Evaluates on test set
   - **Uploads to HuggingFace Hub**

**Output:**
```
model_phase/results/xgboost_lr0.1_n2500_d6_mcw3/
├── xgboost_model.json
├── label_encoder.pkl
├── config.json
├── results.json
├── raw_outputs_validation.jsonl
└── raw_outputs_test.jsonl
```

**Arguments:**
- `--checkpoint_dir` - Path to embeddings checkpoint (required)
- `--dataset` - HuggingFace dataset for metadata (optional)
- `--gridsearch_subset` - Fraction for grid search (default: 0.1)
- `--final_subset` - Fraction for final training (default: 1.0)
- `--use_wandb` - Enable WandB logging
- `--skip_gridsearch` - Skip grid search if already done

---

## TF-IDF Complete Pipeline (Automated)

The complete pipeline automates everything from hyperparameter search to final model deployment.

### Basic Usage

```powershell
# Windows (uses .env)
.\model_phase\train_tfidf_baseline.ps1

# Linux (uses .env)
bash model_phase/train_tfidf_baseline.sh
```

### Advanced Options

```powershell
# Windows - Full options
.\model_phase\train_tfidf_baseline.ps1 `
    -Dataset "username/game-reviews-sentiment" `
    -GridsearchSubset 0.1 `
    -FinalSubset 1.0 `
    -NJobs 4

# Linux - Full options
bash model_phase/train_tfidf_baseline.sh \
    --dataset username/game-reviews-sentiment \
    --gridsearch_subset 0.1 \
    --final_subset 1.0 \
    --n_jobs 4
```

### What It Does

1. **Grid Search** (5-8 min)
   - Tests 5 C (regularization) values: 0.01, 0.1, 1.0, 10.0, 100.0
   - Uses 10% of data on validation set only
   - Finds optimal regularization strength
   - **Does NOT upload to HuggingFace** (exploration models)
   - TF-IDF settings fixed: max_features=20000, ngram=(1,2)

2. **Extract Best Config** (<1 sec)
   - Automatically parses best C parameter
   - No manual intervention needed

3. **Final Training** (3-7 min)
   - Trains on 100% data with best C value
   - Evaluates on test set (first time!)
   - **Uploads to HuggingFace Hub** (production model)

---

## Grid Search Only

If you want to run just the hyperparameter search without final training:

```powershell
# Windows
.\model_phase\train_tfidf_baseline.ps1 -SkipGridsearch:$false -Dataset "username/game-reviews-sentiment"

# Linux  
bash model_phase/train_tfidf_baseline.sh --dataset username/game-reviews-sentiment --skip_final_training
```

Note: The train_tfidf_baseline scripts now include grid search functionality inline and read dataset from `.env` by default.

**Grid Search Parameters Tested:**
- `C` (regularization): 0.01, 0.1, 1.0, 10.0, 100.0

**Fixed TF-IDF Parameters:**
- `max_features`: 20000
- `ngram_range`: (1, 2) - unigrams and bigrams

**Total**: 5 configurations tested automatically

**Output**: Results saved to `model_phase/results/gridsearch/`
- `best_config.txt` - Best hyperparameters found
- `gridsearch_results.txt` - All 27 results
- `config_*/` - Individual trained models

**Important**: Grid search models are NOT uploaded to HuggingFace.

---

## Manual Training

Train a model with specific hyperparameters:

### Basic Usage

```powershell
python model_phase/main_tfidf_baseline.py --dataset username/game-reviews-sentiment
```

### With Best Hyperparameters from Grid Search

After running grid search, check `best_config.txt` and use those values:

```powershell
python model_phase/main_tfidf_baseline.py \
    --dataset username/game-reviews-sentiment \
    --max_features 10000 \
    --ngram_min 1 \
    --ngram_max 2 \
    --max_iter 1000
```

### Without HuggingFace Upload

```powershell
python model_phase/main_tfidf_baseline.py \
    --dataset username/game-reviews-sentiment \
    --no_upload
```

---

## Quick Start

### Option A: One-Command Complete Pipeline (Recommended)

Run the entire process from grid search to final training with one command:

**Windows:**
```powershell
.\model_phase\train_tfidf_baseline.ps1 -Dataset "username/game-reviews-sentiment"
```

**Linux:**
```bash
bash model_phase/train_tfidf_baseline.sh --dataset username/game-reviews-sentiment
```

This automatically:
1. Runs grid search on 10% data (validation set only)
2. Finds best hyperparameters
3. Trains final model on 100% data
4. Uploads only the final model to HuggingFace

### Option B: Manual Step-by-Step

If you prefer manual control:

### 1. Install Dependencies

```powershell
pip install scikit-learn datasets tqdm numpy pandas python-dotenv huggingface_hub
# Optional: pip install wandb
```

### 2. Configure Dataset

Set your dataset in `.env` file (in project root):
```env
HF_DATASET_NAME=your-username/game-reviews-sentiment
HF_TOKEN=your_huggingface_token
```

### 3. Run Complete Pipeline

```powershell
# Run everything at once
.\model_phase\train_tfidf_baseline.ps1 -Dataset "username/game-reviews-sentiment"
```

---

## Command-Line Arguments

### Complete Pipeline Script

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | (required) | HuggingFace dataset name |
| `--gridsearch_subset` | 0.1 | Subset for grid search (10%) |
| `--final_subset` | 1.0 | Subset for final training (100%) |
| `--output_dir` | `model_phase/results` | Base output directory |
| `--n_jobs` | CPU-1 | Number of CPU cores to use |
| `--skip_gridsearch` | false | Skip grid search if already done |

### Main Training Script

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | From `.env` | HuggingFace dataset name (or set HF_DATASET_NAME in .env) |
| `--max_features` | int | 10000 | Maximum TF-IDF features |
| `--ngram_min` | int | 1 | Minimum n-gram size |
| `--ngram_max` | int | 2 | Maximum n-gram size |
| `--max_iter` | int | 1000 | Max iterations for Logistic Regression |
| `--subset` | float | 1.0 | Fraction of data to use (0-1) |
| `--output_dir` | str | Auto | Output directory for results |
| `--use_wandb` | flag | False | Use WandB for experiment tracking |
| `--n_jobs` | int | CPU-1 | Number of CPU cores to use |
| `--upload_to_hf` | flag | True | Upload results to HuggingFace Hub |
| `--no_upload` | flag | False | Skip HuggingFace upload |
| `--hf_repo` | str | Auto | HuggingFace repo name for results |

---

## Output

After training, results are saved to `model_phase/results/tfidf_baseline_TIMESTAMP/`:

```
results/tfidf_baseline_YYYYMMDD_HHMMSS/
├── results.json              # Complete metrics and analysis
├── vectorizer.pkl            # Trained TF-IDF vectorizer
├── classifier.pkl            # Trained Logistic Regression model
├── label_encoder.pkl         # Label encoding mapping
├── config.json              # Model configuration
└── README.md                # Model card (auto-generated)
```

**HuggingFace Upload**: 
- Grid search models: ❌ NOT uploaded (exploration only)
- Final model: ✅ Automatically uploaded to HuggingFace Hub at `username/tfidf_baseline-results`

View your uploaded model at: `https://huggingface.co/username/tfidf_baseline-results`

---

## Usage Examples

### Usage Examples

### Complete Workflow (Recommended)

```powershell
# One command does everything!
.\model_phase\train_tfidf_baseline.ps1 -Dataset "username/game-reviews-sentiment"
```

### Grid Search Examples

```powershell
# Quick test with small subset
bash model_phase/train_tfidf_baseline.sh (grid search only) --dataset username/game-reviews-sentiment --subset 0.05
```

### Manual Training Examples

```powershell
# Train with default settings
python model_phase/main_tfidf_baseline.py --dataset username/game-reviews-sentiment

# Quick test (10% of data)

```powershell
# More features, larger n-grams
python model_phase/main_tfidf_baseline.py \
    --dataset username/game-reviews-sentiment \
    --max_features 20000 \
    --ngram_min 1 \
    --ngram_max 3 \
    --max_iter 2000
```

### With WandB Tracking

```powershell
# Track experiments with WandB
python model_phase/main_tfidf_baseline.py \
    --dataset username/game-reviews-sentiment \
    --use_wandb
```

### Skip HuggingFace Upload

```powershell
# Train without uploading to HuggingFace
python model_phase/main_tfidf_baseline.py \
    --dataset username/game-reviews-sentiment \
    --no_upload
```

### Custom HuggingFace Repository

```powershell
# Upload to a specific HuggingFace repo
python model_phase/main_tfidf_baseline.py \
    --dataset username/game-reviews-sentiment \
    --hf_repo username/my-custom-model-repo
```

## Model Details

### TF-IDF Vectorizer

**Parameters**:
- `max_features`: Limit vocabulary size to most important words
- `ngram_range`: (1, 2) captures both single words and word pairs
- `stop_words`: 'english' - removes common words like "the", "is", etc.
- `lowercase`: True - normalizes text case
- `strip_accents`: 'unicode' - removes accents

### Logistic Regression

**Parameters**:
- `multi_class`: 'multinomial' - for 3-class classification
- `solver`: 'lbfgs' - efficient optimizer
- `max_iter`: 1000 - maximum training iterations
- `n_jobs`: -1 - use all CPU cores

## Evaluation Metrics

The model reports:

1. **Accuracy**: Overall correctness
2. **Precision**: How many predicted positives are correct
3. **Recall**: How many actual positives are found
4. **F1-score**: Harmonic mean of precision and recall
5. **Per-class metrics**: Performance for each sentiment class
6. **Confusion matrix**: Detailed prediction breakdown
7. **Inference speed**: Samples per second
8. **Feature importance**: Most important words for each class

## Feature Importance Analysis

The model provides interpretable feature importance:

```
POSITIVE:
  Top positive features (predict this class):
    amazing: 0.5234
    great: 0.4912
    excellent: 0.4567
    
  Top negative features (predict against this class):
    bad: -0.4123
    worst: -0.3876
    disappointing: -0.3654

NEGATIVE:
  Top positive features:
    terrible: 0.6123
    worst: 0.5892
    garbage: 0.5234
  ...
```

This helps understand what words the model associates with each sentiment.

## Loading Saved Models

```python
from model_phase.main_tfidf_baseline import TFIDFSentimentClassifier

# Load trained model
model = TFIDFSentimentClassifier.load('model_phase/results/tfidf_baseline_20250101_120000')

# Predict on new data
texts = ["This game is amazing!", "Terrible gameplay"]
predictions = model.predict(texts)
probabilities = model.predict_proba(texts)

print(predictions)  # ['positive', 'negative']
print(probabilities)  # [[0.05, 0.15, 0.80], [0.85, 0.10, 0.05]]
```

## Performance Expectations

### Typical Results

On game review sentiment data:
- **Accuracy**: 75-85%
- **Training time**: 30-120 seconds (depends on data size)
- **Inference speed**: 3000-5000 samples/second
- **Model size**: 10-50 MB

### Strengths

- ✅ Very fast training and inference
- ✅ Works well with limited data
- ✅ Interpretable results
- ✅ No GPU required
- ✅ Good baseline performance

### Limitations

- ⚠️ Ignores word order ("not good" = "good not")
- ⚠️ Can't capture context ("good" in "not good")
- ⚠️ Fixed vocabulary (can't handle new words well)
- ⚠️ Simple feature representation

## Comparison with Deep Learning

Use this baseline to compare against more complex models:

| Aspect | TF-IDF + LR | Deep Learning |
|--------|-------------|---------------|
| Training Time | Minutes | Hours |
| Inference Speed | Very Fast | Moderate |
| Accuracy | Good | Better |
| Interpretability | High | Low |
| GPU Required | No | Yes |
| Data Needed | Less | More |

## Planned Methods (Under Development)

The following advanced methods are planned for future implementation:

### Method 1: Recurrent Neural Network (LSTM/GRU) with Word Embeddings

**Core Idea**: The sequence of words matters.

**How it Works**: 
- Reads the review word-by-word, using Word Embeddings (like GloVe) to understand word meaning
- The LSTM "remembers" previous words (like "not") to understand the context of later words (like "good")
- Processes text sequentially, maintaining memory of previous context

**Key Feature**: 
- ✅ Understands context and word order
- ✅ Reads left-to-right with memory
- ✅ Can capture negations and context-dependent meanings
- ⚠️ More complex than TF-IDF baseline
- ⚠️ Requires GPU for efficient training

### Method 2: Fine-Tuning RoBERTa

**Core Idea**: Sentiment is best understood by looking at a word's context from both left and right simultaneously, using a deeply pre-trained model.

**How it Works**: 
- Uses RoBERTa, a model that was pre-trained on a massive amount of text
- "Fine-tunes" this model on specific game reviews, adapting its powerful language understanding to the task
- Processes entire review bidirectionally, understanding context from all directions

**Key Feature**: 
- ✅ RoBERTa (a variant of BERT) generally provides state-of-the-art accuracy
- ✅ Uses bidirectional context for deep understanding
- ✅ Highly effective at understanding nuance and complex sentiment
- ⚠️ Computationally expensive
- ⚠️ Requires significant GPU resources
- ⚠️ Longer training time

## Next Steps

After establishing the baseline:

1. **Error Analysis**: Look at misclassified samples
2. **Feature Engineering**: Try different n-grams, preprocessing
3. **Hyperparameter Tuning**: Optimize max_features, max_iter
4. **Deep Learning**: Implement LSTM/GRU with word embeddings
5. **Transformer Models**: Fine-tune RoBERTa or other BERT variants
6. **Ensemble Methods**: Combine multiple models

## Troubleshooting

**Memory Error:**
- Reduce `--max_features`
- Use `--subset 0.5` to train on less data

**Slow Training:**
- Reduce `--max_iter`
- Use smaller dataset with `--subset`

**Poor Performance:**
- Increase `--max_features` (more vocabulary)
- Try `--ngram_max 3` (capture longer phrases)
- Increase `--max_iter` (more training iterations)

**WandB not working:**
- Install: `pip install wandb`
- Login: `wandb login`

**HuggingFace upload fails:**
- Verify `HF_TOKEN` in `.env` is set
- Check token has write permissions
- Ensure internet connection is stable

---

## Dependencies

### BGE-M3 + XGBoost Requirements:
```
torch>=2.0.0
transformers>=4.30.0
xgboost>=2.0.0
scikit-learn>=1.3.0
datasets>=2.14.0
numpy>=1.24.0
```

### TF-IDF Baseline Requirements:
```
scikit-learn>=1.3.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.66.0
```

### Optional:
```
wandb>=0.16.0  # For experiment tracking
```

---

## Project Structure

```
model_phase/
├── generate_bge_m3_embeddings.py  # BGE-M3 embedding generation
├── main_xgboost.py                # XGBoost training
├── main_tfidf_baseline.py         # TF-IDF baseline
├── utilities.py                   # Shared utility functions
├── generate_embeddings.sh         # BGE-M3 embedding pipeline
├── train_xgboost.sh               # XGBoost training pipeline
├── train_bge_m3.sh                # Legacy: BGE-M3 + SVM (archived)
├── train_tfidf_baseline.sh        # TF-IDF baseline pipeline
├── README.md                      # This file
├── archive/                       # Archived scripts
│   ├── main_bge_m3.py            # BGE-M3 + SVM (no checkpoints)
│   └── main_bge_m3_checkpoint.py # BGE-M3 + SVM (with checkpoints)
└── results/                       # Training results
    ├── bge_m3_embeddings/        # BGE-M3 embedding outputs
    ├── xgboost_*/                # XGBoost training runs
    ├── tfidf_baseline_*/         # TF-IDF training runs
    └── gridsearch*/              # Grid search results
```

---

## Archive

### Archived: BGE-M3 + SVM Implementation

The original BGE-M3 implementation used SVM as the classifier. This has been archived in favor of the cleaner two-script workflow with XGBoost.

**Archived Files:**
- `archive/main_bge_m3.py` - Original BGE-M3 + SVM (no checkpoints)
- `archive/main_bge_m3_checkpoint.py` - BGE-M3 + SVM with checkpoint support
- `train_bge_m3.sh` - Shell script for BGE-M3 + SVM training

**Why Archived:**
- Mixed embedding generation with classifier training in single script
- SVM training was computationally expensive (~30-60 minutes)
- Difficult to experiment with different classifiers
- XGBoost provides better performance with faster training

**Migration to New Workflow:**
The new workflow separates concerns:
1. **Embedding Generation** (`generate_bge_m3_embeddings.py`) - Generate embeddings once
2. **Classifier Training** (`main_xgboost.py`) - Train XGBoost on cached embeddings

This allows:
- Reusing embeddings for multiple experiments
- Faster iteration on classifier hyperparameters
- Better separation of concerns
- Easier maintenance

**To use archived scripts:**
```bash
# Still functional but not recommended
bash model_phase/train_bge_m3.sh --dataset username/game-reviews-sentiment
```

---

**Simple baselines to state-of-the-art models for sentiment analysis!** 🚀
