# Model Phase - Sentiment Analysis

This folder contains machine learning models for sentiment analysis on game reviews.

## � Quick Start (Recommended)

**Run the complete automated pipeline:**

```powershell
# Windows
.\model_phase\train_tfidf_baseline.ps1 -Dataset "username/game-reviews-sentiment"

# Linux
bash model_phase/train_tfidf_baseline.sh --dataset username/game-reviews-sentiment
```

This single command will:
1. Run grid search on 10% data (finds best C regularization parameter)
2. Train final model on 100% data with optimal settings
3. Upload final model to HuggingFace Hub

**Time**: ~5-10 minutes | **Result**: Production-ready model on HuggingFace

---

## 📚 Documentation

This README contains everything you need. No other documentation files needed.

---

## 📖 Table of Contents

- [Quick Start](#-quick-start-recommended)
- [Models](#models)
- [Complete Pipeline](#complete-pipeline-automated)
- [Grid Search Only](#grid-search-only)
- [Manual Training](#manual-training)
- [Command-Line Arguments](#command-line-arguments)
- [Output Structure](#output)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

---

## Models

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

## Complete Pipeline (Automated)

The complete pipeline automates everything from hyperparameter search to final model deployment.

### Basic Usage

```powershell
# Windows
.\model_phase\train_tfidf_baseline.ps1 -Dataset "username/game-reviews-sentiment"

# Linux
bash model_phase/train_tfidf_baseline.sh --dataset username/game-reviews-sentiment
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

1. **Grid Search** (3-5 min)
   - Tests 3 C (regularization) values: 0.1, 1.0, 10.0
   - Uses 10% of data on validation set only
   - Finds optimal regularization strength
   - **Does NOT upload to HuggingFace** (exploration models)
   - TF-IDF settings fixed: max_features=10000, ngram=(1,2)

2. **Extract Best Config** (<1 sec)
   - Automatically parses best C parameter
   - No manual intervention needed

3. **Final Training** (2-5 min)
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

Note: The train_tfidf_baseline scripts now include grid search functionality inline. To run only grid search, use the -SkipGridsearch parameter or manually run only Step 1 of the script.

**Grid Search Parameters Tested:**
- `C` (regularization): 0.1, 1.0, 10.0

**Fixed TF-IDF Parameters:**
- `max_features`: 10000
- `ngram_range`: (1, 2) - unigrams and bigrams

**Total**: 3 configurations tested automatically

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

Minimal requirements:
```
scikit-learn>=1.3.0
datasets>=2.14.0
numpy>=1.24.0
pandas>=2.0.0
tqdm>=4.66.0
```

Optional:
```
wandb>=0.16.0  # For experiment tracking
```

---

## Project Structure

```
model_phase/
├── main_tfidf_baseline.py       # Main training script
├── utilities.py                 # Utility functions
├── train_tfidf_baseline.ps1    # Complete pipeline (Windows)
├── train_tfidf_baseline.sh     # Complete pipeline (Linux)
├── README.md                    # This file
└── results/                     # Training results
    ├── tfidf_baseline_*/        # Single training runs
    └── gridsearch/              # Grid search results
```

---

**A simple, fast, and interpretable baseline for sentiment analysis!** 🚀
