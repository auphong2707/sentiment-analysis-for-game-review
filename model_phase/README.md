# Model Phase - Sentiment Analysis

This folder contains machine learning models for sentiment analysis on game reviews.

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

## Quick Start

### 1. Install Dependencies

```powershell
pip install scikit-learn datasets tqdm numpy pandas python-dotenv
# Optional: pip install wandb
```

### 2. Configure Dataset (Optional)

Set your dataset in `.env` file (in project root):
```env
HF_DATASET_NAME=your-username/game-reviews-sentiment
```

### 3. Train the Baseline Model

**Option A: Use dataset from .env**
```powershell
python model_phase/main_tfidf_baseline.py
```

**Option B: Specify dataset via command line**
```powershell
python model_phase/main_tfidf_baseline.py --dataset your-username/game-reviews-sentiment
```

### 4. Full Options

```powershell
python model_phase/main_tfidf_baseline.py \
    --dataset your-username/game-reviews-sentiment \
    --max_features 10000 \
    --ngram_min 1 \
    --ngram_max 2 \
    --max_iter 1000 \
    --subset 1.0 \
    --output_dir model_phase/results/my_experiment \
    --use_wandb
```

## Command-Line Arguments

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

## Output

After training, results are saved to `model_phase/results/tfidf_baseline_TIMESTAMP/`:

```
results/tfidf_baseline_YYYYMMDD_HHMMSS/
├── results.json              # Complete metrics and analysis
├── vectorizer.pkl            # Trained TF-IDF vectorizer
├── classifier.pkl            # Trained Logistic Regression model
├── label_encoder.pkl         # Label encoding mapping
└── config.json              # Model configuration
```

### Results JSON Structure

```json
{
  "model_config": {
    "max_features": 10000,
    "ngram_range": [1, 2],
    "max_iter": 1000
  },
  "training_time": 45.23,
  "test_accuracy": 0.8542,
  "test_precision": 0.8523,
  "test_recall": 0.8542,
  "test_f1": 0.8531,
  "test_inference_time": 2.15,
  "test_samples_per_second": 4651.16,
  "feature_importance": {
    "positive": {
      "top_positive": [["amazing", 0.523], ["great", 0.491], ...],
      "top_negative": [["bad", -0.412], ["worst", -0.387], ...]
    },
    ...
  }
}
```

## Usage Examples

### Basic Training

```powershell
# Train with default settings
python model_phase/main_tfidf_baseline.py --dataset username/game-reviews-sentiment
```

### Quick Experiment (10% of data)

```powershell
# Fast training for testing
python model_phase/main_tfidf_baseline.py \
    --dataset username/game-reviews-sentiment \
    --subset 0.1
```

### Custom Configuration

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

## Next Steps

After establishing the baseline:

1. **Error Analysis**: Look at misclassified samples
2. **Feature Engineering**: Try different n-grams, preprocessing
3. **Hyperparameter Tuning**: Optimize max_features, max_iter
4. **Deep Learning**: Try BERT, RoBERTa, etc.
5. **Ensemble Methods**: Combine multiple models

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

## Project Structure

```
model_phase/
├── main_tfidf_baseline.py    # Main training script
├── README.md                  # This file
└── results/                   # Training results
    └── tfidf_baseline_*/
        ├── results.json
        ├── vectorizer.pkl
        ├── classifier.pkl
        └── config.json
```

---

**A simple, fast, and interpretable baseline for sentiment analysis!** 🚀
