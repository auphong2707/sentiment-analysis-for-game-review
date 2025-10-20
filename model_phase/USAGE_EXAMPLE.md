# TF-IDF Baseline Model - Usage Examples

## ✅ Updated: Now Supports Default Dataset from .env

The model now reads the dataset name from your `.env` file, making it easier to run!

## Configuration

Your `.env` file should contain:
```env
HF_DATASET_NAME=auphong2707/game-reviews-sentiment
HF_TOKEN=your_token_here
```

## Usage Examples

### Example 1: Simple Run (Uses .env Dataset)

The **easiest** way to run the model:

```powershell
python model_phase/main_tfidf_baseline.py
```

✅ This will automatically use `HF_DATASET_NAME` from `.env`

### Example 2: Override Dataset from Command Line

```powershell
python model_phase/main_tfidf_baseline.py --dataset username/different-dataset
```

### Example 3: Quick Test with Subset

Test with 10% of the data:

```powershell
python model_phase/main_tfidf_baseline.py --subset 0.1
```

### Example 4: Full Training with All Options

```powershell
python model_phase/main_tfidf_baseline.py \
    --max_features 15000 \
    --ngram_min 1 \
    --ngram_max 3 \
    --max_iter 2000 \
    --subset 1.0 \
    --output_dir model_phase/results/my_experiment \
    --use_wandb
```

### Example 5: Different N-gram Configurations

**Unigrams only (single words):**
```powershell
python model_phase/main_tfidf_baseline.py --ngram_min 1 --ngram_max 1
```

**Bigrams only (word pairs):**
```powershell
python model_phase/main_tfidf_baseline.py --ngram_min 2 --ngram_max 2
```

**Unigrams + Bigrams + Trigrams:**
```powershell
python model_phase/main_tfidf_baseline.py --ngram_min 1 --ngram_max 3
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset` | str | From `.env` | HuggingFace dataset (optional if set in .env) |
| `--max_features` | int | 10000 | Maximum TF-IDF features |
| `--ngram_min` | int | 1 | Minimum n-gram size |
| `--ngram_max` | int | 2 | Maximum n-gram size |
| `--max_iter` | int | 1000 | Max iterations for Logistic Regression |
| `--subset` | float | 1.0 | Fraction of data to use (0-1) |
| `--output_dir` | str | Auto | Output directory for results |
| `--use_wandb` | flag | False | Use WandB for experiment tracking |

## Expected Output

```
============================================================
TF-IDF + Logistic Regression Baseline
============================================================

============================================================
Loading dataset: auphong2707/game-reviews-sentiment
============================================================

Dataset splits:
  Train: 50,000 samples
  Validation: 6,250 samples
  Test: 6,250 samples

============================================================
Initializing model
============================================================

============================================================
Training model
============================================================

✓ Training completed in 15.32s

============================================================
Evaluating on Validation set
============================================================

Validation Results:
  Accuracy: 0.8245
  Precision (weighted): 0.8231
  Recall (weighted): 0.8245
  F1-score (weighted): 0.8234
  ...

============================================================
Feature Importance Analysis
============================================================

POSITIVE:
  Top positive features (predict this class):
    great: 2.3456
    amazing: 2.1234
    excellent: 2.0987
    ...

============================================================
Training Complete!
============================================================
Results saved to: model_phase/results/tfidf_baseline_20251021_143022

Test Accuracy: 0.8198
Test F1-score: 0.8187
```

## Output Files

After training, you'll find:

```
model_phase/results/tfidf_baseline_YYYYMMDD_HHMMSS/
├── results.json           # Complete metrics
├── vectorizer.pkl         # TF-IDF vectorizer
├── classifier.pkl         # Logistic Regression model
├── label_encoder.pkl      # Label encoder
└── config.json           # Model configuration
```

## Troubleshooting

### Error: "dataset is required"

Make sure your `.env` file contains:
```env
HF_DATASET_NAME=auphong2707/game-reviews-sentiment
```

Or provide it via command line:
```powershell
python model_phase/main_tfidf_baseline.py --dataset username/dataset-name
```

### Missing Dependencies

Install required packages:
```powershell
pip install scikit-learn datasets tqdm numpy pandas python-dotenv
```

For WandB tracking:
```powershell
pip install wandb
```

## Next Steps

After training:
1. Check `results.json` for detailed metrics
2. Use the saved model for predictions
3. Compare against deep learning models (LSTM, BERT)
4. Experiment with different hyperparameters

Happy training! 🚀
