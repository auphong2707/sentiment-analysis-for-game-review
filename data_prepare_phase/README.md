# Data Preparation Phase

This folder contains scripts for preparing and uploading the game review dataset to HuggingFace.

## Contents

### Scripts
- `aggregate_dataset.py` - Aggregates scraped review data and filters English reviews
- `analyze_reviews.py` - Analyzes review data and generates statistics reports
- `combine_data.py` - Combines multiple review JSON files into a single dataset
- `prepare_and_upload_hf_dataset.py` - Processes, splits, and uploads dataset to HuggingFace

### Documentation
- `UPLOAD_DATASET_GUIDE.md` - Detailed guide on using the upload script
- `README.md` - This file

## Workflow

The typical data preparation workflow is:

1. **Aggregate reviews** (after scraping): `python aggregate_dataset.py`
   - Filters English reviews from raw scraped data
   - Outputs to `data/aggregated_review_english/`

2. **Analyze reviews** (optional): `python analyze_reviews.py data/aggregated_review_english/aggregated_reviews_english.jsonl`
   - Generates statistics and insights about the dataset
   - Shows distributions, samples, and quality metrics

3. **Prepare and upload** (when ready to share): `python prepare_and_upload_hf_dataset.py`
   - Splits data into train/val/test
   - Uploads to HuggingFace Hub

## Quick Start (HuggingFace Upload)

### 1. Configure Environment

Make sure you have a `.env` file in the project root with:

```env
HF_TOKEN=hf_your_token_here
HF_DATASET_NAME=your-username/game-reviews-sentiment
```

### 2. Run the Script

From the project root:
```powershell
python data_prepare_phase/prepare_and_upload_hf_dataset.py
```

Or from this folder:
```powershell
cd data_prepare_phase
python prepare_and_upload_hf_dataset.py
```

## What It Does

1. ✅ Loads all `.jsonl` files from `data/aggregated_review_english/`
2. ✅ Shuffles and splits data into:
   - Train: 80%
   - Validation: 10%
   - Test: 10%
3. ✅ Creates HuggingFace DatasetDict format
4. ✅ Uploads to HuggingFace Hub

## Requirements

Make sure these packages are installed (already in `requirements.txt`):
```
datasets>=2.14.0
huggingface-hub>=0.17.0
python-dotenv>=1.0.0
```

Install with:
```powershell
pip install datasets huggingface-hub python-dotenv
```

## Path Handling

The script automatically detects the project root directory and constructs correct paths, so it works whether you run it from:
- Project root: `python data_prepare_phase/prepare_and_upload_hf_dataset.py`
- This folder: `python prepare_and_upload_hf_dataset.py`

## Output

After successful execution, your dataset will be available at:
```
https://huggingface.co/datasets/YOUR-USERNAME/YOUR-DATASET-NAME
```

And can be loaded with:
```python
from datasets import load_dataset
dataset = load_dataset("your-username/your-dataset-name")
```
