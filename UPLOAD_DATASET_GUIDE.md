# Upload Dataset to HuggingFace Guide

This guide explains how to use `prepare_and_upload_hf_dataset.py` to process and upload your game review dataset to HuggingFace.

## Prerequisites

1. Install required packages:
```powershell
pip install datasets huggingface-hub python-dotenv
```

Or install all requirements:
```powershell
pip install -r requirements.txt
```

2. Get your HuggingFace token:
   - Go to https://huggingface.co/settings/tokens
   - Create a new token with **write** permissions
   - Copy the token

## Configuration

1. **Copy the example environment file**:
```powershell
Copy-Item .env.example .env
```

2. **Edit `.env` file** and update with your values:
```env
HF_TOKEN=hf_your_actual_token_here
HF_DATASET_NAME=your-username/game-reviews-sentiment
```

Replace:
- `hf_your_actual_token_here` with your actual HuggingFace token
- `your-username` with your HuggingFace username
- `game-reviews-sentiment` with your desired dataset name

## Usage

Simply run the script (it will automatically read from `.env`):

```powershell
python prepare_and_upload_hf_dataset.py
```

The script will:
- Load configuration from `.env` file
- Process all data files
- Split into train/validation/test sets
- Upload to HuggingFace Hub

## What the script does

1. **Loads all data**: Reads all `.jsonl` files from `data/aggregated_review_english/`
2. **Splits data**: Randomly splits into:
   - Train: 80%
   - Validation: 10%
   - Test: 10%
3. **Creates HuggingFace dataset**: Formats data as a `DatasetDict`
4. **Uploads to HuggingFace Hub**: Pushes the dataset to your HuggingFace account

## Dataset Format

The dataset will have the following structure:

```python
DatasetDict({
    train: Dataset({
        features: ['review_text', 'review_score', 'review_category'],
        num_rows: ~80% of total
    })
    validation: Dataset({
        features: ['review_text', 'review_score', 'review_category'],
        num_rows: ~10% of total
    })
    test: Dataset({
        features: ['review_text', 'review_score', 'review_category'],
        num_rows: ~10% of total
    })
})
```

Each sample contains:
- `review_text`: The review content (string)
- `review_score`: The review score 0-100 (integer)
- `review_category`: Sentiment category - "positive", "mixed", or "negative" (string)

## After Upload

Once uploaded, your dataset will be available at:
```
https://huggingface.co/datasets/your-username/dataset-name
```

You can then use it in your code:
```python
from datasets import load_dataset

dataset = load_dataset("your-username/dataset-name")
```

## Troubleshooting

- **"HuggingFace token not found"**: Make sure you set the `HF_TOKEN` environment variable or enter it when prompted
- **"Repository not found"**: Make sure the dataset name format is correct (`username/dataset-name`)
- **Permission error**: Make sure your token has write permissions
- **No data found**: Check that the `data/aggregated_review_english/` folder contains `.jsonl` files
