"""
Simple script to process aggregated reviews, split into train/val/test, and upload to HuggingFace.
"""
import json
import os
import sys
from pathlib import Path
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Get the project root directory (parent of data_prepare_phase)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")

# Configuration - paths relative to project root
DATA_DIR = PROJECT_ROOT / "data" / "aggregated_review_english"
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "your-username/game-reviews-sentiment")
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.8"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.1"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.1"))


def load_all_jsonl_files(data_dir):
    """Load all JSONL files from the directory."""
    all_data = []
    
    # Get all .jsonl files in the directory
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        print(f"Loading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    all_data.append(data)
    
    print(f"Loaded {len(all_data)} total reviews")
    return all_data


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data into train/validation/test sets."""
    import random
    
    # Shuffle data for random split
    random.seed(42)  # For reproducibility
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total = len(shuffled_data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")
    
    return train_data, val_data, test_data


def create_huggingface_dataset(train_data, val_data, test_data):
    """Create HuggingFace DatasetDict."""
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict


def upload_to_huggingface(dataset_dict, dataset_name, token=None):
    """Upload dataset to HuggingFace Hub."""
    if token is None:
        token = os.environ.get("HF_TOKEN")
        if token is None:
            raise ValueError(
                "HuggingFace token not found. Please set HF_TOKEN environment variable "
                "or pass token as parameter."
            )
    
    print(f"\nUploading dataset to HuggingFace Hub: {dataset_name}")
    
    # Push to hub
    dataset_dict.push_to_hub(
        dataset_name,
        token=token,
        private=False  # Set to True if you want a private dataset
    )
    
    print(f"✓ Dataset successfully uploaded to: https://huggingface.co/datasets/{dataset_name}")


def main():
    """Main function to process and upload dataset."""
    print("=" * 60)
    print("Processing Game Reviews for HuggingFace Dataset")
    print("=" * 60)
    
    # 1. Load all data
    print("\n[1/4] Loading data...")
    all_data = load_all_jsonl_files(DATA_DIR)
    
    if len(all_data) == 0:
        print("ERROR: No data found!")
        return
    
    # 2. Split data
    print("\n[2/4] Splitting data...")
    train_data, val_data, test_data = split_data(
        all_data, 
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO
    )
    
    # 3. Create HuggingFace dataset
    print("\n[3/4] Creating HuggingFace dataset...")
    dataset_dict = create_huggingface_dataset(train_data, val_data, test_data)
    
    print("\nDataset structure:")
    print(dataset_dict)
    print("\nSample from train set:")
    print(dataset_dict['train'][0])
    
    # 4. Upload to HuggingFace
    print("\n[4/4] Uploading to HuggingFace...")
    
    # Get token from environment or prompt user
    token = os.environ.get("HF_TOKEN")
    if token is None:
        print("\nHF_TOKEN environment variable not found.")
        print("Please set it using: $env:HF_TOKEN='your_token_here' (PowerShell)")
        print("Or pass your token when prompted.")
        token = input("\nEnter your HuggingFace token (or press Enter to skip upload): ").strip()
        
        if not token:
            print("\nSkipping upload. Dataset prepared but not uploaded.")
            return dataset_dict
    
    upload_to_huggingface(dataset_dict, HF_DATASET_NAME, token)
    
    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)
    
    return dataset_dict


if __name__ == "__main__":
    # Check if .env is configured
    if HF_DATASET_NAME == "your-username/game-reviews-sentiment":
        print("⚠️  WARNING: Please configure your .env file first!")
        print("   Copy .env.example to .env and update with your settings.\n")
    
    print(f"Dataset name: {HF_DATASET_NAME}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}\n")
    
    dataset = main()
