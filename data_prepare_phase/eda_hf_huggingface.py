import datasets
import pandas as pd
import numpy as np
from collections import Counter

# Load the dataset from HuggingFace
print("Loading dataset from HuggingFace...")
dataset = datasets.load_dataset("auphong2707/game-reviews-sentiment", split="train")

# Convert to pandas DataFrame for easier analysis
df = pd.DataFrame(dataset)

print("\n===== Dataset Overview =====")
print(f"Total reviews: {len(df):,}")
print(f"Columns: {df.columns.tolist()}")

# Class distribution
print("\n===== Class Distribution =====")
if 'label' in df.columns:
    label_col = 'label'
elif 'review_category' in df.columns:
    label_col = 'review_category'
else:
    label_col = df.columns[-1]  # fallback

label_counts = df[label_col].value_counts()
for label, count in label_counts.items():
    pct = count / len(df) * 100
    print(f"  {label}: {count:,} ({pct:.2f}%)")

# Review length statistics
print("\n===== Review Length Statistics =====")
df['text_length'] = df['review_text'].str.len()
df['word_count'] = df['review_text'].str.split().str.len()

print(f"Character length: Mean={df['text_length'].mean():.1f}, Median={df['text_length'].median():.1f}, Std={df['text_length'].std():.1f}")
print(f"Word count: Mean={df['word_count'].mean():.1f}, Median={df['word_count'].median():.1f}, Std={df['word_count'].std():.1f}")

# Show a few example reviews
print("\n===== Example Reviews =====")
for i, row in df.sample(3, random_state=42).iterrows():
    print(f"[{row[label_col]}] {row['review_text'][:200]}{'...' if len(row['review_text']) > 200 else ''}\n")

# Save summary statistics to CSV
summary = df[[label_col, 'text_length', 'word_count']].describe(include='all')
summary.to_csv('eda_hf_summary_statistics.csv')
print("\nSummary statistics saved to eda_hf_summary_statistics.csv")
