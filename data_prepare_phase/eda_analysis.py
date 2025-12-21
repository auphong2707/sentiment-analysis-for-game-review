"""
Exploratory Data Analysis (EDA) for Game Review Sentiment Dataset

This script performs comprehensive EDA including:
1. Basic statistics and distribution analysis
2. Text analysis per category (positive, negative, mixed)
3. Common words and themes in each category
4. Deep dive into "mixed" reviews to understand their characteristics
5. Sample analysis to understand why reviews are classified as mixed vs positive/negative
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Path to dataset
DATA_PATH = Path(__file__).parent.parent / "data" / "aggregated_review_english" / "aggregated_reviews_english.jsonl"
OUTPUT_DIR = Path(__file__).parent / "eda_results"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - GAME REVIEW SENTIMENT DATASET")
print("="*80)

# ============================================================================
# 1. LOAD AND BASIC STATISTICS
# ============================================================================
print("\n[1/6] Loading dataset...")
df = pd.read_json(DATA_PATH, lines=True)
print(f"✓ Loaded {len(df):,} reviews")
print(f"✓ Columns: {list(df.columns)}")

print("\n" + "="*80)
print("BASIC STATISTICS")
print("="*80)

# Dataset shape
print(f"\nDataset shape: {df.shape}")
print(f"Total reviews: {len(df):,}")
print(f"Features: {len(df.columns)}")

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())

# Category distribution
print("\nCategory Distribution:")
category_counts = df['review_category'].value_counts()
print(category_counts)
print("\nPercentages:")
print(df['review_category'].value_counts(normalize=True) * 100)

# Text length statistics
df['text_length'] = df['review_text'].str.len()
df['word_count'] = df['review_text'].str.split().str.len()

print("\nText Length Statistics:")
print(df[['text_length', 'word_count']].describe())

print("\nText Length by Category:")
print(df.groupby('review_category')[['text_length', 'word_count']].describe())

# ============================================================================
# 2. VISUALIZATIONS - BASIC DISTRIBUTIONS
# ============================================================================
print("\n[2/6] Creating distribution visualizations...")

# Category distribution bar plot
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Category counts
category_counts.plot(kind='bar', ax=axes[0, 0], color=['green', 'gray', 'red'])
axes[0, 0].set_title('Review Count by Category', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=0)
for i, v in enumerate(category_counts.values):
    axes[0, 0].text(i, v + 500, f'{v:,}', ha='center', fontweight='bold')

# Plot 2: Text length distribution by category
for category in df['review_category'].unique():
    subset = df[df['review_category'] == category]['text_length']
    axes[0, 1].hist(subset, bins=50, alpha=0.5, label=category)
axes[0, 1].set_title('Text Length Distribution by Category', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Text Length (characters)')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].legend()
axes[0, 1].set_xlim(0, 2000)

# Plot 3: Word count distribution by category
for category in df['review_category'].unique():
    subset = df[df['review_category'] == category]['word_count']
    axes[1, 0].hist(subset, bins=50, alpha=0.5, label=category)
axes[1, 0].set_title('Word Count Distribution by Category', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('Word Count')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].set_xlim(0, 400)

# Plot 4: Box plot of word count by category
df.boxplot(column='word_count', by='review_category', ax=axes[1, 1])
axes[1, 1].set_title('Word Count Distribution by Category (Box Plot)', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Category')
axes[1, 1].set_ylabel('Word Count')
plt.suptitle('')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_basic_distributions.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / '01_basic_distributions.png'}")

# ============================================================================
# 3. TEXT ANALYSIS PER CATEGORY
# ============================================================================
print("\n[3/6] Analyzing text patterns per category...")

def clean_text(text):
    """Clean and tokenize text"""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    # Remove common stop words
    stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                      'of', 'with', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had',
                      'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
                      'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it',
                      'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their',
                      'me', 'him', 'them', 'us', 'what', 'which', 'who', 'when', 'where',
                      'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
                      'other', 'some', 'such', 'no', 'not', 'only', 'own', 'same', 'so',
                      'than', 'too', 'very', 's', 't', 'just', 'don', 'now', 'as'])
    words = [w for w in words if w not in stop_words and len(w) > 2]
    return words

def get_common_words(texts, top_n=30):
    """Get most common words from texts"""
    all_words = []
    for text in texts:
        all_words.extend(clean_text(text))
    return Counter(all_words).most_common(top_n)

# Analyze each category
categories = df['review_category'].unique()
category_analysis = {}

for category in categories:
    print(f"\n{'='*80}")
    print(f"ANALYZING: {category.upper()}")
    print(f"{'='*80}")
    
    category_df = df[df['review_category'] == category]
    texts = category_df['review_text'].tolist()
    
    # Get common words
    common_words = get_common_words(texts, top_n=30)
    
    print(f"\nTop 30 most common words in {category} reviews:")
    for i, (word, count) in enumerate(common_words, 1):
        print(f"{i:2d}. {word:20s} - {count:6,} times")
    
    category_analysis[category] = {
        'common_words': common_words,
        'sample_reviews': category_df.sample(min(10, len(category_df)))['review_text'].tolist()
    }

# ============================================================================
# 4. WORD CLOUDS FOR EACH CATEGORY
# ============================================================================
print("\n[4/6] Creating word clouds...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, category in enumerate(categories):
    category_df = df[df['review_category'] == category]
    text = ' '.join(category_df['review_text'].tolist())
    
    # Create word cloud
    wordcloud = WordCloud(width=800, height=400, 
                          background_color='white',
                          max_words=100,
                          colormap='viridis' if category == 'positive' else 'plasma' if category == 'negative' else 'coolwarm',
                          relative_scaling=0.5).generate(text)
    
    axes[idx].imshow(wordcloud, interpolation='bilinear')
    axes[idx].set_title(f'{category.upper()} Reviews - Word Cloud', fontsize=14, fontweight='bold')
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_word_clouds.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / '02_word_clouds.png'}")

# ============================================================================
# 5. COMMON WORDS BAR CHARTS
# ============================================================================
print("\n[5/6] Creating common words visualizations...")

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

for idx, category in enumerate(categories):
    words, counts = zip(*category_analysis[category]['common_words'][:20])
    
    axes[idx].barh(range(len(words)), counts, 
                   color='green' if category == 'positive' else 'red' if category == 'negative' else 'gray')
    axes[idx].set_yticks(range(len(words)))
    axes[idx].set_yticklabels(words)
    axes[idx].invert_yaxis()
    axes[idx].set_xlabel('Frequency')
    axes[idx].set_title(f'Top 20 Words in {category.upper()} Reviews', fontsize=12, fontweight='bold')
    axes[idx].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_common_words_by_category.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: {OUTPUT_DIR / '03_common_words_by_category.png'}")

# ============================================================================
# 6. SYSTEMATIC SAMPLE SELECTION FOR REPORT
# ============================================================================
def select_representative_samples(category_df, category_name, n_samples=5, n_candidates=100):
    """
    Systematically select representative samples from a category.
    
    Strategy:
    1. Random sample large candidate pool (default: 100 candidates)
    2. Filter outliers (< 10 words or > 300 words)
    3. For mixed reviews: prioritize those with contrast markers
    4. Stratify by length (below median, near mean, above 75th percentile)
    5. Select diverse examples across the length spectrum
    
    Args:
        category_df: DataFrame for the category
        category_name: Name of category ('positive', 'negative', 'mixed')
        n_samples: Number of samples to select (default: 5)
        n_candidates: Number of initial random candidates (default: 100)
    
    Returns:
        DataFrame with selected samples
    """
    print(f"\n{'='*80}")
    print(f"SYSTEMATIC SAMPLE SELECTION: {category_name.upper()}")
    print(f"{'='*80}")
    
    # Step 0: Random sample from category to get candidate pool
    n_candidates = min(n_candidates, len(category_df))
    candidate_pool = category_df.sample(n=n_candidates, random_state=42)
    print(f"Random candidate pool: {n_candidates} reviews")
    
    # Step 0: Random sample from category to get candidate pool
    n_candidates = min(n_candidates, len(category_df))
    candidate_pool = category_df.sample(n=n_candidates, random_state=42)
    print(f"Random candidate pool: {n_candidates} reviews")
    
    # Step 1: Filter outliers (focus on interquartile range)
    filtered_df = candidate_pool[
        (candidate_pool['word_count'] >= 10) & 
        (candidate_pool['word_count'] <= 300)
    ].copy()
    
    print(f"After filtering outliers (10-300 words): {len(filtered_df)}")
    
    # Step 2: For mixed reviews, prioritize contrast markers
    if category_name == 'mixed':
        contrast_words = ['but', 'however', 'although', 'though', 'despite', 'unfortunately', 'except']
        filtered_df['has_contrast'] = filtered_df['review_text'].apply(
            lambda x: any(word in x.lower() for word in contrast_words)
        )
        # Prioritize reviews with contrast markers (79.6% have them)
        contrast_df = filtered_df[filtered_df['has_contrast']].copy()
        print(f"Reviews with contrast markers: {len(contrast_df)}")
        
        if len(contrast_df) >= n_samples:
            filtered_df = contrast_df
    
    # Check if we have enough samples
    if len(filtered_df) < n_samples:
        print(f"Warning: Only {len(filtered_df)} samples available after filtering, requested {n_samples}")
        n_samples = len(filtered_df)
    
    # Step 3: Calculate length statistics
    median_wc = filtered_df['word_count'].median()
    mean_wc = filtered_df['word_count'].mean()
    q75_wc = filtered_df['word_count'].quantile(0.75)
    
    print(f"\nLength statistics after filtering:")
    print(f"  Median word count: {median_wc:.0f}")
    print(f"  Mean word count: {mean_wc:.0f}")
    print(f"  75th percentile: {q75_wc:.0f}")
    
    # Step 4: Stratified selection by length
    selected_samples = []
    
    # Define length strata
    strata = [
        ('short', 0, median_wc, max(1, n_samples // 3)),
        ('medium', median_wc, q75_wc, max(1, n_samples // 3)),
        ('long', q75_wc, float('inf'), max(1, n_samples - 2 * (n_samples // 3)))
    ]
    
    print(f"\nStratified sampling:")
    for stratum_name, lower, upper, n_from_stratum in strata:
        stratum_df = filtered_df[
            (filtered_df['word_count'] > lower) & 
            (filtered_df['word_count'] <= upper)
        ]
        
        if len(stratum_df) > 0:
            # Random sample from this stratum
            n_to_sample = min(n_from_stratum, len(stratum_df))
            samples = stratum_df.sample(n=n_to_sample, random_state=42)
            selected_samples.append(samples)
            print(f"  {stratum_name:8s} ({lower:5.0f}-{upper:5.0f} words): selected {n_to_sample} from {len(stratum_df):,} available")
    
    # Combine all selected samples
    if selected_samples:
        result_df = pd.concat(selected_samples, ignore_index=True)
        print(f"\nTotal samples selected: {len(result_df)}")
        print(f"Word count range: {result_df['word_count'].min():.0f}-{result_df['word_count'].max():.0f}")
        return result_df
    else:
        print("\nWarning: No samples met criteria, falling back to random selection")
        return filtered_df.sample(n=min(n_samples, len(filtered_df)), random_state=42)


# ============================================================================
# 7. DEEP DIVE INTO MIXED REVIEWS
# ============================================================================
print("\n[6/6] Deep dive into MIXED reviews...")
print("\n" + "="*80)
print("UNDERSTANDING MIXED REVIEWS")
print("="*80)

mixed_df = df[df['review_category'] == 'mixed']
positive_df = df[df['review_category'] == 'positive']
negative_df = df[df['review_category'] == 'negative']

print(f"\nMixed reviews: {len(mixed_df):,} ({len(mixed_df)/len(df)*100:.2f}%)")

# Characteristics of mixed reviews
print("\n" + "-"*80)
print("MIXED REVIEW CHARACTERISTICS")
print("-"*80)

print("\nAverage text length comparison:")
print(f"Mixed:    {mixed_df['text_length'].mean():.0f} characters, {mixed_df['word_count'].mean():.0f} words")
print(f"Positive: {positive_df['text_length'].mean():.0f} characters, {positive_df['word_count'].mean():.0f} words")
print(f"Negative: {negative_df['text_length'].mean():.0f} characters, {negative_df['word_count'].mean():.0f} words")

# Look for contrasting words in mixed reviews
print("\n" + "-"*80)
print("CONTRASTING INDICATORS IN MIXED REVIEWS")
print("-"*80)

def find_contrasting_patterns(text):
    """Find contrasting patterns in text"""
    text_lower = text.lower()
    
    positive_indicators = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'fun', 'enjoy', 'awesome', 'perfect']
    negative_indicators = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'boring', 'disappointing', 'frustrating', 'annoying']
    contrast_words = ['but', 'however', 'although', 'though', 'despite', 'unfortunately', 'except']
    
    has_positive = sum(1 for word in positive_indicators if word in text_lower)
    has_negative = sum(1 for word in negative_indicators if word in text_lower)
    has_contrast = sum(1 for word in contrast_words if word in text_lower)
    
    return has_positive, has_negative, has_contrast

# Analyze mixed reviews
mixed_df_copy = mixed_df.copy()
mixed_df_copy[['pos_count', 'neg_count', 'contrast_count']] = mixed_df_copy['review_text'].apply(
    lambda x: pd.Series(find_contrasting_patterns(x))
)

both_sentiments = ((mixed_df_copy['pos_count'] > 0) & (mixed_df_copy['neg_count'] > 0)).sum()
has_contrast = (mixed_df_copy['contrast_count'] > 0).sum()

print(f"\nMixed reviews with both positive and negative words: {both_sentiments:,}")
print(f"Mixed reviews with contrast words: {has_contrast:,}")

print("\nAverage sentiment indicators per review:")
print(f"Mixed reviews    - Positive words: {mixed_df_copy['pos_count'].mean():.2f}, Negative words: {mixed_df_copy['neg_count'].mean():.2f}, Contrast words: {mixed_df_copy['contrast_count'].mean():.2f}")

# ============================================================================
# SYSTEMATIC SAMPLE SELECTION
# ============================================================================
print("\n" + "="*80)
print("SYSTEMATIC REPRESENTATIVE SAMPLE SELECTION")
print("="*80)

# Select representative samples for each category
positive_samples = select_representative_samples(positive_df, 'positive', n_samples=5)
negative_samples = select_representative_samples(negative_df, 'negative', n_samples=5)
mixed_samples = select_representative_samples(mixed_df, 'mixed', n_samples=5)

print("\n" + "="*80)
print("SELECTED REPRESENTATIVE SAMPLES")
print("="*80)

print("\n" + "="*80)
print("SELECTED REPRESENTATIVE SAMPLES")
print("="*80)

# Display positive samples
print(f"\n{'='*80}")
print("POSITIVE REVIEW SAMPLES")
print(f"{'='*80}")
for i, (idx, row) in enumerate(positive_samples.iterrows(), 1):
    print(f"\nSample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters")
    print(f"{'-'*80}")
    print(row['review_text'][:500] + "..." if len(row['review_text']) > 500 else row['review_text'])

# Display negative samples
print(f"\n\n{'='*80}")
print("NEGATIVE REVIEW SAMPLES")
print(f"{'='*80}")
for i, (idx, row) in enumerate(negative_samples.iterrows(), 1):
    print(f"\nSample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters")
    print(f"{'-'*80}")
    print(row['review_text'][:500] + "..." if len(row['review_text']) > 500 else row['review_text'])

# Display mixed samples with analysis
print(f"\n\n{'='*80}")
print("MIXED REVIEW SAMPLES (with contrast analysis)")
print(f"{'='*80}")
for i, (idx, row) in enumerate(mixed_samples.iterrows(), 1):
    text = row['review_text']
    pos, neg, contrast = find_contrasting_patterns(text)
    
    print(f"\nSample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters")
    print(f"Positive indicators: {pos}, Negative indicators: {neg}, Contrast words: {contrast}")
    print(f"{'-'*80}")
    print(text[:500] + "..." if len(text) > 500 else text)

# Get random diverse samples for comparison
print("\n" + "="*80)
print("COMPARATIVE ANALYSIS: POSITIVE vs NEGATIVE vs MIXED")
print("="*80)

for i in range(5):
    print(f"\n{'='*80}")
    print(f"COMPARISON SET #{i+1}")
    print(f"{'='*80}")
    
    # Get one sample from each category
    pos_sample = positive_df.sample(1).iloc[0]['review_text']
    neg_sample = negative_df.sample(1).iloc[0]['review_text']
    mix_sample = mixed_df.sample(1).iloc[0]['review_text']
    
    print(f"\n[POSITIVE REVIEW]")
    print(pos_sample[:300] + "..." if len(pos_sample) > 300 else pos_sample)
    
    print(f"\n[NEGATIVE REVIEW]")
    print(neg_sample[:300] + "..." if len(neg_sample) > 300 else neg_sample)
    
    print(f"\n[MIXED REVIEW]")
    print(mix_sample[:300] + "..." if len(mix_sample) > 300 else mix_sample)

# ============================================================================
# 7. SAVE DETAILED ANALYSIS TO FILE
# ============================================================================
print("\n" + "="*80)
print("SAVING DETAILED ANALYSIS")
print("="*80)

with open(OUTPUT_DIR / 'detailed_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED EDA REPORT - GAME REVIEW SENTIMENT DATASET\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. BASIC STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total reviews: {len(df):,}\n")
    f.write(f"\nCategory distribution:\n{category_counts}\n")
    f.write(f"\nText length statistics:\n{df[['text_length', 'word_count']].describe()}\n")
    
    for category in categories:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"CATEGORY: {category.upper()}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Top 30 most common words:\n")
        for i, (word, count) in enumerate(category_analysis[category]['common_words'], 1):
            f.write(f"{i:2d}. {word:20s} - {count:6,} times\n")

# Save systematically selected samples
with open(OUTPUT_DIR / 'representative_samples.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("SYSTEMATICALLY SELECTED REPRESENTATIVE SAMPLES\n")
    f.write("="*80 + "\n\n")
    
    f.write("SELECTION METHODOLOGY:\n")
    f.write("-"*80 + "\n")
    f.write("1. Random sampling: 100 reviews randomly selected from each category\n")
    f.write("2. Statistical filtering: Exclude outliers (< 10 words or > 300 words)\n")
    f.write("3. For mixed reviews: Prioritize samples with contrast markers\n")
    f.write("4. Stratified selection: Select samples across length spectrum:\n")
    f.write("   - Short: below median word count\n")
    f.write("   - Medium: median to 75th percentile\n")
    f.write("   - Long: above 75th percentile\n")
    f.write("5. Final output: 5 representative samples per category\n\n")
    
    # Positive samples
    f.write("="*80 + "\n")
    f.write("POSITIVE REVIEW SAMPLES\n")
    f.write("="*80 + "\n\n")
    for i, (idx, row) in enumerate(positive_samples.iterrows(), 1):
        f.write(f"Sample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters\n")
        f.write("-"*80 + "\n")
        f.write(row['review_text'] + "\n\n")
    
    # Negative samples
    f.write("\n" + "="*80 + "\n")
    f.write("NEGATIVE REVIEW SAMPLES\n")
    f.write("="*80 + "\n\n")
    for i, (idx, row) in enumerate(negative_samples.iterrows(), 1):
        f.write(f"Sample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters\n")
        f.write("-"*80 + "\n")
        f.write(row['review_text'] + "\n\n")
    
    # Mixed samples
    f.write("\n" + "="*80 + "\n")
    f.write("MIXED REVIEW SAMPLES\n")
    f.write("="*80 + "\n\n")
    for i, (idx, row) in enumerate(mixed_samples.iterrows(), 1):
        text = row['review_text']
        pos, neg, contrast = find_contrasting_patterns(text)
        f.write(f"Sample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters\n")
        f.write(f"Positive indicators: {pos}, Negative indicators: {neg}, Contrast words: {contrast}\n")
        f.write("-"*80 + "\n")
        f.write(text + "\n\n")

print(f"✓ Saved representative samples: {OUTPUT_DIR / 'representative_samples.txt'}")

# Save original detailed analysis (keeping legacy samples for reference)
with open(OUTPUT_DIR / 'detailed_analysis_legacy.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED EDA REPORT - GAME REVIEW SENTIMENT DATASET (LEGACY)\n")
    f.write("="*80 + "\n\n")
    
    f.write("1. BASIC STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total reviews: {len(df):,}\n")
    f.write(f"\nCategory distribution:\n{category_counts}\n")
    f.write(f"\nText length statistics:\n{df[['text_length', 'word_count']].describe()}\n")
    
    for category in categories:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"CATEGORY: {category.upper()}\n")
        f.write(f"{'='*80}\n\n")
        
        f.write(f"Top 30 most common words:\n")
        for i, (word, count) in enumerate(category_analysis[category]['common_words'], 1):
            f.write(f"{i:2d}. {word:20s} - {count:6,} times\n")
print(f"✓ Saved detailed analysis: {OUTPUT_DIR / 'detailed_analysis.txt'}")

# Save summary statistics to CSV
summary_stats = df.groupby('review_category').agg({
    'text_length': ['mean', 'median', 'std'],
    'word_count': ['mean', 'median', 'std'],
    'review_text': 'count'
}).round(2)
summary_stats.to_csv(OUTPUT_DIR / 'summary_statistics.csv')
print(f"✓ Saved summary statistics: {OUTPUT_DIR / 'summary_statistics.csv'}")

print("\n" + "="*80)
print("EDA COMPLETE!")
print("="*80)
print(f"\nResults saved to: {OUTPUT_DIR}")
print("\nGenerated files:")
print("  1. 01_basic_distributions.png - Distribution visualizations")
print("  2. 02_word_clouds.png - Word clouds per category")
print("  3. 03_common_words_by_category.png - Top words visualization")
print("  4. representative_samples.txt - Systematically selected samples")
print("  5. detailed_analysis_legacy.txt - Comprehensive text report (legacy)")
print("  6. summary_statistics.csv - Statistical summary")
print("\n" + "="*80)
