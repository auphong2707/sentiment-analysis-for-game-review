"""
Exploratory Data Analysis (EDA) for Game Review Sentiment Dataset (HuggingFace version)

This script performs comprehensive EDA including:
1. Basic statistics and distribution analysis
2. Text analysis per category (positive, negative, mixed)
3. Common words and themes in each category
4. Deep dive into "mixed" reviews to understand their characteristics
5. Sample analysis to understand why reviews are classified as mixed vs positive/negative

Loads data directly from HuggingFace: auphong2707/game-reviews-sentiment
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
import re
from wordcloud import WordCloud
import datasets
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Output directory for HF EDA results
OUTPUT_DIR = Path(__file__).parent / "eda_results_hf"
OUTPUT_DIR.mkdir(exist_ok=True)

print("="*80)
print("EXPLORATORY DATA ANALYSIS - GAME REVIEW SENTIMENT DATASET (HuggingFace)")
print("="*80)

# ============================================================================
# 1. LOAD AND BASIC STATISTICS
# ============================================================================
print("\n[1/6] Loading dataset from HuggingFace...")
ds = datasets.load_dataset("auphong2707/game-reviews-sentiment", split="train")
df = pd.DataFrame(ds)
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
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
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
    all_words = []
    for text in texts:
        all_words.extend(clean_text(text))
    return Counter(all_words).most_common(top_n)

categories = df['review_category'].unique()
category_analysis = {}

for category in categories:
    print(f"\n{'='*80}")
    print(f"ANALYZING: {category.upper()}")
    print(f"{'='*80}")
    category_df = df[df['review_category'] == category]
    texts = category_df['review_text'].tolist()
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
    n_candidates = min(n_candidates, len(category_df))
    candidate_pool = category_df.sample(n=n_candidates, random_state=42)
    filtered_df = candidate_pool[(candidate_pool['word_count'] >= 10) & (candidate_pool['word_count'] <= 300)].copy()
    if category_name == 'mixed':
        contrast_words = ['but', 'however', 'although', 'though', 'despite', 'unfortunately', 'except']
        filtered_df['has_contrast'] = filtered_df['review_text'].apply(
            lambda x: any(word in x.lower() for word in contrast_words)
        )
        contrast_df = filtered_df[filtered_df['has_contrast']].copy()
        if len(contrast_df) >= n_samples:
            filtered_df = contrast_df
    if len(filtered_df) < n_samples:
        n_samples = len(filtered_df)
    median_wc = filtered_df['word_count'].median()
    mean_wc = filtered_df['word_count'].mean()
    q75_wc = filtered_df['word_count'].quantile(0.75)
    selected_samples = []
    strata = [
        ('short', 0, median_wc, max(1, n_samples // 3)),
        ('medium', median_wc, q75_wc, max(1, n_samples // 3)),
        ('long', q75_wc, float('inf'), max(1, n_samples - 2 * (n_samples // 3)))
    ]
    for stratum_name, lower, upper, n_from_stratum in strata:
        stratum_df = filtered_df[(filtered_df['word_count'] > lower) & (filtered_df['word_count'] <= upper)]
        if len(stratum_df) > 0:
            n_to_sample = min(n_from_stratum, len(stratum_df))
            samples = stratum_df.sample(n=n_to_sample, random_state=42)
            selected_samples.append(samples)
    if selected_samples:
        result_df = pd.concat(selected_samples, ignore_index=True)
        return result_df
    else:
        return filtered_df.sample(n=min(n_samples, len(filtered_df)), random_state=42)

# ============================================================================
# 7. DEEP DIVE INTO MIXED REVIEWS
# ============================================================================
def find_contrasting_patterns(text):
    text_lower = text.lower()
    positive_indicators = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'fun', 'enjoy', 'awesome', 'perfect']
    negative_indicators = ['bad', 'poor', 'terrible', 'awful', 'hate', 'worst', 'boring', 'disappointing', 'frustrating', 'annoying']
    contrast_words = ['but', 'however', 'although', 'though', 'despite', 'unfortunately', 'except']
    has_positive = sum(1 for word in positive_indicators if word in text_lower)
    has_negative = sum(1 for word in negative_indicators if word in text_lower)
    has_contrast = sum(1 for word in contrast_words if word in text_lower)
    return has_positive, has_negative, has_contrast

mixed_df = df[df['review_category'] == 'mixed']
positive_df = df[df['review_category'] == 'positive']
negative_df = df[df['review_category'] == 'negative']

mixed_df_copy = mixed_df.copy()
mixed_df_copy[['pos_count', 'neg_count', 'contrast_count']] = mixed_df_copy['review_text'].apply(
    lambda x: pd.Series(find_contrasting_patterns(x))
)

# ============================================================================
# 8. SAVE DETAILED ANALYSIS TO FILES
# ============================================================================
print("\n[6/6] Saving detailed analysis and samples...")

categories = df['review_category'].unique()

with open(OUTPUT_DIR / 'detailed_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("="*80 + "\n")
    f.write("DETAILED EDA REPORT - GAME REVIEW SENTIMENT DATASET (HuggingFace)\n")
    f.write("="*80 + "\n\n")
    f.write("1. BASIC STATISTICS\n")
    f.write("-"*80 + "\n")
    f.write(f"Total reviews: {len(df):,}\n")
    f.write(f"\nCategory distribution:\n{category_counts}\n")
    f.write(f"\nText length statistics (overall):\n{df[['text_length', 'word_count']].describe()}\n")

    # Per-category statistics for LaTeX table
    f.write("\nText length and word count statistics by category:\n")
    stats = {}
    for category in categories:
        cat_df = df[df['review_category'] == category]
        stats[category] = {
            'text_length_mean': cat_df['text_length'].mean(),
            'text_length_median': cat_df['text_length'].median(),
            'text_length_std': cat_df['text_length'].std(),
            'text_length_min': cat_df['text_length'].min(),
            'text_length_max': cat_df['text_length'].max(),
            'word_count_mean': cat_df['word_count'].mean(),
            'word_count_median': cat_df['word_count'].median(),
            'word_count_25': cat_df['word_count'].quantile(0.25),
            'word_count_75': cat_df['word_count'].quantile(0.75),
            'word_count_std': cat_df['word_count'].std(),
        }
        f.write(f"\nCategory: {category}\n")
        f.write(f"  text_length: mean={stats[category]['text_length_mean']:.2f}, median={stats[category]['text_length_median']:.0f}, std={stats[category]['text_length_std']:.2f}, min={stats[category]['text_length_min']}, max={stats[category]['text_length_max']}\n")
        f.write(f"  word_count: mean={stats[category]['word_count_mean']:.2f}, median={stats[category]['word_count_median']:.0f}, 25%={stats[category]['word_count_25']:.0f}, 75%={stats[category]['word_count_75']:.0f}, std={stats[category]['word_count_std']:.2f}\n")

    # Also print overall for easy copy-paste
    f.write(f"\nOverall:\n")
    f.write(f"  text_length: mean={df['text_length'].mean():.2f}, median={df['text_length'].median():.0f}, std={df['text_length'].std():.2f}, min={df['text_length'].min()}, max={df['text_length'].max()}\n")
    f.write(f"  word_count: mean={df['word_count'].mean():.2f}, median={df['word_count'].median():.0f}, 25%={df['word_count'].quantile(0.25):.0f}, 75%={df['word_count'].quantile(0.75):.0f}, std={df['word_count'].std():.2f}\n")

    for category in categories:
        f.write(f"\n\n{'='*80}\n")
        f.write(f"CATEGORY: {category.upper()}\n")
        f.write(f"{'='*80}\n\n")
        f.write(f"Top 30 most common words:\n")
        for i, (word, count) in enumerate(category_analysis[category]['common_words'], 1):
            f.write(f"{i:2d}. {word:20s} - {count:6,} times\n")

# Save systematically selected samples
positive_samples = select_representative_samples(positive_df, 'positive', n_samples=5)
negative_samples = select_representative_samples(negative_df, 'negative', n_samples=5)
mixed_samples = select_representative_samples(mixed_df, 'mixed', n_samples=5)

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
    f.write("="*80 + "\n")
    f.write("POSITIVE REVIEW SAMPLES\n")
    f.write("="*80 + "\n\n")
    for i, (idx, row) in enumerate(positive_samples.iterrows(), 1):
        f.write(f"Sample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters\n")
        f.write("-"*80 + "\n")
        f.write(row['review_text'] + "\n\n")
    f.write("\n" + "="*80 + "\n")
    f.write("NEGATIVE REVIEW SAMPLES\n")
    f.write("="*80 + "\n\n")
    for i, (idx, row) in enumerate(negative_samples.iterrows(), 1):
        f.write(f"Sample {i} - {row['word_count']:.0f} words, {row['text_length']:.0f} characters\n")
        f.write("-"*80 + "\n")
        f.write(row['review_text'] + "\n\n")
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
print("  5. detailed_analysis.txt - Comprehensive text report")
print("  6. summary_statistics.csv - Statistical summary")
print("\n" + "="*80)
