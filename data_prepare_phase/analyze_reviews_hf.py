"""
Review Data Analysis Script (HuggingFace version)
Analyzes the review data from the HuggingFace dataset
dataset: auphong2707/game-reviews-sentiment
"""

import random
from collections import Counter, defaultdict
from datetime import datetime
import statistics
import datasets


def load_reviews_hf():
    """Load reviews from HuggingFace dataset"""
    print("Loading dataset from HuggingFace (auphong2707/game-reviews-sentiment)...")
    ds = datasets.load_dataset("auphong2707/game-reviews-sentiment", split="train")
    reviews = ds.to_dict()
    # Convert to list of dicts
    reviews = [dict(zip(reviews, t)) for t in zip(*reviews.values())]
    return reviews


def analyze_reviews(reviews):
    """Perform comprehensive analysis on reviews (same as local version)"""
    print("=" * 80)
    print("REVIEW DATA ANALYSIS REPORT (HuggingFace)")
    print("=" * 80)
    print()
    # Basic Statistics
    print("\U0001F4CA BASIC STATISTICS")
    print("-" * 80)
    total_reviews = len(reviews)
    print(f"Total number of reviews: {total_reviews:,}")
    print()
    # Review Categories
    print("\U0001F4C8 REVIEW CATEGORIES")
    print("-" * 80)
    categories = Counter(review['review_category'] for review in reviews)
    for category, count in categories.most_common():
        percentage = (count / total_reviews) * 100
        print(f"{category.capitalize()}: {count:,} ({percentage:.2f}%)")
    print()
    # Score Statistics
    print("\u2B50 SCORE STATISTICS")
    print("-" * 80)
    scores = [review['review_score'] for review in reviews]
    print(f"Average score: {statistics.mean(scores):.2f}")
    print(f"Median score: {statistics.median(scores):.2f}")
    print(f"Min score: {min(scores)}")
    print(f"Max score: {max(scores)}")
    print(f"Standard deviation: {statistics.stdev(scores):.2f}")
    print()
    # Score Distribution
    print("\U0001F4CA SCORE DISTRIBUTION")
    print("-" * 80)
    score_ranges = {
        '0-20': 0,
        '21-40': 0,
        '41-60': 0,
        '61-80': 0,
        '81-100': 0
    }
    for score in scores:
        if score <= 20:
            score_ranges['0-20'] += 1
        elif score <= 40:
            score_ranges['21-40'] += 1
        elif score <= 60:
            score_ranges['41-60'] += 1
        elif score <= 80:
            score_ranges['61-80'] += 1
        else:
            score_ranges['81-100'] += 1
    for range_name, count in sorted(score_ranges.items()):
        percentage = (count / total_reviews) * 100
        bar = '█' * int(percentage / 2)
        print(f"{range_name:>8}: {bar} {count:,} ({percentage:.2f}%)")
    print()
    # Review Text Length Statistics
    print("\U0001F4DD REVIEW TEXT STATISTICS")
    print("-" * 80)
    review_lengths = [len(review['review_text']) for review in reviews if review.get('review_text')]
    word_counts = [len(review['review_text'].split()) for review in reviews if review.get('review_text')]
    if review_lengths:
        print(f"Average review length (characters): {statistics.mean(review_lengths):.2f}")
        print(f"Average word count: {statistics.mean(word_counts):.2f}")
        print(f"Shortest review: {min(review_lengths)} characters")
        print(f"Longest review: {max(review_lengths)} characters")
        empty_reviews = len([r for r in reviews if not r.get('review_text')])
        if empty_reviews > 0:
            print(f"Empty/missing review texts: {empty_reviews}")
    else:
        print("No review text data available")
    print()
    # Show random samples
    show_random_reviews(reviews, count=5)
    print()
    # Find extreme reviews
    positive_reviews = get_reviews_by_category(reviews, 'positive')
    negative_reviews = get_reviews_by_category(reviews, 'negative')
    print(f"Positive reviews: {len(positive_reviews):,}")
    print(f"Negative reviews: {len(negative_reviews):,}")
    print(f"Mixed reviews: {len(get_reviews_by_category(reviews, 'mixed')):,}")
    print()
    # Perfect scores
    perfect_scores = get_reviews_by_score_range(reviews, 100, 100)
    print(f"Perfect score (100) reviews: {len(perfect_scores):,}")
    # Zero scores
    zero_scores = get_reviews_by_score_range(reviews, 0, 0)
    print(f"Zero score reviews: {len(zero_scores):,}")
    print()
    print("=" * 80)
    print("Analysis complete!")
    print("=" * 80)


def show_random_reviews(reviews, count=5):
    """Show random review samples"""
    print("=" * 80)
    print(f"\U0001F3B2 RANDOM REVIEW PREVIEWS ({count} samples)")
    print("=" * 80)
    print()
    sample_reviews = random.sample(reviews, min(count, len(reviews)))
    for i, review in enumerate(sample_reviews, 1):
        print(f"Review #{i}")
        print("-" * 80)
        print(f"Score: {review['review_score']}/100 ({review['review_category']})")
        review_text = review.get('review_text', '[No review text]')
        if review_text:
            print(f"Review: {review_text[:200]}{'...' if len(review_text) > 200 else ''}")
        else:
            print(f"Review: [No review text]")
        print()

def get_reviews_by_category(reviews, category):
    """Get reviews filtered by category"""
    return [r for r in reviews if r['review_category'] == category]

def get_reviews_by_score_range(reviews, min_score, max_score):
    """Get reviews within a score range"""
    return [r for r in reviews if min_score <= r['review_score'] <= max_score]

def main():
    reviews = load_reviews_hf()
    print(f"\u2705 Loaded {len(reviews):,} reviews\n")
    analyze_reviews(reviews)

if __name__ == "__main__":
    main()
