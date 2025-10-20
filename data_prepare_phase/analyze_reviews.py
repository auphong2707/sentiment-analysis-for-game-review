"""
Review Data Analysis Script
Analyzes the review data from JSONL files
"""

import json
import random
import argparse
from collections import Counter, defaultdict
from datetime import datetime
import statistics


def load_reviews(file_path):
    """Load reviews from JSONL file"""
    reviews = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))
    return reviews


def analyze_reviews(reviews):
    """Perform comprehensive analysis on reviews"""
    
    print("=" * 80)
    print("REVIEW DATA ANALYSIS REPORT")
    print("=" * 80)
    print()
    
    # Basic Statistics
    print("📊 BASIC STATISTICS")
    print("-" * 80)
    total_reviews = len(reviews)
    print(f"Total number of reviews: {total_reviews:,}")
    print()
    
    # Review Categories
    print("📈 REVIEW CATEGORIES")
    print("-" * 80)
    categories = Counter(review['review_category'] for review in reviews)
    for category, count in categories.most_common():
        percentage = (count / total_reviews) * 100
        print(f"{category.capitalize()}: {count:,} ({percentage:.2f}%)")
    print()
    
    # Score Statistics
    print("⭐ SCORE STATISTICS")
    print("-" * 80)
    scores = [review['review_score'] for review in reviews]
    print(f"Average score: {statistics.mean(scores):.2f}")
    print(f"Median score: {statistics.median(scores):.2f}")
    print(f"Min score: {min(scores)}")
    print(f"Max score: {max(scores)}")
    print(f"Standard deviation: {statistics.stdev(scores):.2f}")
    print()
    
    # Score Distribution
    print("📊 SCORE DISTRIBUTION")
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
    
    # Game Statistics
    print("🎮 GAME STATISTICS")
    print("-" * 80)
    games = Counter(review['game_title'] for review in reviews)
    print(f"Total unique games: {len(games)}")
    print(f"\nTop 10 most reviewed games:")
    for i, (game, count) in enumerate(games.most_common(10), 1):
        print(f"{i:2d}. {game[:60]:<60} - {count:,} reviews")
    print()
    
    # Platform Statistics
    print("🕹️  PLATFORM STATISTICS")
    print("-" * 80)
    platforms = Counter(review['game_platform'] for review in reviews)
    print(f"Total unique platforms: {len(platforms)}")
    for platform, count in platforms.most_common():
        percentage = (count / total_reviews) * 100
        print(f"{platform:<30} - {count:,} reviews ({percentage:.2f}%)")
    print()
    
    # Genre Statistics
    print("🎭 GENRE STATISTICS")
    print("-" * 80)
    genres = Counter(review['game_genre'] for review in reviews)
    print(f"Total unique genres: {len(genres)}")
    for genre, count in genres.most_common(10):
        percentage = (count / total_reviews) * 100
        print(f"{genre:<30} - {count:,} reviews ({percentage:.2f}%)")
    print()
    
    # Review Text Length Statistics
    print("📝 REVIEW TEXT STATISTICS")
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
    
    # Top Reviewers
    print("👥 TOP REVIEWERS")
    print("-" * 80)
    reviewers = Counter(review['reviewer_name'] for review in reviews)
    print(f"Total unique reviewers: {len(reviewers):,}")
    print(f"\nTop 10 most active reviewers:")
    for i, (reviewer, count) in enumerate(reviewers.most_common(10), 1):
        print(f"{i:2d}. {reviewer:<30} - {count:,} reviews")
    print()
    
    # Date Statistics
    print("📅 DATE STATISTICS")
    print("-" * 80)
    review_dates = Counter(review['review_date'] for review in reviews)
    print(f"Date range: {min(review_dates.keys())} to {max(review_dates.keys())}")
    print(f"\nMost active review dates:")
    for i, (date, count) in enumerate(review_dates.most_common(10), 1):
        print(f"{i:2d}. {date:<20} - {count:,} reviews")
    print()
    
    # Developer/Publisher Statistics
    print("🏢 DEVELOPER/PUBLISHER STATISTICS")
    print("-" * 80)
    developers = Counter(review.get('game_developer', 'Unknown') for review in reviews)
    publishers = Counter(review.get('game_publisher', 'Unknown') for review in reviews)
    
    print(f"Total unique developers: {len(developers)}")
    print(f"Top 5 developers by review count:")
    for i, (dev, count) in enumerate(developers.most_common(5), 1):
        dev_name = dev if dev else 'Unknown'
        print(f"{i}. {dev_name:<40} - {count:,} reviews")
    
    print(f"\nTotal unique publishers: {len(publishers)}")
    print(f"Top 5 publishers by review count:")
    for i, (pub, count) in enumerate(publishers.most_common(5), 1):
        pub_name = pub if pub else 'Unknown'
        print(f"{i}. {pub_name:<40} - {count:,} reviews")
    print()
    
    return reviews


def show_random_reviews(reviews, count=5):
    """Show random review samples"""
    print("=" * 80)
    print(f"🎲 RANDOM REVIEW PREVIEWS ({count} samples)")
    print("=" * 80)
    print()
    
    sample_reviews = random.sample(reviews, min(count, len(reviews)))
    
    for i, review in enumerate(sample_reviews, 1):
        print(f"Review #{i}")
        print("-" * 80)
        print(f"Game: {review['game_title']}")
        print(f"Platform: {review['game_platform']}")
        print(f"Score: {review['review_score']}/100 ({review['review_category']})")
        print(f"Reviewer: {review['reviewer_name']}")
        print(f"Date: {review['review_date']}")
        review_text = review.get('review_text', '[No review text]')
        if review_text:
            print(f"Review: {review_text[:200]}{'...' if len(review_text) > 200 else ''}")
        else:
            print(f"Review: [No review text]")
        print()


def get_reviews_by_category(reviews, category):
    """Get reviews filtered by category"""
    return [r for r in reviews if r['review_category'] == category]


def get_reviews_by_game(reviews, game_title):
    """Get reviews for a specific game"""
    return [r for r in reviews if r['game_title'].lower() == game_title.lower()]


def get_reviews_by_score_range(reviews, min_score, max_score):
    """Get reviews within a score range"""
    return [r for r in reviews if min_score <= r['review_score'] <= max_score]


def export_summary(reviews, output_file):
    """Export summary statistics to a file"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # Redirect print to file
        import sys
        old_stdout = sys.stdout
        sys.stdout = f
        
        analyze_reviews(reviews)
        
        sys.stdout = old_stdout
    
    print(f"✅ Summary exported to {output_file}")


def main():
    """Main analysis function"""
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Analyze review data from JSONL files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_reviews.py data/review_data/review_data_part1.jsonl
  python analyze_reviews.py "D:/Git/sentiment-analysis-for-game-review/data/review_data/review_data_part1.jsonl"
        """
    )
    parser.add_argument(
        'filepath',
        nargs='?',
        default=r"./data/review_data/review_data_part1.jsonl",
        help='Path to the JSONL file containing review data (default: data/review_data/review_data_part1.jsonl)'
    )
    
    args = parser.parse_args()
    file_path = args.filepath
    
    print(f"Loading reviews from: {file_path}")
    try:
        reviews = load_reviews(file_path)
        print(f"✅ Loaded {len(reviews):,} reviews\n")
    except FileNotFoundError:
        print(f"❌ Error: File not found: {file_path}")
        return
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in file: {e}")
        return
    except Exception as e:
        print(f"❌ Error loading file: {e}")
        return
    
    # Perform analysis
    analyze_reviews(reviews)
    
    # Show random samples
    show_random_reviews(reviews, count=5)
    
    # Additional analyses examples
    print("=" * 80)
    print("📌 ADDITIONAL INSIGHTS")
    print("=" * 80)
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
    
    # Export option
    export_choice = input("\nWould you like to export the summary to a text file? (y/n): ")
    if export_choice.lower() == 'y':
        output_file = r"d:\Git\sentiment-analysis-for-game-review\review_analysis_summary.txt"
        export_summary(reviews, output_file)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
