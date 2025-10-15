"""
Combine Data Script

This script combines multiple scraped review JSON files into a single dataset.
Useful after scraping many games to create one unified dataset for analysis.

Usage:
    python combine_data.py
    python combine_data.py --input data/ --output combined_reviews.json
    python combine_data.py --format csv
"""

import json
import csv
import argparse
from pathlib import Path
from datetime import datetime
from collections import defaultdict


class DataCombiner:
    """Combines multiple review JSON files into one dataset"""
    
    def __init__(self, input_dir='data'):
        self.input_dir = Path(input_dir)
        self.reviews = []
        self.stats = defaultdict(int)
    
    def find_review_files(self):
        """Find all review JSON files in the input directory"""
        pattern = 'metacritic_reviews_*.json'
        files = list(self.input_dir.glob(pattern))
        
        print(f"Found {len(files)} review files in '{self.input_dir}'")
        return files
    
    def load_reviews_from_file(self, filepath):
        """Load reviews from a single JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Handle both array and single object formats
            if isinstance(data, list):
                return data
            elif isinstance(data, dict):
                return [data]
            else:
                print(f"  ⚠ Unknown format in {filepath.name}")
                return []
                
        except json.JSONDecodeError as e:
            print(f"  ✗ Error reading {filepath.name}: Invalid JSON - {e}")
            return []
        except Exception as e:
            print(f"  ✗ Error reading {filepath.name}: {e}")
            return []
    
    def combine_all_files(self):
        """Combine all review files into one list"""
        
        print("\n" + "=" * 70)
        print("COMBINING REVIEW DATA")
        print("=" * 70)
        
        files = self.find_review_files()
        
        if not files:
            print("\n⚠ No review files found!")
            print(f"Make sure JSON files exist in '{self.input_dir}'")
            return []
        
        print()
        
        for i, filepath in enumerate(files, 1):
            print(f"[{i}/{len(files)}] Loading {filepath.name}...", end=' ')
            
            reviews = self.load_reviews_from_file(filepath)
            
            if reviews:
                self.reviews.extend(reviews)
                print(f"✓ ({len(reviews)} reviews)")
                
                # Update stats
                for review in reviews:
                    self.stats['total_reviews'] += 1
                    
                    category = review.get('review_category', 'unknown')
                    self.stats[f'category_{category}'] += 1
                    
                    platform = review.get('game_platform', 'unknown')
                    self.stats[f'platform_{platform}'] += 1
            else:
                print("✗ (no reviews)")
        
        print("\n" + "=" * 70)
        print(f"TOTAL REVIEWS COMBINED: {len(self.reviews)}")
        print("=" * 70)
        
        return self.reviews
    
    def remove_duplicates(self):
        """Remove duplicate reviews based on review_id"""
        
        print("\nRemoving duplicates...", end=' ')
        
        seen_ids = set()
        unique_reviews = []
        duplicates = 0
        
        for review in self.reviews:
            review_id = review.get('review_id')
            
            if review_id and review_id in seen_ids:
                duplicates += 1
                continue
            
            seen_ids.add(review_id)
            unique_reviews.append(review)
        
        original_count = len(self.reviews)
        self.reviews = unique_reviews
        
        print(f"✓ Removed {duplicates} duplicates")
        print(f"Final count: {len(self.reviews)} unique reviews")
        
        return self.reviews
    
    def print_statistics(self):
        """Print statistics about the combined dataset"""
        
        print("\n" + "=" * 70)
        print("DATASET STATISTICS")
        print("=" * 70)
        
        if not self.reviews:
            print("No reviews to analyze")
            return
        
        # Count by category
        categories = defaultdict(int)
        platforms = defaultdict(int)
        games = defaultdict(int)
        scores = []
        
        for review in self.reviews:
            category = review.get('review_category', 'unknown')
            categories[category] += 1
            
            platform = review.get('game_platform', 'unknown')
            platforms[platform] += 1
            
            game = review.get('game_title', 'unknown')
            games[game] += 1
            
            score = review.get('review_score')
            if score is not None:
                scores.append(score)
        
        # Print statistics
        print(f"\nTotal Reviews: {len(self.reviews)}")
        print(f"Unique Games: {len(games)}")
        
        print("\nReviews by Sentiment:")
        for category in ['positive', 'mixed', 'negative']:
            count = categories.get(category, 0)
            percentage = (count / len(self.reviews) * 100) if self.reviews else 0
            print(f"  {category.capitalize()}: {count} ({percentage:.1f}%)")
        
        print("\nTop 10 Games by Review Count:")
        top_games = sorted(games.items(), key=lambda x: x[1], reverse=True)[:10]
        for game, count in top_games:
            print(f"  {game}: {count} reviews")
        
        print("\nReviews by Platform:")
        sorted_platforms = sorted(platforms.items(), key=lambda x: x[1], reverse=True)
        for platform, count in sorted_platforms:
            print(f"  {platform}: {count} reviews")
        
        if scores:
            avg_score = sum(scores) / len(scores)
            print(f"\nAverage Score: {avg_score:.1f}")
            print(f"Score Range: {min(scores)} - {max(scores)}")
        
        print("=" * 70)
    
    def save_to_json(self, output_file):
        """Save combined reviews to JSON file"""
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.reviews, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Saved to JSON: {output_file}")
            
        except Exception as e:
            print(f"\n✗ Error saving JSON: {e}")
    
    def save_to_csv(self, output_file):
        """Save combined reviews to CSV file"""
        
        if not self.reviews:
            print("\n⚠ No reviews to save")
            return
        
        try:
            # Get all possible fields
            fieldnames = set()
            for review in self.reviews:
                fieldnames.update(review.keys())
            
            fieldnames = sorted(fieldnames)
            
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.reviews)
            
            print(f"\n✓ Saved to CSV: {output_file}")
            
        except Exception as e:
            print(f"\n✗ Error saving CSV: {e}")
    
    def save(self, output_format='json', output_file=None):
        """Save combined data in specified format"""
        
        if not self.reviews:
            print("\n⚠ No reviews to save")
            return
        
        if not output_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"data/combined_reviews_{timestamp}.{output_format}"
        
        if output_format == 'json':
            self.save_to_json(output_file)
        elif output_format == 'csv':
            self.save_to_csv(output_file)
        else:
            print(f"\n✗ Unsupported format: {output_format}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Combine multiple review JSON files into one dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Combine all reviews in data/ folder
  python combine_data.py
  
  # Save as CSV
  python combine_data.py --format csv
  
  # Custom input/output
  python combine_data.py --input my_data/ --output results.json
  
  # Keep duplicates
  python combine_data.py --no-dedupe
        """
    )
    
    parser.add_argument('--input', '-i', type=str, default='data',
                      help='Input directory with JSON files (default: data)')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Output filename (default: auto-generated)')
    parser.add_argument('--format', '-f', type=str, default='json',
                      choices=['json', 'csv'],
                      help='Output format (default: json)')
    parser.add_argument('--no-dedupe', action='store_true',
                      help='Keep duplicate reviews')
    parser.add_argument('--no-stats', action='store_true',
                      help='Skip statistics display')
    
    args = parser.parse_args()
    
    # Create combiner
    combiner = DataCombiner(input_dir=args.input)
    
    # Combine files
    reviews = combiner.combine_all_files()
    
    if not reviews:
        print("\n⚠ No reviews found. Exiting.")
        return
    
    # Remove duplicates (unless disabled)
    if not args.no_dedupe:
        combiner.remove_duplicates()
    
    # Show statistics (unless disabled)
    if not args.no_stats:
        combiner.print_statistics()
    
    # Save combined data
    combiner.save(output_format=args.format, output_file=args.output)
    
    print("\n" + "=" * 70)
    print("COMBINATION COMPLETE")
    print("=" * 70)
    print(f"\nYour combined dataset is ready!")
    print(f"Total reviews: {len(combiner.reviews)}")
    print("\nNext steps:")
    print("  1. Load the combined file for analysis")
    print("  2. Perform sentiment analysis")
    print("  3. Train machine learning models")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
