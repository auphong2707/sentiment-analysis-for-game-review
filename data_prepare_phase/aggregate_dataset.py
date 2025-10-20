"""
Script to aggregate all review data files into a single dataset.
Filters for English reviews only and extracts specific fields.
"""

import json
import os
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm


def is_english(text):
    """
    Check if the text is in English.
    
    Args:
        text: The text to check
        
    Returns:
        bool: True if the text is in English, False otherwise
    """
    if not text or len(text.strip()) < 3:
        return False
    
    try:
        lang = detect(text)
        return lang == 'en'
    except LangDetectException:
        return False


def aggregate_reviews(input_dir, output_file, max_file_size_mb=95):
    """
    Aggregate all review data files into a single dataset.
    Split into multiple files if size exceeds max_file_size_mb.
    
    Args:
        input_dir: Directory containing the review data files
        output_file: Path to the output dataset file (will be split if needed)
        max_file_size_mb: Maximum size per output file in MB (default: 95MB)
    """
    input_path = Path(input_dir)
    
    # Get all JSONL files
    review_files = sorted(input_path.glob('review_data_part*.jsonl'))
    
    if not review_files:
        print(f"No review data files found in {input_dir}")
        return
    
    print(f"Found {len(review_files)} review data files")
    print(f"Max file size per output: {max_file_size_mb} MB")
    print("=" * 80)
    
    total_reviews = 0
    english_reviews = 0
    duplicates = 0
    seen_reviews = set()  # Track unique reviews by text content
    
    # Setup for file splitting
    max_file_size_bytes = max_file_size_mb * 1024 * 1024
    output_path = Path(output_file)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    
    current_part = 1
    current_size = 0
    output_files = []
    
    # Generate output filename for current part
    def get_output_filename(part_num):
        if part_num == 1:
            return output_path
        return output_dir / f"{output_stem}_part{part_num}{output_suffix}"
    
    current_output_file = get_output_filename(current_part)
    output_files.append(current_output_file)
    out_f = open(current_output_file, 'w', encoding='utf-8')
    
    try:
        # Process each file
        for idx, file_path in enumerate(review_files, 1):
            file_total = 0
            file_english = 0
            file_duplicates = 0
            
            print(f"\n[{idx}/{len(review_files)}] Processing: {file_path.name}")
            
            with open(file_path, 'r', encoding='utf-8') as in_f:
                lines = in_f.readlines()
                
                for line in tqdm(lines, desc=f"  Filtering reviews", leave=False):
                    file_total += 1
                    total_reviews += 1
                    
                    try:
                        # Parse the JSON line
                        review = json.loads(line.strip())
                        
                        # Extract required fields
                        review_text = review.get('review_text', '')
                        review_score = review.get('review_score')
                        review_category = review.get('review_category', '')
                        
                        # Check if review is in English
                        if is_english(review_text):
                            # Check for duplicates (using normalized text)
                            review_hash = review_text.strip().lower()
                            
                            if review_hash in seen_reviews:
                                file_duplicates += 1
                                duplicates += 1
                                continue
                            
                            # Add to seen set
                            seen_reviews.add(review_hash)
                            
                            # Create filtered review object
                            filtered_review = {
                                'review_text': review_text,
                                'review_score': review_score,
                                'review_category': review_category
                            }
                            
                            # Convert to JSON string
                            json_line = json.dumps(filtered_review, ensure_ascii=False) + '\n'
                            json_line_bytes = json_line.encode('utf-8')
                            
                            # Check if we need to split to a new file
                            if current_size + len(json_line_bytes) > max_file_size_bytes and english_reviews > 0:
                                out_f.close()
                                current_part += 1
                                current_output_file = get_output_filename(current_part)
                                output_files.append(current_output_file)
                                out_f = open(current_output_file, 'w', encoding='utf-8')
                                current_size = 0
                                print(f"  -> Splitting to new file: {current_output_file.name}")
                            
                            # Write to output file
                            out_f.write(json_line)
                            current_size += len(json_line_bytes)
                            file_english += 1
                            english_reviews += 1
                    
                    except json.JSONDecodeError as e:
                        print(f"\n  Error parsing JSON in {file_path}: {e}")
                        continue
            
            # Print file statistics
            percentage = (file_english / file_total * 100) if file_total > 0 else 0
            print(f"  Total: {file_total:,} | English: {file_english:,} | Duplicates: {file_duplicates:,} | Percentage: {percentage:.2f}%")
    
    finally:
        out_f.close()
    
    print("\n" + "=" * 80)
    print(f"Aggregation complete!")
    print(f"Total reviews processed: {total_reviews:,}")
    print(f"English reviews extracted: {english_reviews:,}")
    print(f"Duplicates filtered: {duplicates:,}")
    print(f"Unique English reviews: {english_reviews:,}")
    print(f"Percentage of English reviews: {(english_reviews/total_reviews*100):.2f}%")
    print(f"Number of output files: {len(output_files)}")
    for i, out_file in enumerate(output_files, 1):
        file_size = out_file.stat().st_size / (1024 * 1024)
        print(f"  [{i}] {out_file.name} - {file_size:.2f} MB")
    
    return output_files


def analyze_dataset(dataset_files):
    """
    Analyze the aggregated dataset and print statistics.
    
    Args:
        dataset_files: Single file path or list of file paths to analyze
    """
    # Ensure we have a list
    if isinstance(dataset_files, (str, Path)):
        dataset_files = [dataset_files]
    
    # Check if files exist
    existing_files = []
    for f in dataset_files:
        if Path(f).exists():
            existing_files.append(Path(f))
    
    if not existing_files:
        print(f"No dataset files found")
        return
    
    print("\n" + "=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)
    
    total_count = 0
    category_counts = {}
    score_distribution = {}
    text_lengths = []
    
    print(f"Analyzing {len(existing_files)} file(s)...")
    
    for dataset_file in existing_files:
        with open(dataset_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            for line in tqdm(lines, desc=f"Reading {dataset_file.name}", leave=False):
                try:
                    review = json.loads(line.strip())
                    total_count += 1
                    
                    # Count categories
                    category = review.get('review_category', 'unknown')
                    category_counts[category] = category_counts.get(category, 0) + 1
                    
                    # Count score distribution
                    score = review.get('review_score')
                    if score is not None:
                        score_distribution[score] = score_distribution.get(score, 0) + 1
                    
                    # Track text length
                    text = review.get('review_text', '')
                    text_lengths.append(len(text))
                    
                except json.JSONDecodeError:
                    continue
    
    # Print statistics
    print(f"\n📊 Total Data Points: {total_count:,}")
    print(f"   (All unique English reviews after deduplication)")
    print(f"\n📝 Average Review Length: {sum(text_lengths)/len(text_lengths):.1f} characters")
    print(f"   Min Length: {min(text_lengths):,} characters")
    print(f"   Max Length: {max(text_lengths):,} characters")
    
    print(f"\n📁 Category Distribution:")
    for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_count * 100)
        print(f"   {category.capitalize():12} : {count:6,} ({percentage:5.2f}%)")
    
    print(f"\n⭐ Score Distribution:")
    for score in sorted(score_distribution.keys()):
        count = score_distribution[score]
        percentage = (count / total_count * 100)
        bar = '█' * int(percentage / 2)
        print(f"   Score {score:3} : {count:6,} ({percentage:5.2f}%) {bar}")
    
    # Calculate total file size
    total_file_size = sum(f.stat().st_size for f in existing_files)
    if total_file_size < 1024 * 1024:
        size_str = f"{total_file_size / 1024:.2f} KB"
    else:
        size_str = f"{total_file_size / (1024 * 1024):.2f} MB"
    
    print(f"\n💾 Total File Size: {size_str} across {len(existing_files)} file(s)")
    for f in existing_files:
        f_size = f.stat().st_size / (1024 * 1024)
        print(f"   {f.name}: {f_size:.2f} MB")
    print("=" * 80)


def main():
    # Define paths
    script_dir = Path(__file__).parent
    input_dir = script_dir / 'data' / 'review_data'
    output_file = script_dir / 'data' / 'aggregated_reviews_english.jsonl'
    
    # Create output directory if it doesn't exist
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("Starting review aggregation...")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")
    print()
    
    # Aggregate reviews
    result_files = aggregate_reviews(input_dir, output_file)
    
    # Analyze the final dataset
    if result_files:
        analyze_dataset(result_files)


if __name__ == "__main__":
    main()
