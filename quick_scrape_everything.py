"""
Quick Scrape Everything Script

This is a simplified script that runs the complete workflow:
1. Discover games from Metacritic
2. Scrape reviews for all discovered games
3. Combine all results into a single file

Usage:
    python quick_scrape_everything.py
    
    # With options
    python quick_scrape_everything.py --max-pages 3 --max-reviews 50 --platform playstation-5
"""

import subprocess
import argparse
import sys
from datetime import datetime
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print("\n" + "=" * 70)
    print(f"STEP: {description}")
    print("=" * 70)
    print(f"Command: {' '.join(command)}")
    print()
    
    try:
        result = subprocess.run(command, check=True)
        print(f"\n✓ {description} - COMPLETE")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ {description} - FAILED")
        print(f"Error: {e}")
        return False
    except FileNotFoundError:
        print(f"\n✗ Command not found: {command[0]}")
        print("Make sure all dependencies are installed:")
        print("  pip install -r requirements.txt")
        return False


def find_latest_file(directory, pattern):
    """Find the most recent file matching a pattern"""
    files = list(Path(directory).glob(pattern))
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Quick scrape everything - Complete workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the complete workflow:
  1. Discover games from Metacritic browse pages
  2. Scrape reviews for all discovered games
  3. Combine all results into single dataset

Examples:
  # Quick start with defaults (5 pages, all platforms)
  python quick_scrape_everything.py
  
  # Scrape PS5 games only, 100 reviews per game
  python quick_scrape_everything.py --platform playstation-5 --max-reviews 100 --max-pages 3
  
  # High-quality dataset (minimum score 70, max 200 reviews each)
  python quick_scrape_everything.py --min-score 70 --max-reviews 200 --max-pages 10
        """
    )
    
    parser.add_argument('--platform', type=str, default='all',
                      help='Platform filter (playstation-5, pc, switch, all)')
    parser.add_argument('--max-pages', type=int, default=5,
                      help='Maximum discovery pages (default: 5, use 0 for unlimited)')
    parser.add_argument('--max-reviews', type=int, default=None,
                      help='Maximum reviews per game (default: unlimited)')
    parser.add_argument('--min-score', type=int, default=None,
                      help='Minimum metascore filter (0-100)')
    parser.add_argument('--delay', type=float, default=5.0,
                      help='Delay between games in seconds (default: 5.0)')
    parser.add_argument('--skip-discovery', action='store_true',
                      help='Skip discovery, use existing file')
    parser.add_argument('--discovery-file', type=str, default=None,
                      help='Use specific discovery file instead of latest')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("QUICK SCRAPE EVERYTHING - Complete Workflow")
    print("=" * 70)
    print(f"Platform: {args.platform}")
    print(f"Max discovery pages: {args.max_pages}")
    print(f"Max reviews per game: {args.max_reviews or 'unlimited'}")
    print(f"Min metascore: {args.min_score or 'none'}")
    print("=" * 70)
    
    # Confirm
    response = input("\nProceed with scraping? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    discovery_file = None
    
    # STEP 1: Discover games
    if not args.skip_discovery:
        discovery_cmd = [
            "python", "discover_games.py",
            "--platform", args.platform,
            "--max-pages", str(args.max_pages),
            "--format", "txt"
        ]
        
        if args.min_score:
            discovery_cmd.extend(["--min-score", str(args.min_score)])
        
        success = run_command(discovery_cmd, "DISCOVER GAMES")
        
        if not success:
            print("\n✗ Discovery failed. Exiting.")
            sys.exit(1)
        
        # Find the most recent discovery file
        discovery_file = find_latest_file("data", "discovered_games_*.txt")
        
        if not discovery_file:
            print("\n✗ Could not find discovery output file. Exiting.")
            sys.exit(1)
        
        print(f"\n✓ Using discovery file: {discovery_file}")
    
    else:
        # Use specified file or find latest
        if args.discovery_file:
            discovery_file = Path(args.discovery_file)
        else:
            discovery_file = find_latest_file("data", "discovered_games_*.txt")
        
        if not discovery_file or not discovery_file.exists():
            print("\n✗ Discovery file not found. Run without --skip-discovery first.")
            sys.exit(1)
        
        print(f"\n✓ Using existing discovery file: {discovery_file}")
    
    # STEP 2: Scrape all games
    scrape_cmd = [
        "python", "scrape_all_games.py",
        "--input", str(discovery_file),
        "--delay", str(args.delay),
        "--skip-errors"
    ]
    
    if args.max_reviews:
        scrape_cmd.extend(["--max-reviews", str(args.max_reviews)])
    
    success = run_command(scrape_cmd, "SCRAPE ALL REVIEWS")
    
    if not success:
        print("\n⚠ Scraping completed with errors. Check logs.")
    
    # STEP 3: Summary
    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)
    print("\n📁 Check the 'data/' folder for results:")
    print("  - metacritic_reviews_*.json - Individual game reviews")
    print("  - scraping_log_*.json - Scraping results log")
    print("\nNext steps:")
    print("  1. Review the scraping log to check for any failures")
    print("  2. Use combine_data.py to merge all reviews into one dataset")
    print("  3. Start your sentiment analysis!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user. Goodbye!")
        sys.exit(1)
