"""
Scrape All Games Script

This script reads a list of discovered games and scrapes reviews for each one.
It supports parallel execution, error handling, and progress tracking.

Usage:
    # First, discover games
    python discover_games.py --max-pages 5 --format txt
    
    # Then scrape all discovered games
    python scrape_all_games.py --input data/discovered_games_XXXXXX.txt
    
    # Or use JSON format
    python scrape_all_games.py --input data/discovered_games_XXXXXX.json --max-reviews 100
"""

import subprocess
import argparse
import json
import time
import sys
from pathlib import Path
from datetime import datetime
import csv


class GameScraper:
    """Scrapes reviews for multiple games"""
    
    def __init__(self, max_reviews=None, delay=5, skip_errors=True, skip_existing=True, output_file=None):
        self.max_reviews = max_reviews
        self.delay = delay
        self.skip_errors = skip_errors
        self.skip_existing = skip_existing
        self.results = {
            'success': [],
            'failed': [],
            'skipped': []
        }
        self.progress_file = Path('data/scraping_progress.json')
        self.scraped_games = self._load_progress()
        
        # Setup JSONL output file
        if output_file:
            self.output_file = Path(output_file)
        else:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.output_file = Path(f'data/all_reviews_{timestamp}.jsonl')
        
        # Create output file and parent directories
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize file if it doesn't exist (don't overwrite if resuming)
        if not self.output_file.exists():
            self.output_file.touch()
            print(f"📝 Output file: {self.output_file}")
        else:
            print(f"📝 Appending to existing file: {self.output_file}")
    
    def _load_progress(self):
        """Load previously scraped games from progress file"""
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r', encoding='utf-8') as f:
                    progress = json.load(f)
                    scraped = set()
                    for game in progress.get('success', []):
                        key = f"{game['game']}|{game['platform']}"
                        scraped.add(key)
                    return scraped
            except Exception as e:
                print(f"⚠ Could not load progress file: {e}")
                return set()
        return set()
    
    def _save_progress(self):
        """Save current progress to file"""
        try:
            self.progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.progress_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"⚠ Could not save progress: {e}")
    
    def _is_already_scraped(self, game_name, platform):
        """Check if game was already scraped"""
        key = f"{game_name}|{platform}"
        return key in self.scraped_games
    
    def _mark_as_scraped(self, game_name, platform):
        """Mark game as scraped"""
        key = f"{game_name}|{platform}"
        self.scraped_games.add(key)
    
    def _append_to_jsonl(self, temp_json_file):
        """
        Read JSON array from temp file and append each review as a line to JSONL
        
        Returns: Number of reviews appended
        """
        try:
            temp_path = Path(temp_json_file)
            if not temp_path.exists():
                print(f"⚠ Warning: Temp file not found: {temp_json_file}")
                return 0
            
            # Read the JSON array
            with open(temp_path, 'r', encoding='utf-8') as f:
                reviews = json.load(f)
            
            if not reviews:
                print(f"⚠ Warning: No reviews found in {temp_json_file}")
                return 0
            
            # Append each review as a JSON line
            with open(self.output_file, 'a', encoding='utf-8') as f:
                for review in reviews:
                    json.dump(review, f, ensure_ascii=False)
                    f.write('\n')
            
            return len(reviews)
            
        except json.JSONDecodeError as e:
            print(f"⚠ Warning: Could not parse JSON from {temp_json_file}: {e}")
            return 0
        except Exception as e:
            print(f"⚠ Warning: Error appending to JSONL: {e}")
            return 0
    
    def load_games_from_txt(self, filepath):
        """
        Load games from TXT file
        Format: game_name_slug|platform_slug
        """
        games = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    
                    # Support both old format (game|platform) and new format (game only)
                    if '|' in line:
                        parts = line.split('|')
                        games.append({
                            'game_name_slug': parts[0].strip(),
                            'platform_slug': parts[1].strip() if len(parts) > 1 else 'multiplatform'
                        })
                    else:
                        # New simplified format: just game name
                        games.append({
                            'game_name_slug': line,
                            'platform_slug': 'multiplatform'
                        })
            
            print(f"✓ Loaded {len(games)} games from {filepath}")
            return games
            
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return []
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return []
    
    def load_games_from_json(self, filepath):
        """Load games from JSON file"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                games = json.load(f)
            
            print(f"✓ Loaded {len(games)} games from {filepath}")
            return games
            
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            print(f"✗ Invalid JSON file: {e}")
            return []
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return []
    
    def load_games_from_csv(self, filepath):
        """Load games from CSV file"""
        games = []
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                games = list(reader)
            
            print(f"✓ Loaded {len(games)} games from {filepath}")
            return games
            
        except FileNotFoundError:
            print(f"✗ File not found: {filepath}")
            return []
        except Exception as e:
            print(f"✗ Error reading file: {e}")
            return []
    
    def load_games(self, filepath):
        """Auto-detect format and load games"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            print(f"✗ File not found: {filepath}")
            return []
        
        # Detect format from extension
        if filepath.suffix == '.txt':
            return self.load_games_from_txt(filepath)
        elif filepath.suffix == '.json':
            return self.load_games_from_json(filepath)
        elif filepath.suffix == '.csv':
            return self.load_games_from_csv(filepath)
        else:
            print(f"✗ Unsupported file format: {filepath.suffix}")
            return []
    
    def scrape_game(self, game_name, platform, game_num, total_games):
        """Scrape reviews for a single game"""
        
        # Check if already scraped
        if self.skip_existing and self._is_already_scraped(game_name, platform):
            print(f"\n[{game_num}/{total_games}] ⏭️  Skipping {game_name} ({platform}) - Already scraped")
            self.results['skipped'].append({
                'game': game_name,
                'platform': platform,
                'reason': 'already_scraped',
                'timestamp': datetime.now().isoformat()
            })
            return True
        
        print("\n" + "=" * 70)
        print(f"[{game_num}/{total_games}] Scraping: {game_name} ({platform})")
        print("=" * 70)
        
        # Build scrapy command - output to temporary JSON file first
        temp_output = f"data/temp_{game_name}_{platform}.json"
        
        # Use Python from virtual environment to ensure scrapy-playwright is available
        python_exe = sys.executable  # This gets the current Python interpreter
        
        command = [
            python_exe, "-m", "scrapy", "crawl", "metacritic_reviews",
            "-a", f"game_name={game_name}",
            "-a", f"platform={platform}",
            "-o", temp_output
        ]
        
        if self.max_reviews:
            command.extend(["-a", f"max_reviews={self.max_reviews}"])
        
        print(f"Command: {' '.join(command)}")
        print()
        
        try:
            # Run scraper (increased timeout for Playwright)
            # Playwright takes longer: ~15-30 seconds per game for JavaScript rendering
            timeout = 600  # 10 minute timeout (was 5 minutes)
            if self.max_reviews:
                # Estimate: ~0.5 seconds per review for loading + 30 seconds base
                estimated_time = 30 + (self.max_reviews * 0.5)
                timeout = max(600, int(estimated_time * 1.5))  # 1.5x buffer
            
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode == 0:
                # Read temporary JSON file and append to JSONL
                review_count = self._append_to_jsonl(temp_output)
                
                # Clean up temporary file
                try:
                    Path(temp_output).unlink()
                except:
                    pass
                
                print(f"✓ Successfully scraped: {game_name} ({platform})")
                print(f"✓ Appended {review_count} reviews to: {self.output_file}")
                self.results['success'].append({
                    'game': game_name,
                    'platform': platform,
                    'review_count': review_count,
                    'timestamp': datetime.now().isoformat()
                })
                self._mark_as_scraped(game_name, platform)
                self._save_progress()  # Save progress after each successful scrape
                return True
            else:
                print(f"✗ Failed to scrape: {game_name} ({platform})")
                print(f"Error output:\n{result.stderr[:500]}")
                self.results['failed'].append({
                    'game': game_name,
                    'platform': platform,
                    'error': result.stderr[:200],
                    'timestamp': datetime.now().isoformat()
                })
                self._save_progress()  # Save progress even on failure
                return False
                
        except subprocess.TimeoutExpired:
            print(f"✗ Timeout scraping: {game_name} ({platform})")
            self.results['failed'].append({
                'game': game_name,
                'platform': platform,
                'error': 'Timeout',
                'timestamp': datetime.now().isoformat()
            })
            self._save_progress()
            return False
            
        except Exception as e:
            print(f"✗ Error scraping {game_name} ({platform}): {e}")
            self.results['failed'].append({
                'game': game_name,
                'platform': platform,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            })
            self._save_progress()
            return False
    
    def scrape_all(self, games):
        """Scrape reviews for all games in the list"""
        
        if not games:
            print("No games to scrape!")
            return
        
        total_games = len(games)
        start_time = datetime.now()
        
        # Check how many already scraped
        already_scraped = sum(1 for g in games 
                             if self._is_already_scraped(
                                 g.get('game_name_slug', g.get('game_name', '')),
                                 g.get('platform_slug', g.get('platform', ''))
                             ))
        
        print("\n" + "=" * 70)
        print("STARTING BULK SCRAPING")
        print("=" * 70)
        print(f"Total games: {total_games}")
        print(f"Already scraped: {already_scraped}")
        print(f"Remaining: {total_games - already_scraped}")
        print(f"Max reviews per game: {self.max_reviews or 'unlimited'}")
        print(f"Delay between games: {self.delay}s")
        print(f"Skip existing: {'Yes' if self.skip_existing else 'No'}")
        print(f"Progress file: {self.progress_file}")
        print("=" * 70)
        
        for i, game in enumerate(games, 1):
            game_name = game.get('game_name_slug', game.get('game_name', ''))
            platform = game.get('platform_slug', game.get('platform', ''))
            
            if not game_name or not platform:
                print(f"\n⚠ Skipping game {i}: Missing game_name or platform")
                self.results['skipped'].append(game)
                continue
            
            # Scrape the game
            success = self.scrape_game(game_name, platform, i, total_games)
            
            # If error and not skipping errors, stop
            if not success and not self.skip_errors:
                print("\n⚠ Stopping due to error (--skip-errors not set)")
                break
            
            # Delay between games (except after last game)
            if i < total_games:
                print(f"\nWaiting {self.delay}s before next game...")
                time.sleep(self.delay)
        
        # Print summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Count total reviews
        total_reviews = sum(r.get('review_count', 0) for r in self.results['success'])
        
        print("\n" + "=" * 70)
        print("SCRAPING COMPLETE")
        print("=" * 70)
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Successful: {len(self.results['success'])} games")
        print(f"Failed: {len(self.results['failed'])} games")
        print(f"Skipped: {len(self.results['skipped'])} games")
        print(f"Total reviews: {total_reviews}")
        print(f"Output file: {self.output_file}")
        print("=" * 70)
        
        # Save results log
        self.save_results_log()
    
    def save_results_log(self):
        """Save scraping results to a log file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = f"data/scraping_log_{timestamp}.json"
        
        try:
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ Results log saved to: {log_file}")
            
        except Exception as e:
            print(f"\n✗ Error saving results log: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Scrape reviews for multiple games',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scrape all games to a single JSONL file (auto-skips already scraped)
  python scrape_all_games.py --input data/discovered_games.txt --skip-errors
  
  # Scrape with review limit per game to custom output file
  python scrape_all_games.py --input data/discovered_games.txt --output data/reviews.jsonl --max-reviews 100 --skip-errors
  
  # Force re-scrape everything (ignore progress)
  python scrape_all_games.py --input data/discovered_games.txt --force-rescrape --skip-errors
  
  # Custom delay between games
  python scrape_all_games.py --input data/discovered_games.txt --delay 10 --skip-errors

Features:
  - JSONL output: All reviews in one file, one JSON object per line
  - Incremental scraping: Each game appends immediately after scraping
  - Auto-resume: Skips games that were already scraped
  - Progress tracking: Saves progress to data/scraping_progress.json
  - Stream-friendly: JSONL format is easy to process line-by-line

Workflow:
  1. Run discover_games.py to find games
  2. Run this script to scrape all discovered games into JSONL
  3. Script automatically appends after each game
  4. If interrupted, just run again - it will skip completed games and continue appending
  5. Process JSONL directly (no need to combine files!)
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input file with game list (txt, json, or csv)')
    parser.add_argument('--output', '-o', type=str, default=None,
                      help='Output JSONL file (default: data/all_reviews_TIMESTAMP.jsonl)')
    parser.add_argument('--max-reviews', type=int, default=None,
                      help='Maximum reviews per game (default: unlimited)')
    parser.add_argument('--delay', type=float, default=3.0,
                      help='Delay between games in seconds (default: 3.0, Playwright handles rate limiting)')
    parser.add_argument('--skip-errors', action='store_true',
                      help='Continue scraping even if a game fails')
    parser.add_argument('--skip-existing', action='store_true', default=True,
                      help='Skip games that were already scraped (default: True)')
    parser.add_argument('--force-rescrape', action='store_true',
                      help='Force re-scrape all games (ignore progress file)')
    parser.add_argument('--start-from', type=int, default=1,
                      help='Start from game number N (for resuming)')
    
    args = parser.parse_args()
    
    # Create scraper
    scraper = GameScraper(
        max_reviews=args.max_reviews,
        delay=args.delay,
        skip_errors=args.skip_errors,
        skip_existing=not args.force_rescrape,  # If force rescrape, don't skip existing
        output_file=args.output if hasattr(args, 'output') else None
    )
    
    # Load games
    games = scraper.load_games(args.input)
    
    if not games:
        print("\n✗ No games loaded. Exiting.")
        sys.exit(1)
    
    # Start from specific game (for resuming)
    if args.start_from > 1:
        games = games[args.start_from - 1:]
        print(f"\nStarting from game #{args.start_from}")
    
    # Start scraping immediately
    print(f"\nReady to scrape {len(games)} games")
    print(f"Max reviews per game: {args.max_reviews or 'unlimited'}")
    print(f"Delay between games: {args.delay}s")
    print()
    
    # Start scraping
    try:
        scraper.scrape_all(games)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        scraper.save_results_log()
        sys.exit(1)


if __name__ == "__main__":
    main()
