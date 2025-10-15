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
    
    def __init__(self, max_reviews=None, delay=5, skip_errors=True, skip_existing=True):
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
                    
                    parts = line.split('|')
                    if len(parts) >= 2:
                        games.append({
                            'game_name_slug': parts[0].strip(),
                            'platform_slug': parts[1].strip()
                        })
                    else:
                        print(f"⚠ Skipping invalid line {line_num}: {line}")
            
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
        
        # Build scrapy command with custom output file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"data/{game_name}_{platform}_{timestamp}.json"
        
        command = [
            "scrapy", "crawl", "metacritic_reviews",
            "-a", f"game_name={game_name}",
            "-a", f"platform={platform}",
            "-o", output_file  # Output to specific file immediately
        ]
        
        if self.max_reviews:
            command.extend(["-a", f"max_reviews={self.max_reviews}"])
        
        print(f"Command: {' '.join(command)}")
        print(f"Output: {output_file}")
        print()
        
        try:
            # Run scraper
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print(f"✓ Successfully scraped: {game_name} ({platform})")
                print(f"✓ Saved to: {output_file}")
                self.results['success'].append({
                    'game': game_name,
                    'platform': platform,
                    'output_file': output_file,
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
        
        print("\n" + "=" * 70)
        print("SCRAPING COMPLETE")
        print("=" * 70)
        print(f"Duration: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        print(f"Successful: {len(self.results['success'])} games")
        print(f"Failed: {len(self.results['failed'])} games")
        print(f"Skipped: {len(self.results['skipped'])} games")
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
  # Scrape all games from discovered list (auto-skips already scraped)
  python scrape_all_games.py --input data/discovered_games.txt --skip-errors
  
  # Scrape with review limit per game
  python scrape_all_games.py --input data/discovered_games.json --max-reviews 100 --skip-errors
  
  # Force re-scrape everything (ignore progress)
  python scrape_all_games.py --input data/discovered_games.txt --force-rescrape --skip-errors
  
  # Custom delay between games
  python scrape_all_games.py --input data/discovered_games.txt --delay 10 --skip-errors

Features:
  - Incremental scraping: Each game saves immediately after scraping
  - Auto-resume: Skips games that were already scraped
  - Progress tracking: Saves progress to data/scraping_progress.json
  - Individual outputs: Each game gets its own JSON file

Workflow:
  1. Run discover_games.py to find games
  2. Run this script to scrape all discovered games
  3. Script automatically saves after each game
  4. If interrupted, just run again - it will skip completed games
  5. Use combine_data.py to merge all files when done
        """
    )
    
    parser.add_argument('--input', '-i', type=str, required=True,
                      help='Input file with game list (txt, json, or csv)')
    parser.add_argument('--max-reviews', type=int, default=None,
                      help='Maximum reviews per game (default: unlimited)')
    parser.add_argument('--delay', type=float, default=5.0,
                      help='Delay between games in seconds (default: 5.0)')
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
        skip_existing=not args.force_rescrape  # If force rescrape, don't skip existing
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
    
    # Confirm before starting
    print(f"\nReady to scrape {len(games)} games")
    print(f"Max reviews per game: {args.max_reviews or 'unlimited'}")
    print(f"Delay between games: {args.delay}s")
    
    response = input("\nProceed? (y/n): ").strip().lower()
    
    if response != 'y':
        print("Cancelled.")
        sys.exit(0)
    
    # Start scraping
    try:
        scraper.scrape_all(games)
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        scraper.save_results_log()
        sys.exit(1)


if __name__ == "__main__":
    main()
