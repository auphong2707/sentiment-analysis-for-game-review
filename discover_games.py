"""
Discover Games Script for Metacritic Scraper

This script browses Metacritic game listings and saves titles to a text file.

Usage:
    python discover_games.py --max-pages 10
"""

import requests
from bs4 import BeautifulSoup
import time
import argparse
from datetime import datetime
from urllib.parse import urljoin
from pathlib import Path
import random


class GameDiscoverer:
    """Discovers games from Metacritic browse pages"""
    
    def __init__(self, delay=3, max_pages=None):
        self.base_url = "https://www.metacritic.com"
        self.delay = delay
        self.max_pages = max_pages
        self.discovered_games = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive',
        })
    
    def discover_from_browse(self):
        """Discover game titles from Metacritic browse pages"""
        
        print("=" * 70)
        print("GAME DISCOVERY - Metacritic Browse")
        print("=" * 70)
        print(f"Max Pages: {self.max_pages or 'unlimited'}")
        print("=" * 70)
        print()
        
        browse_url = f"{self.base_url}/browse/game/"
        page = 1
        games_found = 0
        
        while True:
            if self.max_pages and page > self.max_pages:
                print(f"\n Reached max pages limit ({self.max_pages})")
                break
            
            url = f"{browse_url}?page={page}"
            print(f"\n[Page {page}] Fetching: {url}")
            
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find game cards
                game_cards = (
                    soup.select('div.c-finderProductCard') or
                    soup.select('td.clamp-summary-wrap') or
                    soup.select('div.product_wrap') or
                    []
                )
                
                if not game_cards:
                    print(f"   No game cards found on page {page}")
                    break
                
                print(f"  Found {len(game_cards)} game cards")
                
                for card in game_cards:
                    title = self._extract_title_from_card(card)
                    
                    if title:
                        self.discovered_games.append(title)
                        games_found += 1
                        print(f"  [{games_found}] {title}")
                
                page += 1
                
                # Respectful delay
                delay_time = self.delay + random.uniform(0, 1)
                print(f"  Waiting {delay_time:.1f}s before next request...")
                time.sleep(delay_time)
                
            except requests.RequestException as e:
                print(f"   Error fetching page {page}: {e}")
                break
            except Exception as e:
                print(f"   Error processing page {page}: {e}")
                break
        
        print("\n" + "=" * 70)
        print(f"DISCOVERY COMPLETE - Found {len(self.discovered_games)} games")
        print("=" * 70)
        
        return self.discovered_games
    
    def _extract_title_from_card(self, card):
        """Extract game title from a card element"""
        try:
            title_elem = (
                card.select_one('h3.c-finderProductCard_titleHeading') or
                card.select_one('a.title h3') or
                card.select_one('h3')
            )
            
            if title_elem:
                return title_elem.get_text(strip=True)
            
            return None
            
        except Exception as e:
            print(f"     Error extracting title: {e}")
            return None
    
    def save_to_file(self, filename=None, split_size=None):
        """Save discovered game titles to a text file (or multiple files if split_size is set)"""
        
        if not self.discovered_games:
            print("No games to save!")
            return
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"discovered_games_{timestamp}.txt"
        
        # If split_size is set, split into multiple files
        if split_size and split_size > 0:
            return self._save_split_files(filename, split_size)
        
        # Otherwise save to a single file
        filepath = Path(f"data/{filename}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                for game in self.discovered_games:
                    f.write(f"{game}\n")
            
            print(f"\n✓ Saved {len(self.discovered_games)} games to: {filepath}")
            return filepath
            
        except Exception as e:
            print(f"✗ Error saving file: {e}")
    
    def _save_split_files(self, base_filename, split_size):
        """Save games to multiple files with specified number of games per file"""
        
        # Create base path and ensure directory exists
        base_path = Path(f"data/{base_filename}")
        base_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get filename components
        stem = base_path.stem  # filename without extension
        suffix = base_path.suffix or '.txt'  # extension
        
        total_games = len(self.discovered_games)
        num_files = (total_games + split_size - 1) // split_size  # ceiling division
        
        print(f"\n📂 Splitting {total_games} games into {num_files} files ({split_size} games per file)")
        
        saved_files = []
        
        try:
            for file_num in range(num_files):
                start_idx = file_num * split_size
                end_idx = min(start_idx + split_size, total_games)
                
                # Create filename with number: discovered_games_20231016_part1.txt
                split_filename = f"{stem}_part{file_num + 1}{suffix}"
                filepath = base_path.parent / split_filename
                
                # Write chunk to file
                with open(filepath, 'w', encoding='utf-8') as f:
                    for game in self.discovered_games[start_idx:end_idx]:
                        f.write(f"{game}\n")
                
                saved_files.append(filepath)
                games_in_file = end_idx - start_idx
                print(f"  ✓ Saved {games_in_file} games to: {filepath}")
            
            print(f"\n✓ Successfully saved to {num_files} files")
            return saved_files
            
        except Exception as e:
            print(f"✗ Error saving split files: {e}")
            return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Discover games from Metacritic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover all games (unlimited pages)
  python discover_games.py
  
  # Discover with max pages limit
  python discover_games.py --max-pages 10
  
  # Discover with custom output filename
  python discover_games.py --max-pages 5 --output my_games.txt
  
  # Split output into multiple files (100 games per file)
  python discover_games.py --split-size 100
  
  # Combine options
  python discover_games.py --max-pages 20 --split-size 50 --output games.txt
        """
    )
    
    parser.add_argument('--max-pages', type=int, default=None,
                      help='Maximum number of pages to browse (default: unlimited)')
    parser.add_argument('--delay', type=float, default=3.0,
                      help='Delay between requests in seconds (default: 3.0)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output filename (default: auto-generated)')
    parser.add_argument('--split-size', type=int, default=None,
                      help='Split output into multiple files with N games per file (default: single file)')
    
    args = parser.parse_args()
    
    # Create discoverer
    discoverer = GameDiscoverer(delay=args.delay, max_pages=args.max_pages)
    
    # Discover games
    discoverer.discover_from_browse()
    
    # Save results
    if discoverer.discovered_games:
        discoverer.save_to_file(filename=args.output, split_size=args.split_size)
    else:
        print("\n⚠ No games discovered!")


if __name__ == "__main__":
    main()
