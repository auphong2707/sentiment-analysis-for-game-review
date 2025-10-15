"""
Discover Games Script for Metacritic Scraper

This script browses Metacritic's game listings to discover all available games,
their platforms, and URLs. It saves this information to a file that can then be
used by the scraping script.

Usage:
    python discover_games.py --max-pages 10 --platform all
    python discover_games.py --platform playstation-5 --min-score 70
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import csv
import argparse
from datetime import datetime
from urllib.parse import urljoin, urlparse, parse_qs
from pathlib import Path
import random


class GameDiscoverer:
    """Discovers games from Metacritic browse pages"""
    
    def __init__(self, delay=3, max_pages=None, incremental=True):
        self.base_url = "https://www.metacritic.com"
        self.delay = delay
        self.max_pages = max_pages
        self.incremental = incremental
        self.discovered_games = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
        })
        self.output_file = None
        self.discovered_count = 0
    
    def _initialize_output_file(self, output_format='txt', filename=None):
        """Initialize the output file for incremental saving"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"discovered_games_{timestamp}.{output_format}"
        
        self.output_file = Path(f"data/{filename}")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # For JSON, start with an array
        if output_format == 'json':
            with open(self.output_file, 'w', encoding='utf-8') as f:
                f.write('[\n')
        
        # For CSV, write headers
        elif output_format == 'csv':
            with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['title', 'game_name_slug', 'platform', 'platform_slug', 
                               'url', 'metascore', 'release_date', 'discovered_at'])
        
        return self.output_file
    
    def _append_game_to_file(self, game_data, output_format='txt'):
        """Append a single game to the output file immediately"""
        if not self.output_file:
            return
        
        try:
            if output_format == 'txt':
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(f"{game_data['game_name_slug']}|{game_data['platform_slug']}\n")
            
            elif output_format == 'json':
                with open(self.output_file, 'r+', encoding='utf-8') as f:
                    # Move to end, overwrite the last closing bracket
                    f.seek(0, 2)  # Go to end
                    pos = f.tell()
                    
                    # Add comma if not first entry
                    if self.discovered_count > 0:
                        f.seek(pos - 2)  # Move back before \n
                        f.write(',\n')
                    
                    # Write the game data
                    json_line = json.dumps(game_data, indent=2, ensure_ascii=False)
                    # Indent the JSON properly
                    json_line = '\n'.join('  ' + line for line in json_line.split('\n'))
                    f.write(json_line + '\n')
            
            elif output_format == 'csv':
                with open(self.output_file, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        game_data.get('title', ''),
                        game_data.get('game_name_slug', ''),
                        game_data.get('platform', ''),
                        game_data.get('platform_slug', ''),
                        game_data.get('url', ''),
                        game_data.get('metascore', ''),
                        game_data.get('release_date', ''),
                        game_data.get('discovered_at', '')
                    ])
            
            self.discovered_count += 1
            
        except Exception as e:
            print(f"    ⚠ Error saving game to file: {e}")
    
    def _finalize_output_file(self, output_format='txt'):
        """Finalize the output file (close JSON array, etc.)"""
        if not self.output_file:
            return
        
        try:
            if output_format == 'json':
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(']\n')
        except Exception as e:
            print(f"⚠ Error finalizing output file: {e}")
    
    def discover_from_browse(self, platform=None, min_score=None, sort='score', output_format='txt', filename=None):
        """
        Discover games from Metacritic browse pages
        
        Args:
            platform: Platform filter (e.g., 'playstation-5', 'pc', 'all')
            min_score: Minimum metascore filter (0-100)
            sort: Sort order ('score', 'date', 'name')
            output_format: Output file format ('txt', 'json', 'csv')
            filename: Custom filename (optional)
        """
        # Initialize output file for incremental saving
        if self.incremental:
            output_file = self._initialize_output_file(output_format, filename)
            print(f"Incremental output: {output_file}")
        
        print("=" * 70)
        print("GAME DISCOVERY - Metacritic Browse")
        print("=" * 70)
        print(f"Platform: {platform or 'all'}")
        print(f"Min Score: {min_score or 'none'}")
        print(f"Max Pages: {self.max_pages or 'unlimited'}")
        print(f"Incremental save: {'Yes' if self.incremental else 'No'}")
        print("=" * 70)
        print()
        
        # Build browse URL
        # Example: https://www.metacritic.com/browse/game/
        browse_url = f"{self.base_url}/browse/game/"
        
        page = 1
        games_found = 0
        
        while True:
            if self.max_pages and self.max_pages > 0 and page > self.max_pages:
                print(f"\n✓ Reached max pages limit ({self.max_pages})")
                break
            
            # Metacritic uses pagination like: /browse/game/?page=1
            url = f"{browse_url}?page={page}"
            
            print(f"\n[Page {page}] Fetching: {url}")
            
            try:
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Find game cards - Metacritic uses various selectors
                # Try multiple selectors for different page layouts
                game_cards = (
                    soup.select('div.c-finderProductCard') or
                    soup.select('td.clamp-summary-wrap') or
                    soup.select('div.product_wrap') or
                    []
                )
                
                if not game_cards:
                    print(f"  ⚠ No game cards found on page {page}")
                    break
                
                print(f"  Found {len(game_cards)} game cards")
                
                page_games = 0
                for card in game_cards:
                    game_data = self._extract_game_from_card(card, soup)
                    
                    if game_data:
                        # Apply filters
                        if min_score and game_data.get('metascore'):
                            try:
                                if int(game_data['metascore']) < min_score:
                                    continue
                            except (ValueError, TypeError):
                                pass
                        
                        if platform and platform != 'all':
                            if game_data.get('platform_slug') != platform:
                                continue
                        
                        self.discovered_games.append(game_data)
                        page_games += 1
                        games_found += 1
                        
                        # Save incrementally if enabled
                        if self.incremental:
                            self._append_game_to_file(game_data, output_format)
                        
                        print(f"  [{games_found}] {game_data['title']} ({game_data['platform']}) - Score: {game_data.get('metascore', 'N/A')}")
                
                if page_games == 0 and page > 1:
                    # No games found and we're past page 1 - might be end or filter issue
                    print(f"  No games matched filters on page {page}")
                    # Don't break immediately, might just be empty page
                
                # For Metacritic, if we got no game cards at all, we've reached the end
                if len(game_cards) == 0:
                    print(f"\n✓ No more pages (last page: {page - 1})")
                    break
                
                page += 1
                
                # Respectful delay
                delay_time = self.delay + random.uniform(0, 1)
                print(f"  Waiting {delay_time:.1f}s before next request...")
                time.sleep(delay_time)
                
            except requests.RequestException as e:
                print(f"  ✗ Error fetching page {page}: {e}")
                break
            except Exception as e:
                print(f"  ✗ Error processing page {page}: {e}")
                break
        
        # Finalize output file if incremental
        if self.incremental:
            self._finalize_output_file(output_format)
        
        print("\n" + "=" * 70)
        print(f"DISCOVERY COMPLETE - Found {len(self.discovered_games)} games")
        if self.incremental and self.output_file:
            print(f"Saved to: {self.output_file}")
        print("=" * 70)
        
        return self.discovered_games
    
    def _extract_game_from_card(self, card, soup):
        """Extract game information from a card element"""
        try:
            # Try to extract title
            title_elem = (
                card.select_one('h3.c-finderProductCard_titleHeading') or
                card.select_one('a.title h3') or
                card.select_one('h3')
            )
            
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            
            # Try to extract URL
            link_elem = (
                card.select_one('a.c-finderProductCard_container') or
                card.select_one('a.title') or
                card.select_one('a[href*="/game/"]')
            )
            
            if not link_elem:
                return None
            
            game_url = urljoin(self.base_url, link_elem.get('href', ''))
            
            # Extract platform from URL
            # URL format: /game/{platform}/{game-name}
            url_parts = urlparse(game_url).path.split('/')
            platform_slug = url_parts[2] if len(url_parts) > 2 else 'unknown'
            game_name_slug = url_parts[3] if len(url_parts) > 3 else 'unknown'
            
            # Try to extract platform display name
            platform_elem = (
                card.select_one('div.c-finderProductCard_platform') or
                card.select_one('span.platform') or
                card.select_one('.c-globalProductCard_platform')
            )
            platform = platform_elem.get_text(strip=True) if platform_elem else platform_slug
            
            # Try to extract metascore
            score_elem = (
                card.select_one('div.c-siteReviewScore span') or
                card.select_one('div.c-finderProductCard_score span') or
                card.select_one('div.metascore_w') or
                card.select_one('.c-siteReviewScore')
            )
            metascore = score_elem.get_text(strip=True) if score_elem else None
            
            # Try to extract release date
            date_elem = (
                card.select_one('div.c-finderProductCard_meta span') or
                card.select_one('span.release_date') or
                card.select_one('.c-finderProductCard_releaseDate')
            )
            release_date = date_elem.get_text(strip=True) if date_elem else None
            
            return {
                'title': title,
                'game_name_slug': game_name_slug,
                'platform': platform,
                'platform_slug': platform_slug,
                'url': game_url,
                'metascore': metascore,
                'release_date': release_date,
                'discovered_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"    ⚠ Error extracting game data: {e}")
            return None
    
    def discover_from_search(self, search_term, max_results=50):
        """
        Discover games from search results
        
        Args:
            search_term: Search query
            max_results: Maximum number of results to collect
        """
        print("=" * 70)
        print(f"GAME DISCOVERY - Search for: {search_term}")
        print("=" * 70)
        
        # Metacritic search URL
        search_url = f"{self.base_url}/search/{search_term}/game"
        
        print(f"\nFetching: {search_url}")
        
        try:
            response = self.session.get(search_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find search results
            results = soup.select('div.search_results div.result')
            
            print(f"Found {len(results)} search results\n")
            
            for i, result in enumerate(results[:max_results], 1):
                game_data = self._extract_game_from_search_result(result)
                
                if game_data:
                    self.discovered_games.append(game_data)
                    print(f"[{i}] {game_data['title']} ({game_data['platform']}) - Score: {game_data.get('metascore', 'N/A')}")
            
        except requests.RequestException as e:
            print(f"✗ Error searching: {e}")
        
        print("\n" + "=" * 70)
        print(f"DISCOVERY COMPLETE - Found {len(self.discovered_games)} games")
        print("=" * 70)
        
        return self.discovered_games
    
    def _extract_game_from_search_result(self, result):
        """Extract game information from a search result"""
        try:
            # Extract title and URL
            title_elem = result.select_one('h3.product_title a')
            if not title_elem:
                return None
            
            title = title_elem.get_text(strip=True)
            game_url = urljoin(self.base_url, title_elem.get('href', ''))
            
            # Extract platform from URL
            url_parts = urlparse(game_url).path.split('/')
            platform_slug = url_parts[2] if len(url_parts) > 2 else 'unknown'
            game_name_slug = url_parts[3] if len(url_parts) > 3 else 'unknown'
            
            # Extract platform display name
            platform_elem = result.select_one('span.platform')
            platform = platform_elem.get_text(strip=True) if platform_elem else platform_slug
            
            # Extract metascore
            score_elem = result.select_one('div.metascore_w')
            metascore = score_elem.get_text(strip=True) if score_elem else None
            
            # Extract release date
            date_elem = result.select_one('span.release_date')
            release_date = date_elem.get_text(strip=True) if date_elem else None
            
            return {
                'title': title,
                'game_name_slug': game_name_slug,
                'platform': platform,
                'platform_slug': platform_slug,
                'url': game_url,
                'metascore': metascore,
                'release_date': release_date,
                'discovered_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"  ⚠ Error extracting search result: {e}")
            return None
    
    def save_to_file(self, output_format='json', filename=None):
        """Save discovered games to a file"""
        
        if not self.discovered_games:
            print("No games to save!")
            return
        
        # If incremental saving was used, file is already saved
        if self.incremental and self.output_file and self.output_file.exists():
            print(f"\n✓ Games already saved incrementally to: {self.output_file}")
            return self.output_file
        
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"discovered_games_{timestamp}.{output_format}"
        
        filepath = Path(f"data/{filename}")
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if output_format == 'json':
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(self.discovered_games, f, indent=2, ensure_ascii=False)
            
            elif output_format == 'csv':
                with open(filepath, 'w', newline='', encoding='utf-8') as f:
                    if self.discovered_games:
                        writer = csv.DictWriter(f, fieldnames=self.discovered_games[0].keys())
                        writer.writeheader()
                        writer.writerows(self.discovered_games)
            
            elif output_format == 'txt':
                # Simple text format for easy scraping
                with open(filepath, 'w', encoding='utf-8') as f:
                    for game in self.discovered_games:
                        f.write(f"{game['game_name_slug']}|{game['platform_slug']}\n")
            
            print(f"\n✓ Saved {len(self.discovered_games)} games to: {filepath}")
            
        except Exception as e:
            print(f"✗ Error saving file: {e}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Discover games from Metacritic',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Discover all games (first 10 pages)
  python discover_games.py --max-pages 10
  
  # Discover PS5 games with score >= 80
  python discover_games.py --platform playstation-5 --min-score 80 --max-pages 5
  
  # Search for specific game
  python discover_games.py --search "god of war"
  
  # Save as CSV
  python discover_games.py --max-pages 5 --format csv
        """
    )
    
    parser.add_argument('--platform', type=str, default='all',
                      help='Platform filter (e.g., playstation-5, pc, switch, all)')
    parser.add_argument('--min-score', type=int, default=None,
                      help='Minimum metascore (0-100)')
    parser.add_argument('--max-pages', type=int, default=5,
                      help='Maximum number of pages to browse (default: 5, use 0 for unlimited)')
    parser.add_argument('--delay', type=float, default=3.0,
                      help='Delay between requests in seconds (default: 3.0)')
    parser.add_argument('--format', type=str, default='json',
                      choices=['json', 'csv', 'txt'],
                      help='Output format (default: json)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output filename (default: auto-generated)')
    parser.add_argument('--search', type=str, default=None,
                      help='Search for specific game instead of browsing')
    
    args = parser.parse_args()
    
    # Create discoverer with incremental saving enabled by default
    discoverer = GameDiscoverer(delay=args.delay, max_pages=args.max_pages, incremental=True)
    
    # Discover games
    if args.search:
        discoverer.discover_from_search(args.search)
    else:
        discoverer.discover_from_browse(
            platform=args.platform if args.platform != 'all' else None,
            min_score=args.min_score,
            output_format=args.format,
            filename=args.output
        )
    
    # Save results (will skip if already saved incrementally)
    if discoverer.discovered_games:
        discoverer.save_to_file(output_format=args.format, filename=args.output)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total games discovered: {len(discoverer.discovered_games)}")
        
        # Show platform breakdown
        platforms = {}
        for game in discoverer.discovered_games:
            platform = game['platform']
            platforms[platform] = platforms.get(platform, 0) + 1
        
        print("\nGames by platform:")
        for platform, count in sorted(platforms.items(), key=lambda x: x[1], reverse=True):
            print(f"  {platform}: {count}")
        
        print("\n" + "=" * 70)
    else:
        print("\n⚠ No games discovered!")


if __name__ == "__main__":
    main()
