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
                writer.writerow(['title', 'game_name_slug', 'platform_slug', 'url', 'metascore', 'release_date', 'discovered_at'])
        
        return self.output_file
    
    def _append_game_to_file(self, game_data, output_format='txt'):
        """Append a single game to the output file immediately"""
        if not self.output_file:
            return
        
        try:
            if output_format == 'txt':
                # Format: game_name_slug|platform_slug (for scraper input)
                # If platform_slug exists, use it; otherwise just game name
                platform_slug = game_data.get('platform_slug', '')
                game_name_slug = game_data['game_name_slug']
                
                if platform_slug:
                    line = f"{game_name_slug}|{platform_slug}\n"
                else:
                    line = f"{game_name_slug}\n"
                
                with open(self.output_file, 'a', encoding='utf-8') as f:
                    f.write(line)
            
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
    
    def discover_from_browse(self, platform=None, min_score=None, sort='score', output_format='txt', filename=None, fetch_all_platforms=False):
        """
        Discover games from Metacritic browse pages
        
        Args:
            platform: Platform filter (e.g., 'playstation-5', 'pc', 'all')
            min_score: Minimum metascore filter (0-100)
            sort: Sort order ('score', 'date', 'name')
            output_format: Output file format ('txt', 'json', 'csv')
            filename: Custom filename (optional)
            fetch_all_platforms: If True, fetch all available platforms for each game and create separate entries
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
        print(f"Fetch all platforms: {'Yes' if fetch_all_platforms else 'No'}")
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
                    game_data = self._extract_game_from_card(card, soup, fetch_all_platforms=fetch_all_platforms)
                    
                    if game_data:
                        # Apply filters
                        if min_score and game_data.get('metascore'):
                            try:
                                if int(game_data['metascore']) < min_score:
                                    continue
                            except (ValueError, TypeError):
                                pass
                        
                        # If we fetched all platforms, create separate entries for each platform
                        if fetch_all_platforms and game_data.get('platforms'):
                            for platform_slug in game_data['platforms']:
                                platform_entry = game_data.copy()
                                platform_entry['platform_slug'] = platform_slug
                                platform_entry['game_name_slug'] = game_data['game_name_slug']
                                
                                # Apply platform filter if specified
                                if platform and platform != 'all':
                                    if platform_slug != platform:
                                        continue
                                
                                self.discovered_games.append(platform_entry)
                                page_games += 1
                                games_found += 1
                                
                                # Save incrementally if enabled
                                if self.incremental:
                                    self._append_game_to_file(platform_entry, output_format)
                                
                                print(f"  [{games_found}] {platform_entry['title']} ({platform_slug}) - Score: {platform_entry.get('metascore', 'N/A')}")
                        else:
                            # Original behavior - single entry per game
                            if platform and platform != 'all':
                                if game_data.get('platform_slug') != platform:
                                    continue
                            
                            self.discovered_games.append(game_data)
                            page_games += 1
                            games_found += 1
                            
                            # Save incrementally if enabled
                            if self.incremental:
                                self._append_game_to_file(game_data, output_format)
                            
                            print(f"  [{games_found}] {game_data['title']} - Score: {game_data.get('metascore', 'N/A')}")
                
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
    
    def _get_all_platforms_for_game(self, game_name_slug, game_url):
        """
        Fetch all available platforms for a game by visiting the game page
        
        Args:
            game_name_slug: The game slug (e.g., 'grand-theft-auto-iv')
            game_url: The game's URL
            
        Returns:
            List of platform slugs (e.g., ['pc', 'playstation-4', 'xbox-360'])
        """
        try:
            print(f"    🔍 Fetching platforms for: {game_name_slug}")
            
            response = self.session.get(game_url, timeout=30)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            platforms = []
            
            # Method 1: Look for platform switcher/selector on the page
            # Metacritic shows platform tabs/buttons like: "PC", "PS4", "Xbox One", etc.
            platform_selectors = (
                soup.select('li.c-gamePlatformTabList_item a') or
                soup.select('ul.c-gamePlatformTabList a') or
                soup.select('.c-gamePlatformTab a') or
                soup.select('li.product_platform a') or
                []
            )
            
            if platform_selectors:
                for link in platform_selectors:
                    href = link.get('href', '')
                    # Extract platform from URL: /game/{platform}/{game-name} or /game/{game-name}/{platform}
                    url_parts = [p for p in href.split('/') if p]
                    if len(url_parts) >= 3 and url_parts[0] == 'game':
                        platform_slug = url_parts[1]
                        # Sometimes game name comes first, platform second
                        if platform_slug == game_name_slug and len(url_parts) > 2:
                            platform_slug = url_parts[2]
                        
                        if platform_slug != game_name_slug and platform_slug not in platforms:
                            platforms.append(platform_slug)
                            print(f"      ✓ Found platform: {platform_slug}")
            
            # Method 2: Look for platform badges/spans in the title area
            if not platforms:
                platform_badges = (
                    soup.select('.c-productHero_platforms span') or
                    soup.select('.c-ProductHeroGamePlatformInfo span') or
                    []
                )
                
                platform_map = {
                    'pc': 'pc',
                    'playstation 5': 'playstation-5',
                    'ps5': 'playstation-5',
                    'playstation 4': 'playstation-4', 
                    'ps4': 'playstation-4',
                    'playstation 3': 'playstation-3',
                    'ps3': 'playstation-3',
                    'xbox series x': 'xbox-series-x',
                    'xbox one': 'xbox-one',
                    'xbox 360': 'xbox-360',
                    'switch': 'switch',
                    'nintendo switch': 'switch',
                    'wii u': 'wii-u',
                    'wii': 'wii',
                    'gamecube': 'gamecube',
                    'n64': 'nintendo-64',
                }
                
                for badge in platform_badges:
                    platform_text = badge.get_text(strip=True).lower()
                    platform_slug = platform_map.get(platform_text)
                    if platform_slug and platform_slug not in platforms:
                        platforms.append(platform_slug)
                        print(f"      ✓ Found platform: {platform_slug}")
            
            # Method 3: Parse the main URL if it contains platform info
            if not platforms:
                url_parts = [p for p in urlparse(game_url).path.split('/') if p]
                # Check if URL has format: /game/{platform}/{game-name}
                if len(url_parts) >= 3 and url_parts[0] == 'game':
                    potential_platform = url_parts[1]
                    # Verify it's not the game name
                    if potential_platform != game_name_slug:
                        platforms.append(potential_platform)
                        print(f"      ✓ Found platform from URL: {potential_platform}")
            
            if not platforms:
                print(f"      ⚠ No platforms found for {game_name_slug}")
                return []
            
            print(f"      📊 Total platforms: {len(platforms)}")
            return platforms
            
        except Exception as e:
            print(f"      ⚠ Error fetching platforms: {e}")
            return []
    
    def _extract_game_from_card(self, card, soup, fetch_all_platforms=False):
        """
        Extract game information from a card element
        
        Args:
            card: BeautifulSoup element containing game card
            soup: Full page soup (not used but kept for compatibility)
            fetch_all_platforms: If True, fetch all available platforms for each game
        """
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
            
            # Extract game name from URL
            # URL format: /game/{game-name}/ (new Metacritic format - no platform in browse URL)
            url_parts = [p for p in urlparse(game_url).path.split('/') if p]  # Filter out empty strings
            
            # Expected: ['game', 'game-name']
            if len(url_parts) >= 2 and url_parts[0] == 'game':
                game_name_slug = url_parts[1]
            else:
                print(f"    ⚠ Cannot parse URL: {game_url}")
                return None
            
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
            
            game_data = {
                'title': title,
                'game_name_slug': game_name_slug,
                'url': game_url,
                'metascore': metascore,
                'release_date': release_date,
                'discovered_at': datetime.now().isoformat()
            }
            
            # Fetch all platforms if requested
            if fetch_all_platforms:
                platforms = self._get_all_platforms_for_game(game_name_slug, game_url)
                game_data['platforms'] = platforms
                # Add delay after fetching platforms to be respectful
                time.sleep(1)
            
            return game_data
            
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
                    print(f"[{i}] {game_data['title']} - Score: {game_data.get('metascore', 'N/A')}")
            
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
            
            # Extract game name from URL
            url_parts = [p for p in urlparse(game_url).path.split('/') if p]
            game_name_slug = url_parts[2] if len(url_parts) > 2 else url_parts[1] if len(url_parts) > 1 else 'unknown'
            
            # Extract metascore
            score_elem = result.select_one('div.metascore_w')
            metascore = score_elem.get_text(strip=True) if score_elem else None
            
            # Extract release date
            date_elem = result.select_one('span.release_date')
            release_date = date_elem.get_text(strip=True) if date_elem else None
            
            return {
                'title': title,
                'game_name_slug': game_name_slug,
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
                # Simple text format: one game per line
                with open(filepath, 'w', encoding='utf-8') as f:
                    for game in self.discovered_games:
                        f.write(f"{game['game_name_slug']}\n")
            
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
  
  # Discover all games with ALL platforms (creates separate entry per platform)
  python discover_games.py --max-pages 5 --fetch-all-platforms
  
  # Discover PS5 games with score >= 80
  python discover_games.py --platform playstation-5 --min-score 80 --max-pages 5
  
  # Discover and fetch all platforms for each game (for comprehensive scraping)
  python discover_games.py --max-pages 3 --fetch-all-platforms --format txt
  
  # Search for specific game
  python discover_games.py --search "god of war"
  
  # Save as CSV with all platforms
  python discover_games.py --max-pages 5 --format csv --fetch-all-platforms
        """
    )
    
    parser.add_argument('--platform', type=str, default='all',
                      help='Platform filter (e.g., playstation-5, pc, switch, all)')
    parser.add_argument('--min-score', type=int, default=None,
                      help='Minimum metascore (0-100)')
    parser.add_argument('--max-pages', type=int, default=0,
                      help='Maximum number of pages to browse (default: 0, use 0 for unlimited)')
    parser.add_argument('--delay', type=float, default=3.0,
                      help='Delay between requests in seconds (default: 3.0)')
    parser.add_argument('--format', type=str, default='txt',
                      choices=['json', 'csv', 'txt'],
                      help='Output format (default: txt)')
    parser.add_argument('--output', type=str, default=None,
                      help='Output filename (default: auto-generated)')
    parser.add_argument('--search', type=str, default=None,
                      help='Search for specific game instead of browsing')
    parser.add_argument('--fetch-all-platforms', action='store_true',
                      help='Fetch all available platforms for each game (creates separate entries per platform)')
    
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
            filename=args.output,
            fetch_all_platforms=args.fetch_all_platforms
        )
    
    # Save results (will skip if already saved incrementally)
    if discoverer.discovered_games:
        discoverer.save_to_file(output_format=args.format, filename=args.output)
        
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Total games discovered: {len(discoverer.discovered_games)}")
        
        # Show score distribution
        score_ranges = {'90+': 0, '80-89': 0, '70-79': 0, '60-69': 0, '<60': 0, 'N/A': 0}
        for game in discoverer.discovered_games:
            score = game.get('metascore')
            if score:
                try:
                    score_val = int(score)
                    if score_val >= 90:
                        score_ranges['90+'] += 1
                    elif score_val >= 80:
                        score_ranges['80-89'] += 1
                    elif score_val >= 70:
                        score_ranges['70-79'] += 1
                    elif score_val >= 60:
                        score_ranges['60-69'] += 1
                    else:
                        score_ranges['<60'] += 1
                except (ValueError, TypeError):
                    score_ranges['N/A'] += 1
            else:
                score_ranges['N/A'] += 1
        
        print("\nScore distribution:")
        for range_name, count in score_ranges.items():
            if count > 0:
                print(f"  {range_name}: {count}")
        
        print("\n" + "=" * 70)
    else:
        print("\n⚠ No games discovered!")


if __name__ == "__main__":
    main()
