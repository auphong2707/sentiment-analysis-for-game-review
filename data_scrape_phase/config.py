"""
Configuration file for the Metacritic scraper
"""
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Scraper Configuration
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36'

# Request delays (in seconds) - be respectful to the server
DOWNLOAD_DELAY = 2  # Delay between requests
RANDOMIZE_DOWNLOAD_DELAY = True

# Retry settings
RETRY_TIMES = 3
RETRY_HTTP_CODES = [500, 502, 503, 504, 408, 429]

# Output settings
OUTPUT_DIR = 'data'
OUTPUT_FORMAT = 'json'  # Options: 'csv', 'json', 'excel' - JSON is better for multi-line text

# Metacritic specific settings
METACRITIC_BASE_URL = 'https://www.metacritic.com'

# Score categories (Metacritic's classification)
SCORE_CATEGORIES = {
    'positive': (75, 100),  # Green
    'mixed': (50, 74),      # Yellow
    'negative': (0, 49)     # Red
}

# Scraping limits (set to None for unlimited)
MAX_REVIEWS_PER_GAME = None  # Maximum reviews to scrape per game
MAX_GAMES = None  # Maximum games to scrape

# Headers
DEFAULT_HEADERS = {
    'User-Agent': USER_AGENT,
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}
