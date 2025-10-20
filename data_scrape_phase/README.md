# Data Scraping Phase

This folder contains all scripts and tools for scraping game reviews from Metacritic.

## Contents

### Main Scripts
- `discover_games.py` - Discovers games from Metacritic browse pages
- `scrape_all_games.py` - Scrapes reviews for multiple games from a discovered list
- `run_scraper.py` - Interactive script to start scraping individual games
- `config.py` - Configuration settings for the scraper

### Scrapy Project
- `metacritic_scraper/` - Scrapy spider project for scraping reviews
  - `spiders/metacritic_reviews.py` - Main spider for review scraping
  - `items.py` - Data models
  - `pipelines.py` - Data processing pipelines
  - `settings.py` - Scrapy settings
  - `middlewares.py` - Custom middlewares
- `scrapy.cfg` - Scrapy project configuration

### Installation Scripts
- `install_playwright.ps1` - PowerShell script to install Playwright (Windows)
- `install_playwright.sh` - Bash script to install Playwright (Linux/Mac)

## Workflow

### 1. Discover Games

First, discover games from Metacritic:

```powershell
python data_scrape_phase/discover_games.py --max-pages 10 --platform all
```

**Options:**
- `--max-pages N` - Maximum pages to browse (default: all)
- `--platform PLATFORM` - Platform filter (e.g., playstation-5, xbox-series-x, pc, all)
- `--min-score N` - Minimum Metacritic score filter
- `--format FORMAT` - Output format: txt, json, csv (default: txt)

**Output:** `data/discovered_games/discovered_games_YYYYMMDD_HHMMSS.txt`

### 2. Scrape Reviews

Scrape reviews for all discovered games:

```powershell
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/discovered_games_XXXXXX.txt
```

**Options:**
- `--input FILE` - Input file with discovered games
- `--max-reviews N` - Maximum reviews per game (default: all)
- `--delay N` - Delay between games in seconds (default: 5)
- `--skip-existing` - Skip games that already have data
- `--output FILE` - Custom output file pattern

**Output:** `data/review_data/review_data_partN.jsonl`

### 3. Or Run Interactive Scraper

For single game scraping:

```powershell
python data_scrape_phase/run_scraper.py
```

Follow the prompts to enter game details.

## Configuration

Edit `config.py` to adjust scraper settings:

```python
# Request delays (be respectful!)
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True

# Retry settings
RETRY_TIMES = 3

# Output settings
OUTPUT_DIR = 'data'
OUTPUT_FORMAT = 'json'

# Scraping limits
MAX_REVIEWS_PER_GAME = None  # None = unlimited
```

## Path Handling

All scripts work from the project root. Run them as:

```powershell
# From project root
python data_scrape_phase/discover_games.py
python data_scrape_phase/scrape_all_games.py
python data_scrape_phase/run_scraper.py
```

Or from within the folder:

```powershell
cd data_scrape_phase
python discover_games.py
python scrape_all_games.py
python run_scraper.py
```

## Requirements

Make sure these packages are installed (in `requirements.txt`):

```
scrapy>=2.11.0
scrapy-playwright>=0.0.34
selenium>=4.15.0
webdriver-manager>=4.0.1
beautifulsoup4>=4.12.0
requests>=2.31.0
```

Install with:
```powershell
pip install -r requirements.txt
```

### Install Playwright

After installing scrapy-playwright:

**Windows:**
```powershell
.\data_scrape_phase\install_playwright.ps1
```

**Linux/Mac:**
```bash
bash data_scrape_phase/install_playwright.sh
```

## Output Data Structure

### Discovered Games Format
```json
{
  "game_name": "the-last-of-us-part-ii",
  "platform": "playstation-4",
  "url": "/game/playstation-4/the-last-of-us-part-ii/",
  "title": "The Last of Us Part II",
  "score": 93
}
```

### Review Data Format
```json
{
  "game_name": "the-last-of-us-part-ii",
  "platform": "playstation-4",
  "review_text": "Amazing storytelling and gameplay...",
  "review_score": 100,
  "review_category": "positive",
  "reviewer": "John_Doe",
  "review_date": "Jun 19, 2020",
  "url": "https://www.metacritic.com/..."
}
```

## Tips

1. **Be Respectful**: Use delays between requests (configured in config.py)
2. **Incremental Discovery**: Use `--max-pages` to test with small batches first
3. **Resume Scraping**: Use `--skip-existing` to resume interrupted scraping
4. **Monitor Progress**: Scripts show progress bars and status updates
5. **Check Output**: Review data goes to `data/review_data/` folder

## Troubleshooting

**"Module not found" errors:**
```powershell
pip install -r requirements.txt
python data_scrape_phase/install_playwright.ps1
```

**Scrapy not finding the spider:**
- Make sure you're running from project root
- Or update `scrapy.cfg` if running from data_scrape_phase

**Rate limiting / blocks:**
- Increase `DOWNLOAD_DELAY` in config.py
- Add more randomization to delays
- Use proxy middleware (advanced)

## Next Steps

After scraping, use the data preparation scripts:

```powershell
# Aggregate and filter English reviews
python dataset_prepare_phase/aggregate_dataset.py

# Analyze the data
python dataset_prepare_phase/analyze_reviews.py data/aggregated_review_english/aggregated_reviews_english.jsonl

# Upload to HuggingFace
python dataset_prepare_phase/prepare_and_upload_hf_dataset.py
```

## File Organization

This folder is part of the complete workflow:

```
sentiment-analysis-for-game-review/
├── data_scrape_phase/          # ← You are here
│   ├── discover_games.py
│   ├── scrape_all_games.py
│   ├── run_scraper.py
│   └── metacritic_scraper/
├── dataset_prepare_phase/      # Next: Data preparation
│   ├── aggregate_dataset.py
│   └── prepare_and_upload_hf_dataset.py
└── data/                       # Output data
    ├── discovered_games/
    ├── review_data/
    └── aggregated_review_english/
```
