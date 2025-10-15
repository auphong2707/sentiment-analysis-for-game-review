# Sentiment Analysis for Game Reviews

A comprehensive project for scraping video game reviews from Metacritic and performing sentiment analysis on the collected data. This project uses Scrapy for web scraping and includes tools for collecting labeled review data (positive, mixed, negative) for machine learning analysis.

## Features

- 🕷️ **Web Scraping**: Robust Scrapy-based scraper for Metacritic game reviews
- 📊 **Labeled Data**: Automatically categorizes reviews as Positive (75-100), Mixed (50-74), or Negative (0-49)
- 🎮 **Game Metadata**: Collects comprehensive game information (title, platform, metascore, etc.)
- 💾 **Multiple Export Formats**: Supports CSV, JSON, and Excel output
- ⚙️ **Configurable**: Easy-to-configure settings for rate limiting, retry logic, and more
- 🔄 **Duplicate Filtering**: Prevents duplicate reviews in your dataset
- 📝 **Detailed Logging**: Comprehensive logging for debugging and monitoring

## Project Structure

```
sentiment-analysis-for-game-review/
├── metacritic_scraper/          # Scrapy project directory
│   ├── spiders/                 # Spider implementations
│   │   └── metacritic_reviews.py  # Main review scraper spider
│   ├── items.py                 # Data models for scraped items
│   ├── middlewares.py           # Custom middlewares
│   ├── pipelines.py             # Data processing pipelines
│   └── settings.py              # Scrapy settings
├── data/                        # Output directory for scraped data
├── config.py                    # Configuration file
├── run_scraper.py               # Interactive scraper runner
├── requirements.txt             # Python dependencies
├── scrapy.cfg                   # Scrapy project configuration
├── .env.example                 # Environment variables template
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- **Stable internet connection**

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/auphong2707/sentiment-analysis-for-game-review.git
   cd sentiment-analysis-for-game-review
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   # On Windows
   .\venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Test connectivity** (recommended)
   ```bash
   python test_connection.py
   ```
   This will verify you can connect to Metacritic before scraping.

5. **Configure environment** (optional)
   ```bash
   copy .env.example .env
   # Edit .env with your preferred settings
   ```

## Usage

### Method 1: Interactive Mode (Recommended for Beginners)

The easiest way to start scraping:

```bash
python run_scraper.py
```

This will guide you through:
- Entering the game name
- Selecting the platform
- Setting the maximum number of reviews (optional)

### Method 2: Command Line

Scrape reviews for a specific game by providing the game name and platform:

```bash
scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4"
```

### Using Full URL

```bash
scrapy crawl metacritic_reviews -a game_url="https://www.metacritic.com/game/playstation-4/the-last-of-us-part-ii"
```

### Limit Number of Reviews

```bash
scrapy crawl metacritic_reviews -a game_name="elden-ring" -a platform="playstation-5" -a max_reviews=100
```

### Custom Output Format

```bash
# Export to JSON
scrapy crawl metacritic_reviews -a game_name="hades" -a platform="switch" -o data/hades_reviews.json

# Export to CSV
scrapy crawl metacritic_reviews -a game_name="hades" -a platform="switch" -o data/hades_reviews.csv
```

### Platform Values

Common platform values for Metacritic URLs:

**PlayStation:**
- `playstation-5` (PS5)
- `playstation-4` (PS4)
- `playstation-3` (PS3)

**Xbox:**
- `xbox-series-x` (Xbox Series X/S)
- `xbox-one` (Xbox One)
- `xbox-360` (Xbox 360)

**Nintendo:**
- `switch` (Nintendo Switch)
- `wii-u` (Wii U)
- `3ds` (Nintendo 3DS)

**PC:**
- `pc` (PC)

### Method 3: Batch Processing with Python Script

Create a custom script to scrape multiple games:

```python
import subprocess

games = [
    ("the-last-of-us-part-ii", "playstation-4"),
    ("god-of-war", "playstation-4"),
    ("elden-ring", "playstation-5"),
]

for game_name, platform in games:
    subprocess.run([
        "scrapy", "crawl", "metacritic_reviews",
        "-a", f"game_name={game_name}",
        "-a", f"platform={platform}",
        "-a", "max_reviews=50"
    ])
```

### Popular Games to Try

1. **The Last of Us Part II** - `the-last-of-us-part-ii`, `playstation-4`
2. **Elden Ring** - `elden-ring`, `playstation-5`
3. **Baldur's Gate III** - `baldurs-gate-iii`, `pc`
4. **The Legend of Zelda: Breath of the Wild** - `the-legend-of-zelda-breath-of-the-wild`, `switch`
5. **God of War** - `god-of-war`, `playstation-4`
6. **Hades** - `hades`, `switch`
7. **Cyberpunk 2077** - `cyberpunk-2077`, `pc`

## Data Format

The scraper collects the following data for each review:

### Review Data
- `review_id`: Unique identifier for the review
- `review_text`: Full text of the review
- `review_score`: Numerical score (0-100 scale)
- `review_category`: Sentiment category (positive/mixed/negative)
- `reviewer_name`: Username of the reviewer
- `review_date`: Date the review was posted
- `review_url`: URL of the reviews page
- `scraped_at`: Timestamp when the review was scraped

### Game Metadata
- `game_title`: Title of the game
- `game_platform`: Platform (PlayStation 4, PC, etc.)
- `game_url`: Metacritic URL of the game
- `game_metascore`: Critic metascore (0-100)
- `game_release_date`: Game release date (e.g., "Jun 19, 2020")
- `game_genre`: Primary game genre (e.g., "Survival", "Action")
- `game_developer`: Developer name (e.g., "Naughty Dog")
- `game_publisher`: Publisher name (e.g., "Sony Interactive Entertainment")

### Sentiment Categories

Reviews are automatically categorized based on their score:

- **Positive** (Green): Score 75-100
- **Mixed** (Yellow): Score 50-74
- **Negative** (Red): Score 0-49

## Configuration

### Scrapy Settings

Edit `metacritic_scraper/settings.py` to customize:

- `DOWNLOAD_DELAY`: Delay between requests (default: 2 seconds)
- `CONCURRENT_REQUESTS`: Number of concurrent requests (default: 8)
- `AUTOTHROTTLE_ENABLED`: Enable automatic throttling (default: True)
- `ROBOTSTXT_OBEY`: Respect robots.txt (default: False)

### Rate Limiting

To be respectful to Metacritic's servers:

- Default delay: 2 seconds between requests
- AutoThrottle enabled to adjust speed automatically
- Concurrent requests limited to 4-8

### Pipelines

Active pipelines (in order):

1. `MetacriticScraperPipeline`: Clean and process data
2. `DuplicatesPipeline`: Filter duplicate reviews
3. `CsvExportPipeline`: Export to CSV with timestamp

To enable JSON export instead, edit `settings.py`:

```python
ITEM_PIPELINES = {
    "metacritic_scraper.pipelines.MetacriticScraperPipeline": 100,
    "metacritic_scraper.pipelines.DuplicatesPipeline": 200,
    "metacritic_scraper.pipelines.JsonExportPipeline": 300,  # Enable this
    # "metacritic_scraper.pipelines.CsvExportPipeline": 300,  # Disable this
}
```

## Output

Scraped data is saved to the `data/` directory with timestamps:

- CSV format: `data/metacritic_reviews_YYYYMMDD_HHMMSS.csv`
- JSON format: `data/metacritic_reviews_YYYYMMDD_HHMMSS.json`

## Troubleshooting

### Connection Timeouts

If you see `TCP connection timed out` errors:

1. **Test connectivity first:**
   ```bash
   python test_connection.py
   ```

2. **Verify the game exists:**
   - Go to https://www.metacritic.com/
   - Search for your game
   - Make sure it has user reviews
   - Copy the exact game name from the URL

3. **Try a known working game:**
   ```bash
   scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4" -a max_reviews=10
   ```

4. **Increase delays** in `metacritic_scraper/settings.py`:
   ```python
   DOWNLOAD_DELAY = 5  # Increase delay
   CONCURRENT_REQUESTS = 1  # Reduce concurrency
   ```

### No reviews found

- Check if the game has user reviews on Metacritic
- Verify the game name matches exactly (use hyphens, lowercase)
- The game might be too new or unreleased (e.g., "Hollow Knight: Silksong")
- Try using the full URL with `-a game_url="..."`

### Rate limiting errors

- Increase `DOWNLOAD_DELAY` in settings
- Reduce `CONCURRENT_REQUESTS`
- Take breaks between scraping sessions

### SSL errors

Add to `settings.py`:
```python
DOWNLOAD_HANDLERS = {
    "http": "scrapy.core.downloader.handlers.http.HTTPDownloadHandler",
    "https": "scrapy.core.downloader.handlers.http.HTTPDownloadHandler",
}
```

**For more detailed troubleshooting, see [TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

## Legal and Ethical Considerations

⚠️ **Important**: Web scraping should be done responsibly and ethically.

- **Terms of Service**: Metacritic's robots.txt disallows scraping. Use this tool at your own risk and responsibility.
- **Rate Limiting**: This scraper includes delays and throttling to minimize server load.
- **Personal Use**: This project is intended for educational and research purposes.
- **Data Usage**: Respect copyright and terms of use for any scraped content.

## Next Steps for Sentiment Analysis

After collecting the data, you can:

1. **Data Preprocessing**: Clean and normalize review text
2. **Exploratory Data Analysis**: Analyze score distribution, review length, etc.
3. **Feature Engineering**: Extract features from review text
4. **Model Training**: Train ML models using the labeled data (positive/mixed/negative)
5. **Model Evaluation**: Test model performance on held-out data

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the terms included in the LICENSE file.

## Disclaimer

This project is for educational purposes only. The authors are not responsible for any misuse of this tool. Always respect website terms of service and robots.txt files.