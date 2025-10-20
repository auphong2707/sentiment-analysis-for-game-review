# Data Scraping Phase - Complete Setup

## ✅ What Was Done

All scraping-related scripts and tools have been organized into the `data_scrape_phase/` folder.

## 📁 New Structure

```
data_scrape_phase/
├── README.md                        # Complete documentation
├── discover_games.py                # Discover games from Metacritic
├── scrape_all_games.py              # Scrape multiple games
├── run_scraper.py                   # Interactive scraping tool
├── config.py                        # Scraper configuration
├── scrapy.cfg                       # Scrapy project config
├── install_playwright.ps1           # Playwright setup (Windows)
├── install_playwright.sh            # Playwright setup (Linux/Mac)
└── metacritic_scraper/              # Scrapy spider project
    ├── __init__.py
    ├── items.py                     # Data models
    ├── middlewares.py               # Custom middlewares
    ├── pipelines.py                 # Data pipelines
    ├── settings.py                  # Scrapy settings
    └── spiders/
        ├── __init__.py
        └── metacritic_reviews.py    # Main spider
```

**Total: 4 Python scripts + 1 Scrapy project + 2 setup scripts + 1 config file**

## 🎯 Complete Workflow

### Phase 1: Data Scraping (This Folder)

```powershell
# 1. Discover games
python data_scrape_phase/discover_games.py --max-pages 10

# 2. Scrape reviews
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/discovered_games_XXXXXX.txt

# Or use interactive mode
python data_scrape_phase/run_scraper.py
```

### Phase 2: Data Preparation (dataset_prepare_phase)

```powershell
# 3. Aggregate and filter
python dataset_prepare_phase/aggregate_dataset.py

# 4. Analyze (optional)
python dataset_prepare_phase/analyze_reviews.py data/aggregated_review_english/aggregated_reviews_english.jsonl

# 5. Upload to HuggingFace
python dataset_prepare_phase/prepare_and_upload_hf_dataset.py
```

## 📋 Scripts Overview

| Script | Purpose | Input | Output |
|--------|---------|-------|--------|
| `discover_games.py` | Browse Metacritic and find games | Metacritic browse pages | `data/discovered_games/*.txt` |
| `scrape_all_games.py` | Scrape reviews for multiple games | Discovered games list | `data/review_data/*.jsonl` |
| `run_scraper.py` | Interactive single-game scraper | User input | `data/review_data/*.jsonl` |
| `config.py` | Configuration settings | N/A | Used by all scripts |

## ⚙️ Configuration

Edit `data_scrape_phase/config.py`:

```python
# Request delays (be respectful to servers!)
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True

# Retry settings
RETRY_TIMES = 3

# Output directory
OUTPUT_DIR = 'data'  # Relative to project root

# Scraping limits
MAX_REVIEWS_PER_GAME = None  # None = unlimited
```

## 🚀 Quick Start

### First Time Setup

1. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

2. **Install Playwright**:
   ```powershell
   .\data_scrape_phase\install_playwright.ps1
   ```

### Start Scraping

```powershell
# Discover games (test with 5 pages)
python data_scrape_phase/discover_games.py --max-pages 5

# Scrape discovered games
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/discovered_games_*.txt
```

## 🔍 Path Handling

All scripts use paths relative to the **project root**:
- Data output: `data/discovered_games/`, `data/review_data/`
- Configuration: Reads from `config.py` in same folder
- Scrapy project: References `metacritic_scraper/` subdirectory

**Always run from project root**:
```powershell
# ✅ Correct
python data_scrape_phase/discover_games.py

# ❌ Avoid (unless paths are adjusted)
cd data_scrape_phase
python discover_games.py
```

## 📦 Dependencies

Required packages (in `requirements.txt`):

```
# Web Scraping
scrapy>=2.11.0
scrapy-playwright>=0.0.34
selenium>=4.15.0
webdriver-manager>=4.0.1
beautifulsoup4>=4.12.0
requests>=2.31.0

# Data Processing
pandas>=2.1.3
numpy>=1.26.2
tqdm>=4.66.1

# Language Detection
langdetect>=1.0.9
```

## 📊 Output Data

### Discovered Games
Location: `data/discovered_games/`

Format:
```json
{
  "game_name": "the-last-of-us-part-ii",
  "platform": "playstation-4",
  "url": "/game/playstation-4/the-last-of-us-part-ii/",
  "title": "The Last of Us Part II",
  "score": 93
}
```

### Review Data
Location: `data/review_data/`

Format:
```json
{
  "game_name": "the-last-of-us-part-ii",
  "platform": "playstation-4",
  "review_text": "Amazing storytelling...",
  "review_score": 100,
  "review_category": "positive",
  "reviewer": "John_Doe",
  "review_date": "Jun 19, 2020"
}
```

## 🛡️ Best Practices

1. **Respect Rate Limits**: Use delays (configured in config.py)
2. **Test First**: Start with `--max-pages 5` when discovering
3. **Monitor Progress**: Scripts show progress bars
4. **Resume Interrupted**: Use `--skip-existing` flag
5. **Check Data Quality**: Review output files periodically

## 🆘 Troubleshooting

**Scrapy can't find spider:**
```powershell
# Make sure you're in project root
cd d:\Git\sentiment-analysis-for-game-review
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/file.txt
```

**Playwright not working:**
```powershell
playwright install
```

**Import errors:**
```powershell
pip install -r requirements.txt
```

**Rate limiting:**
- Increase `DOWNLOAD_DELAY` in config.py
- Add `--delay N` flag to scrape_all_games.py

## 🎓 Usage Examples

### Discover games for a specific platform
```powershell
python data_scrape_phase/discover_games.py --platform playstation-5 --max-pages 10
```

### Scrape with limits
```powershell
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/file.txt --max-reviews 50
```

### Interactive scraping
```powershell
python data_scrape_phase/run_scraper.py
# Follow prompts to enter game details
```

## 📁 Project Structure

```
sentiment-analysis-for-game-review/
├── data_scrape_phase/              # ← Scraping scripts (this folder)
│   ├── discover_games.py
│   ├── scrape_all_games.py
│   ├── run_scraper.py
│   ├── config.py
│   └── metacritic_scraper/
├── dataset_prepare_phase/          # ← Data preparation scripts
│   ├── aggregate_dataset.py
│   ├── analyze_reviews.py
│   └── prepare_and_upload_hf_dataset.py
├── data/                           # ← Output data
│   ├── discovered_games/
│   ├── review_data/
│   └── aggregated_review_english/
├── requirements.txt
└── .env
```

## ✨ Next Steps

After scraping data, proceed to data preparation:

1. **Aggregate reviews**: `python dataset_prepare_phase/aggregate_dataset.py`
2. **Analyze data**: `python dataset_prepare_phase/analyze_reviews.py`
3. **Upload to HF**: `python dataset_prepare_phase/prepare_and_upload_hf_dataset.py`

---

**All scraping tools organized and ready to use!** 🎉
