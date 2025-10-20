# Sentiment Analysis for Game Reviews - Project Overview

Complete workflow for scraping, processing, and preparing game review data from Metacritic for sentiment analysis.

## 📁 Project Structure

```
sentiment-analysis-for-game-review/
├── data_scrape_phase/              # Phase 1: Data Collection
│   ├── README.md
│   ├── SETUP_COMPLETE.md
│   ├── discover_games.py           # Discover games from Metacritic
│   ├── scrape_all_games.py         # Scrape reviews for multiple games
│   ├── run_scraper.py              # Interactive scraping tool
│   ├── config.py                   # Scraper configuration
│   ├── scrapy.cfg
│   ├── install_playwright.ps1/.sh
│   └── metacritic_scraper/         # Scrapy spider project
│
├── data_prepare_phase/             # Phase 2: Data Preparation
│   ├── README.md
│   ├── SETUP_COMPLETE.md
│   ├── aggregate_dataset.py        # Aggregate & filter English reviews
│   ├── analyze_reviews.py          # Generate statistics
│   ├── combine_data.py             # Combine review files
│   └── prepare_and_upload_hf_dataset.py  # Upload to HuggingFace
│
├── data/                           # Data Storage
│   ├── discovered_games/           # Discovered game lists
│   ├── review_data/                # Raw scraped reviews
│   └── aggregated_review_english/  # Processed English reviews
│
├── .env                            # Environment configuration (HF token, etc.)
├── .env.example                    # Template
├── requirements.txt                # Python dependencies
└── LICENSE
```

## 🎯 Complete Workflow

### Phase 1: Data Scraping

Scrape game reviews from Metacritic.

```powershell
# 1. Discover games
python data_scrape_phase/discover_games.py --max-pages 10 --platform all

# 2. Scrape reviews for all discovered games
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/discovered_games_XXXXXX.txt

# Or use interactive mode for single games
python data_scrape_phase/run_scraper.py
```

**Output**: `data/review_data/*.jsonl`

### Phase 2: Data Preparation

Process, analyze, and upload the scraped data.

```powershell
# 3. Aggregate and filter English reviews
python data_prepare_phase/aggregate_dataset.py

# 4. Analyze the dataset (optional)
python data_prepare_phase/analyze_reviews.py data/aggregated_review_english/aggregated_reviews_english.jsonl

# 5. Split and upload to HuggingFace
python data_prepare_phase/prepare_and_upload_hf_dataset.py
```

**Output**: HuggingFace dataset (80% train, 10% validation, 10% test)

## 🚀 Quick Start

### 1. Setup Environment

```powershell
# Clone repository
git clone https://github.com/auphong2707/sentiment-analysis-for-game-review.git
cd sentiment-analysis-for-game-review

# Install dependencies
pip install -r requirements.txt

# Install Playwright (for web scraping)
.\data_scrape_phase\install_playwright.ps1

# Configure environment
Copy-Item .env.example .env
notepad .env  # Add your HuggingFace token
```

### 2. Run the Pipeline

```powershell
# Phase 1: Scrape Data
python data_scrape_phase/discover_games.py --max-pages 5
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/discovered_games_*.txt

# Phase 2: Prepare Data
python data_prepare_phase/aggregate_dataset.py
python data_prepare_phase/prepare_and_upload_hf_dataset.py
```

## 📦 Requirements

Main dependencies (see `requirements.txt` for full list):

```
# Web Scraping
scrapy>=2.11.0
scrapy-playwright>=0.0.34
beautifulsoup4>=4.12.0
selenium>=4.15.0

# Data Processing
pandas>=2.1.3
numpy>=1.26.2

# HuggingFace
datasets>=2.14.0
huggingface-hub>=0.17.0

# Language Detection
langdetect>=1.0.9

# Utilities
tqdm>=4.66.1
python-dotenv>=1.0.0
```

## ⚙️ Configuration

### Environment Variables (.env)

```env
# HuggingFace Configuration
HF_TOKEN=hf_your_token_here
HF_DATASET_NAME=your-username/game-reviews-sentiment

# Dataset Split Ratios
TRAIN_RATIO=0.8
VAL_RATIO=0.1
TEST_RATIO=0.1
```

### Scraper Configuration (data_scrape_phase/config.py)

```python
# Request delays (be respectful!)
DOWNLOAD_DELAY = 2
RANDOMIZE_DOWNLOAD_DELAY = True

# Retry settings
RETRY_TIMES = 3

# Scraping limits
MAX_REVIEWS_PER_GAME = None  # None = unlimited
```

## 📊 Data Format

### Discovered Games
```json
{
  "game_name": "the-last-of-us-part-ii",
  "platform": "playstation-4",
  "title": "The Last of Us Part II",
  "score": 93,
  "url": "/game/playstation-4/the-last-of-us-part-ii/"
}
```

### Raw Reviews
```json
{
  "game_name": "the-last-of-us-part-ii",
  "platform": "playstation-4",
  "review_text": "Amazing game with great storytelling...",
  "review_score": 100,
  "review_category": "positive",
  "reviewer": "John_Doe",
  "review_date": "Jun 19, 2020"
}
```

### Aggregated Reviews (English only)
```json
{
  "review_text": "Amazing game with great storytelling...",
  "review_score": 100,
  "review_category": "positive"
}
```

## 🎓 Usage Examples

### Scraping Examples

```powershell
# Discover PlayStation 5 games with high scores
python data_scrape_phase/discover_games.py --platform playstation-5 --min-score 80 --max-pages 10

# Scrape with review limit
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/file.txt --max-reviews 100 --delay 3

# Resume interrupted scraping
python data_scrape_phase/scrape_all_games.py --input data/discovered_games/file.txt --skip-existing
```

### Data Preparation Examples

```powershell
# Aggregate with custom output
python data_prepare_phase/aggregate_dataset.py

# Analyze specific file
python data_prepare_phase/analyze_reviews.py data/aggregated_review_english/aggregated_reviews_english_part1.jsonl

# Upload with custom ratios (edit .env first)
python data_prepare_phase/prepare_and_upload_hf_dataset.py
```

## 📚 Documentation

Each phase has detailed documentation:

- **data_scrape_phase/README.md** - Scraping guide
- **data_scrape_phase/SETUP_COMPLETE.md** - Scraping setup summary
- **data_prepare_phase/README.md** - Data preparation guide
- **data_prepare_phase/SETUP_COMPLETE.md** - Preparation setup summary

## 🔍 Features

### Data Scraping
- ✅ Automatic game discovery from Metacritic
- ✅ Multi-game scraping with progress tracking
- ✅ Playwright integration for JavaScript rendering
- ✅ Configurable delays and rate limiting
- ✅ Resume capability for interrupted scraping
- ✅ Multiple platform support

### Data Preparation
- ✅ Language detection and filtering (English only)
- ✅ Automatic file splitting for large datasets
- ✅ Statistical analysis and reporting
- ✅ Train/validation/test splitting (80-10-10)
- ✅ HuggingFace dataset upload
- ✅ Deduplication support

## 🛡️ Best Practices

1. **Respect Rate Limits**: Use appropriate delays between requests
2. **Test Small First**: Start with `--max-pages 5` when discovering
3. **Monitor Progress**: Check console output and data files
4. **Backup Data**: Keep backups of `data/` folder
5. **Environment Security**: Never commit `.env` file (already in `.gitignore`)

## 🆘 Troubleshooting

### Common Issues

**"Module not found" errors:**
```powershell
pip install -r requirements.txt
playwright install
```

**Scrapy can't find spider:**
- Make sure you're running from project root
- Check that `scrapy.cfg` is in correct location

**HuggingFace upload fails:**
- Verify `HF_TOKEN` is set in `.env`
- Ensure token has write permissions
- Check dataset name format: `username/dataset-name`

**No data found:**
- Run scraping phase first
- Check `data/review_data/` has `.jsonl` files
- Run `aggregate_dataset.py` before upload

### Getting Help

1. Check README files in each phase folder
2. Review SETUP_COMPLETE.md documents
3. Check script help: `python script.py --help`

## 📈 Pipeline Flow

```
┌─────────────────────────────────────────────────────────┐
│                  Phase 1: Data Scraping                 │
├─────────────────────────────────────────────────────────┤
│  discover_games.py                                      │
│       ↓                                                 │
│  discovered_games/*.txt                                 │
│       ↓                                                 │
│  scrape_all_games.py                                    │
│       ↓                                                 │
│  review_data/*.jsonl                                    │
└─────────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────────┐
│               Phase 2: Data Preparation                 │
├─────────────────────────────────────────────────────────┤
│  aggregate_dataset.py (filter English)                  │
│       ↓                                                 │
│  aggregated_review_english/*.jsonl                      │
│       ↓                                                 │
│  analyze_reviews.py (optional analysis)                 │
│       ↓                                                 │
│  prepare_and_upload_hf_dataset.py                       │
│       ↓                                                 │
│  HuggingFace Dataset (80-10-10 split)                   │
└─────────────────────────────────────────────────────────┘
```

## 🎉 Result

After completing the pipeline, you'll have:

1. **Raw scraped data** in `data/review_data/`
2. **Processed English reviews** in `data/aggregated_review_english/`
3. **Statistics and insights** from analysis
4. **HuggingFace dataset** ready for ML training

Access your dataset:
```python
from datasets import load_dataset
dataset = load_dataset("your-username/game-reviews-sentiment")
```

## 📝 License

See LICENSE file for details.

## 🤝 Contributing

This is a data collection and preparation pipeline. Feel free to fork and adapt for your needs.

---

**Happy scraping and analyzing!** 🎮📊
