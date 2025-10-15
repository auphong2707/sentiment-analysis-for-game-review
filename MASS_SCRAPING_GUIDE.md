# Complete Mass Scraping Guide

This guide walks you through scraping **all games and reviews** from Metacritic.

## Overview

The complete workflow consists of two main phases:

1. **Discovery Phase**: Browse Metacritic to find all available games
2. **Scraping Phase**: Scrape reviews for each discovered game

## Quick Start (Easiest Method)

Run everything with one command:

```powershell
python quick_scrape_everything.py
```

This will:
- Discover games (5 pages by default)
- Scrape all reviews for each game
- Save results to the `data/` folder

### Quick Start Options

```powershell
# Scrape PS5 games only, max 100 reviews per game
python quick_scrape_everything.py --platform playstation-5 --max-reviews 100 --max-pages 3

# High-quality dataset (score >= 70, max 200 reviews each)
python quick_scrape_everything.py --min-score 70 --max-reviews 200 --max-pages 10

# Scrape PC games
python quick_scrape_everything.py --platform pc --max-pages 5

# UNLIMITED: Scrape ALL games (no page limit)
python quick_scrape_everything.py --max-pages 0 --max-reviews 300
```

### 🚀 Unlimited Scraping

To scrape **all games without page limits**, use `--max-pages 0`:

```powershell
# All games, 300 reviews each (RECOMMENDED)
python quick_scrape_everything.py --max-pages 0 --max-reviews 300

# All PS5 games, unlimited pages
python quick_scrape_everything.py --platform playstation-5 --max-pages 0

# True unlimited (will take weeks!)
python quick_scrape_everything.py --max-pages 0
```

**Note:** `--max-pages 0` means unlimited. This can take days or weeks depending on review limits.

---

## Manual Workflow (Step by Step)

If you want more control, run each step manually:

### Step 1: Discover Games

First, discover all available games from Metacritic:

```powershell
python discover_games.py --max-pages 10 --format txt
```

**Options:**
- `--platform`: Filter by platform (playstation-5, pc, switch, all)
- `--max-pages`: Number of browse pages to scan (each page has ~100 games)
- `--min-score`: Minimum metascore filter (0-100)
- `--format`: Output format (txt, json, csv)

**Examples:**

```powershell
# Discover all PS5 games
python discover_games.py --platform playstation-5 --max-pages 5

# Discover highly-rated games (score >= 75)
python discover_games.py --min-score 75 --max-pages 10

# Save as JSON for detailed metadata
python discover_games.py --max-pages 5 --format json
```

**Output:**
- Creates a file like `data/discovered_games_20231015_143022.txt`
- Contains game names and platforms, one per line
- Format: `game-name-slug|platform-slug`

### Step 2: Scrape All Games

Use the discovered games file to scrape reviews:

```powershell
python scrape_all_games.py --input data/discovered_games_20231015_143022.txt
```

**Options:**
- `--input`: Path to discovery file (required)
- `--max-reviews`: Maximum reviews per game (optional)
- `--delay`: Seconds between games (default: 5)
- `--skip-errors`: Continue even if a game fails
- `--start-from`: Resume from game number N

**Examples:**

```powershell
# Scrape all games, 100 reviews each
python scrape_all_games.py --input data/discovered_games.txt --max-reviews 100

# Resume from game #50 (if interrupted)
python scrape_all_games.py --input data/discovered_games.txt --start-from 50 --skip-errors

# Scrape with longer delays (safer for rate limiting)
python scrape_all_games.py --input data/discovered_games.txt --delay 10 --skip-errors
```

**Output:**
- Individual JSON files per game: `data/metacritic_reviews_20231015_143530.json`
- Scraping log: `data/scraping_log_20231015_143530.json`

### Step 3: Combine Data (Optional)

Merge all scraped reviews into a single dataset:

```powershell
python combine_data.py
```

This creates a unified dataset from all JSON files in the `data/` folder.

---

## Platform Values

Use these values for the `--platform` option:

### PlayStation
- `playstation-5` (PS5)
- `playstation-4` (PS4)
- `playstation-3` (PS3)

### Xbox
- `xbox-series-x` (Xbox Series X/S)
- `xbox-one` (Xbox One)
- `xbox-360` (Xbox 360)

### Nintendo
- `switch` (Nintendo Switch)
- `wii-u` (Wii U)
- `3ds` (Nintendo 3DS)

### PC
- `pc` (PC)

### All Platforms
- `all` (No filter - all platforms)

---

## Recommended Strategies

### Strategy 1: Quick Sample (Fast)
Get a representative sample quickly:

```powershell
python quick_scrape_everything.py --max-pages 3 --max-reviews 50 --skip-errors
```

- **Time**: ~30-60 minutes
- **Games**: ~300 games
- **Reviews**: ~15,000 reviews
- **Good for**: Testing, prototyping, quick analysis

### Strategy 2: Quality Dataset (Balanced)
Focus on high-quality, popular games:

```powershell
python quick_scrape_everything.py --min-score 70 --max-pages 10 --max-reviews 200
```

- **Time**: 2-4 hours
- **Games**: ~500-800 games
- **Reviews**: ~100,000-160,000 reviews
- **Good for**: Sentiment analysis, ML training

### Strategy 3: Complete Dataset (Thorough)
Scrape everything available:

```powershell
# Phase 1: Discover all games
python discover_games.py --max-pages 50 --format txt

# Phase 2: Scrape with limits
python scrape_all_games.py --input data/discovered_games.txt --max-reviews 500 --skip-errors
```

- **Time**: 1-3 days
- **Games**: 2,000-5,000+ games
- **Reviews**: 500,000-1,000,000+ reviews
- **Good for**: Comprehensive research, large-scale ML

### Strategy 4: Platform-Specific
Focus on one platform:

```powershell
# PS5 games only
python quick_scrape_everything.py --platform playstation-5 --max-pages 10 --max-reviews 200

# PC games only
python quick_scrape_everything.py --platform pc --max-pages 10 --max-reviews 200
```

- **Time**: 1-3 hours
- **Games**: 300-1,000 games
- **Reviews**: 60,000-200,000 reviews
- **Good for**: Platform-specific analysis

---

## Search for Specific Games

You can also search for specific games instead of browsing:

```powershell
python discover_games.py --search "god of war"
```

This finds all games matching the search term.

---

## Handling Interruptions

If scraping is interrupted:

1. **Find where you stopped**: Check `data/scraping_log_*.json`
2. **Resume from that point**:
   ```powershell
   python scrape_all_games.py --input data/discovered_games.txt --start-from 123 --skip-errors
   ```

---

## Rate Limiting & Best Practices

### Recommended Delays
- **Conservative**: `--delay 10` (safer, slower)
- **Normal**: `--delay 5` (default, balanced)
- **Fast**: `--delay 3` (riskier, faster)

### Avoid Being Blocked
1. Use reasonable delays (5+ seconds)
2. Don't run multiple scrapers simultaneously
3. Scrape during off-peak hours
4. Use `--max-reviews` to limit per game
5. Take breaks between large batches

### If You Get Blocked
- Wait 1-2 hours before trying again
- Increase delays: `--delay 10` or higher
- Reduce concurrency in `settings.py`
- Consider using VPN (change IP)

---

## Output Files

### Discovery Files
- `data/discovered_games_TIMESTAMP.txt` - Game list (simple)
- `data/discovered_games_TIMESTAMP.json` - Game list (detailed)
- `data/discovered_games_TIMESTAMP.csv` - Game list (spreadsheet)

### Scraping Files
- `data/metacritic_reviews_TIMESTAMP.json` - Review data per scrape
- `data/scraping_log_TIMESTAMP.json` - Success/failure log

### Combined Files
- `data/combined_reviews_TIMESTAMP.json` - All reviews merged
- `data/combined_reviews_TIMESTAMP.csv` - All reviews as CSV

---

## Troubleshooting

### "No games discovered"
- Check your internet connection
- Try without filters first: `python discover_games.py --max-pages 2`
- Metacritic might have changed their HTML structure

### "Timeout" errors during scraping
- Increase delay: `--delay 10`
- Check `settings.py` - increase `DOWNLOAD_DELAY` to 5+
- Reduce `CONCURRENT_REQUESTS` to 1

### "Failed to scrape" many games
- Use `--skip-errors` to continue despite failures
- Check `scraping_log_*.json` for error details
- Some games might not have reviews or might be unreleased

### Script stops unexpectedly
- Resume with `--start-from N`
- Always use `--skip-errors` for bulk scraping
- Check available disk space

---

## Estimating Time & Data

### Time Estimates
Based on default delays (5s between games):

| Games | Reviews/Game | Estimated Time |
|-------|--------------|----------------|
| 100   | 50          | 30-60 min     |
| 500   | 100         | 2-4 hours     |
| 1000  | 200         | 8-12 hours    |
| 5000  | 500         | 3-5 days      |

### Data Size Estimates
Approximate JSON file sizes:

| Reviews   | Approximate Size |
|-----------|------------------|
| 10,000    | 10-20 MB        |
| 100,000   | 100-200 MB      |
| 500,000   | 500 MB - 1 GB   |
| 1,000,000 | 1-2 GB          |

---

## Example Complete Workflow

Here's a complete example:

```powershell
# 1. Test connection
python test_connection.py

# 2. Discover games (5 pages, PS5 only, score >= 75)
python discover_games.py --platform playstation-5 --min-score 75 --max-pages 5 --format txt

# 3. Check discovered games
# Look at: data/discovered_games_TIMESTAMP.txt

# 4. Scrape all discovered games (max 100 reviews each)
python scrape_all_games.py --input data/discovered_games_TIMESTAMP.txt --max-reviews 100 --skip-errors

# 5. Check results
# Look at: data/scraping_log_TIMESTAMP.json

# 6. Combine all data
python combine_data.py

# 7. Analyze!
# Use: data/combined_reviews_TIMESTAMP.json
```

---

## Advanced: Custom Game Lists

You can also create your own game list manually:

**Create `my_games.txt`:**
```
the-last-of-us-part-ii|playstation-4
god-of-war|playstation-4
elden-ring|playstation-5
baldurs-gate-iii|pc
hades|switch
```

**Then scrape:**
```powershell
python scrape_all_games.py --input my_games.txt --max-reviews 200
```

---

## Next Steps

After scraping:

1. **Check data quality**: Review the JSON files
2. **Data cleaning**: Remove duplicates, handle missing values
3. **Exploratory analysis**: Visualize score distributions
4. **Sentiment analysis**: Train ML models on labeled data
5. **Insights**: Discover patterns in game reviews

---

## Need Help?

- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md) for common issues
- Review [README.md](README.md) for basic usage
- Check scraping logs in `data/scraping_log_*.json`
- Test with small batches first (1-2 pages, 10-20 reviews)

---

## Legal Notice

⚠️ **Important**: 
- This is for educational/research purposes only
- Respect Metacritic's terms of service
- Use reasonable rate limits
- Don't resell or republish scraped data
- The authors are not responsible for misuse

---

Happy scraping! 🕷️
