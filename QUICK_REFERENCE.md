# Quick Reference - Mass Scraping Commands

## 🚀 Fastest Way (One Command)

```powershell
python quick_scrape_everything.py
```

## 📋 Common Commands

### Discover Games
```powershell
# All platforms, 5 pages
python discover_games.py --max-pages 5

# PS5 only
python discover_games.py --platform playstation-5 --max-pages 5

# High-rated games (score >= 75)
python discover_games.py --min-score 75 --max-pages 10

# Search specific game
python discover_games.py --search "god of war"
```

### Scrape All Games
```powershell
# Basic (use latest discovery file)
python scrape_all_games.py --input data/discovered_games_TIMESTAMP.txt --skip-errors

# With review limit
python scrape_all_games.py --input data/discovered_games_TIMESTAMP.txt --max-reviews 100 --skip-errors

# Resume from game #50
python scrape_all_games.py --input data/discovered_games_TIMESTAMP.txt --start-from 50 --skip-errors

# With longer delays
python scrape_all_games.py --input data/discovered_games_TIMESTAMP.txt --delay 10 --skip-errors
```

### Quick Complete Workflow
```powershell
# PS5 games, 100 reviews each
python quick_scrape_everything.py --platform playstation-5 --max-reviews 100 --max-pages 3

# Quality dataset
python quick_scrape_everything.py --min-score 70 --max-reviews 200 --max-pages 10
```

### Combine Data
```powershell
python combine_data.py
```

## 🎮 Platform Values

| Platform | Value |
|----------|-------|
| PlayStation 5 | `playstation-5` |
| PlayStation 4 | `playstation-4` |
| Xbox Series X/S | `xbox-series-x` |
| Xbox One | `xbox-one` |
| Nintendo Switch | `switch` |
| PC | `pc` |
| All platforms | `all` |

## ⚙️ Common Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-pages` | Browse pages to scan | 5 |
| `--max-reviews` | Reviews per game | unlimited |
| `--platform` | Platform filter | all |
| `--min-score` | Min metascore (0-100) | none |
| `--delay` | Seconds between games | 5 |
| `--skip-errors` | Continue on errors | false |
| `--start-from` | Resume from game N | 1 |

## 📊 Recommended Strategies

### Quick Sample (30-60 min)
```powershell
python quick_scrape_everything.py --max-pages 3 --max-reviews 50
```
~300 games, ~15,000 reviews

### Quality Dataset (2-4 hours)
```powershell
python quick_scrape_everything.py --min-score 70 --max-pages 10 --max-reviews 200
```
~800 games, ~160,000 reviews

### Platform-Specific (1-3 hours)
```powershell
python quick_scrape_everything.py --platform playstation-5 --max-pages 10 --max-reviews 200
```
~500-1,000 games

### Complete (1-3 days)
```powershell
python discover_games.py --max-pages 50
python scrape_all_games.py --input data/discovered_games_TIMESTAMP.txt --max-reviews 500 --skip-errors
```
2,000-5,000+ games, 500,000-1,000,000+ reviews

## 🔧 Troubleshooting Quick Fixes

### Connection Issues
```powershell
python test_connection.py
```

### Increase Delays (if blocked)
```powershell
python scrape_all_games.py --input data/discovered_games.txt --delay 10 --skip-errors
```

### Resume After Interruption
```powershell
# Check log for last game number, then:
python scrape_all_games.py --input data/discovered_games.txt --start-from 123 --skip-errors
```

## 📁 Output Files

- `data/discovered_games_TIMESTAMP.txt` - Game list
- `data/metacritic_reviews_TIMESTAMP.json` - Review data
- `data/scraping_log_TIMESTAMP.json` - Scraping results
- `data/combined_reviews_TIMESTAMP.json` - All reviews merged

## 💡 Pro Tips

1. **Always use `--skip-errors`** for bulk scraping
2. **Start small** (2-3 pages) to test first
3. **Use `--max-reviews`** to control dataset size
4. **Check logs** in `data/scraping_log_*.json`
5. **Resume interrupted scrapes** with `--start-from`
6. **Combine data last** with `combine_data.py`

## 🆘 Getting Help

- Full guide: See [MASS_SCRAPING_GUIDE.md](MASS_SCRAPING_GUIDE.md)
- Basic usage: See [README.md](README.md)
- Issues: See [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

## ⚠️ Rate Limiting

**Conservative** (safer):
```powershell
--delay 10
```

**Normal** (default):
```powershell
--delay 5
```

**Fast** (riskier):
```powershell
--delay 3
```

## 📝 Example Workflow

```powershell
# 1. Test connection
python test_connection.py

# 2. Discover games
python discover_games.py --platform playstation-5 --max-pages 5 --min-score 75

# 3. Scrape reviews
python scrape_all_games.py --input data/discovered_games_TIMESTAMP.txt --max-reviews 100 --skip-errors

# 4. Combine data
python combine_data.py

# 5. Analyze!
```

---

**Need more details?** Read [MASS_SCRAPING_GUIDE.md](MASS_SCRAPING_GUIDE.md)
