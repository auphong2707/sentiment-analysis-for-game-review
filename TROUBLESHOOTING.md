# Troubleshooting Guide

## Common Issues and Solutions

### 1. TCP Connection Timeout / Connection Failed

**Error Message:**
```
ERROR: TCP connection timed out: 10060: A connection attempt failed...
```

**Possible Causes:**
- Network connectivity issues
- Metacritic's website blocking automated requests
- The game URL doesn't exist or has been changed
- Too many concurrent requests
- Firewall or antivirus blocking the connection

**Solutions:**

#### A. Verify the Game Exists
First, check if the game exists on Metacritic:
1. Go to https://www.metacritic.com/
2. Search for your game
3. Check the URL format: `https://www.metacritic.com/game/{platform}/{game-name}`
4. Copy the exact game name from the URL

Example: For "Hollow Knight: Silksong", check if it has user reviews yet. If it's unreleased, it won't have reviews to scrape.

#### B. Test with a Known Working Game
Try scraping a popular game with many reviews:
```bash
scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4" -a max_reviews=10
```

#### C. Increase Timeouts and Delays
The settings have been updated to be more conservative. If still failing, manually increase in `metacritic_scraper/settings.py`:

```python
DOWNLOAD_DELAY = 5  # Increase from 3 to 5 seconds
DOWNLOAD_TIMEOUT = 60  # Increase from 30 to 60 seconds
CONCURRENT_REQUESTS = 1  # Reduce to 1 request at a time
```

#### D. Check Your Internet Connection
```bash
# Test if you can reach Metacritic
ping metacritic.com

# Test with curl or browser
curl -I https://www.metacritic.com/
```

#### E. Use a Different Network
If your network or ISP is blocking Metacritic:
- Try a different network (mobile hotspot, VPN, etc.)
- Check if your firewall/antivirus is blocking Python or Scrapy

### 2. No Reviews Found

**Solutions:**
- Verify the game has user reviews on Metacritic
- Check if the game name matches exactly (use hyphens, lowercase)
- Try using the full URL with `-a game_url="..."`
- The game might be too new or not yet released

### 3. Blocked by Metacritic (403 Forbidden)

**Solutions:**
- Increase `DOWNLOAD_DELAY` to 5-10 seconds
- Reduce `CONCURRENT_REQUESTS` to 1
- Take breaks between scraping sessions
- Use respectful scraping practices

### 4. Import Errors

**Solutions:**
```bash
# Reinstall dependencies
pip install -r requirements.txt

# Or reinstall Scrapy specifically
pip install --upgrade scrapy
```

### 5. Spider Not Found

**Error:** `KeyError: 'Spider not found: metacritic_reviews'`

**Solutions:**
- Make sure you're in the project root directory
- Check that `scrapy.cfg` exists in the current directory
```bash
# Navigate to project root
cd D:\Git\sentiment-analysis-for-game-review

# Then run the spider
scrapy crawl metacritic_reviews -a game_name="..." -a platform="..."
```

## Best Practices to Avoid Issues

### 1. Start Small
```bash
# Always test with a small number first
scrapy crawl metacritic_reviews -a game_name="..." -a platform="..." -a max_reviews=10
```

### 2. Use Known Good Games
Test with these popular games that definitely have reviews:
- `the-last-of-us-part-ii` (playstation-4)
- `elden-ring` (playstation-5)
- `baldurs-gate-iii` (pc)
- `god-of-war` (playstation-4)

### 3. Be Patient
- Use delays between requests (3-5 seconds)
- Don't scrape too many games in quick succession
- Respect Metacritic's servers

### 4. Monitor the Output
Watch for:
- "No reviews found" warnings
- 404 errors (game doesn't exist)
- 403 errors (blocked)
- Timeout errors (network issues)

## Updated Settings

The scraper settings have been optimized for better reliability:

- **Download Delay**: 3 seconds (was 2)
- **Concurrent Requests**: 4 (was 8)
- **Timeout**: 30 seconds (new)
- **Retry Attempts**: 5 (was 3)
- **AutoThrottle**: Enabled with conservative settings

## Still Having Issues?

### Debug Mode
Run with debug logging to see more details:
```bash
scrapy crawl metacritic_reviews -a game_name="..." -a platform="..." --loglevel=DEBUG
```

### Test Connection
```python
# test_connection.py
import requests

url = "https://www.metacritic.com/game/playstation-4/the-last-of-us-part-ii"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
}

try:
    response = requests.get(url, headers=headers, timeout=30)
    print(f"Status Code: {response.status_code}")
    print(f"Success! Can reach Metacritic.")
except Exception as e:
    print(f"Error: {e}")
```

### Alternative Approach
If persistent issues with Metacritic, consider:
1. Using their API (if available)
2. Manual data collection for small datasets
3. Using alternative review sources (Steam, IGN, GameSpot)
4. Pre-collected datasets (Kaggle, etc.)

## Contact

If you continue to experience issues:
1. Check if Metacritic's website structure has changed
2. Verify your network/firewall settings
3. Try from a different machine/network
4. Consider if Metacritic has implemented stricter anti-scraping measures
