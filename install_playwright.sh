#!/bin/bash
# Shell script to install Scrapy-Playwright and Playwright browsers

echo "=================================================================="
echo "Installing Scrapy-Playwright for JavaScript Rendering"
echo "=================================================================="
echo ""

# Step 1: Install Python packages
echo "Step 1: Installing Python packages..."
pip install scrapy-playwright

if [ $? -ne 0 ]; then
    echo "❌ Failed to install scrapy-playwright"
    exit 1
fi

echo "✓ Python packages installed"
echo ""

# Step 2: Install Playwright browsers
echo "Step 2: Installing Playwright browsers (Chromium, Firefox, WebKit)..."
echo "This may take a few minutes..."
playwright install

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Playwright browsers"
    exit 1
fi

echo "✓ Playwright browsers installed"
echo ""

# Step 3: Verify installation
echo "Step 3: Verifying installation..."
python -c "import scrapy_playwright; print('✓ scrapy-playwright imported successfully')"

if [ $? -ne 0 ]; then
    echo "❌ Verification failed"
    exit 1
fi

echo ""
echo "=================================================================="
echo "✓ Installation Complete!"
echo "=================================================================="
echo ""
echo "You can now scrape with JavaScript rendering enabled!"
echo ""
echo "Usage:"
echo "  python scrape_all_games.py --input data/discovered_games.txt --output data/reviews.jsonl --max-reviews 100 --skip-errors"
echo ""
echo "Expected performance:"
echo "  - Reviews per game: 100-1000+ (vs ~50 without Playwright)"
echo "  - Speed: ~15-30 seconds per game (vs 3-5 seconds)"
echo "  - More complete data for sentiment analysis"
echo ""
