# PowerShell script to install Scrapy-Playwright and Playwright browsers

Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "Installing Scrapy-Playwright for JavaScript Rendering" -ForegroundColor Cyan
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Install Python packages
Write-Host "Step 1: Installing Python packages..." -ForegroundColor Yellow
pip install scrapy-playwright

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install scrapy-playwright" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Python packages installed" -ForegroundColor Green
Write-Host ""

# Step 2: Install Playwright browsers
Write-Host "Step 2: Installing Playwright browsers (Chromium, Firefox, WebKit)..." -ForegroundColor Yellow
Write-Host "This may take a few minutes..." -ForegroundColor Gray
playwright install

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Failed to install Playwright browsers" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Playwright browsers installed" -ForegroundColor Green
Write-Host ""

# Step 3: Verify installation
Write-Host "Step 3: Verifying installation..." -ForegroundColor Yellow
python -c "import scrapy_playwright; print('✓ scrapy-playwright imported successfully')"

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Verification failed" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host "✓ Installation Complete!" -ForegroundColor Green
Write-Host "==================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "You can now scrape with JavaScript rendering enabled!" -ForegroundColor Green
Write-Host ""
Write-Host "Usage:" -ForegroundColor Yellow
Write-Host "  python scrape_all_games.py --input data/discovered_games.txt --output data/reviews.jsonl --max-reviews 100 --skip-errors" -ForegroundColor White
Write-Host ""
Write-Host "Expected performance:" -ForegroundColor Yellow
Write-Host "  - Reviews per game: 100-1000+ (vs ~50 without Playwright)" -ForegroundColor White
Write-Host "  - Speed: ~15-30 seconds per game (vs 3-5 seconds)" -ForegroundColor White
Write-Host "  - More complete data for sentiment analysis" -ForegroundColor White
Write-Host ""
