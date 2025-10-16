"""
Metacritic Game Reviews Spider (with Playwright for JavaScript rendering)

This spider scrapes user reviews from Metacritic for video games.
It collects review text, scores, sentiment categories, and metadata.

Uses Scrapy-Playwright to render JavaScript and load all reviews dynamically.

Usage:
    scrapy crawl metacritic_reviews -a game_url="https://www.metacritic.com/game/pc/..."
    scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4"
    scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4" -a max_reviews_per_platform=100
"""

import scrapy
from datetime import datetime
import re
from urllib.parse import urljoin
from metacritic_scraper.items import GameReviewItem, GameItem


class MetacriticReviewsSpider(scrapy.Spider):
    name = "metacritic_reviews"
    allowed_domains = ["metacritic.com"]
    
    custom_settings = {
        'DOWNLOAD_DELAY': 3,
        'CONCURRENT_REQUESTS': 2,
        'AUTOTHROTTLE_ENABLED': True,
        'DOWNLOAD_TIMEOUT': 30,
        'RETRY_TIMES': 5,
    }
    
    def __init__(self, game_url=None, game_name=None, platform=None, max_reviews_per_platform=None, *args, **kwargs):
        super(MetacriticReviewsSpider, self).__init__(*args, **kwargs)
        
        self.max_reviews_per_platform = int(max_reviews_per_platform) if max_reviews_per_platform else None
        self.total_reviews_count = 0  # Total across all platforms
        self.platform_reviews_count = {}  # Track count per platform
        
        # Build the game URL from components or use provided URL
        if game_url:
            self.start_urls = [game_url]
        elif game_name and platform:
            # Construct URL: https://www.metacritic.com/game/{platform}/{game-name}
            self.start_urls = [f"https://www.metacritic.com/game/{platform}/{game_name}"]
        else:
            # Default example game if no parameters provided
            self.logger.warning("No game specified. Using default example game.")
            self.start_urls = ["https://www.metacritic.com/game/playstation-4/the-last-of-us-part-ii"]
    
    def start_requests(self):
        """Generate initial requests for game pages"""
        for url in self.start_urls:
            self.logger.info(f"Starting to scrape: {url}")
            # Use Playwright for JavaScript rendering
            yield scrapy.Request(
                url=url,
                callback=self.parse_game_page,
                errback=self.handle_error,
                meta={
                    "playwright": True,
                    "playwright_include_page": True,
                    "playwright_page_goto_kwargs": {
                        "wait_until": "domcontentloaded",  # Wait until DOM is ready
                        "timeout": 60000,  # 60 seconds timeout
                    },
                    "playwright_page_methods": [
                        {"wait_for_timeout": 5000},  # Additional 5 second wait
                    ],
                }
            )
    
    def parse_game_page(self, response):
        """Parse the main game page to extract game metadata and navigate to reviews"""
        
        self.logger.info(f"Parsing game page: {response.url}")
        
        # Extract game metadata
        game_title = response.css('div.product_title h1::text').get() or \
                    response.css('h1.c-productHero_title span::text').get() or \
                    response.css('h1::text').get()
        
        # Platform - try multiple selectors
        platform = response.css('span.platform a::text').get() or \
                  response.css('.c-ProductHeroGamePlatformInfo ::text').get() or \
                  response.css('[class*="Platform"]::text').get()
        
        # Metascore
        metascore = response.css('div.metascore_w span::text').get() or \
                   response.css('div.c-siteReviewScore span::text').get() or \
                   response.css('[class*="Metascore"] span::text').get()
        
        # Release date - extract from c-gameDetails_ReleaseDate class
        release_date_elem = response.css('.c-gameDetails_ReleaseDate span.g-color-gray70::text').get() or \
                           response.css('.c-gameDetails_ReleaseDate span:nth-of-type(2)::text').get() or \
                           response.css('li.release_data span.data::text').get()
        
        # Genre - extract from c-genreList_item 
        genre = response.css('.c-genreList_item .c-globalButton_label::text').get() or \
               response.css('.c-genreList_item a span::text').get() or \
               response.css('li.genre span.data::text').get()
        
        # Developer - extract from c-gameDetails_Developer
        developer = response.css('.c-gameDetails_Developer a::text').get() or \
                   response.css('.c-gameDetails_Developer li a::text').get() or \
                   response.css('li.developer span.data::text').get()
        
        # Publisher - extract from c-gameDetails_Distributor (note: class is "Distributor" not "Publisher")
        publisher = response.css('.c-gameDetails_Distributor a::text').get() or \
                   response.css('.c-gameDetails_Distributor span:nth-of-type(2)::text').get() or \
                   response.css('li.publisher span.data::text').get()
        
        # Clean up the extracted data
        if game_title:
            game_title = game_title.strip()
        if platform:
            platform = platform.strip()
        if release_date_elem:
            release_date = release_date_elem.strip()
        else:
            release_date = None
        if genre:
            genre = genre.strip()
        if developer:
            developer = developer.strip()
        if publisher:
            publisher = publisher.strip()
        
        self.logger.info(f"Game: {game_title} ({platform})")
        self.logger.info(f"Release: {release_date}, Genre: {genre}")
        self.logger.info(f"Developer: {developer}, Publisher: {publisher}")
        
        # Store game info for later use
        self.game_info = {
            'game_title': game_title,
            'game_platform': platform,
            'game_url': response.url,
            'game_metascore': metascore,
            'game_release_date': release_date,
            'game_genre': genre,
            'game_developer': developer,
            'game_publisher': publisher,
        }
        
        # Navigate to user reviews page
        # To get ALL reviews across all platforms, we need to remove the platform from URL
        # Convert: /game/platform/game-name -> /game/game-name/user-reviews
        # Note: Metacritic's review page may default to showing one platform at a time
        # even on the cross-platform URL. The platform filter is handled via JavaScript
        # on the client side. Individual reviews may not always include platform metadata.
        import re
        url_match = re.match(r'(https://www\.metacritic\.com/game)/([^/]+)/([^/]+)', response.url)
        if url_match:
            base_url = url_match.group(1)
            platform_part = url_match.group(2)
            game_name_part = url_match.group(3)
            # Create cross-platform URL (without platform specification)
            user_reviews_url = f"{base_url}/{game_name_part}/user-reviews"
            self.logger.info(f"Cross-platform reviews URL: {user_reviews_url}")
        else:
            # Fallback to original behavior if URL doesn't match expected pattern
            user_reviews_url = response.url.rstrip('/') + '/user-reviews'
            self.logger.warning(f"Using platform-specific reviews URL: {user_reviews_url}")
        
        # Use Playwright to load and scroll through reviews
        yield scrapy.Request(
            url=user_reviews_url,
            callback=self.parse_reviews_page,
            errback=self.handle_error,
            meta={
                "playwright": True,
                "playwright_include_page": True,
                "playwright_page_goto_kwargs": {
                    "wait_until": "domcontentloaded",  # Wait until DOM is ready
                    "timeout": 60000,  # 60 seconds timeout
                },
                "playwright_page_methods": [
                    {"wait_for_timeout": 5000},  # Additional 5 second wait
                ],
            }
        )
    
    async def parse_reviews_page(self, response):
        """
        Parse a page of user reviews with Playwright
        
        First discovers all available platforms, then scrapes reviews for each platform
        """
        
        self.logger.info(f"Parsing reviews page: {response.url}")
        
        # Get the Playwright page object
        page = response.meta.get("playwright_page")
        
        if page:
            try:
                # Wait for reviews to load
                await page.wait_for_selector('div.c-siteReview, div.review_content', timeout=10000)
                self.logger.info("✓ Initial reviews loaded")
                
                # Discover available platforms from dropdown
                platforms = []
                try:
                    # First, click on the platform filter dropdown to make options visible
                    # Look for the dropdown button (usually says "Filter by platform" or shows current platform)
                    dropdown_button = await page.query_selector('button[data-testid="siteDropdown"]')
                    if dropdown_button:
                        self.logger.info("Clicking platform dropdown to reveal options...")
                        await dropdown_button.click()
                        await page.wait_for_timeout(1000)  # Wait for dropdown to open
                    
                    # Now get all dropdown options (should be visible after clicking)
                    platform_options = await page.query_selector_all('div[data-testid="siteDropdownOptions"] div.c-siteDropdown_option span.u-text-overflow-ellipsis')
                    
                    # Known valid platforms - filter out review filter options
                    valid_platforms = []
                    filter_keywords = ['all reviews', 'positive', 'mixed', 'negative', 'recently added', 'score', 'reviews']
                    
                    for option in platform_options:
                        platform_text = await option.text_content()
                        if platform_text:
                            platform_text = platform_text.strip()
                            # Check if this is a platform (not a filter option)
                            is_filter = any(keyword in platform_text.lower() for keyword in filter_keywords)
                            if not is_filter and platform_text:
                                valid_platforms.append(platform_text)
                    
                    platforms = valid_platforms
                    
                    if platforms:
                        self.logger.info(f"✓ Found {len(platforms)} platform(s): {platforms}")
                    else:
                        self.logger.warning("No valid platforms found in dropdown, will scrape current page only")
                        platforms = [None]  # Scrape whatever is currently displayed
                        
                except Exception as e:
                    self.logger.warning(f"Could not find platform dropdown: {e}")
                    platforms = [None]  # Scrape whatever is currently displayed
                
                # Close the initial page
                await page.close()
                
                # Now scrape reviews for each platform
                for platform in platforms:
                    if platform:
                        self.logger.info(f"🎮 Scraping reviews for platform: {platform}")
                        # Construct platform-specific URL using query parameter
                        # Convert platform text to URL format (e.g., "PlayStation 3" -> "playstation-3")
                        platform_slug = platform.lower().replace(' ', '-')
                        
                        # Use query parameter for platform filtering
                        platform_url = f"{response.url.rstrip('/')}?platform={platform_slug}"
                        
                        # Scrape this platform's reviews
                        yield scrapy.Request(
                            url=platform_url,
                            callback=self.parse_platform_reviews,
                            errback=self.handle_error,
                            meta={
                                "playwright": True,
                                "playwright_include_page": True,
                                "playwright_page_goto_kwargs": {
                                    "wait_until": "domcontentloaded",
                                    "timeout": 60000,
                                },
                                "playwright_page_methods": [
                                    {"wait_for_timeout": 5000},
                                ],
                                "platform": platform,
                            }
                        )
                    else:
                        # No platform info, scrape current page
                        yield scrapy.Request(
                            url=response.url,
                            callback=self.parse_platform_reviews,
                            errback=self.handle_error,
                            meta={
                                "playwright": True,
                                "playwright_include_page": True,
                                "playwright_page_goto_kwargs": {
                                    "wait_until": "domcontentloaded",
                                    "timeout": 60000,
                                },
                                "playwright_page_methods": [
                                    {"wait_for_timeout": 5000},
                                ],
                                "platform": None,
                            }
                        )
                        
            except Exception as e:
                self.logger.error(f"Error during platform discovery: {e}")
                if page:
                    await page.close()
        else:
            # No Playwright page, fallback to regular parsing
            for item in self.parse_platform_reviews(response):
                yield item
    
    async def parse_platform_reviews(self, response):
        """
        Parse reviews for a specific platform
        
        Uses Playwright to scroll and load all reviews dynamically
        """
        
        platform = response.meta.get('platform', 'Unknown')
        self.logger.info(f"Parsing reviews for {platform}: {response.url}")
        
        # Initialize counter for this platform if not exists
        if platform not in self.platform_reviews_count:
            self.platform_reviews_count[platform] = 0
        
        # Get the Playwright page object
        page = response.meta.get("playwright_page")
        
        if page:
            try:
                # Wait for reviews to load
                await page.wait_for_selector('div.c-siteReview, div.review_content', timeout=10000)
                self.logger.info(f"✓ Reviews loaded for {platform}")
                
                # Scroll and load more reviews
                previous_count = 0
                scroll_attempts = 0
                max_scroll_attempts = 200  # Prevent infinite loops
                
                while scroll_attempts < max_scroll_attempts:
                    # Count current reviews
                    current_reviews = await page.query_selector_all('div.c-siteReview, div.review_content')
                    current_count = len(current_reviews)
                    
                    self.logger.info(f"📊 {platform}: Currently loaded {current_count} reviews")
                    
                    # Check if we've reached max_reviews_per_platform limit for this platform
                    if self.max_reviews_per_platform and current_count >= self.max_reviews_per_platform:
                        self.logger.info(f"✓ {platform}: Reached platform limit of {self.max_reviews_per_platform} reviews")
                        break
                    
                    # Check if no new reviews loaded
                    if current_count == previous_count:
                        self.logger.info(f"✓ {platform}: No more reviews to load")
                        break
                    
                    previous_count = current_count
                    
                    # Scroll to bottom to trigger loading more reviews
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)  # Wait for new reviews to load
                    
                    # Look for and click "Load More" button if exists
                    try:
                        load_more_button = await page.query_selector('button:has-text("Load More"), a:has-text("Load More"), button.load-more')
                        if load_more_button:
                            self.logger.info(f"🔄 {platform}: Clicking 'Load More' button")
                            await load_more_button.click()
                            await page.wait_for_timeout(2000)
                    except:
                        pass  # No load more button, that's fine
                    
                    scroll_attempts += 1
                
                self.logger.info(f"✓ {platform}: Finished loading reviews. Total: {previous_count}")
                
                # Get the final HTML content
                content = await page.content()
                
                # Close the page
                await page.close()
                
                # Parse the HTML with Scrapy selector
                from scrapy.http import HtmlResponse
                final_response = HtmlResponse(
                    url=response.url,
                    body=content.encode('utf-8'),
                    encoding='utf-8'
                )
                
            except Exception as e:
                self.logger.error(f"Error during Playwright interaction for {platform}: {e}")
                if page:
                    await page.close()
                final_response = response
        else:
            final_response = response
        
        # Find all review containers
        reviews = final_response.css('div.review_content, div.c-siteReview')
        
        if not reviews:
            self.logger.warning(f"No reviews found for {platform} on page: {response.url}")
            reviews = final_response.css('div.user_review')
        
        self.logger.info(f"Found {len(reviews)} reviews to parse for {platform}")
        
        # Parse each review
        for review in reviews:
            # Check if we've reached the limit for this specific platform
            if self.max_reviews_per_platform and self.platform_reviews_count.get(platform, 0) >= self.max_reviews_per_platform:
                self.logger.info(f"✓ {platform}: Reached maximum reviews per platform limit: {self.max_reviews_per_platform}")
                return
            
            item = self.parse_single_review(review, final_response, platform)
            if item:
                self.platform_reviews_count[platform] = self.platform_reviews_count.get(platform, 0) + 1
                self.total_reviews_count += 1
                yield item
        
        self.logger.info(f"✓ {platform}: Scraping complete. Collected {self.platform_reviews_count.get(platform, 0)} reviews for this platform (Total across all platforms: {self.total_reviews_count})")
    
    def parse_single_review(self, review_selector, response, platform=None):
        """Extract data from a single review"""
        
        try:
            # Extract review text - Modern Metacritic structure
            review_text = review_selector.css('div.c-siteReview_quote span::text').get()
            
            if not review_text:
                # Fallback: get all text from review quote
                review_text_parts = review_selector.css('div.c-siteReview_quote::text').getall()
                review_text = ' '.join([t.strip() for t in review_text_parts if t.strip()])
            
            if not review_text:
                # Old structure fallback
                review_text = review_selector.css('div.review_body span.blurb_expanded::text').get()
            
            if review_text:
                review_text = ' '.join(review_text.split())  # Normalize whitespace
            
            # Extract score - Modern Metacritic uses div.c-siteReviewScore
            score = review_selector.css('div.c-siteReviewScore span::text').get()
            
            if not score:
                # Alternative selector
                score = review_selector.css('div.c-siteReviewScore::text').get()
            
            if not score:
                # Old structure fallback
                score = review_selector.css('div.metascore_w::text').get()
            
            if score:
                score = score.strip()
                # Metacritic uses 0-10 scale, convert to 0-100
                try:
                    score = int(score) * 10
                except ValueError:
                    score = None
            
            # Determine category based on score
            category = None
            if score is not None:
                if score >= 75:
                    category = 'positive'
                elif score >= 50:
                    category = 'mixed'
                else:
                    category = 'negative'
            
            # Extract platform from individual review (for cross-platform pages)
            # First, use the platform parameter passed from parse_platform_reviews
            review_platform = platform
            
            if not review_platform:
                # Try to extract from review HTML
                review_platform = review_selector.css('div.c-siteReviewHeader_platform span::text').get()
            
            if not review_platform:
                # Try alternative selectors
                review_platform = review_selector.css('[class*="platform"] span::text').get() or \
                                review_selector.css('[class*="Platform"]::text').get()
            
            if not review_platform:
                # Fallback to game-level platform if not found in review
                review_platform = self.game_info.get('game_platform')
            
            if review_platform:
                review_platform = review_platform.strip()
            
            # Extract reviewer name - Modern selector
            reviewer_name = review_selector.css('a.c-siteReviewHeader_username::text').get()
            
            if not reviewer_name:
                # Try finding any element with 'username' in class
                reviewer_name = review_selector.css('[class*="username"]::text').get()
            
            if not reviewer_name:
                # Old structure fallbacks
                reviewer_name = review_selector.css('div.name a::text').get() or \
                              review_selector.css('span.name a::text').get()
            
            if reviewer_name:
                reviewer_name = reviewer_name.strip()
            
            # Extract review date - Modern selector
            review_date = review_selector.css('div.c-siteReview_reviewDate::text').get()
            
            if not review_date:
                # Try finding any element with 'Date' or 'date' in class
                review_date = review_selector.css('[class*="Date"]::text').get() or \
                            review_selector.css('[class*="date"]::text').get()
            
            if not review_date:
                # Old structure fallbacks
                review_date = review_selector.css('div.date::text').get() or \
                            review_selector.css('span.date::text').get() or \
                            review_selector.css('time::attr(datetime)').get()
            
            if review_date:
                review_date = review_date.strip()
            
            # Generate unique review ID
            review_id = f"{self.game_info.get('game_title', 'unknown')}_{reviewer_name or 'anonymous'}_{review_date or 'no-date'}"
            review_id = re.sub(r'[^\w\-]', '_', review_id)
            
            # Create item
            item = GameReviewItem()
            
            # Game information
            item['game_title'] = self.game_info.get('game_title')
            item['game_platform'] = review_platform  # Use review-specific platform instead of game-level platform
            item['game_url'] = self.game_info.get('game_url')
            item['game_metascore'] = self.game_info.get('game_metascore')
            item['game_release_date'] = self.game_info.get('game_release_date')
            item['game_genre'] = self.game_info.get('game_genre')
            item['game_developer'] = self.game_info.get('game_developer')
            item['game_publisher'] = self.game_info.get('game_publisher')
            
            # Review information
            item['review_id'] = review_id
            item['review_text'] = review_text
            item['review_score'] = score
            item['review_category'] = category
            item['reviewer_name'] = reviewer_name
            item['review_date'] = review_date
            item['review_url'] = response.url
            
            # Metadata
            item['scraped_at'] = datetime.now().isoformat()
            
            return item
            
        except Exception as e:
            self.logger.error(f"Error parsing review: {e}")
            return None
    
    def handle_error(self, failure):
        """Handle request failures"""
        self.logger.error(f"Request failed: {failure.request.url}")
        self.logger.error(f"Error: {failure.value}")
        
        # Try to provide helpful error messages
        if "TimeoutError" in str(failure.type) or "TCPTimedOutError" in str(failure.type):
            self.logger.warning("Connection timeout - This could be due to:")
            self.logger.warning("  1. Network connectivity issues")
            self.logger.warning("  2. Metacritic blocking the requests")
            self.logger.warning("  3. The game URL might not exist")
            self.logger.warning("Try:")
            self.logger.warning("  - Checking your internet connection")
            self.logger.warning("  - Verifying the game name and platform are correct")
            self.logger.warning("  - Increasing DOWNLOAD_DELAY in settings.py")
            self.logger.warning("  - Trying a different game first")
