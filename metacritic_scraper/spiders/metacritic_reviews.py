"""
Metacritic Game Reviews Spider (with Playwright for JavaScript rendering)

This spider scrapes user reviews from Metacritic for video games.
It collects review text, scores, sentiment categories, and metadata.

Uses Scrapy-Playwright to render JavaScript and load all reviews dynamically.

Usage:
    scrapy crawl metacritic_reviews -a game_url="https://www.metacritic.com/game/pc/..."
    scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4"
    scrapy crawl metacritic_reviews -a game_name="the-last-of-us-part-ii" -a platform="playstation-4" -a max_reviews=100
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
    
    def __init__(self, game_url=None, game_name=None, platform=None, max_reviews=None, *args, **kwargs):
        super(MetacriticReviewsSpider, self).__init__(*args, **kwargs)
        
        self.max_reviews = int(max_reviews) if max_reviews else None
        self.reviews_count = 0
        
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
        # Metacritic structure: game page -> user-reviews section
        user_reviews_url = response.url.rstrip('/') + '/user-reviews'
        
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
        
        Uses Playwright to scroll and load all reviews dynamically
        """
        
        self.logger.info(f"Parsing reviews page: {response.url}")
        
        # Get the Playwright page object
        page = response.meta.get("playwright_page")
        
        if page:
            try:
                # Wait for reviews to load
                await page.wait_for_selector('div.c-siteReview, div.review_content', timeout=10000)
                self.logger.info("✓ Initial reviews loaded")
                
                # Scroll and load more reviews
                previous_count = 0
                scroll_attempts = 0
                max_scroll_attempts = 50  # Prevent infinite loops
                
                while scroll_attempts < max_scroll_attempts:
                    # Count current reviews
                    current_reviews = await page.query_selector_all('div.c-siteReview, div.review_content')
                    current_count = len(current_reviews)
                    
                    self.logger.info(f"📊 Currently loaded: {current_count} reviews")
                    
                    # Check if we've reached max_reviews limit
                    if self.max_reviews and current_count >= self.max_reviews:
                        self.logger.info(f"✓ Reached target: {self.max_reviews} reviews")
                        break
                    
                    # Check if no new reviews loaded
                    if current_count == previous_count:
                        self.logger.info("✓ No more reviews to load")
                        break
                    
                    previous_count = current_count
                    
                    # Scroll to bottom to trigger loading more reviews
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)  # Wait for new reviews to load
                    
                    # Look for and click "Load More" button if exists
                    try:
                        load_more_button = await page.query_selector('button:has-text("Load More"), a:has-text("Load More"), button.load-more')
                        if load_more_button:
                            self.logger.info("🔄 Clicking 'Load More' button")
                            await load_more_button.click()
                            await page.wait_for_timeout(2000)
                    except:
                        pass  # No load more button, that's fine
                    
                    scroll_attempts += 1
                
                self.logger.info(f"✓ Finished loading reviews. Total: {previous_count}")
                
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
                self.logger.error(f"Error during Playwright interaction: {e}")
                if page:
                    await page.close()
                final_response = response
        else:
            final_response = response
        
        # Find all review containers
        reviews = final_response.css('div.review_content, div.c-siteReview')
        
        if not reviews:
            self.logger.warning(f"No reviews found on page: {response.url}")
            reviews = final_response.css('div.user_review')
        
        self.logger.info(f"Found {len(reviews)} reviews to parse")
        
        # Parse each review
        for review in reviews:
            if self.max_reviews and self.reviews_count >= self.max_reviews:
                self.logger.info(f"Reached maximum reviews limit: {self.max_reviews}")
                return
            
            item = self.parse_single_review(review, final_response)
            if item:
                self.reviews_count += 1
                yield item
        
        self.logger.info(f"✓ Scraping complete. Total reviews collected: {self.reviews_count}")
    
    def parse_single_review(self, review_selector, response):
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
            item['game_platform'] = self.game_info.get('game_platform')
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
